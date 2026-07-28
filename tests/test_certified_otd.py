from __future__ import annotations

import itertools
import json
import asyncio
import hashlib
import math
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dynamix_trace2skill.certified_otd_pipeline as certified_otd_pipeline
import dynamix_trace2skill.skillbank as skillbank_module
from dynamix_core.certified_otd import (
    ExperienceAtom,
    OtdTreeState,
    audit_observed_beta_separation,
    select_budgeted_antichain,
)
from dynamix_trace2skill.certified_otd_pipeline import (
    ATOM_SCHEMA,
    CertifiedOtdConfig,
    ExperienceAtomAnalyst,
    LocalParentSkillAnalyst,
    _atom_leakage_reasons,
    _atom_protocol_fingerprint,
    _build as _build_certified_otd,
    _cautious_failure_diagnostic,
    _failure_atom_semantic_reasons,
    _load_atom_cache,
    _nodebank_manifest,
    _resolve_skill_output_dir,
    _require_complete_atom_identity,
    _require_unique_record_ids,
    _records_fingerprint,
    _required_string,
    _spreadsheet_reference_audit,
    _validate_source_build_output as _validate_cdost_source_build_output,
    _write_vector_cache_manifest,
)
from dynamix_trace2skill.clients import (
    EmbeddingClient,
    EmbeddingConfig,
    GenerationClient,
    GenerationConfig,
    _SqliteEmbeddingCache,
    embedding_vector_sha256,
)
from dynamix_trace2skill.pipeline import (
    DynaMixRunConfig,
    build_tree_from_records,
)
from dynamix_trace2skill.schemas import RawTrajectoryRecord, TrajectoryStep
from dynamix_trace2skill.skillbank import SkillBankSelector, SkillNodeDocument
from dynamix_trace2skill.tokenization import RegexTokenizer
from dynamix_trace2skill.trace_views import render_compact_analysis_bundle_text


def _atom(
    atom_id: str,
    trigger: tuple[float, ...],
    procedure: tuple[float, ...],
) -> ExperienceAtom:
    return ExperienceAtom(
        atom_id=atom_id,
        source_item_id=f"source-{atom_id}",
        trigger=f"trigger {atom_id}",
        scope="matching tasks",
        decision=f"decision {atom_id}",
        invariant=f"invariant {atom_id}",
        verification="verify the resulting artifact",
        failure_mode="do not apply outside the stated scope",
        evidence_type="single_trace",
        reliability=1.0,
        trigger_embedding=trigger,
        procedure_embedding=procedure,
    )


def _atoms() -> list[ExperienceAtom]:
    return [
        _atom("a", (1.0, 0.0), (1.0, 0.0)),
        _atom("b", (0.9, 0.1), (0.8, 0.2)),
        _atom("c", (0.0, 1.0), (0.1, 0.9)),
        _atom("d", (0.1, 0.9), (0.2, 0.8)),
        _atom("e", (-1.0, 0.0), (-0.8, 0.2)),
    ]


def _build(atoms: list[ExperienceAtom]) -> OtdTreeState:
    state = OtdTreeState(dual_view_lambda=0.6)
    for atom in atoms:
        state.insert(atom)
    return state


def _write_authoritative_tree(
    state: OtdTreeState,
    skill_dir: Path,
) -> dict[str, object]:
    tree_root = skill_dir.parent
    tree_root.mkdir(parents=True, exist_ok=True)
    skill_dir.mkdir(parents=True, exist_ok=True)
    state_text = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    structure_text = json.dumps(
        state.to_structural_dict(),
        ensure_ascii=False,
        indent=2,
    )
    (tree_root / "otd_tree_state.json").write_text(
        state_text,
        encoding="utf-8",
    )
    (tree_root / "otd_tree_structure.json").write_text(
        structure_text,
        encoding="utf-8",
    )
    return {
        "state_relative_path": "../otd_tree_state.json",
        "state_sha256": hashlib.sha256(state_text.encode("utf-8")).hexdigest(),
        "structure_relative_path": "../otd_tree_structure.json",
        "structure_sha256": hashlib.sha256(
            structure_text.encode("utf-8")
        ).hexdigest(),
        "structural_node_count": len(state.nodes),
    }


def _brute_within(state: OtdTreeState, node_id: str) -> float:
    atom_ids = state.descendant_atom_ids(node_id)
    return sum(
        state.pair_similarity(state.atoms[left], state.atoms[right])
        for left, right in itertools.combinations(atom_ids, 2)
    )


def _brute_cross(state: OtdTreeState, left_id: str, right_id: str) -> float:
    return sum(
        state.pair_similarity(state.atoms[left], state.atoms[right])
        for left in state.descendant_atom_ids(left_id)
        for right in state.descendant_atom_ids(right_id)
    )


def test_exact_sufficient_statistics_match_brute_force() -> None:
    state = _build(_atoms())
    for node in state.nodes.values():
        assert node.within_similarity == pytest.approx(
            _brute_within(state, node.node_id),
            abs=1.0e-9,
        )
        if not node.is_leaf:
            assert state.cross_similarity(
                str(node.left_id),
                str(node.right_id),
            ) == pytest.approx(
                _brute_cross(state, str(node.left_id), str(node.right_id)),
                abs=1.0e-9,
            )


def test_insert_preserves_single_parent_binary_tree() -> None:
    state = OtdTreeState()
    for index, atom in enumerate(_atoms(), start=1):
        result = state.insert(atom)
        state.validate()
        assert len(state.atoms) == index
        assert len(state.nodes) == 2 * index - 1
        assert result.root_node_id == state.root_node_id
        parent_counts = {node_id: 0 for node_id in state.nodes}
        for node in state.nodes.values():
            for child_id in node.child_ids:
                parent_counts[child_id] += 1
        assert parent_counts[str(state.root_node_id)] == 0
        assert all(
            count == 1
            for node_id, count in parent_counts.items()
            if node_id != state.root_node_id
        )


def test_static_and_prefix_dynamic_builds_are_structurally_identical() -> None:
    atoms = _atoms()
    static = _build(atoms)
    dynamic = _build(atoms[:2])
    prefix_snapshot = json.dumps(dynamic.to_dict(), sort_keys=True)
    assert prefix_snapshot == json.dumps(_build(atoms[:2]).to_dict(), sort_keys=True)
    for atom in atoms[2:]:
        dynamic.insert(atom)
    assert dynamic.to_dict() == static.to_dict()


def test_only_insertion_path_aggregates_change() -> None:
    state = _build(_atoms()[:4])
    before = state.to_dict()
    result = state.insert(_atoms()[4])
    changed = set(result.changed_internal_node_ids) | set(result.created_node_ids)
    for node_id, payload in before["nodes"].items():
        if node_id not in changed:
            current = state.nodes[node_id].to_dict()
            for field_name in (
                "leaf_count",
                "sum_trigger",
                "sum_procedure",
                "sum_trigger_squared_norms",
                "sum_procedure_squared_norms",
                "within_similarity",
                "skill",
                "retrievable",
                "structural_certificate",
            ):
                assert current[field_name] == payload[field_name]


def test_serialization_round_trip(tmp_path: Path) -> None:
    state = _build(_atoms())
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(state.to_dict()), encoding="utf-8")
    restored = OtdTreeState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert restored.to_dict() == state.to_dict()
    assert restored.moseley_wang_revenue() == pytest.approx(
        state.moseley_wang_revenue()
    )


def test_experience_atom_metadata_is_deeply_immutable() -> None:
    atom = ExperienceAtom.from_dict(
        _atom("immutable", (1.0, 0.0), (1.0, 0.0)).to_dict()
        | {"metadata": {"nested": {"values": [1, 2]}}}
    )
    with pytest.raises(TypeError):
        atom.metadata["new"] = "value"
    with pytest.raises(TypeError):
        atom.metadata["nested"]["new"] = "value"
    assert atom.metadata["nested"]["values"] == (1, 2)
    assert atom.to_dict()["metadata"]["nested"]["values"] == [1, 2]


def test_deterministic_tie_break_uses_subtree_key() -> None:
    atoms = [
        _atom("b", (1.0, 0.0), (1.0, 0.0)),
        _atom("a", (-1.0, 0.0), (-1.0, 0.0)),
        _atom("c", (0.0, 1.0), (0.0, 1.0)),
    ]
    first = _build(atoms)
    second = _build(atoms)
    assert first.to_dict() == second.to_dict()


def test_child_routing_uses_average_not_raw_cross_similarity() -> None:
    state = OtdTreeState(dual_view_lambda=0.5)
    state.insert(_atom("left-a", (1.0, 0.0), (1.0, 0.0)))
    state.insert(_atom("left-b", (1.0, 0.0), (1.0, 0.0)))
    state.insert(_atom("right", (-1.0, 0.0), (-1.0, 0.0)))

    root = state.nodes[str(state.root_node_id)]
    original_left_id = str(root.left_id)
    original_right_id = str(root.right_id)
    child_counts = {
        child_id: state.nodes[child_id].leaf_count
        for child_id in root.child_ids
    }
    large_child = next(
        child_id for child_id, count in child_counts.items() if count == 2
    )
    singleton_child = next(
        child_id for child_id, count in child_counts.items() if count == 1
    )
    value = math.sqrt(0.96)
    result = state.insert(
        _atom("query", (-0.2, value), (-0.2, value))
    )

    root_decision = result.decisions[0]
    assert root_decision.action == "recurse"
    assert root_decision.selected_child_id == singleton_child
    assert root_decision.within_similarity_sum / 3.0 == pytest.approx(
        root_decision.within_similarity
    )
    assert root_decision.cross_similarity_sum / 3.0 == pytest.approx(
        root_decision.cross_similarity
    )
    assert (
        state.cross_similarity(large_child, result.leaf_node_id)
        > state.cross_similarity(singleton_child, result.leaf_node_id)
    )
    assert (
        root_decision.left_cross_similarity
        if original_left_id == singleton_child
        else root_decision.right_cross_similarity
    ) == pytest.approx(0.6)


def test_validation_rejects_tampered_squared_norm_statistics() -> None:
    state = _build(_atoms()[:3])
    root = state.nodes[str(state.root_node_id)]
    root.sum_trigger_squared_norms += 0.25
    with pytest.raises(ValueError, match="trigger squared norms"):
        state.validate()


def test_nonzero_tie_epsilon_disables_theorem_claim() -> None:
    diagnostics = OtdTreeState(tie_epsilon=1.0e-6).structural_diagnostics()
    assert diagnostics["exact_otd_comparisons"] is False
    assert diagnostics["theorem_claim_enabled"] is False
    assert diagnostics["guarantee_scope"].startswith("disabled")
    assert diagnostics["similarity_boundary_tolerance"] == pytest.approx(
        1.0e-10
    )


def test_near_unit_embeddings_use_one_canonical_similarity_kernel() -> None:
    atom = _atom(
        "near-unit",
        (1.0 + 5.0e-13, 0.0),
        (0.0, 1.0 + 5.0e-13),
    )
    assert np.linalg.norm(atom.trigger_embedding) == pytest.approx(1.0, abs=1.0e-15)
    assert np.linalg.norm(atom.procedure_embedding) == pytest.approx(1.0, abs=1.0e-15)

    state = _build(
        [
            atom,
            _atom("same", (1.0, 0.0), (0.0, 1.0)),
            _atom("other", (0.0, 1.0), (1.0, 0.0)),
        ]
    )
    for node in state.nodes.values():
        assert node.within_similarity == pytest.approx(
            _brute_within(state, node.node_id),
            abs=1.0e-12,
        )


def test_structural_configuration_is_immutable_after_construction() -> None:
    state = OtdTreeState(dual_view_lambda=0.7, tie_epsilon=0.0)
    with pytest.raises(AttributeError):
        state.dual_view_lambda = 0.2
    with pytest.raises(AttributeError):
        state.tie_epsilon = 0.1


def test_insert_does_not_run_full_tree_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = OtdTreeState()

    def fail_if_called() -> None:
        raise AssertionError("insert must not traverse the full tree")

    monkeypatch.setattr(state, "validate", fail_if_called)
    for atom in _atoms():
        state.insert(atom)


def test_observed_beta_audit_replays_exact_arrival_assumption() -> None:
    atoms = [
        _atom("a", (1.0, 0.0), (1.0, 0.0)),
        _atom("b", (-1.0, 0.0), (-1.0, 0.0)),
        _atom("c", (0.0, 1.0), (0.0, 1.0)),
    ]
    diagnostic = audit_observed_beta_separation(
        atoms,
        dual_view_lambda=0.6,
    )
    assert diagnostic["scope"] == "observed_arrivals_only"
    assert diagnostic["constraint_count"] == 2
    assert diagnostic["observed_beta"] == 0.0
    assert diagnostic["observed_beta_over_3"] == 0.0
    assert diagnostic["theorem_claim_enabled"] is False
    assert diagnostic["future_arrivals_certified"] is False
    assert diagnostic["similarity_boundary_tolerance"] == pytest.approx(
        1.0e-10
    )
    assert diagnostic["worst_constraint"]["arrival_atom_id"] == "c"
    assert (
        diagnostic["worst_constraint"]["subtree_to_arrival_similarity"]
        > diagnostic["worst_constraint"]["subtree_within_similarity"]
    )


def test_observed_beta_audit_ignores_subtrees_outside_theorem_antecedent() -> None:
    atoms = [
        _atom("a", (1.0, 0.0), (1.0, 0.0)),
        _atom("b", (1.0, 0.0), (1.0, 0.0)),
        _atom("c", (1.0, 0.0), (1.0, 0.0)),
    ]
    diagnostic = audit_observed_beta_separation(atoms)
    assert diagnostic["diagnostic"] == "observed_arrival_beta_separation_v3"
    assert diagnostic["antecedent_count"] == 0
    assert diagnostic["constraint_count"] == 0
    assert diagnostic["vacuous"] is True
    assert diagnostic["observed_beta"] == 1.0
    assert diagnostic["assumption_satisfied_on_observed_stream"] is True
    assert (
        diagnostic["assumption_status"]
        == "satisfied_vacuously_on_observed_stream"
    )
    assert diagnostic["theorem_claim_enabled"] is False


def test_observed_beta_audit_distinguishes_active_zero_rhs_constraint() -> None:
    atoms = [
        _atom("a", (1.0, 0.0), (1.0, 0.0)),
        _atom("b", (-1.0, 0.0), (-1.0, 0.0)),
        _atom("c", (-1.0, 0.0), (-1.0, 0.0)),
    ]
    diagnostic = audit_observed_beta_separation(atoms)
    assert diagnostic["antecedent_count"] == 1
    assert diagnostic["constraint_count"] == 0
    assert diagnostic["zero_rhs_constraint_count"] == 1
    assert diagnostic["vacuous"] is False
    assert diagnostic["assumption_status"] == (
        "satisfied_trivially_zero_rhs_on_observed_stream"
    )
    assert diagnostic["theorem_claim_enabled"] is True


def test_cdost_library_entry_requires_dataset_order_and_dynamic_cache() -> None:
    class _Dynamic:
        resume_from_snapshots = False
        shuffle_seed = None
        snapshot_include_embeddings = True

    class _Config:
        hierarchy = {"otd": {}}
        dynamic = _Dynamic()
        enforce_dataset_order = False

    with pytest.raises(ValueError, match="enforce_dataset_order=true"):
        asyncio.run(_build_certified_otd(_Config(), dynamic=False))

    _Config.enforce_dataset_order = True
    _Dynamic.shuffle_seed = 42
    with pytest.raises(ValueError, match="dataset-order arrivals"):
        asyncio.run(_build_certified_otd(_Config(), dynamic=False))
    _Dynamic.shuffle_seed = None
    _Dynamic.snapshot_include_embeddings = False
    with pytest.raises(ValueError, match="snapshot_include_embeddings=true"):
        asyncio.run(_build_certified_otd(_Config(), dynamic=False))
    _Dynamic.snapshot_include_embeddings = True
    with pytest.raises(ValueError, match="frozen atom_cache_path"):
        asyncio.run(_build_certified_otd(_Config(), dynamic=True))


def test_library_entry_enforces_policy_specific_cache_writes() -> None:
    class _Dynamic:
        resume_from_snapshots = False
        shuffle_seed = None
        snapshot_include_embeddings = True

    class _Embedding:
        cache_path = "/tmp/cdost-cache-policy.sqlite"
        cache_write_policy = "replace"

    class _Config:
        hierarchy = {"otd": {}}
        dynamic = _Dynamic()
        embedding = _Embedding()
        enforce_dataset_order = True

    with pytest.raises(ValueError, match="first_write_wins"):
        asyncio.run(_build_certified_otd(_Config(), dynamic=False))

    legacy = DynaMixRunConfig(
        output_dir="/tmp/legacy-cache-policy",
        records_path="/tmp/unused-records.json",
        embedding=EmbeddingConfig(
            cache_write_policy="first_write_wins",
        ),
        hierarchy={"tree_policy": "projected_gmm_bic"},
    )
    with pytest.raises(ValueError, match="cache_write_policy='replace'"):
        asyncio.run(build_tree_from_records(legacy))


def test_library_source_artifact_must_match_build_marker(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "static"
    tree = source_run / "dynamix_tree"
    marker_dir = source_run / "stage_markers"
    tree.mkdir(parents=True)
    marker_dir.mkdir()
    artifact = tree / "experience_atoms.json"
    artifact.write_text('{"atoms": []}', encoding="utf-8")
    identity = {
        "exists": True,
        "kind": "file",
        "size": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    (marker_dir / "04_build_tree.done").write_text(
        json.dumps({"output_identities": {str(artifact.resolve()): identity}}),
        encoding="utf-8",
    )
    _validate_cdost_source_build_output(artifact)

    artifact.write_text('{"atoms": [{"tampered": true}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="completed stage marker"):
        _validate_cdost_source_build_output(artifact)


def test_node_vector_manifest_binds_the_vector_used_by_index(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "vectors.sqlite"
    embedding = EmbeddingConfig(
        base_url="mock://embedding",
        model="mock-embedding",
        cache_path=str(cache_path),
        cache_write_policy="first_write_wins",
    )
    protocol = {
        "max_model_len": 32000,
        "max_input_tokens": 32000,
        "batch_size": 1,
        "tokenizer_model": "",
        "input_policy": "single_vector_fail_if_over_limit",
        "chunking_active": False,
        "vector_cache_write_policy": "first_write_wins",
        "scoring_vector_policy": (
            "persisted_unit_vector_no_reload_renormalization"
        ),
    }
    namespace = skillbank_module.skillbank_vector_cache_namespace(
        base_url=embedding.base_url,
        model=embedding.model,
        api_key_fingerprint_value="",
        embedding_protocol=protocol,
    )
    cache = _SqliteEmbeddingCache(
        cache_path,
        write_policy="first_write_wins",
    )
    cache.set(namespace, "name: node\ntrigger: x\ncontent: y", [1.0, 0.0])
    cache.close()
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "format": "dynamix_skillbank_embedding_index_v2",
                "base_url": embedding.base_url,
                "model": embedding.model,
                "api_key_fingerprint": "",
                "embedding_protocol": protocol,
                "documents": [
                    {
                        "node_id": "node",
                        "embedding_text": (
                            "name: node\ntrigger: x\ncontent: y"
                        ),
                    }
                ],
                "embeddings": [[0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="normalized embedding cache vector"):
        _write_vector_cache_manifest(
            out=tmp_path,
            config=SimpleNamespace(embedding=embedding),
            atoms=[],
            skillbank_index_path=index,
        )


def test_cdost_skill_output_dir_stays_within_tree_output(
    tmp_path: Path,
) -> None:
    assert _resolve_skill_output_dir(tmp_path, "skills") == (
        tmp_path / "skills"
    ).resolve()
    with pytest.raises(ValueError, match="relative path"):
        _resolve_skill_output_dir(tmp_path, "../escaped")
    with pytest.raises(ValueError, match="relative path"):
        _resolve_skill_output_dir(tmp_path, str(tmp_path / "absolute"))


def _is_ancestor(
    ancestor: str,
    descendant: str,
    children: dict[str, tuple[str, ...]],
) -> bool:
    stack = list(children.get(ancestor, ()))
    while stack:
        current = stack.pop()
        if current == descendant:
            return True
        stack.extend(children.get(current, ()))
    return False


def test_antichain_dp_matches_exhaustive_optimum() -> None:
    children = {
        "root": ("left", "right"),
        "left": ("a", "b"),
        "right": ("c", "d"),
    }
    relevance = {
        "root": 0.95,
        "left": 0.80,
        "right": 0.75,
        "a": 0.60,
        "b": 0.55,
        "c": 0.70,
        "d": 0.40,
    }
    costs = {
        "root": 250,
        "left": 120,
        "right": 110,
        "a": 60,
        "b": 60,
        "c": 70,
        "d": 50,
    }
    result = select_budgeted_antichain(
        root_node_id="root",
        children_by_node=children,
        relevance_by_node=relevance,
        token_cost_by_node=costs,
        max_nodes=3,
        token_budget=180,
        token_unit=10,
    )

    best_score = -1.0
    best_cost = 0
    best_ids: tuple[str, ...] = ()
    node_ids = tuple(relevance)
    for size in range(4):
        for subset in itertools.combinations(node_ids, size):
            if any(
                _is_ancestor(left, right, children)
                or _is_ancestor(right, left, children)
                for left, right in itertools.combinations(subset, 2)
            ):
                continue
            cost = sum(costs[node_id] for node_id in subset)
            if cost > 180:
                continue
            score = sum(relevance[node_id] for node_id in subset)
            key = (score, -cost, tuple(reversed(tuple(sorted(subset)))))
            best_key = (best_score, -best_cost, tuple(reversed(best_ids)))
            if key > best_key:
                best_score = score
                best_cost = cost
                best_ids = tuple(sorted(subset))

    assert result.node_ids == best_ids
    assert result.score == pytest.approx(best_score)
    assert result.token_cost == best_cost
    assert not any(
        _is_ancestor(left, right, children)
        or _is_ancestor(right, left, children)
        for left, right in itertools.combinations(result.node_ids, 2)
    )


def test_antichain_dp_uses_complete_rendered_selection_cost() -> None:
    result = select_budgeted_antichain(
        root_node_id="root",
        children_by_node={
            "root": ("left", "right"),
            "left": (),
            "right": (),
        },
        relevance_by_node={"left": 1.0, "right": 0.9},
        token_cost_by_node={"left": 1, "right": 1},
        max_nodes=2,
        token_budget=10,
        token_unit=1,
        total_cost_for_nodes=lambda node_ids: (
            12 if len(node_ids) == 2 else 6
        ),
    )
    assert result.node_ids == ("left",)
    assert result.token_cost == 6


def test_nonadditive_antichain_search_does_not_drop_later_feasible_pair() -> None:
    costs = {
        frozenset(): 0,
        frozenset({"a1"}): 5,
        frozenset({"a2"}): 5,
        frozenset({"b"}): 5,
        frozenset({"a1", "b"}): 11,
        frozenset({"a2", "b"}): 10,
        frozenset({"a1", "a2"}): 11,
    }
    result = select_budgeted_antichain(
        root_node_id="root",
        children_by_node={
            "root": ("x", "b"),
            "x": ("a1", "a2"),
            "a1": (),
            "a2": (),
            "b": (),
        },
        relevance_by_node={"a1": 10.0, "a2": 9.0, "b": 10.0},
        token_cost_by_node={"a1": 5, "a2": 5, "b": 5},
        max_nodes=2,
        token_budget=10,
        token_unit=1,
        total_cost_for_nodes=lambda node_ids: costs[frozenset(node_ids)],
    )

    assert result.node_ids == ("a2", "b")
    assert result.score == pytest.approx(19.0)
    assert result.token_cost == 10


def test_nonadditive_antichain_search_matches_exhaustive_random_objectives() -> None:
    children = {
        "root": ("left", "right"),
        "left": ("a", "b"),
        "right": ("c", "d"),
        "a": (),
        "b": (),
        "c": (),
        "d": (),
    }
    node_ids = tuple(children)
    feasible_subsets = [
        tuple(sorted(subset))
        for size in range(4)
        for subset in itertools.combinations(node_ids, size)
        if not any(
            _is_ancestor(left, right, children)
            or _is_ancestor(right, left, children)
            for left, right in itertools.combinations(subset, 2)
        )
    ]

    for seed in range(100):
        rng = random.Random(seed)
        relevance = {
            node_id: float(rng.randint(0, 4))
            for node_id in node_ids
        }
        rendered_costs = {
            subset: (
                0
                if not subset
                else rng.randint(1, 16)
            )
            for subset in feasible_subsets
        }
        result = select_budgeted_antichain(
            root_node_id="root",
            children_by_node=children,
            relevance_by_node=relevance,
            token_cost_by_node={node_id: 1 for node_id in node_ids},
            max_nodes=3,
            token_budget=10,
            token_unit=1,
            total_cost_for_nodes=lambda selected, costs=rendered_costs: (
                costs[tuple(sorted(selected))]
            ),
        )

        candidates = []
        for subset in feasible_subsets:
            cost = rendered_costs[subset]
            if cost > 10:
                continue
            score = sum(relevance[node_id] for node_id in subset)
            candidates.append((score, subset, cost))
        expected = max(
            candidates,
            key=lambda candidate: (
                candidate[0],
                -candidate[2],
                tuple(reversed(candidate[1])),
            ),
        )
        assert result.node_ids == expected[1]
        assert result.score == pytest.approx(expected[0])
        assert result.token_cost == expected[2]


def test_nonadditive_budget_uses_exact_tokens_not_discretized_units() -> None:
    result = select_budgeted_antichain(
        root_node_id="root",
        children_by_node={"root": ()},
        relevance_by_node={"root": 1.0},
        token_cost_by_node={"root": 129},
        max_nodes=1,
        token_budget=129,
        token_unit=128,
        total_cost_for_nodes=lambda node_ids: 129 if node_ids else 0,
    )
    assert result.node_ids == ("root",)
    assert result.token_cost == 129


def test_nonadditive_tied_relevance_prefers_lower_complete_prompt_cost() -> None:
    result = select_budgeted_antichain(
        root_node_id="root",
        children_by_node={
            "root": ("left", "right"),
            "left": (),
            "right": (),
        },
        relevance_by_node={
            "root": 2.0,
            "left": 1.0,
            "right": 1.0,
        },
        token_cost_by_node={
            "root": 10,
            "left": 4,
            "right": 4,
        },
        max_nodes=2,
        token_budget=10,
        token_unit=1,
        total_cost_for_nodes=lambda node_ids: {
            (): 0,
            ("root",): 10,
            ("left",): 4,
            ("right",): 4,
            ("left", "right"): 8,
        }[tuple(sorted(node_ids))],
    )
    assert result.node_ids == ("left", "right")
    assert result.score == pytest.approx(2.0)
    assert result.token_cost == 8


def test_nonadditive_exact_search_fails_closed_at_declared_state_limit() -> None:
    leaves = tuple(f"leaf_{index:02d}" for index in range(18))
    children = {"root": leaves} | {leaf: () for leaf in leaves}
    with pytest.raises(RuntimeError, match="state limit"):
        select_budgeted_antichain(
            root_node_id="root",
            children_by_node=children,
            relevance_by_node={leaf: 1.0 for leaf in leaves},
            token_cost_by_node={leaf: 1 for leaf in leaves},
            max_nodes=9,
            token_budget=9,
            token_unit=1,
            total_cost_for_nodes=lambda node_ids: (
                10 if len(node_ids) == 9 else len(node_ids)
            ),
            max_exact_states=100,
        )


def test_additive_antichain_rejects_multi_parent_dag() -> None:
    with pytest.raises(ValueError, match="multiple structural parents"):
        select_budgeted_antichain(
            root_node_id="root",
            children_by_node={
                "root": ("left", "right"),
                "left": ("shared",),
                "right": ("shared",),
                "shared": (),
            },
            relevance_by_node={"shared": 1.0},
            token_cost_by_node={"shared": 1},
            max_nodes=2,
            token_budget=2,
            token_unit=1,
        )


def test_embeddings_are_normalized_and_similarity_is_nonnegative() -> None:
    left = _atom("left", (2.0, 0.0), (0.0, 3.0))
    right = _atom("right", (-4.0, 0.0), (0.0, -5.0))
    state = OtdTreeState()
    assert np.linalg.norm(left.trigger_embedding) == pytest.approx(1.0)
    assert np.linalg.norm(left.procedure_embedding) == pytest.approx(1.0)
    assert state.pair_similarity(left, right) == pytest.approx(0.0)


class _FakeGeneration:
    async def chat_json(self, messages, *, schema_name, **kwargs):
        atom = {
            "trigger": "when artifact state must be changed",
            "scope": "tasks with a verifiable output artifact",
            "decision": "inspect state, make the smallest valid change",
            "invariant": "preserve unrelated artifact content",
            "verification": "re-open and verify the requested state",
            "failure_mode": "do not infer success from a tool exit alone",
        }
        if schema_name == "CertifiedExperienceAtom":
            return atom
        if schema_name == "CertifiedFailureRootCauseAtom":
            return {
                "requested_semantics": "produce the requested artifact state",
                "performed_semantics": "the attempted change missed the requirement",
                "source_target_mapping": (
                    "the source state must be mapped to the requested output state"
                ),
                "operation_audit": (
                    "the attempted operation did not implement the requested change"
                ),
                "verifier_outcome": "authoritative recalculation rejected the result",
                "supported_root_cause": "the action did not satisfy the requested state",
                "atom": atom,
            }
        if schema_name == "CertifiedFailureAtomReview":
            return {
                "verdict": "supported_root_cause",
                "supporting_step_ids": [0],
                "rationale": "Recalculation rejected the claimed completion.",
                "atom": {
                    "trigger": "when artifact state must be corrected",
                    "scope": "tasks with a verifiable output artifact",
                    "decision": "diagnose the mismatch before changing the artifact",
                    "invariant": "preserve unrelated artifact content",
                    "verification": "re-open and verify the requested state",
                    "failure_mode": "do not repeat an action rejected by recalculation",
                },
            }
        if schema_name == "CertifiedOtdParentSkill":
            return {
                "name": "Verified artifact editing",
                "trigger": "when a task changes a structured artifact",
                "content": "Preserve unrelated state and verify the saved result.",
            }
        raise AssertionError(schema_name)


class _LeakThenRepairGeneration(_FakeGeneration):
    def __init__(self, *, always_leak: bool = False) -> None:
        self.calls = 0
        self.always_leak = always_leak
        self.requests = []

    async def chat_json(self, messages, *, schema_name, **kwargs):
        self.calls += 1
        self.requests.append((list(messages), dict(kwargs)))
        if schema_name == "CertifiedExperienceAtom" and (
            self.always_leak or self.calls == 1
        ):
            return {
                "trigger": "when editing E6:E13 in report.xlsx",
                "scope": "the task at /tmp/report.xlsx",
                "decision": "write the value 2602",
                "invariant": "preserve unrelated artifact content",
                "verification": "re-open and verify the requested state",
                "failure_mode": "do not infer success from a tool exit alone",
            }
        return await super().chat_json(
            messages,
            schema_name=schema_name,
            **kwargs,
        )


class _FakeEmbedding:
    truncation_events = []

    async def embed_texts(self, texts, *, cache_namespace):
        if "trigger" in cache_namespace:
            return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]
        return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]


class _CoordinateGeneration(_FakeGeneration):
    async def chat_json(self, messages, *, schema_name, **kwargs):
        if schema_name == "CertifiedExperienceAtom":
            return {
                "trigger": "when formulas must follow the target row",
                "scope": "spreadsheet formula generation",
                "decision": "replace fixed H2 and A1 references with relative references",
                "invariant": "each formula uses inputs from its own row",
                "verification": "inspect every generated formula",
                "failure_mode": "fixed coordinates break copied formulas",
            }
        return await super().chat_json(
            messages,
            schema_name=schema_name,
            **kwargs,
        )


class _UnsafeGeneration(_FakeGeneration):
    async def chat_json(self, messages, *, schema_name, **kwargs):
        if schema_name in {
            "CertifiedExperienceAtom",
            "CertifiedFailureRootCauseAtom",
        }:
            atom = {
                "trigger": "when editing an artifact",
                "scope": "https://example.invalid/private",
                "decision": "make a change",
                "invariant": "preserve unrelated content",
                "verification": "inspect the result",
                "failure_mode": "do not infer success",
            }
            if schema_name == "CertifiedFailureRootCauseAtom":
                return {
                    "requested_semantics": "change the artifact",
                    "performed_semantics": "an unsafe output was proposed",
                    "verifier_outcome": "the result failed",
                    "supported_root_cause": "the output is unsafe",
                    "atom": atom,
                }
            return atom
        return await super().chat_json(
            messages,
            schema_name=schema_name,
            **kwargs,
        )


class _FailureReviewGeneration(_FakeGeneration):
    def __init__(self) -> None:
        self.requests = []

    async def chat_json(self, messages, *, schema_name, **kwargs):
        self.requests.append((schema_name, list(messages)))
        return await super().chat_json(
            messages,
            schema_name=schema_name,
            **kwargs,
        )


def test_atom_extraction_and_local_parent_update_contract() -> None:
    asyncio.run(_atom_extraction_and_local_parent_update_contract())


async def _atom_extraction_and_local_parent_update_contract() -> None:
    records = [
        RawTrajectoryRecord(
            trajectory_id=f"trajectory-{index}",
            task_id=f"task-{index}",
            trial_index=0,
            instruction="Modify the artifact and verify the result.",
            success=bool(index),
            verifier_score=float(index),
        )
        for index in range(2)
    ]
    atom_analyst = ExperienceAtomAnalyst(
        _FakeGeneration(),
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=60000,
    )
    atoms, excluded = await atom_analyst.extract_many(records)
    assert not excluded
    assert len(atoms) == 2
    assert all(atom.embedding_dimension == 2 for atom in atoms)
    assert all(atom.evidence_type == "single_trace" for atom in atoms)
    assert atoms[0].reliability == pytest.approx(0.0)
    assert atoms[1].reliability == pytest.approx(1.0)

    state = OtdTreeState()
    state.insert(atoms[0])
    insertion = state.insert(atoms[1])
    events = await LocalParentSkillAnalyst(
        _FakeGeneration(),
        validation_mode="structural_only",
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
    ).refresh(state, insertion.changed_internal_node_ids)
    assert events == [
        {
            "node_id": state.root_node_id,
            "status": "accepted",
            "structural_certificate": {
                "passed": True,
                "descendant_atom_count": 2,
                "duplicate_child_ids": [],
                "provenance_complete": True,
                "behavioral_replay_performed": False,
            },
        }
    ]
    assert state.nodes[str(state.root_node_id)].retrievable


def test_parent_certificate_computes_provenance_instead_of_assuming_it() -> None:
    async def run() -> None:
        state = OtdTreeState()
        state.insert(_atom("provenance-a", (1.0, 0.0), (1.0, 0.0)))
        insertion = state.insert(
            _atom("provenance-b", (0.9, 0.1), (0.9, 0.1))
        )
        state.atoms.pop("provenance-b")
        events = await LocalParentSkillAnalyst(
            _FakeGeneration(),
            validation_mode="structural_only",
            tokenizer=RegexTokenizer(),
            max_prompt_tokens=10000,
            max_output_tokens=4096,
        ).refresh(state, insertion.changed_internal_node_ids)
        certificate = events[0]["structural_certificate"]
        assert events[0]["status"] == "structural_reject"
        assert certificate["provenance_complete"] is False
        assert not state.nodes[str(state.root_node_id)].retrievable

    asyncio.run(run())


def test_atom_rejects_nonfinite_verifier_evidence() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-nonfinite",
        task_id="task-nonfinite",
        trial_index=0,
        instruction="Modify the artifact.",
        verifier_score=float("nan"),
    )
    draft = {
        "trigger": "when modifying an artifact",
        "scope": "structured artifact tasks",
        "decision": "apply the requested operation",
        "invariant": "preserve unrelated content",
        "verification": "inspect the saved artifact",
        "failure_mode": "do not infer success without verification",
    }
    with pytest.raises(ValueError, match="verifier_score must be finite"):
        ExperienceAtomAnalyst._to_atom(
            record,
            draft,
            (1.0, 0.0),
            (1.0, 0.0),
        )


def test_atom_extraction_repairs_task_specific_leakage() -> None:
    generation = _LeakThenRepairGeneration()
    analyst = ExperienceAtomAnalyst(
        generation,
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=60000,
    )
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-sensitive",
        task_id="task-sensitive",
        trial_index=0,
        instruction="Modify E6:E13 in report.xlsx.",
        answer_position="E6:E13",
        spreadsheet_path="/tmp/report.xlsx",
        success=True,
        verifier_score=1.0,
        extra={"expected_answer": "2602"},
    )
    atoms, excluded = asyncio.run(analyst.extract_many([record]))
    assert generation.calls == 2
    assert len(atoms) == 1
    assert not excluded
    reusable_text = "\n".join(
        (
            atoms[0].trigger,
            atoms[0].scope,
            atoms[0].decision,
            atoms[0].invariant,
            atoms[0].verification,
            atoms[0].failure_mode,
        )
    )
    assert "E6:E13" not in reusable_text
    assert "report.xlsx" not in reusable_text
    assert "2602" not in reusable_text
    repair_prompt = generation.requests[1][0][-1]["content"]
    assert "E6:E13" not in repair_prompt
    assert "report.xlsx" not in repair_prompt
    assert "2602" not in repair_prompt
    assert all(
        request_kwargs["retries"] == 1
        for _, request_kwargs in generation.requests
    )
    system_prompt = generation.requests[0][0][0]["content"]
    assert "successful agent trajectory" in system_prompt
    assert "must generalize beyond this task" in system_prompt


def test_failed_atom_uses_auditable_fallback_after_leakage_retries() -> None:
    analyst = ExperienceAtomAnalyst(
        _UnsafeGeneration(),
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=60000,
    )
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-sensitive",
        task_id="task-sensitive",
        trial_index=0,
        instruction="Modify E6:E13 in report.xlsx.",
        answer_position="E6:E13",
        spreadsheet_path="/tmp/report.xlsx",
        success=False,
        verifier_score=0.0,
        extra={"expected_answer": "2602"},
    )
    atoms, excluded = asyncio.run(analyst.extract_many([record]))
    assert not excluded
    assert len(atoms) == 1
    assert atoms[0].metadata["analysis_mode"] == (
        "deterministic_cautious_fallback"
    )
    assert atoms[0].metadata["root_cause_review_attempts"] == 0
    assert atoms[0].metadata["root_cause_review_rejection_reasons"]


def test_atom_extraction_redacts_source_literals_after_bounded_retries() -> None:
    generation = _LeakThenRepairGeneration(always_leak=True)
    analyst = ExperienceAtomAnalyst(
        generation,
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=60000,
    )
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-sensitive",
        task_id="task-sensitive",
        trial_index=0,
        instruction="Modify E6:E13 in report.xlsx.",
        answer_position="E6:E13",
        spreadsheet_path="/tmp/report.xlsx",
        success=True,
        verifier_score=1.0,
        extra={"expected_answer": "2602"},
    )
    atoms, excluded = asyncio.run(analyst.extract_many([record]))
    reusable_text = "\n".join(
        (
            atoms[0].trigger,
            atoms[0].scope,
            atoms[0].decision,
            atoms[0].invariant,
            atoms[0].verification,
            atoms[0].failure_mode,
        )
    )
    assert generation.calls == 3
    assert not excluded
    assert "<redacted>" in reusable_text
    assert "E6:E13" not in reusable_text
    assert "report.xlsx" not in reusable_text
    assert "2602" not in reusable_text


def test_atom_extraction_preserves_cancellation() -> None:
    class CancelledGeneration:
        async def chat_json(self, messages, *, schema_name, **kwargs):
            raise asyncio.CancelledError()

    analyst = ExperienceAtomAnalyst(
        CancelledGeneration(),
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=60000,
    )
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-cancelled",
        task_id="task-cancelled",
        trial_index=0,
        instruction="Modify the artifact.",
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(analyst.extract_many([record]))


def test_failed_trajectory_requires_verifier_grounded_review() -> None:
    generation = _FailureReviewGeneration()
    analyst = ExperienceAtomAnalyst(
        generation,
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=60000,
    )
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-failed-review",
        task_id="task-failed-review",
        trial_index=0,
        instruction="Repair the artifact.",
        success=False,
        verifier_score=0.0,
        steps=[
            TrajectoryStep(
                step_id=0,
                raw_model_output="Inspect and edit the artifact.",
                action="write output",
                observation="saved output",
            )
        ],
        extra={
            "trace2skill_result": {
                "test_cases": [
                    {
                        "passed": False,
                        "message": "authoritative recalculation mismatch",
                        "raw_passed": True,
                        "raw_message": "stale cache passed",
                    }
                ]
            }
        },
    )
    atoms, excluded = asyncio.run(analyst.extract_many([record]))
    assert not excluded
    assert [schema for schema, _ in generation.requests] == [
        "CertifiedFailureRootCauseAtom",
        "CertifiedFailureAtomReview",
    ]
    failure_prompt = generation.requests[0][1][0]["content"]
    assert "LibreOffice-recalculated verifier result is authoritative" in failure_prompt
    assert "contrast the requested semantics" in failure_prompt
    assert "diagnose the mismatch" in atoms[0].decision
    assert atoms[0].metadata["analysis_mode"] == "supported_root_cause"
    assert atoms[0].metadata["root_cause_review_attempts"] == 1
    assert atoms[0].metadata["root_cause_review_rejection_reasons"] == ()
    assert atoms[0].metadata["root_cause_supporting_step_ids"] == (0,)
    review_prompt = generation.requests[1][1][-1]["content"]
    assert "authoritative recalculation mismatch" in review_prompt
    assert "stale cache passed" not in review_prompt
    assert "instruction_references_not_explicitly_used_by_actions" in review_prompt


def test_invalid_failure_review_uses_auditable_cautious_fallback() -> None:
    class InvalidFailureReviewGeneration(_FakeGeneration):
        async def chat_json(self, messages, *, schema_name, **kwargs):
            if schema_name == "CertifiedFailureAtomReview":
                return {
                    "verdict": "supported_root_cause",
                    "supporting_step_ids": [0],
                    "rationale": "Use the attempted action.",
                    "atom": {
                        "trigger": "when a spreadsheet formula fails",
                        "scope": "source-to-output transformations",
                        "decision": "Use the same row-wise formula again",
                        "invariant": "source and output roles remain distinct",
                        "verification": "the verifier checks the golden artifact",
                        "failure_mode": "the output remains incorrect",
                    },
                }
            return await super().chat_json(
                messages,
                schema_name=schema_name,
                **kwargs,
            )

    analyst = ExperienceAtomAnalyst(
        InvalidFailureReviewGeneration(),
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=60000,
    )
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-fallback-audit",
        task_id="task-fallback-audit",
        trial_index=0,
        instruction="Write output in A1:B5 using source A12:C22.",
        success=False,
        verifier_score=0.0,
        steps=[
            TrajectoryStep(
                step_id=0,
                raw_model_output="Edit the output rows.",
                action="write B2:B5",
                observation="saved",
            )
        ],
    )
    atoms, excluded = asyncio.run(analyst.extract_many([record]))
    assert not excluded
    assert atoms[0].metadata["analysis_mode"] == (
        "deterministic_cautious_fallback"
    )
    assert atoms[0].metadata["root_cause_review_attempts"] == 2
    assert atoms[0].metadata["root_cause_review_rejection_reasons"]
    assert atoms[0].metadata["root_cause_supporting_step_ids"] == ()
    assert atoms[0].decision.startswith("Map each named region")


def test_malformed_failure_review_uses_bounded_auditable_fallback() -> None:
    class MalformedFailureReviewGeneration(_FakeGeneration):
        def __init__(self) -> None:
            self.review_calls = 0

        async def chat_json(self, messages, *, schema_name, **kwargs):
            if schema_name == "CertifiedFailureAtomReview":
                self.review_calls += 1
                raise ValueError("guided JSON response was not strict JSON")
            return await super().chat_json(
                messages,
                schema_name=schema_name,
                **kwargs,
            )

    generation = MalformedFailureReviewGeneration()
    analyst = ExperienceAtomAnalyst(
        generation,
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=60000,
    )
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-malformed-review",
        task_id="task-malformed-review",
        trial_index=0,
        instruction="Write output in A1:B5 using source A12:C22.",
        success=False,
        verifier_score=0.0,
    )
    atoms, excluded = asyncio.run(analyst.extract_many([record]))
    assert not excluded
    assert generation.review_calls == 2
    assert atoms[0].metadata["analysis_mode"] == (
        "deterministic_cautious_fallback"
    )
    assert atoms[0].metadata["root_cause_review_attempts"] == 2
    assert atoms[0].metadata["root_cause_review_rejection_reasons"] == (
        "invalid structured critic response",
    )
    assert atoms[0].metadata["root_cause_supporting_step_ids"] == ()


def test_failure_review_cannot_cite_a_step_hidden_by_evidence_truncation() -> None:
    class HiddenStepReviewGeneration(_FakeGeneration):
        async def chat_json(self, messages, *, schema_name, **kwargs):
            if schema_name == "CertifiedFailureAtomReview":
                return {
                    "verdict": "supported_root_cause",
                    "supporting_step_ids": [10],
                    "rationale": "A hidden middle step establishes the cause.",
                    "atom": {
                        "trigger": "when an artifact transformation fails",
                        "scope": "structured artifact transformations",
                        "decision": "derive the output from the complete source",
                        "invariant": "preserve unrelated content",
                        "verification": "re-open and inspect the result",
                        "failure_mode": "partial evidence can mislead",
                    },
                }
            return await super().chat_json(
                messages,
                schema_name=schema_name,
                **kwargs,
            )

    analyst = ExperienceAtomAnalyst(
        HiddenStepReviewGeneration(),
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=1000,
    )
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-hidden-step",
        task_id="task-hidden-step",
        trial_index=0,
        instruction="Repair the artifact.",
        success=False,
        verifier_score=0.0,
        steps=[
            TrajectoryStep(
                step_id=index,
                raw_model_output="reason " + ("x" * 400),
                action=f"action {index}",
                observation="observation " + ("y" * 400),
            )
            for index in range(20)
        ],
    )
    atoms, excluded = asyncio.run(analyst.extract_many([record]))
    assert not excluded
    assert atoms[0].metadata["analysis_mode"] == (
        "deterministic_cautious_fallback"
    )
    assert atoms[0].metadata["root_cause_supporting_step_ids"] == ()


def test_atom_extraction_error_reports_every_failed_record() -> None:
    analyst = ExperienceAtomAnalyst(
        _UnsafeGeneration(),
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=60000,
    )
    records = [
        RawTrajectoryRecord(
            trajectory_id=f"trajectory-failed-{index}",
            task_id=f"task-failed-{index}",
            trial_index=0,
            instruction="Modify the artifact.",
            success=True,
        )
        for index in range(11)
    ]
    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(analyst.extract_many(records))
    assert "trajectory-failed-0" in str(exc_info.value)
    assert "trajectory-failed-10" in str(exc_info.value)


def test_atom_numeric_guard_preserves_general_rules_but_rejects_source_values() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="X",
        task_id="Y",
        trial_index=0,
        instruction="Return the observed value.",
        success=False,
        verifier_score=0.0,
        extra={"expected_answer": "7"},
    )
    generalized = {
        "trigger": "when an API uses 1-based indexing",
        "scope": "structured artifact operations",
        "decision": "convert indices before calling the API",
        "invariant": "preserve unrelated content",
        "verification": "inspect the saved artifact",
        "failure_mode": "do not assume index conventions",
    }
    assert not _atom_leakage_reasons(record, generalized)

    leaked = dict(generalized)
    leaked["decision"] = "return 7 for task X"
    reasons = _atom_leakage_reasons(record, leaked)
    assert any("source literal '7'" in reason for reason in reasons)
    assert any("source literal 'X'" in reason for reason in reasons)


def test_atom_guard_does_not_reject_incidental_record_numbers() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-numeric",
        task_id="task-numeric",
        trial_index=0,
        instruction="Move values from row 1 to row 2.",
        success=True,
        verifier_score=1.0,
        extra={"expected_answer": "2602"},
    )
    draft = {
        "trigger": "when an API uses 1-based indexing",
        "scope": "structured artifact operations",
        "decision": "perform 2 validation passes after converting indices",
        "invariant": "preserve unrelated content",
        "verification": "inspect the saved artifact",
        "failure_mode": "do not assume index conventions",
    }
    assert not _atom_leakage_reasons(record, draft)


def test_atom_guard_ignores_punctuation_only_final_response() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-punctuation",
        task_id="task-punctuation",
        trial_index=0,
        instruction="Validate a structured artifact.",
        final_response="}",
        success=False,
        verifier_score=0.0,
    )
    draft = {
        "trigger": "when validating structured text",
        "scope": "structured artifact operations",
        "decision": "check that delimiters such as } are balanced",
        "invariant": "preserve unrelated content",
        "verification": "parse the saved artifact",
        "failure_mode": "an unmatched delimiter invalidates the artifact",
    }
    assert not _atom_leakage_reasons(record, draft)


def test_atom_guard_rejects_labeled_verifier_values() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-verifier",
        task_id="task-verifier",
        trial_index=0,
        instruction="Repair the workbook.",
        verifier_feedback=(
            "Value mismatch at H2: expected '1272', got '1200'"
        ),
        success=False,
        verifier_score=0.0,
    )
    draft = {
        "trigger": "when repairing a workbook",
        "scope": "structured artifact operations",
        "decision": "write 1272 instead of 1200",
        "invariant": "preserve unrelated content",
        "verification": "inspect the saved artifact",
        "failure_mode": "do not infer correctness from a successful save",
    }
    reasons = _atom_leakage_reasons(record, draft)
    assert any("1272" in reason for reason in reasons)
    assert any("1200" in reason for reason in reasons)


def test_atom_guard_splits_unquoted_adjacent_verifier_values() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-unquoted-verifier",
        task_id="task-unquoted-verifier",
        trial_index=0,
        instruction="Repair the workbook.",
        verifier_feedback="Value mismatch: expected: 584 got: TOTAL",
        success=False,
        verifier_score=0.0,
    )
    draft = {
        "trigger": "when repairing a workbook",
        "scope": "structured artifact operations",
        "decision": "write 584 instead of TOTAL",
        "invariant": "preserve unrelated content",
        "verification": "inspect the saved artifact",
        "failure_mode": "do not infer correctness from a successful save",
    }
    reasons = _atom_leakage_reasons(record, draft)
    assert any("584" in reason for reason in reasons)
    assert any("TOTAL" in reason for reason in reasons)


def test_generic_error_markers_are_reusable_not_answer_leaks() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-error-marker",
        task_id="task-error-marker",
        trial_index=0,
        instruction="Repair the workbook.",
        verifier_feedback=(
            "Value mismatch at G2: expected '82', got '#VALUE!'"
        ),
        success=False,
        verifier_score=0.0,
    )
    draft = {
        "trigger": "when a spreadsheet formula returns #VALUE!",
        "scope": "formula debugging",
        "decision": "inspect input types and replace None with a valid value",
        "invariant": "formula operands have compatible types",
        "verification": "recalculate and confirm the error marker is gone",
        "failure_mode": "invalid references can instead produce #REF! or #N/A",
    }
    assert not _atom_leakage_reasons(record, draft)


def test_generic_markers_remain_protected_when_they_are_gold() -> None:
    gold_boolean = RawTrajectoryRecord(
        trajectory_id="trajectory-gold-boolean",
        task_id="task-gold-boolean",
        trial_index=0,
        instruction="Return the verified state.",
        success=True,
        verifier_score=1.0,
        extra={"gold_answer": "TRUE"},
    )
    gold_error = RawTrajectoryRecord(
        trajectory_id="trajectory-gold-error",
        task_id="task-gold-error",
        trial_index=0,
        instruction="Return the verified state.",
        success=True,
        verifier_score=1.0,
        verifier_feedback="Value mismatch: expected '#N/A', got '#VALUE!'",
    )
    draft = {
        "trigger": "when returning a verified spreadsheet value",
        "scope": "formula evaluation",
        "decision": "return TRUE or #N/A when required",
        "invariant": "preserve the verified state",
        "verification": "compare against the authoritative evaluator",
        "failure_mode": "do not substitute #VALUE! for another result",
    }
    assert any(
        "source literal 'TRUE'" in reason
        for reason in _atom_leakage_reasons(gold_boolean, draft)
    )
    error_reasons = _atom_leakage_reasons(gold_error, draft)
    assert any("source literal '#N/A'" in reason for reason in error_reasons)
    assert not any(
        "source literal '#VALUE!'" in reason for reason in error_reasons
    )


@pytest.mark.parametrize(
    ("expected", "rendered"),
    [("y", "return Y"), ("Y", "return y"), ("a", "answer: A"), ("A", "label is a")],
)
def test_single_character_answers_are_protected_in_answer_context(
    expected: str,
    rendered: str,
) -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-single-answer",
        task_id="task-single-answer",
        trial_index=0,
        instruction="Return the requested label.",
        success=True,
        verifier_score=1.0,
        extra={"expected_answer": expected},
    )
    draft = {
        "trigger": "when returning a classification",
        "scope": "single-label outputs",
        "decision": rendered,
        "invariant": "preserve label semantics",
        "verification": "validate the returned label",
        "failure_mode": "a different label changes the result",
    }
    assert any(
        f"source literal {expected!r}" in reason
        for reason in _atom_leakage_reasons(record, draft)
    )


def test_single_character_answer_does_not_match_a_generic_article() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-single-article",
        task_id="task-single-article",
        trial_index=0,
        instruction="Return the requested label.",
        success=True,
        verifier_score=1.0,
        extra={"expected_answer": "A"},
    )
    draft = {
        "trigger": "when producing a result",
        "scope": "classification tasks",
        "decision": "output a result after validation",
        "invariant": "preserve label semantics",
        "verification": "validate the returned label",
        "failure_mode": "an unchecked label can be wrong",
    }
    assert not any(
        "source literal 'A'" in reason
        for reason in _atom_leakage_reasons(record, draft)
    )


def test_single_character_answer_does_not_match_sentence_initial_article() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-single-capital-article",
        task_id="task-single-capital-article",
        trial_index=0,
        instruction="Return the requested label.",
        success=True,
        verifier_score=1.0,
        extra={"expected_answer": "A"},
    )
    draft = {
        "trigger": "A spreadsheet task requires a reusable procedure.",
        "scope": "structured artifact tasks",
        "decision": "inspect the artifact before editing",
        "invariant": "preserve unrelated state",
        "verification": "reopen the saved artifact",
        "failure_mode": "unchecked edits can be wrong",
    }
    assert not any(
        "source literal 'A'" in reason
        for reason in _atom_leakage_reasons(record, draft)
    )


def test_failed_atom_rejects_benchmark_language_and_failed_action_narration() -> None:
    draft = {
        "trigger": "when a spreadsheet formula fails",
        "scope": "source-to-output transformations",
        "decision": "The agent decides to reuse a row-wise formula",
        "invariant": "source and output roles remain distinct",
        "verification": "the verifier compares against the golden artifact",
        "failure_mode": "the output can look plausible while remaining wrong",
    }
    reasons = _failure_atom_semantic_reasons(
        draft,
        reference_audit={
            "instruction_references_not_explicitly_used_by_actions": ["A12:C22"]
        },
        verdict="cautious_diagnostic",
    )
    assert "benchmark evaluation artifact in reusable atom" in reasons
    assert "decision narrates the failed agent action" in reasons


def test_cautious_failure_diagnostic_covers_omitted_source_without_benchmark_terms() -> None:
    audit = {
        "instruction_references_not_explicitly_used_by_actions": ["A12:C22"]
    }
    draft = _cautious_failure_diagnostic(audit)
    assert not _failure_atom_semantic_reasons(
        draft,
        reference_audit=audit,
        verdict="cautious_diagnostic",
    )
    assert "source" in "\n".join(draft.values()).casefold()
    assert "output" in "\n".join(draft.values()).casefold()


def test_supported_failure_atom_may_be_prescriptive_after_step_citation() -> None:
    draft = {
        "trigger": "when a source-to-output transformation is required",
        "scope": "structured artifact transformations",
        "decision": "derive the output from the complete source region",
        "invariant": "source and output roles remain distinct",
        "verification": "re-open the output and validate the requested semantics",
        "failure_mode": "a partial source can produce a plausible wrong result",
    }
    assert not _failure_atom_semantic_reasons(
        draft,
        reference_audit={
            "instruction_references_not_explicitly_used_by_actions": []
        },
        verdict="supported_root_cause",
    )


def test_cautious_failure_atom_must_remain_diagnostic() -> None:
    draft = {
        "trigger": "when a source-to-output transformation is required",
        "scope": "structured artifact transformations",
        "decision": "derive the output from the complete source region",
        "invariant": "source and output roles remain distinct",
        "verification": "re-open the output and validate the requested semantics",
        "failure_mode": "a partial source can produce a plausible wrong result",
    }
    assert "cautious diagnostic decision is prescriptive" in (
        _failure_atom_semantic_reasons(
            draft,
            reference_audit={
                "instruction_references_not_explicitly_used_by_actions": []
            },
            verdict="cautious_diagnostic",
        )
    )


def test_redaction_cannot_turn_an_atom_into_only_placeholders() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-placeholder",
        task_id="task-placeholder",
        trial_index=0,
        instruction="Produce a reusable rule.",
    )
    draft = {
        "trigger": "<redacted>",
        "scope": "<redacted>",
        "decision": "<redacted>",
        "invariant": "<redacted>",
        "verification": "<redacted>",
        "failure_mode": "<redacted>",
    }
    reasons = _atom_leakage_reasons(record, draft)
    assert len(
        [
            reason
            for reason in reasons
            if reason.startswith("insufficient reusable content")
        ]
    ) == 6


def test_atom_fields_enforce_schema_max_length_locally() -> None:
    with pytest.raises(ValueError, match="exceeds max length"):
        _required_string(
            {"trigger": "x" * 1201},
            "trigger",
            max_length=1200,
        )


def test_compact_analysis_uses_only_authoritative_recalc_verdict() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-recalc",
        task_id="task-recalc",
        trial_index=0,
        instruction="Repair the workbook.",
        success=True,
        verifier_score=1.0,
        extra={
            "trace2skill_result": {
                "passed_count": 1,
                "raw_passed_count": 0,
                "test_cases": [
                    {
                        "passed": True,
                        "message": "LibreOffice recalculation passed",
                        "raw_passed": False,
                        "raw_message": "cached-value mismatch",
                        "raw_evaluation_mode": "audit_only_no_recalc",
                    }
                ],
            }
        },
    )
    rendered = render_compact_analysis_bundle_text(record)
    assert "LibreOffice recalculation passed" in rendered
    assert "cached-value mismatch" not in rendered
    assert "raw_passed" not in rendered
    assert record.extra["trace2skill_result"]["raw_passed_count"] == 0


def test_spreadsheet_reference_audit_exposes_uncovered_source_range() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-range-audit",
        task_id="task-range-audit",
        trial_index=0,
        instruction=(
            "Write results to A1:B5 using source data in A12:C22; "
            "the formula relates B1 to C1."
        ),
        steps=[
            TrajectoryStep(
                step_id=0,
                raw_model_output="Use the output rows directly.",
                action="write formulas into B2:B5 using A2 and C2",
                observation="saved",
            )
        ],
    )
    audit = _spreadsheet_reference_audit(record)
    assert set(audit["instruction_references"]) == {
        "A1:B5",
        "A12:C22",
        "B1",
        "C1",
    }
    assert set(audit["action_references"]) == {"A2", "B2:B5", "C2"}
    assert "A12:C22" in audit[
        "instruction_references_not_explicitly_used_by_actions"
    ]
    assert "lexical_range_coverage_only" in audit["audit_limitations"]
    assert "absence_is_not_proof_of_omitted_access" in audit[
        "audit_limitations"
    ]


def test_atom_extraction_canonicalizes_spreadsheet_coordinates() -> None:
    analyst = ExperienceAtomAnalyst(
        _CoordinateGeneration(),
        _FakeEmbedding(),
        config=CertifiedOtdConfig(),
        tokenizer=RegexTokenizer(),
        max_prompt_tokens=10000,
        max_output_tokens=4096,
        max_evidence_chars=60000,
    )
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-coordinate",
        task_id="task-coordinate",
        trial_index=0,
        instruction="Generate relative formulas.",
        success=True,
        verifier_score=1.0,
    )
    atoms, excluded = asyncio.run(analyst.extract_many([record]))
    reusable_text = "\n".join(
        (
            atoms[0].trigger,
            atoms[0].scope,
            atoms[0].decision,
            atoms[0].invariant,
            atoms[0].verification,
            atoms[0].failure_mode,
        )
    )
    assert not excluded
    assert "H2" not in reusable_text
    assert "A1" not in reusable_text
    assert "<cell-reference>" in reusable_text


def test_atom_schema_bounds_each_reusable_field() -> None:
    assert all(
        ATOM_SCHEMA["properties"][field_name]["maxLength"] == 1200
        for field_name in (
            "trigger",
            "scope",
            "decision",
            "invariant",
            "verification",
            "failure_mode",
        )
    )


def test_one_character_answer_literal_is_case_sensitive() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-short-answer",
        task_id="task-short-answer",
        trial_index=0,
        instruction="Select a category.",
        success=False,
        verifier_score=0.0,
        extra={"expected_answer": "A"},
    )
    generalized = {
        "trigger": "when a category must be selected",
        "scope": "classification tasks",
        "decision": "derive the category from evidence",
        "invariant": "preserve the category semantics",
        "verification": "compare the category against evidence",
        "failure_mode": "do not guess",
    }
    assert not _atom_leakage_reasons(record, generalized)

    leaked = dict(generalized)
    leaked["decision"] = "return A"
    assert any(
        "source literal 'A'" in reason
        for reason in _atom_leakage_reasons(record, leaked)
    )


def test_short_punctuated_answer_does_not_match_inside_abbreviation() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-short-punctuation",
        task_id="task-short-punctuation",
        trial_index=0,
        instruction="Format an initial.",
        success=True,
        verifier_score=1.0,
        extra={"expected_answer": "G."},
    )
    generalized = {
        "trigger": "when punctuation must be appended conditionally",
        "scope": "structured text transformations",
        "decision": "apply a general rule, e.g. after checking for content",
        "invariant": "preserve empty values",
        "verification": "inspect the transformed value",
        "failure_mode": "do not introduce a circular dependency",
    }
    assert not _atom_leakage_reasons(record, generalized)

    leaked = dict(generalized)
    leaked["decision"] = "return G."
    assert any(
        "source literal 'G.'" in reason
        for reason in _atom_leakage_reasons(record, leaked)
    )


def test_atom_guard_rejects_nested_evaluator_values() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-nested-verifier",
        task_id="task-nested-verifier",
        trial_index=0,
        instruction="Repair the workbook.",
        success=False,
        verifier_score=0.0,
        extra={
            "trace2skill_result": {
                "test_cases": [
                    {
                        "message": (
                            "Value mismatch at E585: expected '584', got 'TOTAL'"
                        ),
                        "raw_message": (
                            "Value mismatch at C9: expected '2026-07-27', "
                            "got 'None'"
                        ),
                    }
                ]
            }
        },
    )
    draft = {
        "trigger": "when repairing a workbook",
        "scope": "structured artifact operations",
        "decision": "write 584 with the date 2026-07-27, not TOTAL",
        "invariant": "preserve unrelated content",
        "verification": "inspect the saved artifact",
        "failure_mode": "do not infer correctness from a successful save",
    }
    reasons = _atom_leakage_reasons(record, draft)
    assert any("584" in reason for reason in reasons)
    assert any("TOTAL" in reason for reason in reasons)
    assert any("2026-07-27" in reason for reason in reasons)


def test_textual_final_answer_is_rejected_as_reusable_content() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-answer",
        task_id="task-answer",
        trial_index=0,
        instruction="Determine the status.",
        final_response=(
            "A long explanation precedes the final response. " * 10
            + "<answer>Approved</answer>"
        ),
        success=True,
        verifier_score=1.0,
    )
    draft = {
        "trigger": "when checking status",
        "scope": "status questions",
        "decision": "return Approved",
        "invariant": "preserve the source wording",
        "verification": "compare against evidence",
        "failure_mode": "do not guess",
    }
    reasons = _atom_leakage_reasons(record, draft)
    assert any("Approved" in reason for reason in reasons)


def test_long_untagged_terminal_answer_is_rejected() -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-untagged-answer",
        task_id="task-untagged-answer",
        trial_index=0,
        instruction="Determine the status.",
        final_response=(
            "The evidence was inspected carefully. " * 20
            + "Approved"
        ),
        success=True,
        verifier_score=1.0,
    )
    draft = {
        "trigger": "when checking status",
        "scope": "status questions",
        "decision": "return Approved",
        "invariant": "preserve the source wording",
        "verification": "compare against evidence",
        "failure_mode": "do not guess",
    }
    reasons = _atom_leakage_reasons(record, draft)
    assert any("Approved" in reason for reason in reasons)


def test_record_and_atom_identity_must_be_unique_and_order_complete() -> None:
    first = RawTrajectoryRecord(
        trajectory_id="trajectory-a",
        task_id="task-a",
        trial_index=0,
        instruction="Modify the artifact.",
    )
    duplicate = RawTrajectoryRecord(
        trajectory_id="trajectory-a",
        task_id="task-b",
        trial_index=0,
        instruction="Modify another artifact.",
    )
    with pytest.raises(ValueError, match="unique trajectory_id"):
        _require_unique_record_ids([first, duplicate])

    second = RawTrajectoryRecord(
        trajectory_id="trajectory-b",
        task_id="task-b",
        trial_index=0,
        instruction="Modify another artifact.",
    )
    atoms = [
        _atom("atom-a", (1.0, 0.0), (1.0, 0.0)),
        _atom("atom-b", (0.0, 1.0), (0.0, 1.0)),
    ]
    atoms[0] = ExperienceAtom.from_dict(
        atoms[0].to_dict() | {"source_item_id": "trajectory-a"}
    )
    atoms[1] = ExperienceAtom.from_dict(
        atoms[1].to_dict() | {"source_item_id": "trajectory-b"}
    )
    _require_complete_atom_identity([first, second], atoms)
    with pytest.raises(ValueError, match="complete ordered trajectory"):
        _require_complete_atom_identity([first, second], list(reversed(atoms)))


def test_otd_nodebank_manifest_carries_lineage_without_changing_format(
    tmp_path: Path,
) -> None:
    state = _build(_atoms()[:3])
    for node in state.nodes.values():
        if not node.is_leaf:
            node.skill = {
                "name": f"parent {node.node_id}",
                "trigger": "related tasks",
                "content": "shared invariant",
                "confidence": 1.0,
            }
            node.retrievable = True
            node.structural_certificate = {"passed": True}

    class _AnalystConfig:
        tokenizer_model = None
        tokenizer_required = False
        allow_regex_tokenizer_fallback = True

    class _EmbeddingConfig:
        tokenizer_model = None

    class _Config:
        analyst = _AnalystConfig()
        embedding = _EmbeddingConfig()

    skill_dir = tmp_path / "skills"
    manifest = _nodebank_manifest(
        state,
        output_dir=skill_dir,
        config=_Config(),
        otd_config=CertifiedOtdConfig(),
        authoritative_tree=_write_authoritative_tree(state, skill_dir),
    )
    assert manifest["format"] == "dynamix_node_skill_bank_v1"
    assert manifest["export_policy"]["heldout_retrieval"] == (
        "tree_antichain_knapsack"
    )
    assert manifest["tree_index"]["root_node_id"] == state.root_node_id
    assert manifest["root_node_id"] == state.root_node_id
    assert len(manifest["nodes"]) == len(state.nodes)
    assert all(int(node["token_cost"]) > 0 for node in manifest["nodes"])
    assert all("parent_node_id" in node for node in manifest["nodes"])
    tokenizer = RegexTokenizer()
    assert manifest["export_policy"]["fixed_prompt_overhead_tokens"] == (
        tokenizer.count(skillbank_module.retrieved_experience_preamble())
    )
    assert all(
        int(node["token_cost"]) == tokenizer.count(str(node["prompt_text"]))
        for node in manifest["nodes"]
    )
    assert manifest["export_policy"]["relevance_transform"] == (
        "shifted_cosine_(1_plus_cosine)_over_2"
    )


def test_cdost_nodebank_rejects_inconsistent_audit_lineage_before_embedding(
    tmp_path: Path,
) -> None:
    state = _build(_atoms()[:3])
    for node in state.nodes.values():
        if not node.is_leaf:
            node.skill = {
                "name": f"parent {node.node_id}",
                "trigger": "related tasks",
                "content": "shared invariant",
                "confidence": 1.0,
            }
            node.retrievable = True
            node.structural_certificate = {"passed": True}

    class _AnalystConfig:
        tokenizer_model = None
        tokenizer_required = False
        allow_regex_tokenizer_fallback = True

    class _EmbeddingConfig:
        tokenizer_model = None

    class _Config:
        analyst = _AnalystConfig()
        embedding = _EmbeddingConfig()

    skill_dir = tmp_path / "skills"
    original = _nodebank_manifest(
        state,
        output_dir=skill_dir,
        config=_Config(),
        otd_config=CertifiedOtdConfig(),
        authoritative_tree=_write_authoritative_tree(state, skill_dir),
    )
    non_root = next(
        node for node in original["nodes"] if node["parent_node_id"] is not None
    )
    internal = next(
        node for node in original["nodes"] if node["child_node_ids"]
    )
    mutations = [
        lambda payload: payload.update(root_node_id="wrong-root"),
        lambda payload: payload.update(node_count=payload["node_count"] + 1),
        lambda payload: next(
            node
            for node in payload["nodes"]
            if node["node_id"] == non_root["node_id"]
        ).update(parent_node_id="wrong-parent"),
        lambda payload: next(
            node
            for node in payload["nodes"]
            if node["node_id"] == internal["node_id"]
        ).update(child_node_ids=list(reversed(internal["child_node_ids"]))),
    ]
    for mutate in mutations:
        payload = json.loads(json.dumps(original))
        mutate(payload)
        (skill_dir / "node_bank_manifest.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        selector = SkillBankSelector(skillbank_root=skill_dir)
        selector._load_or_build_index = lambda: pytest.fail(
            "inconsistent CDOST audit lineage must fail before index loading"
        )
        with pytest.raises(ValueError, match="certified_dual_view_otd"):
            selector.select("query", top_k=1)


def test_cdost_nodebank_binds_to_authoritative_tree_before_embedding(
    tmp_path: Path,
) -> None:
    state = _build(_atoms()[:3])
    for node in state.nodes.values():
        if not node.is_leaf:
            node.skill = {
                "name": f"parent {node.node_id}",
                "trigger": "related tasks",
                "content": "shared invariant",
                "confidence": 1.0,
            }
            node.retrievable = True
            node.structural_certificate = {"passed": True}

    config = SimpleNamespace(
        analyst=SimpleNamespace(
            tokenizer_model=None,
            tokenizer_required=False,
            allow_regex_tokenizer_fallback=True,
        ),
        embedding=SimpleNamespace(tokenizer_model=None),
    )
    skill_dir = tmp_path / "skills"
    manifest = _nodebank_manifest(
        state,
        output_dir=skill_dir,
        config=config,
        otd_config=CertifiedOtdConfig(),
        authoritative_tree=_write_authoritative_tree(state, skill_dir),
    )

    root_node_id = str(manifest["root_node_id"])
    coordinated_rewire = json.loads(json.dumps(manifest))
    root_children = coordinated_rewire["tree_index"]["children_by_node"][
        root_node_id
    ]
    coordinated_rewire["tree_index"]["children_by_node"][root_node_id] = list(
        reversed(root_children)
    )
    next(
        node
        for node in coordinated_rewire["nodes"]
        if node["node_id"] == root_node_id
    )["child_node_ids"] = list(reversed(root_children))

    leaf_only = json.loads(json.dumps(manifest))
    leaf_only["nodes"] = [
        node for node in leaf_only["nodes"] if not node["child_node_ids"]
    ]
    leaf_only["node_count"] = len(leaf_only["nodes"])

    for payload in (coordinated_rewire, leaf_only):
        (skill_dir / "node_bank_manifest.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        selector = SkillBankSelector(skillbank_root=skill_dir)
        selector._load_or_build_index = lambda: pytest.fail(
            "authority mismatch must fail before index loading"
        )
        with pytest.raises(ValueError, match="authoritative tree"):
            selector.select("query", top_k=1)

    (skill_dir / "node_bank_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    state_path = tmp_path / "otd_tree_state.json"
    state_path.write_text(
        state_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    selector = SkillBankSelector(skillbank_root=skill_dir)
    selector._load_or_build_index = lambda: pytest.fail(
        "digest mismatch must fail before index loading"
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        selector.select("query", top_k=1)


def test_atom_cache_requires_exact_record_and_protocol_identity(
    tmp_path: Path,
) -> None:
    record = RawTrajectoryRecord(
        trajectory_id="trajectory-a",
        task_id="task-a",
        trial_index=0,
        instruction="Modify the artifact.",
        success=True,
        verifier_score=1.0,
    )

    class _AnalystConfig:
        max_output_tokens = 4096
        max_prompt_tokens = 85000
        analysis_bundle_max_chars = 60000
        tokenizer_model = "test-tokenizer"

    class _Config:
        generation = GenerationConfig(
            base_url="http://generation.invalid/v1",
            model="test-model",
            thinking_mode=True,
        )
        embedding = EmbeddingConfig(
            base_url="http://embedding.invalid/v1",
            model="test-embedding",
            max_model_len=32000,
            max_input_tokens=28000,
            tokenizer_model="test-tokenizer",
        )
        analyst = _AnalystConfig()

    otd_config = CertifiedOtdConfig()
    record_fingerprint = _records_fingerprint([record])
    protocol_fingerprint = _atom_protocol_fingerprint(_Config(), otd_config)
    cache_path = tmp_path / "experience_atoms.json"
    cache_path.write_text(
        json.dumps(
            {
                "format": "certified_experience_atoms_v1",
                "record_count": 1,
                "atom_count": 1,
                "excluded_records": [],
                "records_fingerprint": record_fingerprint,
                "atom_protocol_fingerprint": protocol_fingerprint,
                "atoms": [
                    _atom(
                        "atom-a",
                        (1.0, 0.0),
                        (1.0, 0.0),
                    ).to_dict()
                    | {"source_item_id": record.trajectory_id}
                ],
            }
        ),
        encoding="utf-8",
    )
    atoms, excluded = _load_atom_cache(
        cache_path,
        [record],
        records_fingerprint=record_fingerprint,
        atom_protocol_fingerprint=protocol_fingerprint,
    )
    assert len(atoms) == 1
    assert not excluded
    with pytest.raises(ValueError, match="record fingerprint"):
        _load_atom_cache(
            cache_path,
            [record],
            records_fingerprint="different",
            atom_protocol_fingerprint=protocol_fingerprint,
        )
    with pytest.raises(ValueError, match="protocol"):
        _load_atom_cache(
            cache_path,
            [record],
            records_fingerprint=record_fingerprint,
            atom_protocol_fingerprint="different",
        )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["atoms"][0]["trigger"] = "when solving task-a"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="leaked task-specific content"):
        _load_atom_cache(
            cache_path,
            [record],
            records_fingerprint=record_fingerprint,
            atom_protocol_fingerprint=protocol_fingerprint,
        )
    payload["atoms"][0]["trigger"] = "apply a reusable validated transformation"
    payload["atoms"][0]["evidence_type"] = "community_summary"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="only single_trace atoms"):
        _load_atom_cache(
            cache_path,
            [record],
            records_fingerprint=record_fingerprint,
            atom_protocol_fingerprint=protocol_fingerprint,
        )


def test_atom_protocol_fingerprint_covers_generation_and_embedding_protocol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    generation = GenerationConfig(
        base_url="http://generation.invalid/v1",
        model="test-model",
        api_key_env_var="CDOST_TEST_GENERATION_KEY",
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    embedding = EmbeddingConfig(
        base_url="http://embedding.invalid/v1",
        model="test-embedding",
        api_key_env_var="CDOST_TEST_EMBEDDING_KEY",
        max_model_len=32000,
        max_input_tokens=28000,
        truncate_long_texts=True,
        tokenizer_model="test-tokenizer",
        tokenizer_required=True,
        truncation_strategy="head",
        deterministic_dim=384,
    )
    analyst = SimpleNamespace(
        max_output_tokens=4096,
        max_prompt_tokens=85000,
        analysis_bundle_max_chars=60000,
        tokenizer_model="test-tokenizer",
    )
    monkeypatch.setenv("CDOST_TEST_GENERATION_KEY", "generation-key-a")
    monkeypatch.setenv("CDOST_TEST_EMBEDDING_KEY", "embedding-key-a")

    def fingerprint(
        *,
        generation_config=generation,
        embedding_config=embedding,
    ) -> str:
        return _atom_protocol_fingerprint(
            SimpleNamespace(
                generation=generation_config,
                embedding=embedding_config,
                analyst=analyst,
            ),
            CertifiedOtdConfig(),
        )

    baseline = fingerprint()
    cache_record = RawTrajectoryRecord(
        trajectory_id="trajectory-cache",
        task_id="task-cache",
        trial_index=0,
        instruction="Modify the artifact.",
    )
    cache_records_fingerprint = _records_fingerprint([cache_record])
    cache_path = tmp_path / "experience_atoms.json"
    cache_path.write_text(
        json.dumps(
            {
                "format": "certified_experience_atoms_v1",
                "record_count": 1,
                "atom_count": 1,
                "excluded_records": [],
                "records_fingerprint": cache_records_fingerprint,
                "atom_protocol_fingerprint": baseline,
                "atoms": [
                    _atom(
                        "atom-cache",
                        (1.0, 0.0),
                        (1.0, 0.0),
                    ).to_dict()
                    | {"source_item_id": cache_record.trajectory_id}
                ],
            }
        ),
        encoding="utf-8",
    )
    embedding_variants = [
        replace(embedding, truncate_long_texts=False),
        replace(embedding, tokenizer_required=False),
        replace(embedding, truncation_strategy="tail"),
        replace(embedding, batch_size=4),
        replace(embedding, max_concurrency=4),
        replace(embedding, deterministic_dim=768),
    ]
    assert all(
        fingerprint(embedding_config=variant) != baseline
        for variant in embedding_variants
    )
    assert (
        fingerprint(
            generation_config=replace(
                generation,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        )
        != baseline
    )
    generation_variants = [
        replace(generation, timeout_seconds=60.0),
        replace(generation, max_concurrency=4),
        replace(generation, retry_wait_seconds=(1.0,)),
    ]
    assert all(
        fingerprint(generation_config=variant) != baseline
        for variant in generation_variants
    )

    monkeypatch.setenv("CDOST_TEST_GENERATION_KEY", "generation-key-b")
    assert fingerprint() != baseline
    monkeypatch.setenv("CDOST_TEST_GENERATION_KEY", "generation-key-a")
    monkeypatch.setenv("CDOST_TEST_EMBEDDING_KEY", "embedding-key-b")
    assert fingerprint() != baseline
    monkeypatch.setenv("CDOST_TEST_EMBEDDING_KEY", "embedding-key-a")

    original_getsource = certified_otd_pipeline.inspect.getsource

    for changed_component in (
        certified_otd_pipeline.certified_otd_module,
        certified_otd_pipeline.clients_module,
        certified_otd_pipeline.openai_compat_module,
        certified_otd_pipeline.tokenization_module,
        certified_otd_pipeline.trace_views_module,
        ExperienceAtomAnalyst,
        GenerationClient,
        EmbeddingClient,
    ):
        def changed_source(component, target=changed_component):
            source = original_getsource(component)
            if component is target:
                return source + "\n# changed protocol implementation"
            return source

        monkeypatch.setattr(
            certified_otd_pipeline.inspect,
            "getsource",
            changed_source,
        )
        changed_fingerprint = fingerprint()
        assert changed_fingerprint != baseline
        with pytest.raises(ValueError, match="protocol"):
            _load_atom_cache(
                cache_path,
                [cache_record],
                records_fingerprint=cache_records_fingerprint,
                atom_protocol_fingerprint=changed_fingerprint,
            )

    monkeypatch.setattr(
        certified_otd_pipeline.inspect,
        "getsource",
        original_getsource,
    )
    original_read_text = Path.read_text
    pipeline_path = Path(certified_otd_pipeline.__file__).resolve()

    def changed_pipeline_source(path, *args, **kwargs):
        source = original_read_text(path, *args, **kwargs)
        if Path(path).resolve() == pipeline_path:
            return source + "\n# changed pipeline helper implementation"
        return source

    monkeypatch.setattr(Path, "read_text", changed_pipeline_source)
    changed_fingerprint = fingerprint()
    assert changed_fingerprint != baseline
    with pytest.raises(ValueError, match="protocol"):
        _load_atom_cache(
            cache_path,
            [cache_record],
            records_fingerprint=cache_records_fingerprint,
            atom_protocol_fingerprint=changed_fingerprint,
        )
    monkeypatch.setattr(Path, "read_text", original_read_text)

    monkeypatch.setattr(
        certified_otd_pipeline.clients_module,
        "_openai_client_identity",
        lambda: {"implementation": "openai", "version": "changed-version"},
    )
    assert fingerprint() != baseline


def test_atom_cache_rejects_reordered_atoms(tmp_path: Path) -> None:
    records = [
        RawTrajectoryRecord(
            trajectory_id=f"trajectory-{name}",
            task_id=f"task-{name}",
            trial_index=0,
            instruction="Modify the artifact.",
        )
        for name in ("a", "b")
    ]
    atoms = [
        _atom("atom-a", (1.0, 0.0), (1.0, 0.0)).to_dict()
        | {"source_item_id": "trajectory-a"},
        _atom("atom-b", (0.0, 1.0), (0.0, 1.0)).to_dict()
        | {"source_item_id": "trajectory-b"},
    ]
    path = tmp_path / "experience_atoms.json"
    path.write_text(
        json.dumps(
            {
                "format": "certified_experience_atoms_v1",
                "record_count": 2,
                "atom_count": 2,
                "excluded_records": [],
                "records_fingerprint": "records",
                "atom_protocol_fingerprint": "protocol",
                "atoms": list(reversed(atoms)),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cache order"):
        _load_atom_cache(
            path,
            records,
            records_fingerprint="records",
            atom_protocol_fingerprint="protocol",
        )


def _manifest_node(node_id: str) -> dict[str, object]:
    embedding_text = f"name: {node_id}\ntrigger: {node_id}\ncontent: {node_id}"
    return {
        "node_id": node_id,
        "item_id": node_id,
        "name": node_id,
        "trigger": node_id,
        "content": node_id,
        "embedding_text": embedding_text,
        "prompt_text": node_id,
        "sha256": hashlib.sha256(embedding_text.encode()).hexdigest(),
    }


def test_skillbank_cache_reorders_embedding_rows_with_manifest(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    manifest_path = bank / "node_bank_manifest.json"
    nodes = [_manifest_node("alpha"), _manifest_node("beta")]
    manifest_path.write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "nodes": nodes,
            }
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "index.json"
    first = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://cache-order",
        model="mock",
        cache_path=cache_path,
    )
    vectors = {
        str(nodes[0]["embedding_text"]): [1.0, 0.0],
        str(nodes[1]["embedding_text"]): [0.0, 1.0],
    }
    first._embed = lambda texts: [vectors[text] for text in texts]
    first._load_or_build_index()

    manifest_path.write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "nodes": list(reversed(nodes)),
            }
        ),
        encoding="utf-8",
    )
    second = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://cache-order",
        model="mock",
        cache_path=cache_path,
    )
    second._embed = lambda texts: [[0.0, 1.0] for _ in texts]
    selected = second.select("beta query", top_k=1)
    assert selected[0].skill.node_id == "beta"


def test_skillbank_single_vector_protocol_fails_closed_on_token_overflow(
    tmp_path: Path,
) -> None:
    selector = SkillBankSelector(
        skillbank_root=tmp_path,
        base_url="mock://single-vector-limit",
        max_model_len=3,
        max_input_tokens=2,
        chunk_tokens=100,
        chunk_overlap_tokens=10,
        expected_tree_policy="certified_dual_view_otd",
    )
    protocol = selector._embedding_protocol_payload()
    assert protocol["input_policy"] == "single_vector_fail_if_over_limit"
    assert protocol["chunking_active"] is False
    assert protocol["scoring_vector_policy"] == (
        "persisted_unit_vector_no_reload_renormalization"
    )
    assert "chunk_tokens" not in protocol
    with pytest.raises(ValueError, match="single-vector embedding input"):
        selector._embed(["one two three"])


def test_cdost_cached_index_vector_is_used_without_second_normalization(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    node = _manifest_node("alpha")
    (bank / "node_bank_manifest.json").write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "tree_policy": "certified_dual_view_otd",
                "nodes": [node],
            }
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "index.json"
    raw = np.asarray(
        [-0.02467038707010525, -1.1334575675104939],
        dtype=float,
    )
    persisted = (
        raw.reshape(1, -1)
        / np.linalg.norm(raw.reshape(1, -1), axis=1, keepdims=True)
    )[0]
    renormalized = persisted / float(np.linalg.norm(persisted))
    assert not np.array_equal(persisted, renormalized)

    builder = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://exact-scoring-vector",
        model="mock",
        cache_path=cache_path,
        expected_tree_policy="certified_dual_view_otd",
    )
    builder._embed = lambda texts: [raw.tolist() for _ in texts]
    _, built = builder._load_or_build_index()
    assert np.array_equal(built[0], persisted)

    loader = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://exact-scoring-vector",
        model="mock",
        cache_path=cache_path,
        expected_tree_policy="certified_dual_view_otd",
        require_cache_match=True,
    )
    loader._embed = lambda texts: pytest.fail("cached index must not rebuild")
    _, loaded = loader._load_or_build_index()
    assert np.array_equal(loaded[0], persisted)


def test_cdost_cached_index_rejects_non_unit_scoring_vectors(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    node = _manifest_node("alpha")
    (bank / "node_bank_manifest.json").write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "tree_policy": "certified_dual_view_otd",
                "nodes": [node],
            }
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "index.json"
    builder = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://unit-validation",
        model="mock",
        cache_path=cache_path,
        expected_tree_policy="certified_dual_view_otd",
    )
    builder._load_or_build_index()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["embeddings"] = [[2.0, 0.0]]
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    loader = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://unit-validation",
        model="mock",
        cache_path=cache_path,
        expected_tree_policy="certified_dual_view_otd",
        require_cache_match=True,
    )
    with pytest.raises(RuntimeError, match="strict skillbank cache"):
        loader._load_or_build_index()


def test_cdost_unit_vector_validation_uses_exact_documented_tolerance() -> None:
    epsilon = np.finfo(float).eps
    accepted = np.asarray([[1.0 + 32.0 * epsilon, 0.0]])
    assert skillbank_module._require_unit_matrix(
        accepted,
        field_name="test vectors",
    ) is accepted
    rejected = np.asarray([[1.0 + 100.0 * epsilon, 0.0]])
    with pytest.raises(ValueError, match="unit vectors"):
        skillbank_module._require_unit_matrix(
            rejected,
            field_name="test vectors",
        )


def test_legacy_skillbank_accepts_pre_cdost_index_protocol(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    node = _manifest_node("alpha")
    (bank / "node_bank_manifest.json").write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "nodes": [node],
            }
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "index.json"
    builder = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://legacy-index",
        model="mock",
        cache_path=cache_path,
        chunk_tokens=28000,
        chunk_overlap_tokens=1000,
    )
    builder._load_or_build_index()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["embedding_protocol"] = {
        "max_model_len": builder.max_model_len,
        "max_input_tokens": builder.max_input_tokens,
        "batch_size": builder.batch_size,
        "tokenizer_model": builder.tokenizer_model,
        "chunk_tokens": builder.chunk_tokens,
        "chunk_overlap_tokens": builder.chunk_overlap_tokens,
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    loader = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://legacy-index",
        model="mock",
        cache_path=cache_path,
        chunk_tokens=28000,
        chunk_overlap_tokens=1000,
    )
    loader._embed = lambda texts: pytest.fail("legacy index must be reused")
    docs, vectors = loader._load_or_build_index()
    assert [doc.node_id for doc in docs] == ["alpha"]
    assert vectors.shape[0] == 1


def test_query_audit_hashes_final_normalized_scoring_vector(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    node = _manifest_node("alpha")
    (bank / "node_bank_manifest.json").write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "nodes": [node],
            }
        ),
        encoding="utf-8",
    )
    selector = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://query-scoring-vector",
        model="mock",
        cache_path=tmp_path / "index.json",
    )
    raw = np.asarray(
        [0.6662754303347858, -0.7457057401765095],
        dtype=float,
    )

    def embed(texts: list[str]) -> list[list[float]]:
        selector.last_embedding_audit = {
            "vector_sha256": [embedding_vector_sha256(raw.tolist())],
        }
        return [raw.tolist() for _ in texts]

    selector._load_or_build_index = lambda: (
        [SkillNodeDocument(**node)],
        np.asarray([[1.0, 0.0]], dtype=float),
    )
    selector._embed = embed
    selector.select("query", top_k=1)
    scoring = raw / float(np.linalg.norm(raw))
    assert selector.last_embedding_audit["scoring_vector_sha256"] == [
        embedding_vector_sha256(scoring.tolist())
    ]


def test_skillbank_content_cache_freezes_first_vector_across_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_cache = tmp_path / "shared_vectors.sqlite"
    first = SkillBankSelector(
        skillbank_root=tmp_path,
        base_url="mock://frozen-vectors",
        vector_cache_path=vector_cache,
    )
    first_vector = first._embed(["same query"])[0]
    assert first.last_embedding_audit["cache_miss_count"] == 1

    monkeypatch.setattr(
        skillbank_module,
        "_deterministic_embedding",
        lambda text, dim=384: [0.5] * dim,
    )
    second = SkillBankSelector(
        skillbank_root=tmp_path,
        base_url="mock://frozen-vectors",
        vector_cache_path=vector_cache,
        require_vector_cache_match=True,
    )
    assert second._embed(["same query"])[0] == first_vector
    assert second.last_embedding_audit["cache_hit_count"] == 1
    with pytest.raises(RuntimeError, match="missing 1 required"):
        second._embed(["uncached query"])


def test_strict_skillbank_cache_mismatch_fails_without_rebuilding(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    node = _manifest_node("alpha")
    (bank / "node_bank_manifest.json").write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "nodes": [node],
            }
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "index.json"
    builder = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://strict-cache",
        model="mock",
        cache_path=cache_path,
        max_input_tokens=32000,
    )
    builder._load_or_build_index()
    before = hashlib.sha256(cache_path.read_bytes()).hexdigest()

    heldout = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://strict-cache",
        model="mock",
        cache_path=cache_path,
        max_input_tokens=28000,
        require_cache_match=True,
    )
    heldout._embed = lambda texts: pytest.fail("strict heldout must not rebuild")
    with pytest.raises(RuntimeError, match="strict skillbank cache"):
        heldout._load_or_build_index()
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == before


def test_strict_skillbank_rejects_manifest_hash_not_matching_live_text(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    node = _manifest_node("alpha")
    manifest_path = bank / "node_bank_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "nodes": [node],
            }
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "index.json"
    SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://strict-live-text",
        model="mock",
        cache_path=cache_path,
    )._load_or_build_index()

    tampered_node = {
        **node,
        "embedding_text": str(node["embedding_text"]) + "\ntampered",
    }
    manifest_path.write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "nodes": [tampered_node],
            }
        ),
        encoding="utf-8",
    )
    selector = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://strict-live-text",
        model="mock",
        cache_path=cache_path,
        require_cache_match=True,
    )
    selector._embed = lambda texts: pytest.fail("strict heldout must not rebuild")
    with pytest.raises(RuntimeError, match="manifest document hash"):
        selector._load_or_build_index()


def test_skillbank_selector_uses_antichain_policy(tmp_path: Path) -> None:
    manifest = {
        "format": "dynamix_node_skill_bank_v1",
        "nodes": [],
        "tree_index": {
            "root_node_id": "root",
            "children_by_node": {
                "root": ["left", "right"],
                "left": [],
                "right": [],
            },
        },
        "export_policy": {
            "heldout_retrieval": "tree_antichain_knapsack",
            "token_budget": 1000,
            "token_unit": 1,
            "tokenizer": {
                "regex_fallback_allowed": True,
            },
        },
    }
    (tmp_path / "node_bank_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    docs = [
        SkillNodeDocument(
            node_id=node_id,
            item_id=node_id,
            name=node_id,
            trigger="trigger",
            content="content",
            embedding_text=node_id,
            prompt_text=node_id,
            sha256=node_id,
            token_cost=10,
        )
        for node_id in ("root", "left", "right")
    ]
    selector = SkillBankSelector(skillbank_root=tmp_path)
    selector._docs = docs
    selector._embeddings = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=float,
    )
    selector._embed = lambda texts: [[1.0, 0.0]]
    selected = selector.select("query", top_k=10)
    assert {item.skill.node_id for item in selected} == {"left", "right"}


def test_antichain_selection_reserves_fixed_injection_overhead(
    tmp_path: Path,
) -> None:
    manifest = {
        "format": "dynamix_node_skill_bank_v1",
        "nodes": [],
        "tree_index": {
            "root_node_id": "root",
            "children_by_node": {"root": []},
        },
        "export_policy": {
            "heldout_retrieval": "tree_antichain_knapsack",
            "token_budget": 10,
            "token_unit": 1,
            "fixed_prompt_overhead_tokens": 6,
            "tokenizer": {
                "regex_fallback_allowed": True,
            },
        },
    }
    (tmp_path / "node_bank_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    selector = SkillBankSelector(skillbank_root=tmp_path)
    selector._docs = [
        SkillNodeDocument(
            node_id="root",
            item_id="root",
            name="root",
            trigger="trigger",
            content="content",
            embedding_text="root",
            prompt_text="root",
            sha256="root",
            token_cost=5,
        )
    ]
    selector._embeddings = np.asarray([[1.0, 0.0]], dtype=float)
    selector._embed = lambda texts: [[1.0, 0.0]]
    assert selector.select("query", top_k=1) == []


def test_cdost_skillbank_cannot_fall_back_to_dense_top_k(tmp_path: Path) -> None:
    manifest = {
        "format": "dynamix_node_skill_bank_v1",
        "tree_policy": "certified_dual_view_otd",
        "nodes": [],
        "export_policy": {},
    }
    (tmp_path / "node_bank_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    selector = SkillBankSelector(skillbank_root=tmp_path)
    selector._docs = [
        SkillNodeDocument(
            node_id="root",
            item_id="root",
            name="root",
            trigger="trigger",
            content="content",
            embedding_text="root",
            prompt_text="root",
            sha256="root",
            token_cost=10,
        )
    ]
    selector._embeddings = np.asarray([[1.0, 0.0]], dtype=float)
    selector._embed = lambda texts: [[1.0, 0.0]]
    with pytest.raises(ValueError, match="requires tree_antichain_knapsack"):
        selector.select("query", top_k=1)


def test_expected_cdost_policy_rejects_manifest_with_identity_removed(
    tmp_path: Path,
) -> None:
    manifest = {
        "format": "dynamix_node_skill_bank_v1",
        "nodes": [],
        "export_policy": {},
    }
    (tmp_path / "node_bank_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    selector = SkillBankSelector(
        skillbank_root=tmp_path,
        expected_tree_policy="certified_dual_view_otd",
    )
    selector._load_or_build_index = lambda: pytest.fail(
        "identity mismatch must fail before index loading"
    )
    with pytest.raises(ValueError, match="tree_policy does not match"):
        selector.select("query", top_k=1)


def test_cdost_skillbank_requires_complete_tree_index(tmp_path: Path) -> None:
    manifest = {
        "format": "dynamix_node_skill_bank_v1",
        "tree_policy": "certified_dual_view_otd",
        "nodes": [],
        "export_policy": {
            "heldout_retrieval": "tree_antichain_knapsack",
            "token_budget": 1000,
        },
        "tree_index": {"root_node_id": "root"},
    }
    (tmp_path / "node_bank_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    selector = SkillBankSelector(skillbank_root=tmp_path)
    selector._load_or_build_index = lambda: pytest.fail(
        "invalid CDOST manifest must fail before index loading"
    )
    with pytest.raises(ValueError, match="requires a complete tree_index"):
        selector.select("query", top_k=1)


@pytest.mark.parametrize(
    "children_by_node",
    [
        {"root": ["left", "right"], "left": []},
        {"root": ["left", "left"], "left": []},
        {
            "root": ["left", "right"],
            "left": ["shared", "left_leaf"],
            "right": ["shared", "right_leaf"],
            "shared": [],
            "left_leaf": [],
            "right_leaf": [],
        },
        {
            "root": ["left", "right"],
            "left": [],
            "right": [],
            "orphan": [],
        },
    ],
)
def test_cdost_skillbank_rejects_malformed_present_tree_map_before_embedding(
    tmp_path: Path,
    children_by_node: dict[str, list[str]],
) -> None:
    manifest = {
        "format": "dynamix_node_skill_bank_v1",
        "tree_policy": "certified_dual_view_otd",
        "nodes": [
            {
                "node_id": "root",
                "name": "root",
                "trigger": "trigger",
                "content": "content",
            }
        ],
        "export_policy": {
            "heldout_retrieval": "tree_antichain_knapsack",
            "token_budget": 1000,
        },
        "tree_index": {
            "root_node_id": "root",
            "children_by_node": children_by_node,
        },
    }
    (tmp_path / "node_bank_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    selector = SkillBankSelector(skillbank_root=tmp_path)
    selector._load_or_build_index = lambda: pytest.fail(
        "invalid CDOST manifest must fail before index loading"
    )
    with pytest.raises(ValueError, match="certified_dual_view_otd"):
        selector.select("query", top_k=1)


def test_cdost_skillbank_requires_every_atom_leaf_in_manifest(
    tmp_path: Path,
) -> None:
    manifest = {
        "format": "dynamix_node_skill_bank_v1",
        "tree_policy": "certified_dual_view_otd",
        "root_node_id": "root",
        "node_count": 1,
        "nodes": [
            {
                "node_id": "left",
                "name": "left",
                "trigger": "trigger",
                "content": "content",
                "parent_node_id": "root",
                "child_node_ids": [],
            }
        ],
        "export_policy": {
            "heldout_retrieval": "tree_antichain_knapsack",
            "token_budget": 1000,
        },
        "tree_index": {
            "root_node_id": "root",
            "children_by_node": {
                "root": ["left", "right"],
                "left": [],
                "right": [],
            },
        },
    }
    (tmp_path / "node_bank_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    selector = SkillBankSelector(skillbank_root=tmp_path)
    selector._load_or_build_index = lambda: pytest.fail(
        "missing atom leaf must fail before index loading"
    )
    with pytest.raises(ValueError, match="omits retrievable atom leaves"):
        selector.select("query", top_k=1)
