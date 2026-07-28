from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dynamix_core.balanced_metric_tree import BalancedMetricTreeState
from dynamix_core.certified_otd import ExperienceAtom
from dynamix_core.skill_capsules import (
    SkillCapsule,
    SkillCapsuleRegistry,
    validate_capsule_candidate,
)
from dynamix_trace2skill.evidence_balanced_skill_pipeline import (
    EvidenceBalancedSkillConfig,
    SkillCapsuleAnalyst,
    _refresh_capsules,
    _write_nodebank_manifest,
)
from dynamix_trace2skill.pipeline import (
    _raise_if_ebst_build_incomplete,
    _refresh_skillbank_index,
)
from dynamix_trace2skill.skillbank import (
    SkillBankSelector,
    SkillNodeDocument,
    SkillSelection,
    selected_experience_to_system_content,
)


def _atom(
    index: int,
    *,
    total: int = 64,
    success: bool = True,
) -> ExperienceAtom:
    angle = 2.0 * math.pi * index / total
    trigger = (math.cos(angle), math.sin(angle), 0.5, 0.25)
    procedure = (
        math.cos(angle + 0.3),
        math.sin(angle + 0.3),
        0.25,
        0.5,
    )
    return ExperienceAtom(
        atom_id=f"atom_{index:03d}",
        source_item_id=f"task_{index:03d}",
        trigger=f"trigger {index}",
        scope="synthetic test",
        decision=f"decision {index}",
        invariant="preserve the tested invariant",
        verification="check the expected result",
        failure_mode="incorrect result",
        evidence_type="single_trace",
        reliability=1.0,
        trigger_embedding=trigger,
        procedure_embedding=procedure,
        metadata={"success": success},
    )


def test_balanced_tree_preserves_invariants_after_every_insert() -> None:
    state = BalancedMetricTreeState(max_entries=8, dual_view_lambda=0.5)
    for index in range(64):
        state.insert(_atom(index))

    audit = state.structural_audit()
    assert audit["atom_count"] == 64
    assert audit["all_leaves_same_depth"] is True
    assert audit["height_within_bound"] is True
    assert audit["cover_radius_certified_upper_bound"] is True
    assert audit["atoms_only_in_leaves"] is True
    assert audit["unique_atom_placement"] is True
    assert audit["split_count"] > 0


def test_static_and_resumed_sequential_insertion_are_identical() -> None:
    direct = BalancedMetricTreeState(max_entries=6, dual_view_lambda=0.4)
    for index in range(40):
        direct.insert(_atom(index))

    resumed = BalancedMetricTreeState(max_entries=6, dual_view_lambda=0.4)
    for index in range(20):
        resumed.insert(_atom(index))
    resumed = BalancedMetricTreeState.from_dict(resumed.to_dict())
    for index in range(20, 40):
        resumed.insert(_atom(index))

    assert resumed.canonical_json() == direct.canonical_json()


def test_exact_nearest_search_matches_brute_force() -> None:
    state = BalancedMetricTreeState(max_entries=8, dual_view_lambda=0.65)
    for index in range(48):
        state.insert(_atom(index))
    query = _atom(17)

    selected = state.nearest_atom_ids(
        trigger_embedding=query.trigger_embedding,
        procedure_embedding=query.procedure_embedding,
        k=7,
    )
    brute_force = tuple(
        atom_id
        for _, atom_id in sorted(
            (
                state.distance(query.atom_id, atom_id),
                atom_id,
            )
            for atom_id in state.atoms
        )[:7]
    )
    assert selected == brute_force


def test_exact_nearest_search_preserves_duplicate_vector_ties() -> None:
    state = BalancedMetricTreeState(max_entries=6, dual_view_lambda=0.5)
    template = _atom(0)
    for index in range(27):
        state.insert(
            replace(
                template,
                atom_id=f"a{index:04d}",
                source_item_id=f"task-{index:04d}",
                trigger=f"duplicate trigger {index}",
                decision=f"duplicate decision {index}",
            )
        )
    selected = state.nearest_atom_ids(
        trigger_embedding=template.trigger_embedding,
        procedure_embedding=template.procedure_embedding,
        k=4,
    )
    assert selected == ("a0000", "a0001", "a0002", "a0003")


def test_randomized_insertions_preserve_structure_search_and_locality() -> None:
    for max_entries in (4, 6, 8):
        order = list(range(160))
        random.Random(1000 + max_entries).shuffle(order)
        state = BalancedMetricTreeState(
            max_entries=max_entries,
            dual_view_lambda=0.35,
        )
        for insertion_index, atom_index in enumerate(order):
            before = {
                node_id: node.to_dict()
                for node_id, node in state.nodes.items()
            }
            result = state.insert(_atom(atom_index, total=160))
            state.validate()
            changed_or_retired = (
                set(result.affected_node_ids)
                | set(result.retired_node_ids)
            )
            for node_id, payload in before.items():
                if node_id not in changed_or_retired:
                    assert state.nodes[node_id].to_dict() == payload

            if insertion_index % 23 == 0:
                query = _atom(
                    (atom_index + 17) % 160,
                    total=160,
                )
                selected = state.nearest_atom_ids(
                    trigger_embedding=query.trigger_embedding,
                    procedure_embedding=query.procedure_embedding,
                    k=min(9, len(state.atoms)),
                )
                brute_force = tuple(
                    candidate_id
                    for _, candidate_id in sorted(
                        (
                            state._query_distance(  # noqa: SLF001
                                query.trigger_embedding,
                                query.procedure_embedding,
                                candidate_id,
                            ),
                            candidate_id,
                        )
                        for candidate_id in state.atoms
                    )[: min(9, len(state.atoms))]
                )
                assert selected == brute_force

        audit = state.structural_audit()
        assert audit["height_within_bound"] is True
        assert audit["all_leaves_same_depth"] is True


def test_atoms_are_not_duplicated_across_leaves() -> None:
    state = BalancedMetricTreeState(max_entries=4, dual_view_lambda=0.5)
    for index in range(25):
        state.insert(_atom(index))

    placements = [
        atom_id
        for node in state.nodes.values()
        if node.is_leaf
        for atom_id in node.atom_ids
    ]
    assert len(placements) == len(set(placements)) == 25
    assert all(
        not node.atom_ids
        for node in state.nodes.values()
        if not node.is_leaf
    )


def _capsule(
    state: BalancedMetricTreeState,
    node_id: str,
    *,
    name: str,
    child_capsule_ids: tuple[str, ...] = (),
) -> SkillCapsule:
    evidence_atom_ids = state.descendant_atom_ids(node_id)
    return SkillCapsule(
        capsule_id=f"skill_{node_id}_v0001",
        tree_node_id=node_id,
        version=1,
        level=1,
        name=name,
        trigger=f"use {name}",
        content=f"apply {name} carefully",
        scope="synthetic scope",
        verification="check the result",
        failure_modes=("incorrect result",),
        evidence_atom_ids=evidence_atom_ids,
        source_item_ids=tuple(
            state.atoms[atom_id].source_item_id
            for atom_id in evidence_atom_ids
        ),
        child_capsule_ids=child_capsule_ids,
        status="active",
    )


def test_singleton_evidence_cannot_become_retrievable_capsule() -> None:
    state = BalancedMetricTreeState(max_entries=4)
    state.insert(_atom(0))
    capsule = _capsule(state, state.root_id, name="singleton")

    assert "insufficient_evidence_atoms" in validate_capsule_candidate(capsule)


def test_parent_capsule_must_add_content_distinct_from_children() -> None:
    state = BalancedMetricTreeState(max_entries=4)
    for index in range(9):
        state.insert(_atom(index))
    root = state.nodes[state.root_id]
    left = _capsule(state, root.child_ids[0], name="left")
    right = _capsule(state, root.child_ids[1], name="right")
    parent = _capsule(
        state,
        state.root_id,
        name="left",
        child_capsule_ids=(left.capsule_id, right.capsule_id),
    )
    parent.trigger = left.trigger
    parent.content = left.content
    parent.scope = left.scope
    parent.verification = left.verification
    parent.failure_modes = left.failure_modes

    assert "duplicates_child_capsule" in validate_capsule_candidate(
        parent,
        child_capsules=(left, right),
    )


def test_registry_archives_replaced_capsules_and_rebuilds_links() -> None:
    state = BalancedMetricTreeState(max_entries=4)
    for index in range(9):
        state.insert(_atom(index))
    root = state.nodes[state.root_id]
    registry = SkillCapsuleRegistry()
    children = [
        _capsule(state, child_id, name=f"child {offset}")
        for offset, child_id in enumerate(root.child_ids)
    ]
    for capsule in children:
        registry.register(capsule, event_reason="initial")
    parent = _capsule(
        state,
        state.root_id,
        name="parent",
        child_capsule_ids=tuple(
            capsule.capsule_id for capsule in children
        ),
    )
    registry.register(parent, event_reason="initial")
    registry.rebuild_active_links()
    registry.validate(state)

    assert all(
        capsule.parent_capsule_id == parent.capsule_id
        for capsule in children
    )
    registry.archive_tree_nodes(
        [root.child_ids[0]],
        reason="evidence_changed",
    )
    assert children[0].status == "archived"
    assert root.child_ids[0] not in registry.active_by_tree_node


class _FakeCapsuleAnalyst:
    async def generate_leaf(
        self,
        tree: BalancedMetricTreeState,
        registry: SkillCapsuleRegistry,
        tree_node_id: str,
    ) -> SkillCapsule:
        capsule_id, version = registry.next_identity(tree_node_id)
        atom_ids = tree.descendant_atom_ids(tree_node_id)
        return SkillCapsule(
            capsule_id=capsule_id,
            tree_node_id=tree_node_id,
            version=version,
            level=1,
            name=f"leaf {tree_node_id}",
            trigger="related synthetic tasks",
            content=f"combine evidence for {tree_node_id}",
            scope="synthetic scope",
            verification="check the result",
            failure_modes=("incorrect result",),
            evidence_atom_ids=atom_ids,
            source_item_ids=tuple(
                tree.atoms[atom_id].source_item_id
                for atom_id in atom_ids
            ),
            status="active",
            metadata={"analyst_mode": "fake_leaf"},
        )

    async def generate_parent(
        self,
        tree: BalancedMetricTreeState,
        registry: SkillCapsuleRegistry,
        tree_node_id: str,
        child_capsules: tuple[SkillCapsule, ...],
    ) -> SkillCapsule:
        capsule_id, version = registry.next_identity(tree_node_id)
        atom_ids = tree.descendant_atom_ids(tree_node_id)
        return SkillCapsule(
            capsule_id=capsule_id,
            tree_node_id=tree_node_id,
            version=version,
            level=2,
            name=f"parent {tree_node_id}",
            trigger="broader synthetic tasks",
            content=f"abstract across {len(child_capsules)} child skills",
            scope="broader synthetic scope",
            verification="check all child invariants",
            failure_modes=("unsupported abstraction",),
            evidence_atom_ids=atom_ids,
            source_item_ids=tuple(
                tree.atoms[atom_id].source_item_id
                for atom_id in atom_ids
            ),
            child_capsule_ids=tuple(
                capsule.capsule_id for capsule in child_capsules
            ),
            status="active",
            metadata={"analyst_mode": "fake_parent"},
        )


class _FakeTokenizer:
    def count(self, text: str) -> int:
        return max(1, len(text.split()))


class _FakeGeneration:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def chat_json(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return dict(self.payload)


class _FailingGeneration:
    def __init__(self, error: Exception):
        self.error = error

    async def chat_json(self, *args, **kwargs):
        raise self.error


def test_leaf_capsule_separates_positive_and_negative_evidence() -> None:
    async def run() -> None:
        tree = BalancedMetricTreeState(max_entries=4)
        tree.insert(_atom(0, success=True))
        tree.insert(_atom(1, success=False))
        generation = _FakeGeneration(
            {
                "promote": True,
                "name": "validated procedure",
                "trigger": "matching tasks",
                "content": "apply the supported procedure",
                "scope": "supported cases",
                "verification": "verify the result",
                "failure_modes": ["unsupported evidence"],
                "rationale": "positive support and negative guardrail",
            }
        )
        validator = _FakeGeneration(
            {
                "uses_negative_for_root_cause_boundary_or_guardrail": True,
                "recommends_observed_failed_action": False,
                "copies_example_specific_content": False,
                "accept": True,
                "reason": "negative evidence is used as a guardrail",
            }
        )
        analyst = SkillCapsuleAnalyst(
            generation,
            validator,
            tokenizer=_FakeTokenizer(),
            max_prompt_tokens=100000,
            max_output_tokens=4096,
            validation_mode="structural_only",
        )
        capsule = await analyst.generate_leaf(
            tree,
            SkillCapsuleRegistry(),
            tree.root_id,
        )
        assert capsule is not None
        messages = generation.calls[0][0][0]
        payload = json.loads(messages[1]["content"])
        assert len(payload["positive_evidence_atoms"]) == 1
        assert len(payload["negative_evidence_atoms"]) == 1
        assert payload["unknown_outcome_evidence_atoms"] == []
        assert capsule.metadata["positive_evidence_count"] == 1
        assert capsule.metadata["negative_evidence_count"] == 1
        assert capsule.status == "active"
        assert len(validator.calls) == 1

    asyncio.run(run())


def test_leaf_capsule_rejects_unsafe_negative_evidence_use() -> None:
    async def run() -> None:
        tree = BalancedMetricTreeState(max_entries=4)
        tree.insert(_atom(0, success=False))
        tree.insert(_atom(1, success=False))
        analyst = SkillCapsuleAnalyst(
            _FakeGeneration(
                {
                    "promote": True,
                    "name": "unsafe procedure",
                    "trigger": "matching tasks",
                    "content": "repeat the observed failed action",
                    "scope": "all cases",
                    "verification": "none",
                    "failure_modes": ["the same failure"],
                    "rationale": "copied from failures",
                }
            ),
            _FakeGeneration(
                {
                    "uses_negative_for_root_cause_boundary_or_guardrail": False,
                    "recommends_observed_failed_action": True,
                    "copies_example_specific_content": False,
                    "accept": True,
                    "reason": "the proposal repeats the failed action",
                }
            ),
            tokenizer=_FakeTokenizer(),
            max_prompt_tokens=100000,
            max_output_tokens=4096,
            validation_mode="structural_only",
        )
        capsule = await analyst.generate_leaf(
            tree,
            SkillCapsuleRegistry(),
            tree.root_id,
        )
        assert capsule is not None
        assert capsule.status == "rejected"
        assert (
            "negative_evidence_validation_rejected"
            in capsule.validation_reasons
        )

    asyncio.run(run())


def test_capsule_generation_classifies_runtime_and_prompt_budget_errors() -> None:
    async def generate(error: Exception) -> SkillCapsule:
        tree = BalancedMetricTreeState(max_entries=4)
        tree.insert(_atom(0))
        tree.insert(_atom(1))
        analyst = SkillCapsuleAnalyst(
            _FailingGeneration(error),
            _FakeGeneration({}),
            tokenizer=_FakeTokenizer(),
            max_prompt_tokens=100000,
            max_output_tokens=4096,
            validation_mode="structural_only",
        )
        capsule = await analyst.generate_leaf(
            tree,
            SkillCapsuleRegistry(),
            tree.root_id,
        )
        assert capsule is not None
        return capsule

    runtime = asyncio.run(generate(RuntimeError("service unavailable")))
    assert runtime.metadata["rejection_classes"] == [
        "runtime_generation_error"
    ]
    prompt_budget = asyncio.run(
        generate(
            ValueError(
                "evidence-balanced skill capsule generation prompt exceeds "
                "configured budget: 101 > 100"
            )
        )
    )
    assert prompt_budget.metadata["rejection_classes"] == [
        "prompt_budget_error"
    ]


def test_capsule_registry_records_candidate_before_final_status() -> None:
    tree = BalancedMetricTreeState(max_entries=4)
    tree.insert(_atom(0))
    tree.insert(_atom(1))
    registry = SkillCapsuleRegistry()
    capsule = asyncio.run(
        _FakeCapsuleAnalyst().generate_leaf(
            tree,
            registry,
            tree.root_id,
        )
    )
    assert capsule is not None
    registry.register(capsule, event_reason="test")
    assert [event["status"] for event in registry.events] == [
        "candidate",
        "active",
    ]


def test_local_capsule_refresh_excludes_atoms_and_singletons(
    tmp_path,
) -> None:
    asyncio.run(_run_local_capsule_refresh_test(tmp_path))


async def _run_local_capsule_refresh_test(tmp_path) -> None:
    tree = BalancedMetricTreeState(max_entries=4)
    for index in range(16):
        tree.insert(_atom(index))
    registry = SkillCapsuleRegistry()
    analyst = _FakeCapsuleAnalyst()
    await _refresh_capsules(
        tree=tree,
        registry=registry,
        analyst=analyst,
        dirty_node_ids=set(tree.nodes),
        retired_node_ids=set(),
        reason="initial",
    )

    active_before = dict(registry.active_by_tree_node)
    assert active_before
    assert all(
        len(registry.capsules[capsule_id].evidence_atom_ids) >= 2
        for capsule_id in active_before.values()
    )

    result = tree.insert(_atom(16))
    await _refresh_capsules(
        tree=tree,
        registry=registry,
        analyst=analyst,
        dirty_node_ids=set(result.capsule_refresh_node_ids),
        retired_node_ids=set(result.retired_node_ids),
        reason="arrival",
    )
    registry.validate(tree)
    unaffected = (
        set(active_before)
        - set(result.capsule_refresh_node_ids)
        - set(result.retired_node_ids)
    )
    assert all(
        registry.active_by_tree_node[node_id] == active_before[node_id]
        for node_id in unaffected
        if node_id in registry.active_by_tree_node
    )

    config = SimpleNamespace(
        analyst=SimpleNamespace(
            tokenizer_model=None,
            tokenizer_required=False,
            allow_regex_tokenizer_fallback=True,
        ),
        embedding=SimpleNamespace(tokenizer_model=None),
    )
    manifest = _write_nodebank_manifest(
        tree=tree,
        registry=registry,
        output_dir=tmp_path / "skills",
        config=config,
        ebst=EvidenceBalancedSkillConfig(),
        tokenizer=_FakeTokenizer(),
    )
    assert manifest["node_count"] == len(registry.active_by_tree_node)
    assert manifest["export_policy"]["experience_atoms_exported"] is False
    assert not {
        node["node_id"] for node in manifest["nodes"]
    }.intersection(tree.atoms)


def test_parent_promotion_validator_can_reject_paraphrase() -> None:
    async def run() -> None:
        tree = BalancedMetricTreeState(max_entries=4)
        for index in range(9):
            tree.insert(_atom(index))
        root = tree.nodes[tree.root_id]
        registry = SkillCapsuleRegistry()
        children = [
            _capsule(tree, child_id, name=f"child {offset}")
            for offset, child_id in enumerate(root.child_ids)
        ]
        for capsule in children:
            registry.register(capsule, event_reason="test")
        analyst = SkillCapsuleAnalyst(
            _FakeGeneration(
                {
                    "promote": True,
                    "name": "shared procedure",
                    "trigger": "related tasks",
                    "content": "apply the shared procedure",
                    "scope": "shared scope",
                    "verification": "verify the result",
                    "failure_modes": ["unsupported input"],
                    "rationale": "claimed shared abstraction",
                }
            ),
            _FakeGeneration(
                {
                    "supported_by_multiple_children": False,
                    "adds_cross_child_abstraction": False,
                    "copies_example_specific_content": False,
                    "accept": True,
                    "reason": "only one child supports the rule",
                }
            ),
            tokenizer=_FakeTokenizer(),
            max_prompt_tokens=100000,
            max_output_tokens=4096,
            validation_mode="structural_only",
        )
        capsule = await analyst.generate_parent(
            tree,
            registry,
            tree.root_id,
            tuple(children),
        )
        assert capsule is not None
        assert capsule.status == "rejected"
        assert "parent_promotion_rejected" in capsule.validation_reasons

    asyncio.run(run())


def test_parent_capsule_records_only_presented_child_evidence() -> None:
    async def run() -> None:
        tree = BalancedMetricTreeState(max_entries=4)
        for index in range(80):
            tree.insert(_atom(index, total=80))
        parent = next(
            node
            for node in tree.nodes.values()
            if not node.is_leaf and len(node.child_ids) >= 3
        )
        children = [
            _capsule(tree, child_id, name=f"child {offset}")
            for offset, child_id in enumerate(parent.child_ids[:2])
        ]
        analyst = SkillCapsuleAnalyst(
            _FakeGeneration(
                {
                    "promote": True,
                    "name": "shared abstraction",
                    "trigger": "related tasks",
                    "content": "apply the cross-child rule",
                    "scope": "shared cases",
                    "verification": "verify both child invariants",
                    "failure_modes": ["unsupported cases"],
                    "rationale": "supported by two branches",
                }
            ),
            _FakeGeneration(
                {
                    "supported_by_multiple_children": True,
                    "adds_cross_child_abstraction": True,
                    "copies_example_specific_content": False,
                    "accept": True,
                    "reason": "two branches support a new abstraction",
                }
            ),
            tokenizer=_FakeTokenizer(),
            max_prompt_tokens=100000,
            max_output_tokens=4096,
            validation_mode="structural_only",
        )
        capsule = await analyst.generate_parent(
            tree,
            SkillCapsuleRegistry(),
            parent.node_id,
            tuple(children),
        )
        assert capsule is not None
        expected = {
            atom_id
            for child in children
            for atom_id in child.evidence_atom_ids
        }
        assert set(capsule.evidence_atom_ids) == expected
        assert expected < set(tree.descendant_atom_ids(parent.node_id))

    asyncio.run(run())


def test_parent_refresh_requires_two_direct_child_branches() -> None:
    async def run() -> None:
        tree = BalancedMetricTreeState(max_entries=4)
        for index in range(80):
            tree.insert(_atom(index, total=80))
        branch = next(
            node
            for node in tree.nodes.values()
            if (
                node.parent_id is not None
                and not node.is_leaf
                and len(node.child_ids) >= 2
                and all(
                    tree.nodes[child_id].is_leaf
                    for child_id in node.child_ids
                )
            )
        )
        parent_id = branch.parent_id
        assert parent_id is not None
        registry = SkillCapsuleRegistry()
        for offset, child_id in enumerate(branch.child_ids[:2]):
            registry.register(
                _capsule(tree, child_id, name=f"same branch {offset}"),
                event_reason="test",
            )
        result = await _refresh_capsules(
            tree=tree,
            registry=registry,
            analyst=_FakeCapsuleAnalyst(),
            dirty_node_ids={parent_id},
            retired_node_ids=set(),
            reason="test",
        )
        assert parent_id in result["skipped_tree_node_ids"]
        assert parent_id not in registry.active_by_tree_node

    asyncio.run(run())


def test_ebst_nodebank_uses_shared_vector_cache_and_full_prompt(
    tmp_path,
) -> None:
    bank = tmp_path / "skills"
    bank.mkdir()
    prompt_text = (
        "### Capsule\nTrigger: matching tasks\nScope: supported cases\n\n"
        "Apply the validated rule.\n\nVerification: check the result\n"
        "Failure modes:\n- unsupported input"
    )
    embedding_text = (
        "name: Capsule\ntrigger: matching tasks\n"
        "content: Apply the validated rule."
    )
    (bank / "node_bank_manifest.json").write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "tree_policy": "evidence_balanced_skill_tree",
                "nodes": [
                    {
                        "node_id": "skill-1",
                        "item_id": "skill-1",
                        "name": "Capsule",
                        "trigger": "matching tasks",
                        "content": "Apply the validated rule.",
                        "embedding_text": embedding_text,
                        "prompt_text": prompt_text,
                        "sha256": hashlib.sha256(
                            embedding_text.encode("utf-8")
                        ).hexdigest(),
                        "analyst_mode": "evidence_bucket_consolidation",
                    }
                ],
                "tree_index": {
                    "root_node_id": "__root__",
                    "children_by_node": {
                        "__root__": ["skill-1"],
                        "skill-1": [],
                    },
                },
                "export_policy": {
                    "heldout_retrieval": "tree_antichain_knapsack",
                    "token_budget": 24000,
                    "token_unit": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    vector_cache = tmp_path / "vectors.sqlite"
    config = SimpleNamespace(
        hierarchy={"tree_policy": "evidence_balanced_skill_tree"},
        scenario="static_build",
        chunked_embedding={"enabled": False},
        embedding=SimpleNamespace(
            base_url="mock://ebst-cache",
            model="mock-embed",
            resolved_api_key="EMPTY",
            cache_path=str(vector_cache),
            max_model_len=32000,
            effective_max_input_tokens=32000,
            batch_size=8,
            tokenizer_model=None,
        ),
    )
    _refresh_skillbank_index(bank, config)
    with sqlite3.connect(vector_cache) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0] == 1
        original_vector = connection.execute(
            "SELECT vector FROM embeddings"
        ).fetchone()[0]

    rendered = selected_experience_to_system_content(
        [
            SkillSelection(
                skill=SkillNodeDocument(
                    node_id="skill-1",
                    item_id="skill-1",
                    name="Capsule",
                    trigger="matching tasks",
                    content="Apply the validated rule.",
                    embedding_text=embedding_text,
                    prompt_text=prompt_text,
                    sha256="hash",
                    analyst_mode="evidence_bucket_consolidation",
                ),
                score=1.0,
            )
        ]
    )
    assert "Scope: supported cases" in rendered
    assert "Verification: check the result" in rendered
    assert "Failure modes:" in rendered

    manifest_path = bank / "node_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    second_embedding_text = (
        "name: Second capsule\ntrigger: other tasks\n"
        "content: Apply a second validated rule."
    )
    manifest["nodes"].append(
        {
            **manifest["nodes"][0],
            "node_id": "skill-2",
            "item_id": "skill-2",
            "name": "Second capsule",
            "trigger": "other tasks",
            "content": "Apply a second validated rule.",
            "embedding_text": second_embedding_text,
            "sha256": hashlib.sha256(
                second_embedding_text.encode("utf-8")
            ).hexdigest(),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    config.scenario = "dynamic_update"
    _refresh_skillbank_index(bank, config)
    with sqlite3.connect(vector_cache) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT vector FROM embeddings ORDER BY rowid LIMIT 1"
        ).fetchone()[0] == original_vector


def test_selected_experience_preserves_legacy_prompt_format() -> None:
    legacy = SkillNodeDocument(
        node_id="legacy-1",
        item_id="legacy-1",
        name="Legacy",
        trigger="matching tasks",
        content="Legacy guidance.",
        embedding_text="legacy",
        prompt_text="THIS FULL PROMPT MUST NOT BE USED",
        sha256="hash",
        analyst_mode="cluster_summary",
    )
    rendered = selected_experience_to_system_content(
        [SkillSelection(skill=legacy, score=1.0)]
    )
    assert "## Node 1: Legacy" in rendered
    assert "THIS FULL PROMPT MUST NOT BE USED" not in rendered

    cdost = replace(
        legacy,
        node_id="cdost-1",
        item_id="cdost-1",
        prompt_text="CDOST FULL PROMPT",
        analyst_mode="cdost_parent",
    )
    rendered = selected_experience_to_system_content(
        [SkillSelection(skill=cdost, score=1.0)]
    )
    assert "CDOST FULL PROMPT" in rendered


def test_ebst_selector_rejects_lineage_mismatch(tmp_path) -> None:
    bank = tmp_path / "skills"
    bank.mkdir()
    (bank / "node_bank_manifest.json").write_text(
        json.dumps(
            {
                "format": "dynamix_node_skill_bank_v1",
                "tree_policy": "evidence_balanced_skill_tree",
                "root_node_id": "__root__",
                "node_count": 1,
                "nodes": [
                    {
                        "node_id": "skill-1",
                        "item_id": "skill-1",
                        "name": "Capsule",
                        "trigger": "matching tasks",
                        "content": "Apply the validated rule.",
                        "embedding_text": (
                            "name: Capsule\n"
                            "trigger: matching tasks\n"
                            "content: Apply the validated rule."
                        ),
                        "prompt_text": "Validated capsule prompt.",
                        "analyst_mode": "evidence_bucket_consolidation",
                        "lifecycle_status": "active",
                        "evidence_atom_ids": ["atom-1", "atom-2"],
                        "parent_node_id": "wrong-parent",
                        "child_node_ids": [],
                    }
                ],
                "tree_index": {
                    "root_node_id": "__root__",
                    "children_by_node": {
                        "__root__": ["skill-1"],
                        "skill-1": [],
                    },
                },
                "export_policy": {
                    "heldout_retrieval": "tree_antichain_knapsack",
                    "token_budget": 24000,
                    "token_unit": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    selector = SkillBankSelector(
        skillbank_root=bank,
        base_url="mock://ebst",
        model="mock-embed",
        expected_tree_policy="evidence_balanced_skill_tree",
    )
    with pytest.raises(ValueError, match="lineage does not match"):
        selector.select("query", top_k=1)


def test_ebst_runtime_errors_fail_the_build_process() -> None:
    config = SimpleNamespace(
        hierarchy={"tree_policy": "evidence_balanced_skill_tree"}
    )
    with pytest.raises(RuntimeError, match="incomplete capsule artifacts"):
        _raise_if_ebst_build_incomplete(
            config,
            {
                "runtime_generation_error_count": 1,
                "prompt_budget_error_count": 0,
            },
        )
    _raise_if_ebst_build_incomplete(
        config,
        {
            "runtime_generation_error_count": 0,
            "prompt_budget_error_count": 0,
        },
    )
