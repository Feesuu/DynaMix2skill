from __future__ import annotations

import asyncio
import importlib.util
import itertools
import json
import math
import shlex
import shutil
import sys
from pathlib import Path

import pytest

from dynamix_core.contract_cut_ebst import (
    ContractAtom,
    ContractCutTreeState,
    ExactContractCutOptimizer,
)
from dynamix_trace2skill.contract_cut_pipeline import (
    CompiledRegion,
    ConditionalRule,
    ContractAtomAnalyst,
    ContractCompiler,
    ContractCutBuildConfig,
    SkillContract,
    StableSkillRegistry,
    _lazy_contract_cut,
    atom_leakage_reasons,
    atom_messages,
    build_contract_cut_skills,
    export_skill_folders,
    render_skill_markdown,
)
from dynamix_trace2skill.schemas import RawTrajectoryRecord, TrajectoryStep
from dynamix_trace2skill.skillbank import (
    SkillBankSelector,
    selected_experience_to_system_content,
)


def _atom(index: int, *, total: int = 128) -> ContractAtom:
    angle = 2.0 * math.pi * index / total
    vector = (
        math.cos(angle),
        math.sin(angle),
        0.35 * math.cos(3.0 * angle),
        0.35 * math.sin(3.0 * angle),
    )
    return ContractAtom(
        atom_id=f"atom_{index:04d}",
        source_item_id=f"task_{index:04d}",
        trigger=f"operation family {index % 7}",
        scope="synthetic structured-artifact tasks",
        decision=f"use decision policy {index % 5}",
        invariant="preserve the requested data relationship",
        verification="reopen the artifact and verify representative outputs",
        failure_mode="the output violates the requested relationship",
        boundary_embedding=vector,
        provenance={"trajectory_id": f"trajectory_{index:04d}"},
    )


class _WhitespaceTokenizer:
    def count(self, text: str) -> int:
        return len(text.split())


class _CharacterTokenizer:
    def count(self, text: str) -> int:
        return len(text)


class _FakeEmbedding:
    async def embed_texts(self, texts, **_kwargs):
        return [[1.0, 0.0, 0.0] for _ in texts]


class _FakeGeneration:
    def __init__(self, *, split_multi_atom: bool = False):
        self.calls = []
        self.split_multi_atom = split_multi_atom

    async def chat_json(self, messages, *, schema_name, **kwargs):
        self.calls.append((schema_name, messages, kwargs))
        if schema_name == "ContractAtom":
            return {
                "trigger": "a structured spreadsheet transformation is requested",
                "scope": "workbook edits that preserve an existing data relationship",
                "decision": "inspect the workbook structure and apply the requested transformation",
                "invariant": "unrelated workbook content remains unchanged",
                "verification": "reopen and recalculate the workbook before checking representative outputs",
                "failure_mode": "an unverified edit can produce plausible but incorrect output",
            }
        marker = "Candidate Contract Atoms:\n"
        payload = json.loads(messages[1]["content"].split(marker, 1)[1])
        atom_ids = [atom["atom_id"] for atom in payload["atoms"]]
        if self.split_multi_atom and len(atom_ids) > 1:
            return {
                "status": "split_required",
                "reason": "synthetic incompatibility",
                "contract": None,
            }
        return {
            "status": "contract",
            "reason": "coherent synthetic contract",
            "contract": {
                "name": "Reliable workbook transformation",
                "applicability": "Use for structured workbook transformation tasks.",
                "objective": "Apply the requested transformation without altering unrelated content.",
                "conditional_rules": [
                    {
                        "condition": "A workbook transformation is requested.",
                        "procedure": "Inspect structure, apply the requested edit, save, and recalculate.",
                        "invariant": "Unrelated workbook content remains unchanged.",
                        "verification": "Reopen the output and inspect representative results.",
                        "recovery": "Diagnose the first mismatching operation before retrying.",
                        "source_atom_ids": atom_ids,
                    }
                ],
                "invariants": ["Unrelated workbook content remains unchanged."],
                "verification": "Reopen, recalculate, and inspect representative results.",
                "recovery": "Diagnose the first mismatching operation before retrying.",
                "source_atom_ids": atom_ids,
            },
        }


class _RevisionGeneration(_FakeGeneration):
    async def chat_json(self, messages, *, schema_name, **kwargs):
        if not self.calls:
            self.calls.append((schema_name, messages, kwargs))
            return {
                "trigger": "Rows must move from RANGES to a target sheet.",
                "scope": "Structured workbook edits.",
                "decision": "Match rows by stable keys.",
                "invariant": "Unrelated content remains unchanged.",
                "verification": "Reopen and inspect representative results.",
                "failure_mode": "Incorrect key matching can duplicate rows.",
            }
        return await super().chat_json(messages, schema_name=schema_name, **kwargs)


class _CompilerRepairGeneration(_FakeGeneration):
    async def chat_json(self, messages, *, schema_name, **kwargs):
        if not self.calls:
            self.calls.append((schema_name, messages, kwargs))
            return {
                "status": "contract",
                "reason": "malformed first draft",
                "contract": {"name": "Incomplete contract"},
            }
        assert messages[-2]["role"] == "assistant"
        assert "Incomplete contract" in messages[-2]["content"]
        return await super().chat_json(messages, schema_name=schema_name, **kwargs)


def _record(
    index: int,
    *,
    success: bool = True,
    observation: str = "ok",
    instruction: str = "Transform the workbook while preserving unrelated content.",
) -> RawTrajectoryRecord:
    return RawTrajectoryRecord(
        trajectory_id=f"trajectory-{index}",
        task_id=f"task-{index}",
        trial_index=0,
        instruction=instruction,
        instruction_type="Cell-Level Manipulation",
        answer_position="Z99",
        final_response="TASK_COMPLETE",
        success=success,
        verifier_score=1.0 if success else 0.0,
        verifier_feedback="" if success else "output mismatch",
        steps=[
            TrajectoryStep(
                step_id=0,
                raw_model_output="inspect and edit",
                action="run a workbook edit",
                observation=observation,
            )
        ],
        extra={"trace2skill_result": {"hard_score": int(success)}},
    )


def _load_experiment_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_contract_cut_ebst_experiment.py"
    spec = importlib.util.spec_from_file_location("contract_cut_experiment_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_atom_normalizes_the_single_boundary_embedding() -> None:
    atom = _atom(3)
    assert math.isclose(
        sum(value * value for value in atom.boundary_embedding),
        1.0,
    )
    assert atom.weight == 1.0
    assert "Decision" not in atom.boundary_text
    assert atom.verification in atom.boundary_text


def test_contract_tree_preserves_balance_and_unique_atom_placement() -> None:
    state = ContractCutTreeState()
    for index in range(200):
        state.insert(_atom(index, total=200))
        state.validate()

    audit = state.structural_audit()
    assert audit["tree_policy"] == "contract_cut_ebst"
    assert audit["atom_count"] == 200
    assert audit["all_leaves_same_depth"] is True
    assert audit["height_within_bound"] is True
    assert audit["unique_atom_placement"] is True
    assert audit["all_atom_weights_one"] is True
    assert audit["cut_domain"] == "ebst_nodes_plus_leaf_atom_entries"
    assert audit["atom_entry_terminal_count"] == 200
    assert audit["optimal_cut_guarantee"] == "exact_on_augmented_candidate_tree"
    assert all(
        4 <= node.occupancy <= 8
        for node_id, node in state.nodes.items()
        if node_id != state.root_id
    )


def test_first_overflow_uses_the_exact_126_partition_objective() -> None:
    state = ContractCutTreeState()
    for index in range(9):
        state.insert(_atom(index, total=17))

    event = state.split_events[0]
    assert event["enumerated_partition_count"] == 126
    entries = tuple(sorted(state.atoms))

    def radius(group: tuple[str, ...]) -> float:
        return min(
            max(state.distance(candidate, member) for member in group)
            for candidate in group
        )

    expected = min(
        (
            max(radius(left), radius(right)),
            radius(left) + radius(right),
            left,
            right,
        )
        for left in itertools.combinations(entries, 4)
        for right in (tuple(sorted(set(entries) - set(left))),)
    )
    actual = (
        event["max_objective_radius"],
        event["sum_objective_radius"],
        tuple(event["left_entries"]),
        tuple(event["right_entries"]),
    )
    assert actual[:2] == pytest.approx(expected[:2])
    assert actual[2:] == expected[2:]


def test_tree_round_trip_preserves_future_online_insertions() -> None:
    direct = ContractCutTreeState()
    for index in range(60):
        direct.insert(_atom(index))

    resumed = ContractCutTreeState()
    for index in range(24):
        resumed.insert(_atom(index))
    resumed = ContractCutTreeState.from_dict(resumed.to_dict())
    for index in range(24, 60):
        resumed.insert(_atom(index))

    assert resumed.canonical_json() == direct.canonical_json()


def test_exact_cut_is_an_antichain_and_covers_each_atom_once() -> None:
    state = ContractCutTreeState()
    for index in range(72):
        state.insert(_atom(index))

    result = ExactContractCutOptimizer(state, beta=1.0).solve()
    assert set(result.covered_atom_ids) == set(state.atoms)
    assert len(result.covered_atom_ids) == len(set(result.covered_atom_ids))
    selected = set(result.selected_node_ids)
    for node_id in selected:
        parent_id = state.candidate_parent_id(node_id)
        while parent_id is not None:
            assert parent_id not in selected
            parent_id = state.nodes[parent_id].parent_id
    assert math.isclose(
        result.objective,
        result.distortion + result.opening_cost,
    )


def test_infeasible_candidate_forces_the_cut_to_its_children() -> None:
    state = ContractCutTreeState()
    duplicate = _atom(0)
    for index in range(24):
        state.insert(
            ContractAtom(
                atom_id=f"duplicate_{index:03d}",
                source_item_id=f"task_{index:03d}",
                trigger=duplicate.trigger,
                scope=duplicate.scope,
                decision=duplicate.decision,
                invariant=duplicate.invariant,
                verification=duplicate.verification,
                failure_mode=duplicate.failure_mode,
                boundary_embedding=duplicate.boundary_embedding,
                provenance={"trajectory_id": f"trace_{index:03d}"},
            )
        )

    optimizer = ExactContractCutOptimizer(state, beta=1.0)
    initial = optimizer.solve()
    assert initial.selected_node_ids == (state.root_id,)
    split = optimizer.solve(infeasible_node_ids=(state.root_id,))
    assert state.root_id not in split.selected_node_ids
    assert len(split.selected_node_ids) > 1
    assert split.covered_atom_ids == initial.covered_atom_ids


def test_infeasible_multi_atom_leaf_falls_back_to_atom_terminals() -> None:
    state = ContractCutTreeState()
    for index in range(6):
        state.insert(_atom(index))
    assert state.nodes[state.root_id].is_leaf

    result = ExactContractCutOptimizer(state, beta=1.0).solve(
        infeasible_node_ids=(state.root_id,)
    )
    assert len(result.selected_node_ids) == 6
    assert all(state.is_atom_entry(value) for value in result.selected_node_ids)
    assert set(result.covered_atom_ids) == set(state.atoms)


def test_batch_boundaries_do_not_change_the_final_geometry() -> None:
    sequential = ContractCutTreeState()
    for index in range(80):
        sequential.insert(_atom(index))

    batched = ContractCutTreeState()
    for offset in range(0, 80, 8):
        for index in range(offset, offset + 8):
            batched.insert(_atom(index))
        batched.validate()

    assert batched.canonical_json() == sequential.canonical_json()


def test_success_and_failure_use_one_prompt_and_keep_full_rollout() -> None:
    marker = "END_OF_FULL_EVIDENCE"
    success_messages = atom_messages(_record(0, observation="x " * 1000 + marker))
    failure_messages = atom_messages(_record(1, success=False))

    assert success_messages[0]["content"] == failure_messages[0]["content"]
    assert marker in success_messages[1]["content"]
    assert "Z99" not in success_messages[1]["content"]
    assert '"success": false' in failure_messages[1]["content"].lower()


def test_atom_leakage_check_rejects_quoted_task_entity_names() -> None:
    record = _record(
        0,
        instruction="Copy matching rows from 'RANGES' into the 'LISTS' sheet.",
    )
    reasons = atom_leakage_reasons(
        record,
        {
            "trigger": "Data must move from RANGES to a target sheet.",
            "scope": "Structured workbook edits.",
            "decision": "Match rows by stable keys.",
            "invariant": "Unrelated content remains unchanged.",
            "verification": "Reopen and inspect representative results.",
            "failure_mode": "Incorrect key matching can duplicate rows.",
        },
    )
    assert "copied source literal" in reasons


def test_atom_leakage_check_rejects_incidental_task_quantities() -> None:
    record = _record(
        0,
        instruction="Filter a workbook with over 13,000 rows; the example has 207 rows.",
    )
    reasons = atom_leakage_reasons(
        record,
        {
            "trigger": "Large filtering jobs with over 13000 rows.",
            "scope": "Structured workbook edits.",
            "decision": "Filter rows by stable criteria.",
            "invariant": "Unrelated rows remain unchanged.",
            "verification": "Reopen and inspect representative results.",
            "failure_mode": "Incorrect range detection can omit rows.",
        },
    )
    assert "copied source literal" in reasons


def test_atom_semantic_revision_includes_the_previous_draft(tmp_path: Path) -> None:
    generation = _RevisionGeneration()
    analyst = ContractAtomAnalyst(
        generation,
        _FakeEmbedding(),
        tokenizer=_WhitespaceTokenizer(),
        max_prompt_tokens=100_000,
        draft_cache_path=tmp_path / "drafts.jsonl",
    )
    atoms, exclusions = asyncio.run(
        analyst.extract_many(
            [
                _record(
                    0,
                    instruction="Copy matching rows from 'RANGES' into a target sheet.",
                )
            ]
        )
    )

    assert len(atoms) == 1
    assert not exclusions
    revision_messages = generation.calls[1][1]
    assert revision_messages[-2]["role"] == "assistant"
    assert "RANGES" in revision_messages[-2]["content"]
    assert "not the schema" in revision_messages[-1]["content"]


def test_over_budget_atom_is_explicitly_excluded(tmp_path: Path) -> None:
    analyst = ContractAtomAnalyst(
        _FakeGeneration(),
        _FakeEmbedding(),
        tokenizer=_CharacterTokenizer(),
        max_prompt_tokens=100,
        draft_cache_path=tmp_path / "drafts.jsonl",
    )
    atoms, exclusions = asyncio.run(analyst.extract_many([_record(0)]))

    assert atoms == []
    assert len(exclusions) == 1
    assert exclusions[0].reason == "analyst_prompt_over_budget"
    assert exclusions[0].prompt_tokens > exclusions[0].prompt_budget


def test_lazy_feasibility_reaches_atom_terminals(tmp_path: Path) -> None:
    state = ContractCutTreeState()
    for index in range(6):
        state.insert(_atom(index))
    compiler = ContractCompiler(
        _FakeGeneration(split_multi_atom=True),
        tokenizer=_WhitespaceTokenizer(),
        max_prompt_tokens=100_000,
        cache_path=tmp_path / "compiler.json",
    )

    cut, regions, attempts = asyncio.run(
        _lazy_contract_cut(state, compiler, beta=1.0)
    )
    assert len(cut.selected_node_ids) == 6
    assert all(state.is_atom_entry(node_id) for node_id in cut.selected_node_ids)
    assert len(regions) == 6
    assert len(attempts) == 2


def test_compiler_rejects_scalar_source_atom_ids(tmp_path: Path) -> None:
    compiler = ContractCompiler(
        _FakeGeneration(),
        tokenizer=_WhitespaceTokenizer(),
        max_prompt_tokens=100_000,
        cache_path=tmp_path / "compiler.json",
    )
    payload = {
        "name": "Invalid source IDs",
        "applicability": "Use for tests.",
        "objective": "Reject malformed model output.",
        "conditional_rules": [
            {
                "condition": "A test runs.",
                "procedure": "Validate the schema.",
                "invariant": "IDs stay atomic.",
                "verification": "Check the parsed IDs.",
                "recovery": "Repair the response.",
                "source_atom_ids": "atom_0001",
            }
        ],
        "invariants": ["IDs stay atomic."],
        "verification": "Check the parsed IDs.",
        "recovery": "Repair the response.",
        "source_atom_ids": "atom_0001",
    }
    with pytest.raises(ValueError, match="list of strings"):
        compiler._contract_from_payload(payload)


def test_compiler_repair_includes_previous_payload(tmp_path: Path) -> None:
    state = ContractCutTreeState()
    state.insert(_atom(0))
    state.insert(_atom(1))
    generation = _CompilerRepairGeneration()
    compiler = ContractCompiler(
        generation,
        tokenizer=_WhitespaceTokenizer(),
        max_prompt_tokens=100_000,
        cache_path=tmp_path / "compiler.json",
    )

    result = asyncio.run(compiler.compile(state, state.root_id))

    assert result.status == "contract"
    assert len(generation.calls) == 2


def test_compiler_over_budget_repair_becomes_split_required(tmp_path: Path) -> None:
    state = ContractCutTreeState()
    state.insert(_atom(0))
    state.insert(_atom(1))
    generation = _CompilerRepairGeneration()
    compiler = ContractCompiler(
        generation,
        tokenizer=_CharacterTokenizer(),
        max_prompt_tokens=100_000,
        cache_path=tmp_path / "compiler.json",
    )
    initial_tokens = sum(
        len(message["role"] + "\n" + message["content"])
        for message in compiler.messages(state, state.root_id)
    )
    compiler.max_prompt_tokens = initial_tokens + 10

    result = asyncio.run(compiler.compile(state, state.root_id))

    assert result.status == "split_required"
    assert result.reason == "compiler_repair_prompt_over_budget"
    assert len(generation.calls) == 1


def test_export_refuses_to_replace_an_unowned_skills_directory(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "user-file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="unowned skill directory"):
        export_skill_folders(
            ContractCutTreeState(),
            (),
            skills_dir,
            tokenizer=_WhitespaceTokenizer(),
            registry=StableSkillRegistry(),
        )
    assert (skills_dir / "user-file.txt").read_text(encoding="utf-8") == "keep"


def test_mock_build_exports_one_top1_full_skill_folder(tmp_path: Path) -> None:
    generation = _FakeGeneration()
    result = asyncio.run(
        build_contract_cut_skills(
            [_record(index, success=index % 2 == 0) for index in range(9)],
            generation=generation,
            embedding=_FakeEmbedding(),
            tokenizer=_WhitespaceTokenizer(),
            run_dir=tmp_path / "run",
            config=ContractCutBuildConfig(),
        )
    )

    assert len(result.atoms) == 9
    assert not result.exclusions
    assert len(result.active_skills) == 1
    skills_dir = Path(result.skills_dir)
    manifest = json.loads(
        (skills_dir / "node_bank_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tree_policy"] == "contract_cut_ebst"
    assert manifest["export_policy"]["default_top_k"] == 1
    assert manifest["node_count"] == 1
    node = manifest["nodes"][0]
    assert "name:" in node["embedding_text"]
    assert "applicability:" in node["embedding_text"]
    assert "objective:" in node["embedding_text"]
    assert "answer_position" not in node["embedding_text"]
    assert "## Decision Rules" in node["prompt_text"]
    assert "references/atoms.json" in node["prompt_text"]
    assert [len(event["accepted_atom_ids"]) for event in result.batch_events] == [8, 1]
    call_sequence = [call[0] for call in generation.calls]
    assert call_sequence[:9] == ["ContractAtom"] * 8 + [
        "ContractCutSkillCompiler"
    ]
    assert call_sequence[9] == "ContractAtom"

    public_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in skills_dir.rglob("*.json")
    ]
    public_text = json.dumps(public_payloads, ensure_ascii=False)
    for forbidden in (
        "source_item_ids",
        "boundary_embedding",
        "verifier_feedback",
        "record_sha256",
        '"task_id"',
        "task-0",
    ):
        assert forbidden not in public_text
    private_registry = json.loads(
        (Path(result.run_dir) / "contract_cut" / "skill_registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert private_registry["versions"]
    assert all(row["skill_md_sha256"] for row in private_registry["versions"])

    selector = SkillBankSelector(
        skillbank_root=skills_dir,
        base_url="mock://deterministic",
        model="mock-embedding",
        expected_tree_policy="contract_cut_ebst",
    )
    selected = selector.select(
        "Transform the workbook.\n\nTask type: Cell-Level Manipulation",
        top_k=1,
    )
    rendered = selected_experience_to_system_content(selected)
    assert len(selected) == 1
    assert "## Decision Rules" in rendered
    assert "Optional reference directory" in rendered
    index_audit = selector.prepare_index()
    assert index_audit["document_count"] == 1
    assert index_audit["embedding_dimension"] > 0


def test_spreadsheet_bash_tool_scrubs_api_keys(tmp_path: Path, monkeypatch) -> None:
    from spreadsheet_agent.tools import create_bash_tool

    monkeypatch.setenv("OPENAI_API_KEY", "should-never-reach-agent-shell")
    monkeypatch.setenv("YD5_API_KEY", "should-never-reach-agent-shell")
    command = (
        f"{shlex.quote(sys.executable)} -c \"import os; "
        "print(os.getenv('OPENAI_API_KEY', 'missing')); "
        "print(os.getenv('YD5_API_KEY', 'missing'))\""
    )
    output = create_bash_tool(str(tmp_path)).execute(command=command)

    assert output.splitlines() == ["missing", "missing"]


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_spreadsheet_bash_tool_hides_parent_proc_and_private_run_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from spreadsheet_agent.tools import create_bash_tool

    working_dir = tmp_path / "work"
    private_dir = Path(__file__).resolve().parents[1] / "research" / "contract_cut_ebst"
    working_dir.mkdir()
    private_file = private_dir / "ALGORITHM_CONTRACT.md"
    assert private_file.is_file()
    monkeypatch.setenv("OPENAI_API_KEY", "parent-process-secret")
    monkeypatch.setenv("DYNAMIX_TOOL_BWRAP", "true")
    monkeypatch.setenv("DYNAMIX_TOOL_MASK_PATHS", str(private_dir))
    command = (
        f"{shlex.quote(sys.executable)} -c \"from pathlib import Path; "
        "print(b'parent-process-secret' in Path('/proc/1/environ').read_bytes()); "
        f"print(Path({str(private_file)!r}).exists())\""
    )

    output = create_bash_tool(str(working_dir)).execute(command=command)

    assert output.splitlines() == ["False", "False"]


def test_contract_cut_run_directory_has_an_exclusive_lock(tmp_path: Path) -> None:
    runner = _load_experiment_runner()
    first = runner._acquire_run_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="owns this run directory"):
            runner._acquire_run_lock(tmp_path)
    finally:
        runner.fcntl.flock(first.fileno(), runner.fcntl.LOCK_UN)
        first.close()


def test_heldout_environment_canonicalizes_custom_api_key_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_experiment_runner()
    args = runner.build_parser().parse_args(["--run-dir", str(tmp_path)])
    args.api_key_env_var = "CUSTOM_AUTH"
    monkeypatch.setenv("CUSTOM_AUTH", "review-sentinel")
    public_root = tmp_path / "public-skills"
    public_root.mkdir()

    env = runner._heldout_env(args, tmp_path, public_root)

    assert "CUSTOM_AUTH" not in env
    assert env["OPENAI_API_KEY"] == "review-sentinel"


def test_stage_marker_seals_all_declared_artifacts(tmp_path: Path) -> None:
    runner = _load_experiment_runner()
    summary = tmp_path / "reports" / "summary.json"
    artifact_dir = tmp_path / "skills"
    summary.parent.mkdir()
    artifact_dir.mkdir()
    summary.write_text('{"ok": true}', encoding="utf-8")
    skill = artifact_dir / "SKILL.md"
    skill.write_text("v1", encoding="utf-8")
    runner._write_completed_stage(
        tmp_path,
        stage="build",
        fingerprint="fingerprint",
        summary_path=summary,
        artifact_paths=(summary, artifact_dir),
    )
    assert runner._load_completed_stage(
        tmp_path,
        stage="build",
        fingerprint="fingerprint",
        summary_path=summary,
        artifact_paths=(summary, artifact_dir),
    ) == {"ok": True}

    skill.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts changed"):
        runner._load_completed_stage(
            tmp_path,
            stage="build",
            fingerprint="fingerprint",
            summary_path=summary,
            artifact_paths=(summary, artifact_dir),
        )


def test_compiler_identity_depends_on_atom_content_not_physical_node_id(
    tmp_path: Path,
) -> None:
    atom = _atom(0)

    class _State:
        atoms = {atom.atom_id: atom}

        @staticmethod
        def candidate_atom_ids(_node_id: str) -> tuple[str, ...]:
            return (atom.atom_id,)

    compiler = ContractCompiler(
        _FakeGeneration(),
        tokenizer=_WhitespaceTokenizer(),
        max_prompt_tokens=100_000,
        cache_path=tmp_path / "compiler.json",
    )

    assert compiler.input_sha256(_State(), "old-node") == compiler.input_sha256(
        _State(),
        "relocated-node",
    )


def test_skill_registry_records_split_and_merge_lineage() -> None:
    contract = SkillContract(
        name="Registry test",
        applicability="Use for registry tests.",
        objective="Track stable skill lineage.",
        conditional_rules=(
            ConditionalRule(
                condition="A region is selected.",
                procedure="Compile one skill.",
                invariant="Membership is auditable.",
                verification="Inspect registry history.",
                recovery="Rebuild from the prior batch.",
                source_atom_ids=("a", "b"),
            ),
        ),
        invariants=("Membership is auditable.",),
        verification="Inspect registry history.",
        recovery="Rebuild from the prior batch.",
        source_atom_ids=("a", "b"),
    )
    def region(node_id: str, atom_ids: tuple[str, ...], digest: str) -> CompiledRegion:
        payload = contract.to_dict()
        payload["source_atom_ids"] = list(atom_ids)
        payload["conditional_rules"][0]["source_atom_ids"] = list(atom_ids)
        return CompiledRegion(
            tree_node_id=node_id,
            member_atom_ids=atom_ids,
            source_item_ids=tuple(f"task-{value}" for value in atom_ids),
            input_sha256=digest,
            contract=SkillContract.from_dict(payload),
            compiler_cache_hit=False,
        )

    registry = StableSkillRegistry()
    initial = registry.update((region("root", ("a", "b"), "v1"),), batch_index=1)
    relocated = registry.update(
        (region("root-relocated", ("a", "b"), "v1"),),
        batch_index=2,
    )
    assert relocated[0].skill_id == initial[0].skill_id
    assert relocated[0].version == 1
    assert relocated[0].tree_node_id == "root-relocated"
    assert len(registry.history) == 1

    split = registry.update(
        (region("left", ("a",), "v2a"), region("right", ("b",), "v2b")),
        batch_index=3,
    )
    merged = registry.update((region("root2", ("a", "b"), "v3"),), batch_index=4)

    assert len(initial) == 1
    assert len(split) == 2
    assert len(merged) == 1
    assert merged[0].skill_id not in {skill.skill_id for skill in split}
    assert set(merged[0].derived_from) == {skill.skill_id for skill in split}
    initial_history = next(
        item for item in registry.history if item.skill_id == initial[0].skill_id
    )
    assert set(initial_history.superseded_by) == {skill.skill_id for skill in split}


def test_skill_markdown_has_the_fixed_contract_sections() -> None:
    contract = SkillContract(
        name="Test skill",
        applicability="Use for tests.",
        objective="Complete the test.",
        conditional_rules=(),
        invariants=("Preserve the invariant.",),
        verification="Verify the result.",
        recovery="Diagnose and retry.",
        source_atom_ids=("atom",),
    )
    markdown = render_skill_markdown(contract)
    for heading in (
        "## When To Use",
        "## Objective",
        "## Decision Rules",
        "## Procedure",
        "## Invariants",
        "## Verification",
        "## Recovery",
        "## References",
    ):
        assert heading in markdown
