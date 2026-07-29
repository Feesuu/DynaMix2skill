from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import pytest

from dynamix_trace2skill.clients import EmbeddingConfig, GenerationConfig
from dynamix_trace2skill.pipeline import DynaMixRunConfig
from dynamix_trace2skill.schemas import RawTrajectoryRecord


def _load_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_ebst_closed_loop_experiment.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_ebst_closed_loop_experiment_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> DynaMixRunConfig:
    return DynaMixRunConfig(
        output_dir=str(tmp_path / "tree"),
        records_path=str(tmp_path / "records.json"),
        generation=GenerationConfig(
            base_url="http://llm.invalid/v1",
            model="Qwen3.5-9B-AWQ",
            api_key="secret",
        ),
        embedding=EmbeddingConfig(
            base_url="http://embedding.invalid/v1",
            model="Qwen3-Embedding-8B",
            cache_path=str(tmp_path / "vectors.sqlite"),
            cache_write_policy="first_write_wins",
        ),
        chunked_embedding={
            "enabled": True,
            "chunk_tokens": 8000,
            "overlap_tokens": 1000,
            "pooling": "mean",
        },
        hierarchy={"tree_policy": "evidence_balanced_skill_tree"},
    )


def test_closed_loop_rollout_command_is_single_task_and_secret_free(
    tmp_path: Path,
) -> None:
    module = _load_runner()
    config = _config(tmp_path)
    args = argparse.Namespace(
        python_executable="/env/bin/python",
        data_path="/dataset",
        nodebank_dir=tmp_path / "nodebank",
        max_turns=30,
        rollout_temperature=0.0,
        rollout_timeout_seconds=1200,
        rollout_retry_wait_seconds=(5.0, 10.0, 30.0),
    )

    command = module._rollout_command(
        args=args,
        config=config,
        task_index=127,
        task_dir=tmp_path / "task",
        nodebank_active=True,
        generation_config_path=tmp_path / "generation.json",
    )

    assert command[command.index("--agent") + 1] == "cli_skill_preloaded"
    assert command[command.index("--start_idx") + 1] == "127"
    assert command[command.index("--end_idx") + 1] == "128"
    assert command[command.index("--workers") + 1] == "1"
    assert command[
        command.index("--llm_retry_wait_seconds") + 1
    ] == "5.0,10.0,30.0"
    assert "secret" not in command


def test_closed_loop_env_exposes_only_current_nodebank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner()
    config = _config(tmp_path)
    args = argparse.Namespace(skillbank_top_k=10)
    monkeypatch.setenv("DYNAMIX_SKILLBANK_ROOT", "/stale")
    monkeypatch.setenv("DYNAMIX_SKILL_SELECTION_LOG", "/stale/log")

    clean = module._rollout_env(
        args=args,
        config=config,
        nodebank_root=tmp_path / "nodebank",
        selection_log=tmp_path / "selection.jsonl",
        nodebank_active=False,
    )
    active = module._rollout_env(
        args=args,
        config=config,
        nodebank_root=tmp_path / "nodebank",
        selection_log=tmp_path / "selection.jsonl",
        nodebank_active=True,
    )

    assert "DYNAMIX_SKILLBANK_ROOT" not in clean
    assert "DYNAMIX_SKILL_SELECTION_LOG" not in clean
    assert active["DYNAMIX_SKILLBANK_ROOT"] == str(
        tmp_path / "nodebank"
    )
    assert active["DYNAMIX_SKILLBANK_TOP_K"] == "10"
    assert active["DYNAMIX_SKILL_SELECTION_LOG"] == str(
        tmp_path / "selection.jsonl"
    )
    assert active["DYNAMIX_SKILLBANK_USAGE_LOG"] == str(
        tmp_path / "skillbank_usage.jsonl"
    )
    assert active["DYNAMIX_SKILLBANK_REQUIRE_CACHE_MATCH"] == "true"
    assert (
        active["DYNAMIX_SKILLBANK_REQUIRE_VECTOR_CACHE_MATCH"]
        == "false"
    )
    assert "DYNAMIX_SKILLBANK_CHUNK_TOKENS" not in active


def test_closed_loop_retry_schedule_is_parsed_as_floats() -> None:
    module = _load_runner()
    assert module._parse_float_csv("5,10,30") == (
        5.0,
        10.0,
        30.0,
    )


def test_closed_loop_launcher_parses_manifest_boolean_strings() -> None:
    module = _load_runner()
    launcher = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "tree_v4"
        / "run_closed_loop.sh"
    ).read_text(encoding="utf-8")

    assert module._parse_bool("false") is False
    assert module._parse_bool(False) is False
    assert '"thinking": _parse_bool(rollout["thinking"])' in launcher
    assert re.search(
        r'"response_cache_enabled": _parse_bool\(\s*'
        r'rollout\["response_cache_enabled"\]',
        launcher,
    )
    assert '"thinking": bool(rollout["thinking"])' not in launcher


def test_closed_loop_feedback_is_attached_without_changing_task() -> None:
    module = _load_runner()
    record = RawTrajectoryRecord(
        trajectory_id="task-120::trial-0",
        task_id="task-120",
        trial_index=0,
        instruction="Update the workbook.",
        instruction_type="spreadsheet",
        answer_position="E6:E13",
        success=True,
        verifier_score=1.0,
    )

    updated = module._record_with_feedback(
        record,
        task_index=120,
        selection={
            "selected_node_ids": ["skill-node-v1"],
            "selected_node_scores": [0.8],
            "query": "Update the workbook.\n\nTask type: spreadsheet",
        },
    )

    assert updated.trajectory_id == record.trajectory_id
    assert updated.instruction == record.instruction
    assert updated.answer_position == record.answer_position
    feedback = updated.extra["closed_loop_skill_evolution"]
    assert feedback["selected_capsule_ids"] == ["skill-node-v1"]
    assert feedback["attribution"] == "exposure_outcome_not_causal"


def test_selection_log_must_have_exactly_one_row(tmp_path: Path) -> None:
    module = _load_runner()
    path = tmp_path / "selection.jsonl"
    instance = argparse.Namespace(
        id="task-1",
        instruction="Update the workbook.",
        instruction_type="spreadsheet",
    )
    row = {
        "instance_id": "task-1",
        "instruction": "Update the workbook.",
        "instruction_type": "spreadsheet",
        "query": "Update the workbook.\n\nTask type: spreadsheet",
        "top_k": 10,
        "selected_node_ids": ["skill-1"],
        "selected_node_scores": [0.7],
    }
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        module._read_selection(
            path,
            expected_instance=instance,
            expected_top_k=10,
            active_node_ids={"skill-1"},
        )


def test_active_nodebank_requires_selection_log(tmp_path: Path) -> None:
    module = _load_runner()

    with pytest.raises(FileNotFoundError, match="required skill selection"):
        module._read_selection(
            tmp_path / "missing.jsonl",
            expected_instance=argparse.Namespace(
                id="task-1",
                instruction="Update the workbook.",
                instruction_type="spreadsheet",
            ),
            expected_top_k=10,
            active_node_ids={"skill-1"},
        )


def test_selection_log_is_bound_to_current_task(tmp_path: Path) -> None:
    module = _load_runner()
    path = tmp_path / "selection.jsonl"
    path.write_text(
        json.dumps(
            {
                "instance_id": "wrong-task",
                "instruction": "Update the workbook.",
                "instruction_type": "spreadsheet",
                "query": (
                    "Update the workbook.\n\nTask type: spreadsheet"
                ),
                "top_k": 10,
                "selected_node_ids": ["skill-1"],
                "selected_node_scores": [0.7],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="wrong instance_id"):
        module._read_selection(
            path,
            expected_instance=argparse.Namespace(
                id="task-1",
                instruction="Update the workbook.",
                instruction_type="spreadsheet",
            ),
            expected_top_k=10,
            active_node_ids={"skill-1"},
        )


def test_resume_contract_rejects_protocol_changes(tmp_path: Path) -> None:
    module = _load_runner()
    path = tmp_path / "experiment_contract.json"
    payload = {"model": "model-a", "top_k": 10}

    expected_hash = module._canonical_sha256(payload)
    assert module._bind_experiment_contract(
        path,
        payload,
        resume=True,
    ) == expected_hash
    assert module._bind_experiment_contract(
        path,
        payload,
        resume=True,
    ) == expected_hash
    with pytest.raises(RuntimeError, match="does not match"):
        module._bind_experiment_contract(
            path,
            {**payload, "top_k": 4},
            resume=True,
        )


def test_checkpoint_protocol_binds_source_records_and_contract() -> None:
    module = _load_runner()
    marker = {
        "trajectory_source": "closed_loop_skill_evolution",
        "atom_protocol_fingerprint": "atom",
        "tree_protocol_fingerprint": "tree",
        "record_prefix_sha256": "records",
        "metadata": {"experiment_contract_sha256": "contract"},
    }

    module._validate_checkpoint_protocol(
        marker,
        trajectory_source="closed_loop_skill_evolution",
        atom_protocol_fingerprint="atom",
        tree_protocol_fingerprint="tree",
        record_prefix_sha256="records",
        experiment_contract_sha256="contract",
    )
    with pytest.raises(RuntimeError, match="protocol mismatch"):
        module._validate_checkpoint_protocol(
            marker,
            trajectory_source="open_loop_replay",
            atom_protocol_fingerprint="atom",
            tree_protocol_fingerprint="tree",
            record_prefix_sha256="records",
        )


def test_task_directory_cleanup_requires_owned_marker(
    tmp_path: Path,
) -> None:
    module = _load_runner()
    arrivals = tmp_path / "arrivals"
    arrivals.mkdir()
    task_dir = arrivals / "task_0120"

    module._prepare_task_dir(
        task_dir=task_dir,
        arrivals_dir=arrivals,
        task_index=120,
    )
    (task_dir / "partial.txt").write_text("partial", encoding="utf-8")
    module._prepare_task_dir(
        task_dir=task_dir,
        arrivals_dir=arrivals,
        task_index=120,
    )
    assert not (task_dir / "partial.txt").exists()

    unmarked = arrivals / "task_0121"
    unmarked.mkdir()
    with pytest.raises(RuntimeError, match="unmarked"):
        module._prepare_task_dir(
            task_dir=unmarked,
            arrivals_dir=arrivals,
            task_index=121,
        )


def test_closed_loop_run_dir_has_exclusive_lock(tmp_path: Path) -> None:
    module = _load_runner()
    run_dir = tmp_path / "closed-loop"
    first = module._acquire_run_lock(run_dir)
    try:
        with pytest.raises(RuntimeError, match="another closed-loop"):
            module._acquire_run_lock(run_dir)
    finally:
        first.close()


def test_dataset_identity_hashes_ordered_task_inputs(tmp_path: Path) -> None:
    module = _load_runner()
    dataset = tmp_path / "dataset"
    task_dir = dataset / "spreadsheet" / "task-a"
    task_dir.mkdir(parents=True)
    workbook = task_dir / "1_task-a_input.xlsx"
    workbook.write_bytes(b"workbook-a")
    golden = task_dir / "1_task-a_answer.xlsx"
    golden.write_bytes(b"golden-a")
    (dataset / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": "task-a",
                    "instruction": "Update the workbook.",
                    "instruction_type": "spreadsheet",
                    "spreadsheet_path": "spreadsheet/task-a",
                }
            ]
        ),
        encoding="utf-8",
    )

    instances, entries = module._dataset_identity(
        str(dataset),
        end_index=1,
    )

    assert [instance.id for instance in instances[:1]] == ["task-a"]
    assert entries[0]["task_id"] == "task-a"
    assert entries[0]["input_workbooks"][0]["sha256"] == (
        module._file_sha256(workbook)
    )
    assert entries[0]["ground_truth_workbooks"][0]["sha256"] == (
        module._file_sha256(golden)
    )
