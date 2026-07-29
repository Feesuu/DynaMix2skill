#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from evaluate_with_official import evaluate, evaluation_runtime_identity
from spreadsheet_agent.runner import SpreadsheetBenchRunner

from dynamix_trace2skill.certified_otd_pipeline import (
    CertifiedOtdConfig,
    ExperienceAtomAnalyst,
    _atom_protocol_fingerprint,
    _prepare_otd_analyst_config,
    _records_fingerprint,
    _tokenizer_for_config,
)
from dynamix_trace2skill.clients import EmbeddingClient, GenerationClient
from dynamix_trace2skill.evidence_balanced_skill_pipeline import (
    EvidenceBalancedOnlineSession,
    EvidenceBalancedSkillConfig,
    SkillCapsuleAnalyst,
    _ebst_tree_protocol_fingerprint,
    _write_nodebank_manifest,
    _write_state_artifacts,
)
from dynamix_trace2skill.log_parser import (
    load_records,
    parse_trace2skill_logs,
)
from dynamix_trace2skill.pipeline import (
    DynaMixRunConfig,
    _refresh_skillbank_index,
)
from dynamix_trace2skill.schemas import RawTrajectoryRecord


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _parse_float_csv(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "expected at least one comma-separated float"
        )
    return tuple(float(part) for part in parts)


def _checkpoint_protocol_config(
    config: DynaMixRunConfig,
    source_tree_dir: Path,
) -> DynaMixRunConfig:
    checkpoint_config = copy.deepcopy(config)
    checkpoint_config.analyst.prompt_token_report_path = str(
        source_tree_dir / "analysis" / "cdost_prompt_token_report.json"
    )
    return checkpoint_config


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _acquire_run_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".closed_loop_run.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            f"another closed-loop process owns this run dir: {run_dir}"
        ) from exc
    return lock_handle


def _write_records_atomic(
    path: Path,
    records: Sequence[RawTrajectoryRecord],
) -> None:
    _write_json_atomic(path, [record.to_dict() for record in records])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_fingerprints() -> dict[str, str]:
    relative_paths = (
        "experiments/tree_v4/run_closed_loop.sh",
        "scripts/run_ebst_closed_loop_experiment.py",
        "run_spreadsheetbench.py",
        "evaluate_with_official.py",
        "src/dynamix_core/balanced_metric_tree.py",
        "src/dynamix_core/skill_capsules.py",
        "src/dynamix_trace2skill/clients.py",
        "src/dynamix_trace2skill/certified_otd_pipeline.py",
        "src/dynamix_trace2skill/evidence_balanced_skill_pipeline.py",
        "src/dynamix_trace2skill/log_parser.py",
        "src/dynamix_trace2skill/pipeline.py",
        "src/dynamix_trace2skill/schemas.py",
        "src/dynamix_trace2skill/skillbank.py",
        "src/react_agent/agent.py",
        "src/react_agent/converter.py",
        "src/react_agent/models.py",
        "src/react_agent/prompts.py",
        "src/react_agent/tools.py",
        "spreadsheet_agent/agents/base.py",
        "spreadsheet_agent/agents/cli_only_agent.py",
        "spreadsheet_agent/agents/cli_skill_preloaded_agent.py",
        "spreadsheet_agent/runner.py",
        "spreadsheet_agent/system_prompts.py",
        "spreadsheet_agent/tools/__init__.py",
        "spreadsheet_agent/tools/bash.py",
    )
    return {
        relative_path: _file_sha256(REPO_ROOT / relative_path)
        for relative_path in relative_paths
    }


def _dataset_identity(
    data_path: str,
    *,
    end_index: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    runner = SpreadsheetBenchRunner(
        agent=None,
        data_path=data_path,
        output_dir=os.devnull,
    )
    instances = runner.load_data()
    if len(instances) < end_index:
        raise ValueError(
            "SpreadsheetBench dataset is shorter than the requested train "
            f"prefix: requested={end_index} observed={len(instances)}"
        )
    entries: list[dict[str, Any]] = []
    data_root = Path(data_path).resolve()
    for task_index, instance in enumerate(instances[:end_index]):
        spreadsheet_dir_text = runner._find_spreadsheet_dir(instance)
        if spreadsheet_dir_text is None:
            raise FileNotFoundError(
                "SpreadsheetBench input directory is missing for "
                f"task_index={task_index} task_id={instance.id}"
            )
        spreadsheet_dir = Path(spreadsheet_dir_text).resolve()
        input_names = runner._find_input_files(str(spreadsheet_dir))
        if not input_names:
            raise FileNotFoundError(
                "SpreadsheetBench input workbook is missing for "
                f"task_index={task_index} task_id={instance.id}"
            )
        ground_truth_names = sorted(
            path.name
            for path in spreadsheet_dir.iterdir()
            if path.name.endswith("_answer.xlsx")
            or path.name.endswith("_golden.xlsx")
            or path.name == "golden.xlsx"
        )
        if not ground_truth_names:
            raise FileNotFoundError(
                "SpreadsheetBench ground-truth workbook is missing for "
                f"task_index={task_index} task_id={instance.id}"
            )
        entries.append(
            {
                "task_index": task_index,
                "task_id": str(instance.id),
                "instruction_sha256": hashlib.sha256(
                    instance.instruction.encode("utf-8")
                ).hexdigest(),
                "instruction_type": str(instance.instruction_type),
                "spreadsheet_path": str(instance.spreadsheet_path),
                "input_workbooks": [
                    {
                        "path": str(
                            (spreadsheet_dir / name).relative_to(data_root)
                        ),
                        "sha256": _file_sha256(
                            spreadsheet_dir / name
                        ),
                    }
                    for name in input_names
                ],
                "ground_truth_workbooks": [
                    {
                        "path": str(
                            (spreadsheet_dir / name).relative_to(data_root)
                        ),
                        "sha256": _file_sha256(
                            spreadsheet_dir / name
                        ),
                    }
                    for name in ground_truth_names
                ],
            }
        )
    return instances, entries


def _prepare_task_dir(
    *,
    task_dir: Path,
    arrivals_dir: Path,
    task_index: int,
) -> None:
    arrivals_root = arrivals_dir.resolve()
    candidate = task_dir.resolve(strict=False)
    if candidate.parent != arrivals_root or task_dir.is_symlink():
        raise RuntimeError(
            f"unsafe closed-loop task directory: {task_dir}"
        )
    marker_path = task_dir / ".ebst_task_dir.json"
    if task_dir.exists():
        if not marker_path.is_file():
            raise RuntimeError(
                "refusing to delete an unmarked closed-loop task directory: "
                f"{task_dir}"
            )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker != {
            "format": "ebst_closed_loop_task_dir_v1",
            "task_index": int(task_index),
        }:
            raise RuntimeError(
                f"closed-loop task directory marker mismatch: {task_dir}"
            )
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(
        marker_path,
        {
            "format": "ebst_closed_loop_task_dir_v1",
            "task_index": int(task_index),
        },
    )


def _bind_experiment_contract(
    path: Path,
    payload: dict[str, Any],
    *,
    resume: bool,
) -> str:
    contract_hash = _canonical_sha256(payload)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "closed-loop experiment contract does not match the "
                "existing run directory"
            )
        if not resume:
            raise FileExistsError(
                "closed-loop experiment contract already exists; pass "
                f"--resume or use a clean run directory: {path}"
            )
        return contract_hash
    _write_json_atomic(path, payload)
    return contract_hash


def _validate_checkpoint_protocol(
    marker: dict[str, Any],
    *,
    trajectory_source: str,
    atom_protocol_fingerprint: str,
    tree_protocol_fingerprint: str,
    record_prefix_sha256: str,
    experiment_contract_sha256: str | None = None,
    require_strict_open_loop: bool = False,
) -> None:
    expected = {
        "trajectory_source": trajectory_source,
        "atom_protocol_fingerprint": atom_protocol_fingerprint,
        "tree_protocol_fingerprint": tree_protocol_fingerprint,
        "record_prefix_sha256": record_prefix_sha256,
    }
    observed_mismatch = {
        key: {"expected": value, "observed": marker.get(key)}
        for key, value in expected.items()
        if marker.get(key) != value
    }
    metadata = dict(marker.get("metadata") or {})
    if require_strict_open_loop and metadata.get("strict_online") is not True:
        observed_mismatch["metadata.strict_online"] = {
            "expected": True,
            "observed": metadata.get("strict_online"),
        }
    if (
        experiment_contract_sha256 is not None
        and metadata.get("experiment_contract_sha256")
        != experiment_contract_sha256
    ):
        observed_mismatch["metadata.experiment_contract_sha256"] = {
            "expected": experiment_contract_sha256,
            "observed": metadata.get("experiment_contract_sha256"),
        }
    if observed_mismatch:
        raise RuntimeError(
            "closed-loop checkpoint protocol mismatch: "
            f"{observed_mismatch}"
        )


def _usage_summary(paths: Sequence[Path]) -> dict[str, Any]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    records = 0
    records_with_usage = 0
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = payload.get("usage")
            if not isinstance(usage, dict):
                continue
            records_with_usage += 1
            for key in totals:
                if usage.get(key) is not None:
                    totals[key] += int(float(usage[key]))
    return {
        "paths": [str(path) for path in paths if path.is_file()],
        "records": records,
        "records_with_usage": records_with_usage,
        "totals": totals,
    }


def _read_selection(
    path: Path,
    *,
    expected_instance: Any,
    expected_top_k: int,
    active_node_ids: set[str],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "closed-loop rollout did not write the required skill selection "
            f"log: {path}"
        )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(
            "closed-loop rollout must write exactly one skill selection row; "
            f"observed={len(rows)} path={path}"
        )
    row = rows[0]
    instruction = str(expected_instance.instruction or "").strip()
    instruction_type = str(
        expected_instance.instruction_type or ""
    ).strip()
    expected_query = (
        f"{instruction}\n\nTask type: {instruction_type}"
        if instruction_type
        else instruction
    )
    if str(row.get("instance_id") or "") != str(expected_instance.id):
        raise RuntimeError("skill selection record has the wrong instance_id")
    if str(row.get("instruction") or "") != str(
        expected_instance.instruction
    ):
        raise RuntimeError("skill selection record has the wrong instruction")
    if str(row.get("instruction_type") or "") != str(
        expected_instance.instruction_type or ""
    ):
        raise RuntimeError(
            "skill selection record has the wrong instruction_type"
        )
    if str(row.get("query") or "") != expected_query:
        raise RuntimeError(
            "skill selection record violates the retrieval query contract"
        )
    if int(row.get("top_k", -1)) != int(expected_top_k):
        raise RuntimeError("skill selection record has the wrong top_k")
    selected_node_ids = [
        str(value) for value in row.get("selected_node_ids", [])
    ]
    selected_node_scores = [
        float(value) for value in row.get("selected_node_scores", [])
    ]
    if len(selected_node_ids) != len(selected_node_scores):
        raise RuntimeError(
            "skill selection IDs and scores have different lengths"
        )
    if len(selected_node_ids) > expected_top_k:
        raise RuntimeError("skill selection exceeds top_k")
    if len(set(selected_node_ids)) != len(selected_node_ids):
        raise RuntimeError("skill selection contains duplicate node IDs")
    unknown = set(selected_node_ids).difference(active_node_ids)
    if unknown:
        raise RuntimeError(
            "skill selection references inactive node IDs: "
            f"{sorted(unknown)}"
        )
    return {
        "selected_node_ids": selected_node_ids,
        "selected_node_scores": selected_node_scores,
        "query": expected_query,
    }


def _export_current_nodebank(
    *,
    session: EvidenceBalancedOnlineSession,
    root: Path,
    config: DynaMixRunConfig,
    policy: EvidenceBalancedSkillConfig,
    tokenizer: Any,
) -> dict[str, Any]:
    manifest = _write_nodebank_manifest(
        tree=session.tree,
        registry=session.registry,
        output_dir=root,
        config=config,
        ebst=policy,
        tokenizer=tokenizer,
    )
    if int(manifest["node_count"]) > 0:
        manifest["index_path"] = _refresh_skillbank_index(root, config)
    else:
        (root / ".dynamix_skillbank_index.json").unlink(missing_ok=True)
        manifest["index_path"] = None
    return manifest


def _rollout_env(
    *,
    args: argparse.Namespace,
    config: DynaMixRunConfig,
    nodebank_root: Path,
    selection_log: Path,
    nodebank_active: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(REPO_ROOT / "src"), existing_pythonpath)
        if value
    )
    env["OPENAI_BASE_URL"] = config.generation.base_url
    env["OPENAI_API_KEY"] = config.generation.resolved_api_key
    env["REACT_AGENT_USAGE_LOG"] = str(
        selection_log.parent / "react_usage.jsonl"
    )
    for name in tuple(env):
        if name.startswith("DYNAMIX_SKILLBANK_") or name == (
            "DYNAMIX_SKILL_SELECTION_LOG"
        ):
            env.pop(name, None)
    if not nodebank_active:
        return env
    env.update(
        {
            "DYNAMIX_SKILLBANK_ROOT": str(nodebank_root),
            "DYNAMIX_SKILLBANK_TOP_K": str(args.skillbank_top_k),
            "DYNAMIX_SKILLBANK_EMBED_BASE_URL": config.embedding.base_url,
            "DYNAMIX_SKILLBANK_EMBED_MODEL": config.embedding.model,
            "DYNAMIX_SKILLBANK_EMBED_API_KEY": (
                config.embedding.resolved_api_key
            ),
            "DYNAMIX_SKILLBANK_EMBED_MAX_MODEL_LEN": str(
                config.embedding.max_model_len
            ),
            "DYNAMIX_SKILLBANK_EMBED_MAX_INPUT_TOKENS": str(
                config.embedding.effective_max_input_tokens
            ),
            "DYNAMIX_SKILLBANK_EMBED_BATCH_SIZE": str(
                config.embedding.batch_size
            ),
            "DYNAMIX_SKILLBANK_CACHE_PATH": str(
                nodebank_root / ".dynamix_skillbank_index.json"
            ),
            "DYNAMIX_SKILLBANK_VECTOR_CACHE_PATH": str(
                config.embedding.cache_path or ""
            ),
            "DYNAMIX_SKILLBANK_REQUIRE_CACHE_MATCH": "true",
            "DYNAMIX_SKILLBANK_REQUIRE_VECTOR_CACHE_MATCH": "false",
            "DYNAMIX_SKILLBANK_EXPECT_TREE_POLICY": (
                "evidence_balanced_skill_tree"
            ),
            "DYNAMIX_SKILL_SELECTION_LOG": str(selection_log),
            "DYNAMIX_SKILLBANK_USAGE_LOG": str(
                selection_log.parent / "skillbank_usage.jsonl"
            ),
        }
    )
    if config.embedding.tokenizer_model:
        env["DYNAMIX_SKILLBANK_EMBED_TOKENIZER"] = (
            config.embedding.tokenizer_model
        )
    return env


def _rollout_command(
    *,
    args: argparse.Namespace,
    config: DynaMixRunConfig,
    task_index: int,
    task_dir: Path,
    nodebank_active: bool,
    generation_config_path: Path,
) -> list[str]:
    return [
        args.python_executable,
        str(REPO_ROOT / "run_spreadsheetbench.py"),
        "--data_path",
        args.data_path,
        "--output_dir",
        str(task_dir / "outputs"),
        "--working_dir",
        str(task_dir / "work"),
        "--agent",
        "cli_skill_preloaded" if nodebank_active else "cli_only",
        "--skills_dir",
        str(args.nodebank_dir),
        "--model",
        config.generation.model,
        "--llm_client",
        "openai",
        "--max_turns",
        str(args.max_turns),
        "--temperature",
        str(args.rollout_temperature),
        "--generation_config",
        str(generation_config_path),
        "--llm_timeout_seconds",
        str(args.rollout_timeout_seconds),
        "--llm_retry_wait_seconds",
        ",".join(
            str(value) for value in args.rollout_retry_wait_seconds
        ),
        "--disable_response_cache",
        "--start_idx",
        str(task_index),
        "--end_idx",
        str(task_index + 1),
        "--results_file",
        str(task_dir / "rollout_results.json"),
        "--log_dir",
        str(task_dir / "logs"),
        "--log_format",
        "markdown",
        "--workers",
        "1",
    ]


def _run_rollout(
    *,
    command: Sequence[str],
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "closed-loop rollout failed; "
            f"returncode={completed.returncode} log={log_path}"
        )


def _record_with_feedback(
    record: RawTrajectoryRecord,
    *,
    task_index: int,
    selection: dict[str, Any],
) -> RawTrajectoryRecord:
    extra = dict(record.extra)
    extra["closed_loop_skill_evolution"] = {
        "task_index": int(task_index),
        "selected_capsule_ids": selection["selected_node_ids"],
        "selected_capsule_scores": selection["selected_node_scores"],
        "selection_query": selection["query"],
        "success": bool(record.success),
        "verifier_score": record.verifier_score,
        "verifier_feedback": record.verifier_feedback,
        "attribution": "exposure_outcome_not_causal",
    }
    return replace(record, extra=extra)


def _load_bootstrap(
    args: argparse.Namespace,
    config: DynaMixRunConfig,
    policy: EvidenceBalancedSkillConfig,
) -> tuple[EvidenceBalancedOnlineSession, list[RawTrajectoryRecord]]:
    if args.bootstrap_checkpoint:
        session = EvidenceBalancedOnlineSession.from_checkpoint(
            args.bootstrap_checkpoint
        )
        records_path = Path(args.bootstrap_records or config.records_path)
        records = load_records(records_path)[: args.arrival_start]
        session.validate_prefix(
            [record.trajectory_id for record in records]
        )
        if len(records) != args.arrival_start:
            raise ValueError(
                "bootstrap records do not cover the requested arrival prefix"
            )
        return session, records
    if args.arrival_start != 0:
        raise ValueError(
            "arrival_start must be zero when no bootstrap checkpoint is used"
        )
    return EvidenceBalancedOnlineSession.empty(policy), []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run policy-dependent SpreadsheetBench train arrivals through "
            "the evidence-balanced online skill tree."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--bootstrap-checkpoint")
    parser.add_argument("--bootstrap-records")
    parser.add_argument("--arrival-start", type=int, default=120)
    parser.add_argument("--arrival-end", type=int, default=200)
    parser.add_argument("--model")
    parser.add_argument("--openai-base-url")
    parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--rollout-temperature", type=float, default=0.0)
    parser.add_argument(
        "--rollout-thinking",
        type=_parse_bool,
        default=False,
    )
    parser.add_argument("--rollout-timeout-seconds", type=float, default=600)
    parser.add_argument(
        "--rollout-retry-wait-seconds",
        type=_parse_float_csv,
        default=(5.0, 10.0, 30.0),
    )
    parser.add_argument("--skillbank-top-k", type=int, default=10)
    parser.add_argument(
        "--evaluator-backend",
        default="auto",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    tree_dir = run_dir / "tree"
    checkpoints_dir = run_dir / "checkpoints"
    arrivals_dir = run_dir / "arrivals"
    nodebank_dir = run_dir / "current_nodebank"
    args.nodebank_dir = nodebank_dir
    for path in (tree_dir, checkpoints_dir, arrivals_dir, nodebank_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = DynaMixRunConfig.from_json(args.config)
    if config.hierarchy.get("tree_policy") != "evidence_balanced_skill_tree":
        raise ValueError(
            "closed-loop driver requires evidence_balanced_skill_tree"
        )
    if (
        config.dataset_path is None
        or Path(config.dataset_path).resolve()
        != Path(args.data_path).resolve()
    ):
        raise ValueError(
            "closed-loop data path must exactly match the source EBST config"
        )
    if int(config.train_start) != 0 or int(config.train_end or -1) != 200:
        raise ValueError(
            "closed-loop source config must bind SpreadsheetBench train 0:200"
        )
    source_tree_dir = Path(config.output_dir).resolve()
    if args.bootstrap_checkpoint:
        bootstrap_root = Path(args.bootstrap_checkpoint).resolve()
        if not bootstrap_root.is_relative_to(
            source_tree_dir / "dynamic_snapshots"
        ):
            raise ValueError(
                "bootstrap checkpoint is not owned by the source open-loop "
                "tree config"
            )
    config.output_dir = str(tree_dir)
    config.scenario = "dynamic_update"
    config.dynamic.trajectory_source = "closed_loop_skill_evolution"
    if args.model:
        config.generation.model = args.model
    if args.openai_base_url:
        config.generation.base_url = args.openai_base_url
    config.generation.api_key = "EMPTY"
    config.generation.api_key_env_var = args.openai_api_key_env
    config.generation.debug_dir = str(
        tree_dir / "analysis" / "generation_debug"
    )
    if not os.environ.get(args.openai_api_key_env):
        raise ValueError(
            f"API key environment variable is unset: "
            f"{args.openai_api_key_env}"
        )
    if not config.embedding.cache_path:
        raise ValueError(
            "closed-loop EBST requires embedding.cache_path"
        )
    if config.embedding.cache_write_policy != "first_write_wins":
        raise ValueError(
            "closed-loop EBST requires embedding cache first_write_wins"
        )
    if not 0 <= args.arrival_start < args.arrival_end <= 200:
        raise ValueError(
            "closed-loop train arrivals must satisfy "
            "0 <= start < end <= 200"
        )

    policy = EvidenceBalancedSkillConfig.from_mapping(
        dict(config.hierarchy or {}).get("ebst", {})
    )
    _prepare_otd_analyst_config(config, tree_dir)
    tokenizer = _tokenizer_for_config(config)
    atom_config = CertifiedOtdConfig(
        dual_view_lambda=policy.dual_view_lambda,
        atom_temperature=policy.atom_temperature,
        parent_temperature=policy.capsule_temperature,
        atom_cache_path=None,
    )
    atom_protocol_fingerprint = _atom_protocol_fingerprint(
        config,
        atom_config,
    )
    checkpoint_protocol_config = _checkpoint_protocol_config(
        config,
        source_tree_dir,
    )
    tree_protocol_fingerprint = _ebst_tree_protocol_fingerprint(
        checkpoint_protocol_config,
        policy,
    )
    dataset_instances, dataset_entries = _dataset_identity(
        args.data_path,
        end_index=args.arrival_end,
    )
    session, records = _load_bootstrap(args, config, policy)
    if len(session.arrived_source_item_ids) != len(records):
        raise RuntimeError(
            "bootstrap tree and bootstrap records have different lengths"
        )
    bootstrap_marker_path = (
        Path(args.bootstrap_checkpoint).resolve()
        / "checkpoint.complete.json"
        if args.bootstrap_checkpoint
        else None
    )
    if bootstrap_marker_path is not None:
        bootstrap_marker = json.loads(
            bootstrap_marker_path.read_text(encoding="utf-8")
        )
        if int(bootstrap_marker.get("arrival_count", -1)) != (
            args.arrival_start
        ):
            raise ValueError(
                "bootstrap checkpoint arrival count does not match "
                "--arrival-start"
            )
        _validate_checkpoint_protocol(
            bootstrap_marker,
            trajectory_source="open_loop_replay",
            atom_protocol_fingerprint=atom_protocol_fingerprint,
            tree_protocol_fingerprint=tree_protocol_fingerprint,
            record_prefix_sha256=_records_fingerprint(records),
            require_strict_open_loop=True,
        )

    generation_config_path = run_dir / "rollout_generation_config.json"
    _write_json_atomic(
        generation_config_path,
        {
            "temperature": float(args.rollout_temperature),
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": bool(args.rollout_thinking)
                }
            },
        },
    )
    dataset_manifest = Path(args.data_path).resolve() / "dataset.json"
    bootstrap_records_path = (
        Path(args.bootstrap_records or config.records_path).resolve()
        if args.bootstrap_checkpoint
        else None
    )
    command_manifest = {
        "format": "ebst_closed_loop_contract_v1",
        "trajectory_source": "closed_loop_skill_evolution",
        "data_path": str(Path(args.data_path).resolve()),
        "train_prefix_range": [0, args.arrival_end],
        "policy_dependent_arrival_range": [
            args.arrival_start,
            args.arrival_end,
        ],
        "bootstrap_count": args.arrival_start,
        "bootstrap_checkpoint": (
            str(Path(args.bootstrap_checkpoint).resolve())
            if args.bootstrap_checkpoint
            else None
        ),
        "repository_commit": _git_commit(),
        "source_fingerprints": _source_fingerprints(),
        "config_sha256": _file_sha256(Path(args.config).resolve()),
        "dataset_manifest_sha256": (
            _file_sha256(dataset_manifest)
            if dataset_manifest.is_file()
            else None
        ),
        "dataset_identity_sha256": _canonical_sha256(dataset_entries),
        "bootstrap_records_sha256": (
            _file_sha256(bootstrap_records_path)
            if bootstrap_records_path is not None
            else None
        ),
        "bootstrap_checkpoint_marker_sha256": (
            _file_sha256(bootstrap_marker_path)
            if bootstrap_marker_path is not None
            else None
        ),
        "model": config.generation.model,
        "openai_base_url": config.generation.base_url,
        "openai_api_key_env": args.openai_api_key_env,
        "rollout_thinking": bool(args.rollout_thinking),
        "rollout_temperature": float(args.rollout_temperature),
        "rollout_max_turns": int(args.max_turns),
        "rollout_timeout_seconds": float(args.rollout_timeout_seconds),
        "rollout_retry_wait_seconds": [
            float(value) for value in args.rollout_retry_wait_seconds
        ],
        "embedding": {
            "model": config.embedding.model,
            "base_url": config.embedding.base_url,
            "max_model_len": config.embedding.max_model_len,
            "max_input_tokens": (
                config.embedding.effective_max_input_tokens
            ),
            "batch_size": config.embedding.batch_size,
            "max_concurrency": config.embedding.max_concurrency,
            "cache_path": config.embedding.cache_path,
            "cache_write_policy": config.embedding.cache_write_policy,
        },
        "chunked_embedding": dict(config.chunked_embedding or {}),
        "skillbank_top_k": int(args.skillbank_top_k),
        "evaluator": "SpreadsheetBench + LibreOffice recalc",
        "evaluator_runtime_identity": evaluation_runtime_identity(
            args.evaluator_backend
        ),
        "atom_protocol_fingerprint": atom_protocol_fingerprint,
        "tree_protocol_fingerprint": tree_protocol_fingerprint,
        "policy_dependent_trajectories": True,
        "arrival_order": "dataset_order",
        "structure_update": (
            "insert -> dirty-path refresh -> validate -> atomic checkpoint"
        ),
        "skill_feedback_attribution": "exposure_outcome_not_causal",
    }
    contract_sha256 = _bind_experiment_contract(
        run_dir / "experiment_contract.json",
        command_manifest,
        resume=bool(args.resume),
    )
    _write_json_atomic(
        run_dir / "dataset_identity.json",
        {
            "format": "spreadsheetbench_ordered_identity_v1",
            "entries": dataset_entries,
            "sha256": command_manifest["dataset_identity_sha256"],
        },
    )

    if args.resume and checkpoints_dir.is_dir():
        completed = sorted(
            checkpoints_dir.glob("arrival_*/checkpoint.complete.json")
        )
        if completed:
            latest_checkpoint = completed[-1].parent
            marker = json.loads(
                completed[-1].read_text(encoding="utf-8")
            )
            resumed = EvidenceBalancedOnlineSession.from_checkpoint(
                latest_checkpoint
            )
            persisted_records = load_records(
                run_dir / "online_records.json"
            )
            committed_count = int(marker["arrival_count"])
            if len(persisted_records) < committed_count:
                raise RuntimeError(
                    "online record ledger is shorter than the last complete "
                    "checkpoint"
                )
            persisted_records = persisted_records[:committed_count]
            _validate_checkpoint_protocol(
                marker,
                trajectory_source="closed_loop_skill_evolution",
                atom_protocol_fingerprint=atom_protocol_fingerprint,
                tree_protocol_fingerprint=tree_protocol_fingerprint,
                record_prefix_sha256=_records_fingerprint(
                    persisted_records
                ),
                experiment_contract_sha256=contract_sha256,
            )
            resumed.validate_prefix(
                [
                    record.trajectory_id
                    for record in persisted_records
                ]
            )
            _write_records_atomic(
                run_dir / "online_records.json",
                persisted_records,
            )
            session = resumed
            records = persisted_records
    elif any(checkpoints_dir.iterdir()):
        raise FileExistsError(
            "closed-loop checkpoints already exist; pass --resume or use "
            f"a clean run directory: {checkpoints_dir}"
        )
    processed_arrivals = len(records) - args.arrival_start
    next_task_index = args.arrival_start + processed_arrivals
    if next_task_index < args.arrival_start:
        raise RuntimeError("checkpoint precedes the bootstrap prefix")
    if next_task_index > args.arrival_end:
        raise RuntimeError("checkpoint exceeds the requested arrival range")

    usage_dir = run_dir / "usage"
    usage_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DYNAMIX_GENERATION_USAGE_LOG"] = str(
        usage_dir / "analysis_generation_usage.jsonl"
    )
    os.environ["DYNAMIX_EMBEDDING_USAGE_LOG"] = str(
        usage_dir / "analysis_embedding_usage.jsonl"
    )
    os.environ["DYNAMIX_SKILLBANK_USAGE_LOG"] = str(
        usage_dir / "nodebank_index_usage.jsonl"
    )
    atom_generation = GenerationClient(
        replace(
            config.generation,
            temperature=float(policy.atom_temperature),
        )
    )
    capsule_generation = GenerationClient(
        replace(
            config.generation,
            temperature=float(policy.capsule_temperature),
        )
    )
    validator_generation = GenerationClient(
        replace(
            config.generation,
            temperature=float(policy.validator_temperature),
        )
    )
    embedding = EmbeddingClient(config.embedding)
    atom_analyst = ExperienceAtomAnalyst(
        atom_generation,
        embedding,
        config=atom_config,
        tokenizer=tokenizer,
        max_prompt_tokens=int(config.analyst.max_prompt_tokens),
        max_output_tokens=config.analyst.max_output_tokens,
        max_evidence_chars=int(
            config.analyst.analysis_bundle_max_chars or 60000
        ),
    )
    capsule_analyst = SkillCapsuleAnalyst(
        capsule_generation,
        validator_generation,
        tokenizer=tokenizer,
        max_prompt_tokens=int(config.analyst.max_prompt_tokens),
        max_output_tokens=config.analyst.max_output_tokens,
        validation_mode=policy.validation_mode,
    )

    started_at = time.time()
    try:
        for task_index in range(next_task_index, args.arrival_end):
            task_started_at = time.time()
            task_dir = arrivals_dir / f"task_{task_index:04d}"
            _prepare_task_dir(
                task_dir=task_dir,
                arrivals_dir=arrivals_dir,
                task_index=task_index,
            )
            expected_instance = dataset_instances[task_index]
            manifest = _export_current_nodebank(
                session=session,
                root=nodebank_dir,
                config=config,
                policy=policy,
                tokenizer=tokenizer,
            )
            nodebank_active = int(manifest["node_count"]) > 0
            selection_log = task_dir / "skill_selection_records.jsonl"
            command = _rollout_command(
                args=args,
                config=config,
                task_index=task_index,
                task_dir=task_dir,
                nodebank_active=nodebank_active,
                generation_config_path=generation_config_path,
            )
            _write_json_atomic(
                task_dir / "rollout_command.json",
                {
                    "command": command,
                    "agent": (
                        "cli_skill_preloaded"
                        if nodebank_active
                        else "cli_only"
                    ),
                },
            )
            _run_rollout(
                command=command,
                env=_rollout_env(
                    args=args,
                    config=config,
                    nodebank_root=nodebank_dir,
                    selection_log=selection_log,
                    nodebank_active=nodebank_active,
                ),
                log_path=task_dir / "rollout.log",
            )
            evaluation = evaluate(
                args.data_path,
                str(task_dir / "outputs"),
                start_idx=task_index,
                end_idx=task_index + 1,
                recalc_dir=str(task_dir / "libreoffice_recalculated"),
                evaluator_backend=args.evaluator_backend,
            )
            evaluation_path = task_dir / "evaluation.json"
            _write_json_atomic(evaluation_path, evaluation)
            evaluation_rows = list(evaluation.get("results") or [])
            if (
                len(evaluation_rows) != 1
                or str(evaluation_rows[0].get("id"))
                != str(expected_instance.id)
            ):
                raise RuntimeError(
                    "closed-loop evaluation identity does not match the "
                    f"dataset task; task_index={task_index} "
                    f"expected={expected_instance.id!r}"
                )
            task_records = parse_trace2skill_logs(
                task_dir / "logs",
                results_file=evaluation_path,
            )
            if len(task_records) != 1:
                raise RuntimeError(
                    "closed-loop task must produce exactly one trajectory; "
                    f"task_index={task_index} observed={len(task_records)}"
                )
            if (
                str(task_records[0].task_id)
                != str(expected_instance.id)
                or task_records[0].instruction
                != expected_instance.instruction
            ):
                raise RuntimeError(
                    "closed-loop parsed trajectory identity does not match "
                    f"dataset task_index={task_index}"
                )
            selection = (
                _read_selection(
                    selection_log,
                    expected_instance=expected_instance,
                    expected_top_k=args.skillbank_top_k,
                    active_node_ids={
                        str(node["node_id"])
                        for node in manifest.get("nodes") or []
                    },
                )
                if nodebank_active
                else {
                    "selected_node_ids": [],
                    "selected_node_scores": [],
                    "query": "",
                }
            )
            record = _record_with_feedback(
                task_records[0],
                task_index=task_index,
                selection=selection,
            )
            session.record_skill_feedback(
                task_id=record.task_id,
                selected_capsule_ids=selection["selected_node_ids"],
                success=record.success,
                verifier_score=record.verifier_score,
            )
            atoms, excluded = await atom_analyst.extract_many([record])
            if excluded or len(atoms) != 1:
                raise RuntimeError(
                    "closed-loop task did not yield exactly one valid atom; "
                    f"task_id={record.task_id} excluded={excluded}"
                )
            await session.insert_atom(
                atoms[0],
                analyst=capsule_analyst,
                arrival_index=len(records) + 1,
                reason="closed_loop_skill_evolution_arrival",
                revise_prior_capsules=True,
            )
            records.append(record)
            session.validate_prefix(
                [item.trajectory_id for item in records]
            )
            _write_records_atomic(run_dir / "online_records.json", records)
            checkpoint = session.write_checkpoint(
                checkpoints_dir / f"arrival_{len(records):04d}",
                trajectory_source="closed_loop_skill_evolution",
                record_prefix_sha256=_records_fingerprint(records),
                atom_protocol_fingerprint=atom_protocol_fingerprint,
                tree_protocol_fingerprint=tree_protocol_fingerprint,
                metadata={
                    "experiment_contract_sha256": contract_sha256,
                    "task_index": task_index,
                    "task_id": record.task_id,
                    "selected_capsule_ids": selection[
                        "selected_node_ids"
                    ],
                    "success": record.success,
                    "verifier_score": record.verifier_score,
                },
            )
            _write_json_atomic(
                task_dir / "task.complete.json",
                {
                    "task_index": task_index,
                    "task_id": record.task_id,
                    "checkpoint": str(checkpoint),
                    "checkpoint_marker_sha256": _file_sha256(
                        checkpoint / "checkpoint.complete.json"
                    ),
                    "runtime_seconds": time.time() - task_started_at,
                },
            )
    finally:
        embedding.close()

    final_manifest = _export_current_nodebank(
        session=session,
        root=nodebank_dir,
        config=config,
        policy=policy,
        tokenizer=tokenizer,
    )
    _write_state_artifacts(session.tree, session.registry, tree_dir)
    summary = {
        **command_manifest,
        "record_count": len(records),
        "arrival_count": len(records) - args.arrival_start,
        "tree_atom_count": len(session.tree.atoms),
        "active_capsule_count": len(session.registry.active_by_tree_node),
        "nodebank_node_count": int(final_manifest["node_count"]),
        "skill_feedback_event_count": len(session.skill_feedback_events),
        "successful_arrivals": sum(
            bool(record.success)
            for record in records[args.arrival_start :]
        ),
        "usage": _usage_summary(
            sorted(run_dir.rglob("*usage.jsonl"))
        ),
        "runtime_seconds": time.time() - started_at,
        "completed": True,
    }
    _write_json_atomic(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    lock_handle = _acquire_run_lock(Path(args.run_dir).resolve())
    try:
        summary = asyncio.run(_run(args))
    finally:
        lock_handle.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
