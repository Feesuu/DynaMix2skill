#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import atexit
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dynamix_trace2skill.clients import (  # noqa: E402
    EmbeddingClient,
    EmbeddingConfig,
    GenerationClient,
    GenerationConfig,
    api_key_fingerprint,
)
from dynamix_trace2skill.contract_cut_pipeline import (  # noqa: E402
    ContractCutBuildConfig,
    build_contract_cut_skills,
    load_raw_trajectory_records,
)
from dynamix_trace2skill.skillbank import SkillBankSelector  # noqa: E402
from dynamix_trace2skill.tokenization import get_tokenizer  # noqa: E402

FROZEN_RECORDS_SHA256 = (
    "f07519085195e8fa77d036e4cea5cc3654c57722d8cd9476fffc93da51075db1"
)
PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_path(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        if "__pycache__" in child.parts or child.suffix in {".pyc", ".pyo"}:
            continue
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _acquire_run_lock(run_dir: Path):
    lock_path = run_dir / ".contract_cut_run.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            f"another Contract-Cut process owns this run directory: {run_dir}"
        ) from exc
    return lock_handle


def _source_fingerprint() -> dict[str, str]:
    relative_paths = (
        "scripts/run_contract_cut_ebst_experiment.py",
        "src/dynamix_core/balanced_metric_tree.py",
        "src/dynamix_core/contract_cut_ebst.py",
        "src/dynamix_trace2skill/contract_cut_pipeline.py",
        "src/dynamix_trace2skill/clients.py",
        "src/dynamix_trace2skill/schemas.py",
        "src/dynamix_trace2skill/skillbank.py",
        "src/dynamix_trace2skill/tokenization.py",
        "src/react_agent/models.py",
        "spreadsheet_agent/agents/cli_skill_preloaded_agent.py",
        "spreadsheet_agent/tools/bash.py",
        "run_spreadsheetbench.py",
        "evaluate_with_official.py",
        "spreadsheetbench_support.py",
    )
    fingerprint = {
        relative_path: _sha256_file(REPO_ROOT / relative_path)
        for relative_path in relative_paths
    }
    fingerprint["src/react_agent/"] = _sha256_path(REPO_ROOT / "src" / "react_agent")
    fingerprint["spreadsheet_agent/"] = _sha256_path(REPO_ROOT / "spreadsheet_agent")
    return fingerprint


def _load_completed_stage(
    run_dir: Path,
    *,
    stage: str,
    fingerprint: str,
    summary_path: Path,
    artifact_paths: Sequence[Path],
) -> dict[str, Any] | None:
    marker_path = run_dir / "stages" / f"{stage}.complete.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("format") != "contract_cut_stage_marker_v1":
        raise ValueError(f"unsupported stage marker: {marker_path}")
    if marker.get("fingerprint") != fingerprint:
        raise ValueError(
            f"stale {stage} resume marker; use a new run directory instead of "
            "mixing experiment protocols"
        )
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{stage} marker exists but its summary is missing: {summary_path}"
        )
    if marker.get("summary_sha256") != _sha256_file(summary_path):
        raise ValueError(f"{stage} summary changed after completion: {summary_path}")
    recorded_artifacts = marker.get("artifact_sha256")
    if not isinstance(recorded_artifacts, dict):
        raise ValueError(f"{stage} marker does not seal stage artifacts")
    expected_artifacts = {
        str(path.resolve().relative_to(run_dir.resolve())): _sha256_path(path)
        for path in artifact_paths
    }
    if recorded_artifacts != expected_artifacts:
        raise ValueError(f"{stage} artifacts changed after completion")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _write_completed_stage(
    run_dir: Path,
    *,
    stage: str,
    fingerprint: str,
    summary_path: Path,
    artifact_paths: Sequence[Path],
) -> None:
    _write_json(
        run_dir / "stages" / f"{stage}.complete.json",
        {
            "format": "contract_cut_stage_marker_v1",
            "stage": stage,
            "fingerprint": fingerprint,
            "summary_path": str(summary_path),
            "summary_sha256": _sha256_file(summary_path),
            "artifact_sha256": {
                str(path.resolve().relative_to(run_dir.resolve())): _sha256_path(path)
                for path in artifact_paths
            },
        },
    )


def _redacted_command(command: Sequence[str]) -> list[str]:
    redacted = []
    for value in command:
        if "api_key" in value.casefold() or "bearer" in value.casefold():
            redacted.append("<redacted>")
        else:
            redacted.append(value)
    return redacted


def _run_logged(
    command: Sequence[str],
    *,
    log_path: Path,
    env: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", shlex.join(_redacted_command(command)), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _load_dataset_ids(dataset_path: Path) -> list[str]:
    dataset_file = dataset_path / "dataset.json"
    payload = json.loads(dataset_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("SpreadsheetBench dataset.json must contain a list")
    return [str(row["id"]) for row in payload]


def _preflight_inputs(args: argparse.Namespace) -> dict[str, Any]:
    records_path = Path(args.records).resolve()
    dataset_path = Path(args.dataset_path).resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    if not (dataset_path / "dataset.json").is_file():
        raise FileNotFoundError(dataset_path / "dataset.json")
    records_sha = _sha256_file(records_path)
    if args.expected_records_sha256 and records_sha != args.expected_records_sha256:
        raise ValueError(
            "records SHA256 does not match the frozen experiment input: "
            f"{records_sha} != {args.expected_records_sha256}"
        )
    records = load_raw_trajectory_records(records_path)
    if len(records) != 200:
        raise ValueError(f"Contract-Cut train requires exactly 200 records, got {len(records)}")
    dataset_ids = _load_dataset_ids(dataset_path)
    expected_ids = dataset_ids[args.train_start : args.train_end]
    record_ids = [record.task_id for record in records]
    if record_ids != expected_ids:
        first_mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(zip(record_ids, expected_ids))
                if actual != expected
            ),
            None,
        )
        raise ValueError(
            "records are not in the frozen SpreadsheetBench dataset order; "
            f"first_mismatch={first_mismatch}"
        )
    if args.train_start != 0 or args.train_end != 200:
        raise ValueError("Contract-Cut v1 freezes train slice to 0:200")
    if args.heldout_start != 200 or args.heldout_end != 400:
        raise ValueError("Contract-Cut v1 freezes heldout slice to 200:400")
    if args.workers != 8 or args.batch_size != 8:
        raise ValueError("Contract-Cut v1 requires workers=8 and batch_size=8")
    if args.embedding_workers != 8 or args.embedding_batch_size != 8:
        raise ValueError(
            "Contract-Cut v1 requires embedding_workers=8 and "
            "embedding_batch_size=8"
        )
    if args.embedding_max_model_len != 32000:
        raise ValueError("Contract-Cut v1 requires embedding_max_model_len=32000")
    if args.max_entries != 8:
        raise ValueError("Contract-Cut v1 requires max_entries=8")
    if args.max_turns != 30:
        raise ValueError("Contract-Cut v1 requires max_turns=30")
    if args.thinking:
        raise ValueError("Contract-Cut v1 requires thinking=false")
    if ".venv" in str(Path(args.python_executable).resolve()):
        raise ValueError("formal Contract-Cut runs must not use a repository .venv")
    return {
        "records_path": str(records_path),
        "records_sha256": records_sha,
        "records_count": len(records),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path / "dataset.json"),
        "first_task_id": record_ids[0],
        "last_task_id": record_ids[-1],
        "success_count": sum(bool(record.success) for record in records),
    }


async def _service_preflight(
    generation: GenerationClient,
    embedding: EmbeddingClient,
) -> dict[str, Any]:
    reply = await generation.chat_text(
        [{"role": "user", "content": "Reply with exactly: ready"}],
        temperature=0.0,
        max_tokens=16,
        debug_metadata={"component": "contract_cut_preflight"},
    )
    vector = (await embedding.embed_texts(["contract cut preflight"]))[0]
    if not vector:
        raise RuntimeError("embedding preflight returned an empty vector")
    return {
        "llm_reply": reply.strip(),
        "embedding_dimension": len(vector),
    }


def _build_clients(args: argparse.Namespace, run_dir: Path):
    generation_config = GenerationConfig(
        base_url=args.openai_base_url,
        model=args.model,
        api_key="EMPTY",
        api_key_env_var=args.api_key_env_var,
        temperature=0.0,
        timeout_seconds=args.llm_timeout_seconds,
        max_concurrency=args.workers,
        thinking_mode=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        debug_dir=str(run_dir / "logs" / "generation_debug"),
        retry_wait_seconds=tuple(args.llm_retry_wait_seconds),
    )
    embedding_config = EmbeddingConfig(
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        api_key="EMPTY",
        max_model_len=args.embedding_max_model_len,
        max_input_tokens=args.embedding_max_model_len,
        truncate_long_texts=False,
        tokenizer_model=args.embedding_tokenizer,
        tokenizer_required=True,
        batch_size=args.embedding_batch_size,
        max_concurrency=args.embedding_workers,
        cache_path=str(run_dir / "cache" / "embedding_cache.sqlite"),
    )
    return (
        GenerationClient(generation_config),
        EmbeddingClient(embedding_config),
        generation_config,
        embedding_config,
    )


async def _run_build(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    records = load_raw_trajectory_records(args.records)
    generation, embedding, generation_config, embedding_config = _build_clients(
        args, run_dir
    )
    tokenizer = get_tokenizer(
        args.embedding_tokenizer,
        allow_regex_fallback=False,
    )
    service = await _service_preflight(generation, embedding)
    started = time.monotonic()
    result = await build_contract_cut_skills(
        records,
        generation=generation,
        embedding=embedding,
        tokenizer=tokenizer,
        run_dir=run_dir,
        config=ContractCutBuildConfig(
            batch_size=args.batch_size,
            max_entries=args.max_entries,
            beta=args.beta,
            analyst_max_prompt_tokens=args.analyst_prompt_tokens,
            compiler_max_prompt_tokens=args.compiler_prompt_tokens,
        ),
    )
    elapsed = time.monotonic() - started
    summary = {
        "stage": "build",
        "elapsed_seconds": elapsed,
        "service_preflight": service,
        "generation": {
            "base_url": generation_config.base_url,
            "model": generation_config.model,
            "api_key": api_key_fingerprint(generation_config.resolved_api_key),
            "temperature": generation_config.temperature,
            "timeout_seconds": generation_config.timeout_seconds,
            "max_concurrency": generation_config.max_concurrency,
            "thinking_mode": generation_config.thinking_mode,
            "max_tokens": None,
        },
        "embedding": {
            "base_url": embedding_config.base_url,
            "model": embedding_config.model,
            "max_model_len": embedding_config.max_model_len,
            "batch_size": embedding_config.batch_size,
            "max_concurrency": embedding_config.max_concurrency,
        },
        "accepted_atom_count": len(result.atoms),
        "excluded_atom_count": len(result.exclusions),
        "tree_audit": result.tree.structural_audit(),
        "final_cut": asdict(result.final_cut),
        "active_skill_count": len(result.active_skills),
        "active_skill_ids": [skill.skill_id for skill in result.active_skills],
        "skills_dir": result.skills_dir,
    }
    _write_json(run_dir / "reports" / "build_summary.json", summary)
    return summary


def _prepare_public_skillbank_mirror(run_dir: Path) -> tuple[Path, str]:
    source = run_dir / "skills"
    source_sha256 = _sha256_path(source)
    mirror_parent = Path("/tmp/dynamix_contract_cut_skillbanks")
    mirror = mirror_parent / source_sha256
    marker_path = mirror / ".contract_cut_public_mirror.json"
    if mirror.is_dir():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("source_sha256") != source_sha256:
            raise ValueError(f"stale public skill mirror: {mirror}")
        return mirror, source_sha256

    mirror_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{source_sha256}.", dir=mirror_parent)
    )
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        manifest_path = temporary / "node_bank_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for node in manifest.get("nodes", []):
            prompt_text = str(node.get("prompt_text") or "")
            node["prompt_text"] = prompt_text.replace(str(source), str(mirror))
            skill_directory = str(node.get("skill_directory") or "")
            node["skill_directory"] = skill_directory.replace(str(source), str(mirror))
        _write_json(manifest_path, manifest)
        _write_json(
            temporary / ".contract_cut_public_mirror.json",
            {
                "format": "contract_cut_public_skill_mirror_v1",
                "source_sha256": source_sha256,
            },
        )
        try:
            os.replace(temporary, mirror)
        except OSError:
            if not mirror.is_dir():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return mirror, source_sha256


def _heldout_env(
    args: argparse.Namespace,
    run_dir: Path,
    public_skillbank_root: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    for name in PROXY_ENV_VARS:
        env.pop(name, None)
    api_key = os.environ.get(args.api_key_env_var, "")
    if not api_key:
        raise ValueError(f"required API key environment variable is empty: {args.api_key_env_var}")
    env.pop(args.api_key_env_var, None)
    python_bin = str(Path(args.python_executable).resolve().parent)
    env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
    env["OPENAI_BASE_URL"] = args.openai_base_url
    env["OPENAI_API_KEY"] = api_key
    env["DYNAMIX_SKILLBANK_ROOT"] = str(public_skillbank_root)
    env["DYNAMIX_SKILLBANK_TOP_K"] = "1"
    env["DYNAMIX_SKILLBANK_EXPECT_TREE_POLICY"] = "contract_cut_ebst"
    env["DYNAMIX_SKILLBANK_EMBED_BASE_URL"] = args.embedding_base_url
    env["DYNAMIX_SKILLBANK_EMBED_MODEL"] = args.embedding_model
    env["DYNAMIX_SKILLBANK_EMBED_API_KEY"] = "EMPTY"
    env["DYNAMIX_SKILLBANK_EMBED_MAX_MODEL_LEN"] = str(args.embedding_max_model_len)
    env["DYNAMIX_SKILLBANK_EMBED_MAX_INPUT_TOKENS"] = str(args.embedding_max_model_len)
    env["DYNAMIX_SKILLBANK_EMBED_BATCH_SIZE"] = str(args.embedding_batch_size)
    env["DYNAMIX_SKILLBANK_EMBED_TOKENIZER"] = args.embedding_tokenizer
    env["DYNAMIX_SKILLBANK_CACHE_PATH"] = str(
        run_dir / "cache" / "heldout_skillbank_index.json"
    )
    env["DYNAMIX_SKILLBANK_REQUIRE_CACHE_MATCH"] = "true"
    env["DYNAMIX_SKILL_SELECTION_LOG"] = str(
        run_dir / "heldout" / "skill_selections.jsonl"
    )
    env["DYNAMIX_TOOL_BWRAP"] = "true"
    env["DYNAMIX_PUBLIC_SKILL_ROOT"] = str(public_skillbank_root)
    env["DYNAMIX_TOOL_MASK_PATHS"] = str(run_dir)
    env["REACT_AGENT_USAGE_LOG"] = str(run_dir / "logs" / "heldout_usage.jsonl")
    env["SAL_DISABLE_OPENCL"] = "1"
    return env


def _prepare_heldout_skillbank_index(
    args: argparse.Namespace,
    run_dir: Path,
    public_skillbank_root: Path,
) -> dict[str, Any]:
    cache_path = run_dir / "cache" / "heldout_skillbank_index.json"
    ledger_path = run_dir / "heldout" / "rollout_results.jsonl"
    selector = SkillBankSelector(
        skillbank_root=public_skillbank_root,
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        api_key="EMPTY",
        cache_path=cache_path,
        max_model_len=args.embedding_max_model_len,
        max_input_tokens=args.embedding_max_model_len,
        batch_size=args.embedding_batch_size,
        tokenizer_model=args.embedding_tokenizer,
        require_cache_match=ledger_path.is_file(),
        expected_tree_policy="contract_cut_ebst",
    )
    audit = selector.prepare_index()
    audit["cache_sha256"] = _sha256_file(cache_path)
    audit["strict_resume_validation"] = ledger_path.is_file()
    _write_json(run_dir / "reports" / "heldout_index_preflight.json", audit)
    return audit


def _run_heldout(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    skills_manifest = run_dir / "skills" / "node_bank_manifest.json"
    if not skills_manifest.is_file():
        raise FileNotFoundError(skills_manifest)
    if shutil.which("bwrap") is None:
        raise RuntimeError("Contract-Cut heldout requires bubblewrap for tool isolation")
    public_skillbank_root, source_skillbank_sha256 = _prepare_public_skillbank_mirror(
        run_dir
    )
    index_preflight = _prepare_heldout_skillbank_index(
        args,
        run_dir,
        public_skillbank_root,
    )
    heldout_dir = run_dir / "heldout"
    output_dir = heldout_dir / "outputs"
    log_dir = heldout_dir / "trajectories"
    results_file = heldout_dir / "rollout_results.json"
    eval_file = heldout_dir / "libreoffice_eval.json"
    env = _heldout_env(args, run_dir, public_skillbank_root)
    generation_config = json.dumps(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        separators=(",", ":"),
    )
    rollout_command = [
        args.python_executable,
        str(REPO_ROOT / "run_spreadsheetbench.py"),
        "--data_path",
        str(Path(args.dataset_path).resolve()),
        "--output_dir",
        str(output_dir),
        "--agent",
        "cli_skill_preloaded",
        "--skills_dir",
        str(public_skillbank_root),
        "--model",
        args.model,
        "--max_turns",
        str(args.max_turns),
        "--temperature",
        "0.0",
        "--generation_config",
        generation_config,
        "--llm_timeout_seconds",
        str(args.llm_timeout_seconds),
        "--llm_retry_wait_seconds",
        ",".join(str(value) for value in args.llm_retry_wait_seconds),
        "--workers",
        str(args.workers),
        "--start_idx",
        str(args.heldout_start),
        "--end_idx",
        str(args.heldout_end),
        "--results_file",
        str(results_file),
        "--log_dir",
        str(log_dir),
        "--log_format",
        "jsonl",
        "--disable_response_cache",
    ]
    if args.resume:
        rollout_command.append("--missing_only")
    started = time.monotonic()
    _run_logged(
        rollout_command,
        log_path=run_dir / "logs" / "heldout_rollout.log",
        env=env,
    )
    rollout_seconds = time.monotonic() - started
    eval_command = [
        args.python_executable,
        str(REPO_ROOT / "evaluate_with_official.py"),
        "--data_path",
        str(Path(args.dataset_path).resolve()),
        "--output_dir",
        str(output_dir),
        "--results_file",
        str(eval_file),
        "--recalc_dir",
        str(heldout_dir / "libreoffice_recalculated"),
        "--start_idx",
        str(args.heldout_start),
        "--end_idx",
        str(args.heldout_end),
        "--evaluator-backend",
        args.evaluator_backend,
    ]
    eval_started = time.monotonic()
    _run_logged(
        eval_command,
        log_path=run_dir / "logs" / "heldout_eval.log",
        env=env,
    )
    eval_seconds = time.monotonic() - eval_started
    evaluation = json.loads(eval_file.read_text(encoding="utf-8"))
    summary = {
        "stage": "heldout",
        "rollout_seconds": rollout_seconds,
        "evaluation_seconds": eval_seconds,
        "protocol": {
            "dataset_slice": [args.heldout_start, args.heldout_end],
            "model": args.model,
            "base_url": args.openai_base_url,
            "workers": args.workers,
            "max_turns": args.max_turns,
            "thinking": False,
            "temperature": 0.0,
            "max_tokens": None,
            "skillbank_top_k": 1,
            "query": "instruction + Task type: instruction_type",
            "injection": "complete SKILL.md",
            "evaluation": "libreoffice_recalc",
        },
        "evaluation_summary": evaluation["summary"],
        "skillbank_index_preflight": index_preflight,
        "source_skillbank_sha256": source_skillbank_sha256,
        "public_skillbank_root": str(public_skillbank_root),
        "results_file": str(results_file),
        "evaluation_file": str(eval_file),
    }
    _write_json(run_dir / "reports" / "heldout_summary.json", summary)
    return summary


def _write_report(
    run_dir: Path,
    *,
    input_audit: dict[str, Any],
    build_summary: dict[str, Any] | None,
    heldout_summary: dict[str, Any] | None,
) -> None:
    payload = {
        "format": "contract_cut_experiment_report_v1",
        "input_audit": input_audit,
        "build": build_summary,
        "heldout": heldout_summary,
    }
    _write_json(run_dir / "reports" / "experiment_report.json", payload)
    lines = [
        "# Contract-Cut EBST Experiment Report",
        "",
        f"- Records: `{input_audit['records_path']}`",
        f"- Records SHA256: `{input_audit['records_sha256']}`",
        f"- Dataset: `{input_audit['dataset_path']}`",
        f"- Train records: {input_audit['records_count']}",
        f"- Train success labels: {input_audit['success_count']}",
    ]
    if build_summary:
        audit = build_summary["tree_audit"]
        lines.extend(
            [
                "",
                "## Build",
                "",
                f"- Accepted Atoms: {build_summary['accepted_atom_count']}",
                f"- Excluded Atoms: {build_summary['excluded_atom_count']}",
                f"- Tree nodes: {audit['node_count']}",
                f"- Tree height: {audit['height']}",
                f"- Exact splits: {audit['split_count']}",
                f"- Final Skills: {build_summary['active_skill_count']}",
                f"- Build seconds: {build_summary['elapsed_seconds']:.1f}",
            ]
        )
    if heldout_summary:
        summary = heldout_summary["evaluation_summary"]
        lines.extend(
            [
                "",
                "## Heldout",
                "",
                f"- Denominator: {summary['total_instances']}",
                f"- Fully correct: {summary['fully_correct_instances']}",
                f"- Instance accuracy: {summary['instance_accuracy']:.4f}",
                f"- Evaluation mode: {summary['evaluation_mode']}",
                f"- Rollout seconds: {heldout_summary['rollout_seconds']:.1f}",
            ]
        )
    (run_dir / "reports" / "experiment_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_artifact_paths(run_dir: Path) -> tuple[Path, ...]:
    return (
        run_dir / "reports" / "build_summary.json",
        run_dir / "contract_cut",
        run_dir / "skills",
        run_dir / "logs" / "build_usage.jsonl",
    )


def _heldout_artifact_paths(run_dir: Path) -> tuple[Path, ...]:
    return (
        run_dir / "reports" / "heldout_summary.json",
        run_dir / "reports" / "heldout_index_preflight.json",
        run_dir / "cache" / "heldout_skillbank_index.json",
        run_dir / "heldout",
        run_dir / "logs" / "heldout_rollout.log",
        run_dir / "logs" / "heldout_eval.log",
        run_dir / "logs" / "heldout_usage.jsonl",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Contract-Cut EBST")
    parser.add_argument("--stage", choices=("build", "heldout", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--records",
        default=(
            "/mnt/data/yaodong/codes/DynaMix2skill/runs/"
            "spreadsheet_splitthink_rolloutfalse_analysttrue_best52_20260720_180410/"
            "ordered_records.json"
        ),
    )
    parser.add_argument(
        "--expected-records-sha256",
        default=FROZEN_RECORDS_SHA256,
    )
    parser.add_argument(
        "--dataset-path",
        default=(
            "/mnt/data/yaodong/codes/DynaMix2skill/data/"
            "spreadsheetbench_verified/spreadsheetbench_verified_400"
        ),
    )
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=200)
    parser.add_argument("--heldout-start", type=int, default=200)
    parser.add_argument("--heldout-end", type=int, default=400)
    parser.add_argument("--openai-base-url", default="http://10.26.1.184:18085/v1")
    parser.add_argument("--model", default="Qwen3.5-9B-AWQ")
    parser.add_argument("--api-key-env-var", default="YD5_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--llm-timeout-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--llm-retry-wait-seconds",
        type=float,
        nargs="*",
        default=[2.0, 5.0, 15.0],
    )
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--embedding-base-url", default="http://10.26.1.184:18007/v1")
    parser.add_argument("--embedding-model", default="Qwen3-Embedding-8B")
    parser.add_argument("--embedding-max-model-len", type=int, default=32000)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-workers", type=int, default=8)
    parser.add_argument(
        "--embedding-tokenizer",
        default="/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-8B",
    )
    parser.add_argument("--max-entries", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--analyst-prompt-tokens", type=int, default=92000)
    parser.add_argument("--compiler-prompt-tokens", type=int, default=92000)
    parser.add_argument("--evaluator-backend", choices=("auto", "official", "local"), default="auto")
    parser.add_argument(
        "--python-executable",
        default="/home/yaodong/miniconda3/envs/stableskill-skillrl/bin/python",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = _acquire_run_lock(run_dir)
    atexit.register(lock_handle.close)
    os.environ["DYNAMIX_GENERATION_USAGE_LOG"] = str(
        run_dir / "logs" / "build_usage.jsonl"
    )
    input_audit = _preflight_inputs(args)
    source_fingerprint = _source_fingerprint()
    protocol_arguments = dict(vars(args))
    protocol_arguments.pop("stage", None)
    protocol_arguments.pop("run_dir", None)
    protocol_arguments.pop("resume", None)
    protocol = {
        "method": "contract_cut_ebst_v1",
        "input_audit": input_audit,
        "source_fingerprint": source_fingerprint,
        "arguments": {
            **protocol_arguments,
            "api_key_env_var": args.api_key_env_var,
            "api_key_fingerprint": api_key_fingerprint(
                os.environ.get(args.api_key_env_var, "")
            ),
        },
        "max_tokens": None,
        "thinking": False,
        "retrieval_top_k": 1,
    }
    protocol["fingerprint"] = _sha256_json(protocol)
    protocol_path = run_dir / "experiment_protocol.json"
    if protocol_path.is_file():
        previous_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise ValueError(
                f"run directory already contains a protocol; use --resume or a new path: {run_dir}"
            )
        if previous_protocol.get("fingerprint") != protocol["fingerprint"]:
            raise ValueError(
                "resume protocol mismatch; use a new run directory instead of mixing runs"
            )
    else:
        _write_json(protocol_path, protocol)

    build_fingerprint = _sha256_json(
        {"stage": "build", "protocol_fingerprint": protocol["fingerprint"]}
    )
    build_summary_path = run_dir / "reports" / "build_summary.json"

    build_summary = None
    heldout_summary = None
    if args.stage in {"build", "all"}:
        if args.resume:
            build_summary = _load_completed_stage(
                run_dir,
                stage="build",
                fingerprint=build_fingerprint,
                summary_path=build_summary_path,
                artifact_paths=_build_artifact_paths(run_dir),
            )
        if build_summary is None:
            build_summary = asyncio.run(_run_build(args, run_dir))
            _write_completed_stage(
                run_dir,
                stage="build",
                fingerprint=build_fingerprint,
                summary_path=build_summary_path,
                artifact_paths=_build_artifact_paths(run_dir),
            )
    else:
        build_summary = _load_completed_stage(
            run_dir,
            stage="build",
            fingerprint=build_fingerprint,
            summary_path=build_summary_path,
            artifact_paths=_build_artifact_paths(run_dir),
        )
        if build_summary is None:
            raise ValueError(
                "heldout requires a completed and artifact-sealed Contract-Cut build"
            )
    if args.stage in {"heldout", "all"}:
        skills_manifest = run_dir / "skills" / "node_bank_manifest.json"
        if not skills_manifest.is_file():
            raise FileNotFoundError(skills_manifest)
        heldout_fingerprint = _sha256_json(
            {
                "stage": "heldout",
                "protocol_fingerprint": protocol["fingerprint"],
                "skills_sha256": _sha256_path(run_dir / "skills"),
            }
        )
        heldout_summary_path = run_dir / "reports" / "heldout_summary.json"
        if args.resume:
            heldout_summary = _load_completed_stage(
                run_dir,
                stage="heldout",
                fingerprint=heldout_fingerprint,
                summary_path=heldout_summary_path,
                artifact_paths=_heldout_artifact_paths(run_dir),
            )
        if heldout_summary is None:
            heldout_summary = _run_heldout(args, run_dir)
            _write_completed_stage(
                run_dir,
                stage="heldout",
                fingerprint=heldout_fingerprint,
                summary_path=heldout_summary_path,
                artifact_paths=_heldout_artifact_paths(run_dir),
            )
    _write_report(
        run_dir,
        input_audit=input_audit,
        build_summary=build_summary,
        heldout_summary=heldout_summary,
    )
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()
    atexit.unregister(lock_handle.close)


if __name__ == "__main__":
    main()
