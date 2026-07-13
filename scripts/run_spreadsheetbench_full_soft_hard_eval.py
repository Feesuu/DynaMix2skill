#!/usr/bin/env python3
"""Run paper-aligned full SpreadsheetBench Soft/Hard evaluation for DynaMix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spreadsheetbench_support import find_spreadsheet_dir as find_spreadsheet_dir_in_dataset

SENSITIVE_ARG_MARKERS = ("api-key", "api_key", "apikey", "token", "secret", "password", "authorization", "bearer")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_json_without_timestamps(payload: Any) -> Any:
    if isinstance(payload, dict):
        volatile_keys = {"created_at", "prepared_dataset"}
        return {key: stable_json_without_timestamps(value) for key, value in payload.items() if key not in volatile_keys}
    if isinstance(payload, list):
        return [stable_json_without_timestamps(value) for value in payload]
    return payload


def file_tree_digest(path: Path) -> dict[str, Any]:
    files = []
    for child in sorted(path.rglob("*")):
        if child.is_file():
            files.append([child.relative_to(path).as_posix(), child.stat().st_size, sha256_file(child)])
    return {"file_count": len(files), "sha256": sha256_json(files), "files": files}


def api_key_fingerprint(value: str | None) -> str:
    if not value:
        return ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def safe_spreadsheet_path(instance: dict[str, Any]) -> str:
    spreadsheet_path = str(instance.get("spreadsheet_path", instance["id"]))
    path = Path(spreadsheet_path)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe spreadsheet_path for {instance.get('id')}: {spreadsheet_path}")
    return spreadsheet_path


def resolve_spreadsheet_dir(data_path: Path, instance: dict[str, Any]) -> Path:
    safe_spreadsheet_path(instance)
    instance_id = str(instance["id"])
    found = find_spreadsheet_dir_in_dataset(str(data_path), instance)
    if found is None:
        raise FileNotFoundError(f"Spreadsheet directory not found for {instance_id} under {data_path}")
    resolved = Path(found).resolve()
    source_root = data_path.resolve()
    if not resolved.is_relative_to(source_root):
        raise RuntimeError(f"Spreadsheet directory escapes source root for {instance_id}: {resolved}")
    return resolved


def prepared_spreadsheet_dir(output_dir: Path, instance: dict[str, Any]) -> Path:
    spreadsheet_path = safe_spreadsheet_path(instance)
    dst_dir = (output_dir / spreadsheet_path).resolve()
    output_root = output_dir.resolve()
    if not dst_dir.is_relative_to(output_root):
        raise RuntimeError(f"Prepared spreadsheet directory escapes output root for {instance.get('id')}: {dst_dir}")
    return dst_dir


def answer_files(spreadsheet_dir: Path) -> list[Path]:
    files = sorted(path for path in spreadsheet_dir.iterdir() if path.name.endswith("_answer.xlsx"))
    if not files:
        files = sorted(path for path in spreadsheet_dir.iterdir() if path.name.endswith("_golden.xlsx"))
    if not files and (spreadsheet_dir / "golden.xlsx").is_file():
        files = [spreadsheet_dir / "golden.xlsx"]
    return files


def verified_input_files(spreadsheet_dir: Path) -> list[Path]:
    files = sorted(path for path in spreadsheet_dir.iterdir() if path.name.endswith("_init.xlsx"))
    if not files:
        for name in ("initial.xlsx", "input.xlsx"):
            path = spreadsheet_dir / name
            if path.is_file():
                return [path]
    return files


def full_input_files(spreadsheet_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in spreadsheet_dir.iterdir()
        if path.suffix == ".xlsx"
        and ("_input" in path.name or path.name in {"initial.xlsx", "input.xlsx"})
        and "answer" not in path.name.lower()
        and "golden" not in path.name.lower()
        and "output" not in path.name.lower()
    )


def testcase_prefix(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(f"expected {path.name} to end with {suffix}")
    return path.name[: -len(suffix)]


def full_testcase_pairs(spreadsheet_dir: Path) -> dict[str, dict[str, Path]]:
    pairs: dict[str, dict[str, Path]] = {}
    for answer in answer_files(spreadsheet_dir):
        if answer.name.endswith("_answer.xlsx"):
            pairs.setdefault(testcase_prefix(answer, "_answer.xlsx"), {})["answer"] = answer
    for input_file in full_input_files(spreadsheet_dir):
        if input_file.name.endswith("_input.xlsx"):
            prefix = testcase_prefix(input_file, "_input.xlsx")
        elif input_file.name.endswith("_input .xlsx"):
            prefix = testcase_prefix(input_file, "_input .xlsx")
        else:
            prefix = input_file.stem
        pairs.setdefault(prefix, {})["input"] = input_file
    return {prefix: pair for prefix, pair in pairs.items() if "input" in pair and "answer" in pair}


def select_excluded_testcase(full_dir: Path, verified_dir: Path, instance_id: str) -> dict[str, Any]:
    verified_inputs = verified_input_files(verified_dir)
    verified_answers = answer_files(verified_dir)
    if len(verified_inputs) != 1 or len(verified_answers) != 1:
        raise RuntimeError(
            f"Expected one verified input/answer for {instance_id}, "
            f"got {len(verified_inputs)} inputs and {len(verified_answers)} answers"
        )
    verified_input_hash = sha256_file(verified_inputs[0])
    verified_answer_hash = sha256_file(verified_answers[0])
    pairs = full_testcase_pairs(full_dir)
    if not pairs:
        raise RuntimeError(f"No full testcase pairs found for {instance_id}")

    def build(prefix: str, method: str) -> dict[str, Any]:
        pair = pairs[prefix]
        return {
            "instance_id": instance_id,
            "prefix": prefix,
            "excluded_input": pair["input"].name,
            "excluded_answer": pair["answer"].name,
            "excluded_input_source": str(pair["input"]),
            "excluded_answer_source": str(pair["answer"]),
            "excluded_input_sha256": sha256_file(pair["input"]),
            "excluded_answer_sha256": sha256_file(pair["answer"]),
            "verified_input_source": str(verified_inputs[0]),
            "verified_answer_source": str(verified_answers[0]),
            "verified_input_sha256": verified_input_hash,
            "verified_answer_sha256": verified_answer_hash,
            "match_method": method,
            "reason": "verified_train_evolution_testcase",
        }

    input_matches = {prefix for prefix, pair in pairs.items() if sha256_file(pair["input"]) == verified_input_hash}
    answer_matches = {prefix for prefix, pair in pairs.items() if sha256_file(pair["answer"]) == verified_answer_hash}
    both = sorted(input_matches & answer_matches)
    if len(both) == 1:
        return build(both[0], "hash_input_and_answer")
    fallback_prefix = f"1_{instance_id}"
    if fallback_prefix in pairs:
        return build(fallback_prefix, "fallback_exact_1_instance_id")
    raise RuntimeError(
        "Could not map verified train testcase to full data for "
        f"{instance_id}; both_hash_matches={both}, input_only={sorted(input_matches)}, "
        f"answer_only={sorted(answer_matches)}, expected_fallback={fallback_prefix}"
    )


def copy_file_no_overwrite(src: Path, dst: Path) -> None:
    if src.is_symlink():
        raise RuntimeError(f"Refusing to copy symlinked source file into prepared dataset: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"Refusing to overwrite prepared dataset file: {dst}")
    shutil.copy2(src, dst)


def count_answer_testcases(data_path: Path) -> int:
    dataset = read_json(data_path / "dataset.json")
    return sum(len(answer_files(resolve_spreadsheet_dir(data_path, item))) for item in dataset)


def materialize_paper_aligned_dataset(
    *,
    full_data_path: Path,
    verified_data_path: Path,
    output_dir: Path,
    train_start: int = 0,
    train_end: int = 200,
    expected_full_testcases: int | None = 2729,
    expected_excluded_testcases: int | None = 200,
    expected_prepared_testcases: int | None = 2529,
    expected_normalized_input_files: int | None = 3,
) -> dict[str, Any]:
    full_data_path = full_data_path.resolve()
    verified_data_path = verified_data_path.resolve()
    output_dir = output_dir.expanduser()
    resolved_output_dir = output_dir.resolve(strict=False)
    for source_dir in (full_data_path, verified_data_path):
        if (
            resolved_output_dir == source_dir
            or resolved_output_dir.is_relative_to(source_dir)
            or source_dir.is_relative_to(resolved_output_dir)
        ):
            raise RuntimeError(
                "prepared dataset output_dir must not overlap a source dataset directory: "
                f"output_dir={resolved_output_dir}, source={source_dir}"
            )
    if output_dir.exists():
        if output_dir.is_symlink():
            raise RuntimeError(f"Refusing to remove symlinked prepared dataset output_dir: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    full_dataset = read_json(full_data_path / "dataset.json")
    verified_dataset = read_json(verified_data_path / "dataset.json")
    full_by_id = {str(item["id"]): item for item in full_dataset}
    excluded_entries: list[dict[str, Any]] = []
    excluded_by_instance: dict[str, set[str]] = {}
    normalized_entries: list[dict[str, Any]] = []

    full_testcases = sum(len(answer_files(resolve_spreadsheet_dir(full_data_path, item))) for item in full_dataset)
    if expected_full_testcases is not None and full_testcases != expected_full_testcases:
        raise RuntimeError(f"Expected {expected_full_testcases} full testcases, got {full_testcases}")

    for item in verified_dataset[train_start:train_end]:
        instance_id = str(item["id"])
        if instance_id not in full_by_id:
            raise RuntimeError(f"Verified train instance missing from full data: {instance_id}")
        full_dir = resolve_spreadsheet_dir(full_data_path, full_by_id[instance_id])
        verified_dir = resolve_spreadsheet_dir(verified_data_path, item)
        excluded = select_excluded_testcase(full_dir, verified_dir, instance_id)
        excluded_entries.append(excluded)
        excluded_by_instance.setdefault(instance_id, set()).update(
            {excluded["excluded_input"], excluded["excluded_answer"]}
        )

    if expected_excluded_testcases is not None and len(excluded_entries) != expected_excluded_testcases:
        raise RuntimeError(f"Expected {expected_excluded_testcases} excluded testcases, got {len(excluded_entries)}")

    write_json(output_dir / "dataset.json", full_dataset)
    for item in full_dataset:
        instance_id = str(item["id"])
        src_dir = resolve_spreadsheet_dir(full_data_path, item)
        dst_dir = prepared_spreadsheet_dir(output_dir, item)
        excluded_names = excluded_by_instance.get(instance_id, set())
        for src in sorted(path for path in src_dir.iterdir() if path.is_file()):
            if src.name in excluded_names:
                continue
            dst_name = src.name
            if src.name.endswith("_input .xlsx"):
                dst_name = src.name.replace("_input .xlsx", "_input.xlsx")
                if (dst_dir / dst_name).exists():
                    raise RuntimeError(f"Cannot normalize {src}: destination exists")
                normalized_entries.append({
                    "instance_id": instance_id,
                    "source": str(src),
                    "normalized_name": dst_name,
                    "reason": "runner_matches_only_standard_input_suffix",
                })
            copy_file_no_overwrite(src, dst_dir / dst_name)

    prepared_testcases = count_answer_testcases(output_dir)
    if expected_prepared_testcases is not None and prepared_testcases != expected_prepared_testcases:
        raise RuntimeError(f"Expected {expected_prepared_testcases} prepared testcases, got {prepared_testcases}")
    if expected_normalized_input_files is not None and len(normalized_entries) != expected_normalized_input_files:
        raise RuntimeError(f"Expected {expected_normalized_input_files} normalized input files, got {len(normalized_entries)}")

    manifest = {
        "format": "dynamix_spreadsheetbench_full_soft_hard_prepared_dataset_v1",
        "created_at": utc_now_iso(),
        "full_data_path": str(full_data_path),
        "verified_data_path": str(verified_data_path),
        "prepared_dataset": str(output_dir),
        "verified_train_range": [train_start, train_end],
        "full_instances": len(full_dataset),
        "full_answer_testcases": full_testcases,
        "excluded_testcases": len(excluded_entries),
        "prepared_answer_testcases": prepared_testcases,
        "exclusions": excluded_entries,
    }
    normalization_manifest = {
        "format": "dynamix_spreadsheetbench_full_soft_hard_normalization_v1",
        "created_at": utc_now_iso(),
        "normalized_input_files": len(normalized_entries),
        "normalizations": normalized_entries,
    }
    return {
        "testcase_filter_manifest": manifest,
        "normalization_manifest": normalization_manifest,
        "stable_materialization_hash": sha256_json({
            "testcase_filter_manifest": stable_json_without_timestamps(manifest),
            "normalization_manifest": stable_json_without_timestamps(normalization_manifest),
            "prepared_dataset_tree": file_tree_digest(output_dir),
        }),
    }


def path_fingerprint(path: Path, *, source_only: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {"path": str(path), "exists": True, "size": path.stat().st_size, "sha256": sha256_file(path)}
    files = []
    for child in sorted(path.rglob("*")):
        if child.is_file():
            if source_only and (child.suffix not in {".py", ".json", ".toml", ".md", ".txt", ".yaml", ".yml"}):
                continue
            rel = child.relative_to(path).as_posix()
            files.append([rel, child.stat().st_size, sha256_file(child)])
    return {"path": str(path), "exists": True, "files": files, "sha256": sha256_json(files)}


def aggregate_usage_jsonl(path: Path) -> dict[str, Any]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
    records = 0
    if not path.is_file():
        return {"path": str(path), "records": 0, "totals": totals}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        records += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            usage = payload
        for key in totals:
            value = usage.get(key) if isinstance(usage, dict) else None
            if isinstance(value, (int, float)):
                totals[key] += int(value)
    return {"path": str(path), "records": records, "totals": totals}


def redact_command(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for part in cmd:
        lower = part.lower()
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if part.startswith("--") and any(marker in lower for marker in SENSITIVE_ARG_MARKERS):
            redacted.append(part)
            redact_next = True
            continue
        if "=" in part and any(marker in lower.split("=", 1)[0] for marker in SENSITIVE_ARG_MARKERS):
            key, _ = part.split("=", 1)
            redacted.append(f"{key}=<redacted>")
            continue
        if lower.startswith("bearer "):
            redacted.append("Bearer <redacted>")
            continue
        redacted.append(part)
    return redacted


def sanitized_subprocess_env(*, args: argparse.Namespace, skill_env: dict[str, str] | None = None) -> dict[str, str]:
    keep_keys = (
        "PATH",
        "PYTHONPATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "no_proxy",
        "SAL_DISABLE_OPENCL",
    )
    env = {key: value for key, value in os.environ.items() if key in keep_keys}
    env["PATH"] = f"{Path(args.python_executable).resolve().parent}:{env.get('PATH', '')}"
    env["OPENAI_BASE_URL"] = args.openai_base_url
    env["OPENAI_API_KEY"] = args.openai_api_key
    if skill_env:
        env.update(skill_env)
    return env


def run_stage(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    marker_path: Path,
    outputs: list[Path],
    fingerprint: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    fingerprint_hash = sha256_json(fingerprint)
    if resume and marker_path.is_file() and all(path.exists() for path in outputs):
        marker = read_json(marker_path)
        if marker.get("fingerprint_hash") == fingerprint_hash and marker.get("done"):
            return {**marker, "skipped_by_resume": True}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"stage": name, "cmd": redact_command(cmd), "started_at": utc_now_iso()}, ensure_ascii=False) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    ended = time.time()
    marker = {
        "stage": name,
        "done": proc.returncode == 0,
        "returncode": proc.returncode,
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "ended_at": datetime.fromtimestamp(ended, timezone.utc).isoformat(),
        "elapsed_seconds": ended - started,
        "cmd": redact_command(cmd),
        "log": str(log_path),
        "fingerprint_hash": fingerprint_hash,
        "fingerprint": fingerprint,
    }
    write_json(marker_path, marker)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, redact_command(cmd))
    return marker


def infer_source_paths(source_run_dir: Path | None, skills_dir: str | None, skillbank_index: str | None, generation_config: str | None) -> dict[str, Path | None]:
    summary = {}
    if source_run_dir is not None and (source_run_dir / "experiment_summary.json").is_file():
        summary = read_json(source_run_dir / "experiment_summary.json")
    resolved_skills = Path(skills_dir) if skills_dir else None
    if resolved_skills is None and summary.get("skills_root"):
        resolved_skills = Path(summary["skills_root"])
    if resolved_skills is None and source_run_dir is not None:
        resolved_skills = source_run_dir / "skills"

    resolved_index = Path(skillbank_index) if skillbank_index else None
    if resolved_index is None and summary.get("skillbank_index"):
        resolved_index = Path(summary["skillbank_index"])
    if resolved_index is None and resolved_skills is not None:
        resolved_index = resolved_skills / ".dynamix_skillbank_index.json"

    resolved_generation_config = Path(generation_config) if generation_config else None
    if resolved_generation_config is None and source_run_dir is not None:
        candidate = source_run_dir / "trace2skill_generation_config.json"
        if candidate.is_file():
            resolved_generation_config = candidate
    return {
        "skills_dir": resolved_skills.resolve() if resolved_skills else None,
        "skillbank_index": resolved_index.resolve() if resolved_index else None,
        "generation_config": resolved_generation_config.resolve() if resolved_generation_config else None,
    }


def source_summary(source_run_dir: Path | None) -> dict[str, Any]:
    if source_run_dir is not None and (source_run_dir / "experiment_summary.json").is_file():
        return read_json(source_run_dir / "experiment_summary.json")
    return {}


def resolve_runtime_args(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, str]:
    sources: dict[str, str] = {}

    def fill(attr: str, *, summary_key: str | None, env_key: str | None, default: Any, cast=lambda value: value) -> None:
        current = getattr(args, attr)
        if current is not None:
            sources[attr] = "cli"
            return
        if summary_key and summary.get(summary_key) not in (None, ""):
            setattr(args, attr, cast(summary[summary_key]))
            sources[attr] = f"source_run_dir:{summary_key}"
            return
        if env_key and os.environ.get(env_key) not in (None, ""):
            setattr(args, attr, cast(os.environ[env_key]))
            sources[attr] = f"env:{env_key}"
            return
        setattr(args, attr, cast(default))
        sources[attr] = "default"

    fill("model", summary_key="model", env_key="GEN_MODEL", default="Qwen3.5-9B")
    fill("openai_base_url", summary_key="openai_base_url", env_key="OPENAI_BASE_URL", default="http://127.0.0.1:18002/v1")
    fill("openai_api_key", summary_key=None, env_key="OPENAI_API_KEY", default="EMPTY")
    fill("embedding_base_url", summary_key="embedding_base_url", env_key="EMBED_BASE_URL", default="http://10.26.1.184:18007/v1")
    fill("embedding_model", summary_key="embedding_model", env_key="EMBED_MODEL", default="Qwen3-Embedding-8B")
    fill("skillbank_top_k", summary_key="skillbank_top_k", env_key="DYNAMIX_SKILLBANK_TOP_K", default=10, cast=int)
    fill("workers", summary_key="workers", env_key=None, default=4, cast=int)
    fill("max_turns", summary_key="max_turns", env_key=None, default=100, cast=int)
    fill("temperature", summary_key="rollout_temperature", env_key=None, default=0.0, cast=float)
    fill("llm_client", summary_key="rollout_llm_client", env_key=None, default="openai")
    fill("llm_timeout_seconds", summary_key="rollout_client_timeout_seconds", env_key=None, default=600.0, cast=float)

    if args.llm_retry_wait_seconds is None:
        waits = summary.get("rollout_client_retry_wait_seconds")
        if isinstance(waits, list):
            args.llm_retry_wait_seconds = ",".join(str(value) for value in waits)
            sources["llm_retry_wait_seconds"] = "source_run_dir:rollout_client_retry_wait_seconds"
        else:
            args.llm_retry_wait_seconds = "5.0,10.0,30.0"
            sources["llm_retry_wait_seconds"] = "default"
    else:
        sources["llm_retry_wait_seconds"] = "cli"

    if int(args.num_random_seeds) != 1 or int(args.repeat) != 1:
        raise ValueError("full Soft/Hard wrapper supports only single-run eval: --num-random-seeds=1 and --repeat=1")
    return sources


def build_rollout_command(args: argparse.Namespace, prepared_dataset: Path, outputs_dir: Path, logs_dir: Path, results_file: Path, skills_dir: Path, generation_config: Path | None) -> list[str]:
    cmd = [
        args.python_executable,
        "run_spreadsheetbench.py",
        "--data_path",
        str(prepared_dataset),
        "--output_dir",
        str(outputs_dir),
        "--agent",
        "cli_skill_preloaded",
        "--skills_dir",
        str(skills_dir),
        "--model",
        args.model,
        "--llm_client",
        args.llm_client,
        "--temperature",
        str(args.temperature),
        "--llm_timeout_seconds",
        str(args.llm_timeout_seconds),
        "--llm_retry_wait_seconds",
        args.llm_retry_wait_seconds,
        "--num_random_seeds",
        str(args.num_random_seeds),
        "--repeat",
        str(args.repeat),
        "--max_turns",
        str(args.max_turns),
        "--workers",
        str(args.workers),
        "--results_file",
        str(results_file),
        "--log_dir",
        str(logs_dir),
        "--log_format",
        "markdown",
    ]
    if generation_config is not None:
        cmd.extend(["--generation_config", str(generation_config)])
    return cmd


def build_eval_command(args: argparse.Namespace, prepared_dataset: Path, outputs_dir: Path, eval_file: Path, recalc_dir: Path) -> list[str]:
    return [
        args.python_executable,
        "evaluate_with_official.py",
        "--data_path",
        str(prepared_dataset),
        "--output_dir",
        str(outputs_dir),
        "--results_file",
        str(eval_file),
        "--recalc_dir",
        str(recalc_dir),
    ]


def render_report_md(report: dict[str, Any]) -> str:
    summary = report.get("evaluation", {}).get("summary", {})
    raw = summary.get("trace2skill_compatible_no_recalc", {})
    lines = [
        "# SpreadsheetBench Full Soft/Hard Report",
        "",
        f"- Created at: `{report.get('created_at')}`",
        f"- Prepared dataset: `{report.get('prepared_dataset')}`",
        f"- Full instances: `{report.get('dataset', {}).get('full_instances')}`",
        f"- Prepared answer testcases: `{report.get('dataset', {}).get('prepared_answer_testcases')}`",
        f"- Excluded train/evolution testcases: `{report.get('dataset', {}).get('excluded_testcases')}`",
        "",
        "## LibreOffice Recalc Main Metrics",
        "",
        f"- Avg Soft Score: `{summary.get('avg_soft_score', 0):.6f}`",
        f"- Avg Hard Score: `{summary.get('avg_hard_score', 0):.6f}`",
        f"- Test Case Accuracy: `{summary.get('test_case_accuracy', 0):.6f}`",
        f"- Instance Accuracy: `{summary.get('instance_accuracy', 0):.6f}`",
        "",
        "## Trace2Skill-Compatible Raw Audit",
        "",
        f"- Avg Soft Score: `{raw.get('avg_soft_score', 0):.6f}`",
        f"- Avg Hard Score: `{raw.get('avg_hard_score', 0):.6f}`",
        f"- Test Case Accuracy: `{raw.get('test_case_accuracy', 0):.6f}`",
        f"- Instance Accuracy: `{raw.get('instance_accuracy', 0):.6f}`",
        "",
        "## Stage Runtime",
        "",
        "| Stage | Done | Elapsed(s) | Log |",
        "| --- | --- | ---: | --- |",
    ]
    for stage in report.get("stages", []):
        lines.append(f"| `{stage.get('stage')}` | {stage.get('done')} | {stage.get('elapsed_seconds', 0):.1f} | `{stage.get('log')}` |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DynaMix full SpreadsheetBench Soft/Hard evaluation.")
    parser.add_argument("--source-run-dir", default=None, help="Existing DynaMix run dir used to infer skills/index/generation config")
    parser.add_argument("--full-data-path", required=True)
    parser.add_argument("--verified-data-path", required=True)
    parser.add_argument("--output-dir", required=True, help="Directory for full_soft_hard artifacts")
    parser.add_argument("--skills-dir", default=None)
    parser.add_argument("--skillbank-index", default=None)
    parser.add_argument("--generation-config", default=None)
    parser.add_argument("--python-executable", default=os.environ.get("DYNAMIX_PYTHON", sys.executable))
    parser.add_argument("--model", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--skillbank-top-k", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--llm-client", default=None, choices=["openai", "api_chat"])
    parser.add_argument("--llm-timeout-seconds", type=float, default=None)
    parser.add_argument("--llm-retry-wait-seconds", default=None)
    parser.add_argument("--num-random-seeds", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verified-train-start", type=int, default=0)
    parser.add_argument("--verified-train-end", type=int, default=200)
    parser.add_argument("--expected-full-testcases", type=int, default=2729)
    parser.add_argument("--expected-excluded-testcases", type=int, default=200)
    parser.add_argument("--expected-prepared-testcases", type=int, default=2529)
    parser.add_argument("--expected-normalized-input-files", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run_dir = Path(args.source_run_dir).resolve() if args.source_run_dir else None
    output_dir = Path(args.output_dir).resolve()
    prepared_dataset = output_dir / "prepared_dataset"
    manifests_dir = output_dir / "manifests"
    logs_dir = output_dir / "logs"
    markers_dir = output_dir / "markers"
    usage_dir = output_dir / "usage"

    materialized = materialize_paper_aligned_dataset(
        full_data_path=Path(args.full_data_path),
        verified_data_path=Path(args.verified_data_path),
        output_dir=prepared_dataset,
        train_start=args.verified_train_start,
        train_end=args.verified_train_end,
        expected_full_testcases=args.expected_full_testcases,
        expected_excluded_testcases=args.expected_excluded_testcases,
        expected_prepared_testcases=args.expected_prepared_testcases,
        expected_normalized_input_files=args.expected_normalized_input_files,
    )
    write_json(manifests_dir / "testcase_filter_manifest.json", materialized["testcase_filter_manifest"])
    write_json(manifests_dir / "normalization_manifest.json", materialized["normalization_manifest"])

    if args.prepare_only:
        manifest = materialized["testcase_filter_manifest"]
        normalization = materialized["normalization_manifest"]
        print(json.dumps({
            "prepared_dataset": str(prepared_dataset),
            "full_instances": manifest["full_instances"],
            "full_answer_testcases": manifest["full_answer_testcases"],
            "excluded_testcases": manifest["excluded_testcases"],
            "prepared_answer_testcases": manifest["prepared_answer_testcases"],
            "normalized_input_files": normalization["normalized_input_files"],
            "testcase_filter_manifest": str(manifests_dir / "testcase_filter_manifest.json"),
            "normalization_manifest": str(manifests_dir / "normalization_manifest.json"),
        }, ensure_ascii=False, indent=2))
        return

    summary = source_summary(source_run_dir)
    runtime_arg_sources = resolve_runtime_args(args, summary)
    paths = infer_source_paths(source_run_dir, args.skills_dir, args.skillbank_index, args.generation_config)
    skills_dir = paths["skills_dir"]
    skillbank_index = paths["skillbank_index"]
    generation_config = paths["generation_config"]
    if skills_dir is None or not skills_dir.is_dir():
        raise FileNotFoundError(f"Could not resolve DynaMix skills/nodebank dir: {skills_dir}")
    if not (skills_dir / "node_bank_manifest.json").is_file():
        raise FileNotFoundError(f"Node bank manifest not found: {skills_dir / 'node_bank_manifest.json'}")
    if skillbank_index is None or not skillbank_index.is_file():
        raise FileNotFoundError(f"Could not resolve DynaMix skillbank index: {skillbank_index}")
    if generation_config is None or not generation_config.is_file():
        raise FileNotFoundError("Generation config was not found; pass --generation-config explicitly")

    raw_dir = output_dir / "raw"
    if raw_dir.is_symlink():
        raise RuntimeError(f"Refusing to write through symlinked artifact directory: {raw_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_skillbank_index = raw_dir / "skillbank_index_for_full_eval.json"
    if local_skillbank_index.exists() or local_skillbank_index.is_symlink():
        local_skillbank_index.unlink()
    copy_file_no_overwrite(skillbank_index, local_skillbank_index)

    outputs_dir = output_dir / "trace2skill_full_outputs"
    rollout_logs_dir = output_dir / "trace2skill_full_logs"
    rollout_results = output_dir / "trace2skill_full_results.json"
    eval_file = output_dir / "full_soft_hard_eval.json"
    recalc_dir = output_dir / "eval_artifacts" / "libreoffice_recalculated_outputs"
    selection_log = raw_dir / "skill_selection_records.jsonl"
    react_usage_log = usage_dir / "full_rollout.react_usage.jsonl"
    skillbank_usage_log = usage_dir / "full_rollout.skillbank_usage.jsonl"
    eval_usage_log = usage_dir / "full_eval.usage.jsonl"

    env = sanitized_subprocess_env(
        args=args,
        skill_env={
            "DYNAMIX_SKILLBANK_ROOT": str(skills_dir),
            "DYNAMIX_SKILLBANK_TOP_K": str(args.skillbank_top_k),
            "DYNAMIX_SKILLBANK_EMBED_BASE_URL": args.embedding_base_url,
            "DYNAMIX_SKILLBANK_EMBED_MODEL": args.embedding_model,
            "DYNAMIX_SKILLBANK_EMBED_API_KEY": "EMPTY",
            "DYNAMIX_SKILLBANK_CACHE_PATH": str(local_skillbank_index),
            "DYNAMIX_SKILL_SELECTION_LOG": str(selection_log),
            "REACT_AGENT_USAGE_LOG": str(react_usage_log),
            "DYNAMIX_SKILLBANK_USAGE_LOG": str(skillbank_usage_log),
        },
    )

    rollout_cmd = build_rollout_command(args, prepared_dataset, outputs_dir, rollout_logs_dir, rollout_results, skills_dir, generation_config)
    eval_cmd = build_eval_command(args, prepared_dataset, outputs_dir, eval_file, recalc_dir)
    common_fp = {
        "prepared_dataset_contract_hash": materialized["stable_materialization_hash"],
        "skills_manifest": path_fingerprint(skills_dir / "node_bank_manifest.json"),
        "skillbank_index": path_fingerprint(local_skillbank_index),
        "generation_config": path_fingerprint(generation_config),
        "script": path_fingerprint(Path(__file__).resolve()),
        "run_spreadsheetbench": path_fingerprint(REPO_ROOT / "run_spreadsheetbench.py"),
        "spreadsheetbench_support": path_fingerprint(REPO_ROOT / "spreadsheetbench_support.py"),
        "spreadsheet_agent": path_fingerprint(REPO_ROOT / "spreadsheet_agent", source_only=True),
        "react_agent": path_fingerprint(REPO_ROOT / "src" / "react_agent", source_only=True),
        "skillbank_source": path_fingerprint(REPO_ROOT / "src" / "dynamix_trace2skill" / "skillbank.py"),
        "runtime_protocol": {
            "model": args.model,
            "openai_base_url": args.openai_base_url,
            "openai_api_key": api_key_fingerprint(args.openai_api_key),
            "skillbank_top_k": int(args.skillbank_top_k),
            "embedding_base_url": args.embedding_base_url,
            "embedding_model": args.embedding_model,
            "llm_client": args.llm_client,
            "temperature": float(args.temperature),
            "max_turns": int(args.max_turns),
            "workers": int(args.workers),
            "llm_timeout_seconds": float(args.llm_timeout_seconds),
            "llm_retry_wait_seconds": args.llm_retry_wait_seconds,
        },
    }
    stages = []
    stages.append(run_stage(
        name="01_full_rollout",
        cmd=rollout_cmd,
        cwd=REPO_ROOT,
        env=env,
        log_path=logs_dir / "01_full_rollout.log",
        marker_path=markers_dir / "01_full_rollout.done.json",
        outputs=[rollout_results],
        fingerprint={"cmd": rollout_cmd, **common_fp},
        resume=args.resume,
    ))
    stages.append(run_stage(
        name="02_full_eval",
        cmd=eval_cmd,
        cwd=REPO_ROOT,
        env={**env, "DYNAMIX_EVAL_USAGE_LOG": str(eval_usage_log)},
        log_path=logs_dir / "02_full_eval.log",
        marker_path=markers_dir / "02_full_eval.done.json",
        outputs=[eval_file],
        fingerprint={
            "cmd": eval_cmd,
            "outputs": path_fingerprint(outputs_dir),
            "evaluator": path_fingerprint(REPO_ROOT / "evaluate_with_official.py"),
            **common_fp,
        },
        resume=args.resume,
    ))

    evaluation = read_json(eval_file)
    report = {
        "format": "dynamix_spreadsheetbench_full_soft_hard_report_v1",
        "created_at": utc_now_iso(),
        "source_run_dir": str(source_run_dir) if source_run_dir else None,
        "prepared_dataset": str(prepared_dataset),
        "dataset": {
            "full_data_path": str(Path(args.full_data_path).resolve()),
            "verified_data_path": str(Path(args.verified_data_path).resolve()),
            "full_instances": materialized["testcase_filter_manifest"]["full_instances"],
            "full_answer_testcases": materialized["testcase_filter_manifest"]["full_answer_testcases"],
            "excluded_testcases": materialized["testcase_filter_manifest"]["excluded_testcases"],
            "prepared_answer_testcases": materialized["testcase_filter_manifest"]["prepared_answer_testcases"],
        },
        "runtime": {
            "model": args.model,
            "workers": args.workers,
            "skillbank_top_k": args.skillbank_top_k,
            "openai_base_url": args.openai_base_url,
            "openai_api_key_fingerprint": api_key_fingerprint(args.openai_api_key),
            "embedding_base_url": args.embedding_base_url,
            "embedding_model": args.embedding_model,
            "python_executable": args.python_executable,
            "argument_sources": runtime_arg_sources,
        },
        "commands": {
            "rollout": redact_command(rollout_cmd),
            "eval": redact_command(eval_cmd),
        },
        "stages": stages,
        "usage": {
            "react": aggregate_usage_jsonl(react_usage_log),
            "skillbank": aggregate_usage_jsonl(skillbank_usage_log),
            "eval": aggregate_usage_jsonl(eval_usage_log),
        },
        "evaluation": evaluation,
        "manifests": {
            "testcase_filter_manifest": str(manifests_dir / "testcase_filter_manifest.json"),
            "normalization_manifest": str(manifests_dir / "normalization_manifest.json"),
            "source_skillbank_index": str(skillbank_index),
            "local_skillbank_index": str(local_skillbank_index),
        },
    }
    write_json(output_dir / "full_soft_hard_report.json", report)
    (output_dir / "full_soft_hard_report.md").write_text(render_report_md(report), encoding="utf-8")
    print(json.dumps({
        "report": str(output_dir / "full_soft_hard_report.json"),
        "eval": str(eval_file),
        "summary": evaluation.get("summary", {}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
