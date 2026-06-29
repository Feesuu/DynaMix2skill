#!/usr/bin/env python3
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
from typing import Any, Iterable
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dynamix_benchmarks.adapters import BenchmarkSlice, EvalCommandSpec, ExtractCommandSpec, RolloutCommandSpec, get_benchmark_adapter
from dynamix_benchmarks.officeqa import resolve_reward_path
from dynamix_core.skill_export import SkillExportConfig, export_skill_files_from_payload
from dynamix_trace2skill.pipeline import default_hierarchy_config
from dynamix_trace2skill.skillbank import SkillBankSelector
from dynamix_trace2skill.summary import ClusterAnalystConfig


SKILLOPT_QWEN_DEFAULT_TEMPERATURE = 0.7


def run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stage_done(marker: Path, outputs: Iterable[Path], *, fingerprint: dict | None = None) -> bool:
    if not marker.exists():
        return False
    if not all(path.exists() for path in outputs):
        return False
    if fingerprint is None:
        return True
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("fingerprint") == fingerprint


def run_stage(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    marker_dir: Path,
    outputs: list[Path],
    resume: bool,
    fingerprint: dict | None = None,
    clear_outputs_before_run: list[Path] | None = None,
) -> None:
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{name}.done"
    if resume and stage_done(marker, outputs, fingerprint=fingerprint):
        print(f"[resume] skip stage {name}", flush=True)
        return
    for path in clear_outputs_before_run or []:
        if path.exists() and path.is_file():
            path.unlink()
    started_at = utc_now_iso()
    started_monotonic = time.monotonic()
    running_payload = {"stage": name, "cmd": cmd, "fingerprint": fingerprint, "started_at": started_at, "log": str(log_path)}
    (marker_dir / f"{name}.running").write_text(json.dumps(running_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        run(cmd, cwd=cwd, env=env, log_path=log_path)
    except Exception as exc:
        fail = marker_dir / f"{name}.failed.json"
        fail.write_text(json.dumps({
            "stage": name,
            "cmd": cmd,
            "error": repr(exc),
            "log": str(log_path),
            "started_at": started_at,
            "ended_at": utc_now_iso(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        fail = marker_dir / f"{name}.failed.json"
        error = f"stage completed but required outputs are missing: {missing}"
        fail.write_text(json.dumps({
            "stage": name,
            "cmd": cmd,
            "error": error,
            "log": str(log_path),
            "started_at": started_at,
            "ended_at": utc_now_iso(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(error)
    ended_at = utc_now_iso()
    elapsed_seconds = time.monotonic() - started_monotonic
    marker.write_text(json.dumps({
        "stage": name,
        "outputs": [str(p) for p in outputs],
        "fingerprint": fingerprint,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed_seconds,
        "log": str(log_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    running = marker_dir / f"{name}.running"
    if running.exists():
        running.unlink()


def write_done_marker(
    name: str,
    *,
    marker_dir: Path,
    outputs: list[Path],
    fingerprint: dict | None,
    log_path: Path,
    started_at: str,
    elapsed_seconds: float,
) -> None:
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{name}.done"
    marker.write_text(json.dumps({
        "stage": name,
        "outputs": [str(p) for p in outputs],
        "fingerprint": fingerprint,
        "started_at": started_at,
        "ended_at": utc_now_iso(),
        "elapsed_seconds": elapsed_seconds,
        "log": str(log_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def write_generation_config(path: Path, *, thinking: bool | None, temperature: float = 0.0) -> None:
    payload: dict = {"temperature": temperature}
    if thinking is not None:
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_float_csv(value: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return [float(part) for part in parts]


def parse_str_csv(value: str) -> list[str]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one comma-separated string")
    return parts


def arg_was_provided(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def apply_officeqa_default_ranges(args: argparse.Namespace, adapter: Any, data_path: Path, argv: list[str]) -> None:
    if args.benchmark != "officeqa":
        return
    if not arg_was_provided(argv, "--rollout-temperature"):
        args.rollout_temperature = SKILLOPT_QWEN_DEFAULT_TEMPERATURE
    if not arg_was_provided(argv, "--generation-temperature"):
        args.generation_temperature = SKILLOPT_QWEN_DEFAULT_TEMPERATURE
    train_total = len(adapter.load_rows(data_path, BenchmarkSlice(split=args.officeqa_train_split, start=0, end=None)))
    heldout_total = len(adapter.load_rows(data_path, BenchmarkSlice(split=args.officeqa_heldout_split, start=0, end=None)))
    if train_total <= 0:
        raise ValueError(f"OfficeQA train split {args.officeqa_train_split!r} is empty under {data_path}")
    if heldout_total <= 0:
        raise ValueError(f"OfficeQA heldout split {args.officeqa_heldout_split!r} is empty under {data_path}")
    if not arg_was_provided(argv, "--train-start"):
        args.train_start = 0
    if not arg_was_provided(argv, "--train-end"):
        args.train_end = train_total
    if not arg_was_provided(argv, "--heldout-start"):
        args.heldout_start = 0
    if not arg_was_provided(argv, "--heldout-end"):
        args.heldout_end = heldout_total
    if not (0 <= int(args.train_start) < int(args.train_end) <= train_total):
        raise ValueError(
            f"Invalid OfficeQA train range [{args.train_start}, {args.train_end}) "
            f"for split {args.officeqa_train_split!r} with {train_total} items"
        )
    if not (0 <= int(args.heldout_start) < int(args.heldout_end) <= heldout_total):
        raise ValueError(
            f"Invalid OfficeQA heldout range [{args.heldout_start}, {args.heldout_end}) "
            f"for split {args.officeqa_heldout_split!r} with {heldout_total} items"
        )
    train_count = max(0, int(args.train_end) - int(args.train_start))
    if args.tree_scenario == "dynamic_update" and train_count > 0:
        if not arg_was_provided(argv, "--dynamic-initial-count"):
            args.dynamic_initial_count = max(1, int(train_count * 0.6))
        if not arg_was_provided(argv, "--dynamic-arrival-count"):
            args.dynamic_arrival_count = max(0, train_count - int(args.dynamic_initial_count))


def _host_from_url(value: str) -> str:
    try:
        return urlparse(value).hostname or ""
    except Exception:
        return ""


def _proxy_bypass_host(host: str) -> bool:
    normalized = host.strip().lower()
    if not normalized:
        return False
    if normalized in {"localhost", "127.0.0.1", "0.0.0.0"} or normalized.endswith(".nip.io"):
        return True
    if normalized.startswith("10.") or normalized.startswith("192.168."):
        return True
    if normalized.startswith("172."):
        parts = normalized.split(".")
        return len(parts) >= 2 and parts[1].isdigit() and 16 <= int(parts[1]) <= 31
    return False


def append_no_proxy_hosts(env: dict[str, str], urls: Iterable[str]) -> None:
    hosts = [host for host in (_host_from_url(url) for url in urls) if _proxy_bypass_host(host)]
    if not hosts:
        return
    existing = [part.strip() for part in env.get("NO_PROXY", env.get("no_proxy", "")).split(",") if part.strip()]
    merged = existing[:]
    for host in hosts:
        if host not in merged:
            merged.append(host)
    value = ",".join(merged)
    env["NO_PROXY"] = value
    env["no_proxy"] = value


def tree_dataset_order_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_path": str(Path(args.data_path).resolve()),
        "train_start": int(args.train_start),
        "train_end": int(args.train_end),
        "enforce_dataset_order": args.benchmark == "spreadsheetbench",
    }


def resolve_python_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"--python-executable does not exist: {candidate}")
        return str(candidate)
    resolved = shutil.which(value)
    if not resolved:
        raise FileNotFoundError(f"--python-executable is not on PATH: {value}")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SOURCE_FINGERPRINT_SUFFIXES = {".py", ".txt", ".md", ".json", ".toml", ".yaml", ".yml", ".jinja", ".j2"}
SOURCE_FINGERPRINT_SKIP_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def api_key_fingerprint(value: str) -> str:
    if value == "EMPTY":
        return "EMPTY"
    if not value:
        return ""
    return f"sha256:{text_sha256(value)}"


def should_fingerprint_source(path: Path) -> bool:
    if any(part in SOURCE_FINGERPRINT_SKIP_DIRS for part in path.parts):
        return False
    return path.suffix in SOURCE_FINGERPRINT_SUFFIXES


def directory_sha256(path: Path, *, source_only: bool = False) -> str:
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file() and (not source_only or should_fingerprint_source(p))):
        rel = child.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(child.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def path_fingerprint(path: Path, *, source_only: bool = False) -> dict[str, str | bool | int]:
    if not path.exists():
        return {"exists": False}
    if path.is_file():
        return {"exists": True, "kind": "file", "size": path.stat().st_size, "sha256": file_sha256(path)}
    if path.is_dir():
        mode = "source" if source_only else "artifact"
        return {"exists": True, "kind": "dir", "mode": mode, "sha256": directory_sha256(path, source_only=source_only)}
    return {"exists": True, "kind": "other"}


def skill_export_config_from_args(args: argparse.Namespace) -> SkillExportConfig:
    min_level = None if int(args.skill_export_min_level) < 0 else int(args.skill_export_min_level)
    max_level = None if int(args.skill_export_max_level) < 0 else int(args.skill_export_max_level)
    return SkillExportConfig(
        output_dir_name=args.skill_output_dir_name,
        max_node_count=None if int(args.skill_export_max_node_count) < 0 else int(args.skill_export_max_node_count),
        min_level=min_level,
        max_level=max_level,
    )


def skill_export_payload_from_args(args: argparse.Namespace) -> dict[str, int | None]:
    export = skill_export_config_from_args(args)
    return {
        "max_node_count": export.max_node_count,
        "min_level": export.min_level,
        "max_level": export.max_level,
    }


def _protocol_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve()) if value else ""


def _selected_protocol_generation(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in ("base_url", "model", "temperature", "thinking_mode", "extra_body")
    }


def _selected_protocol_embedding(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "base_url",
            "model",
            "max_model_len",
            "max_input_tokens",
            "truncate_long_texts",
            "tokenizer_model",
            "tokenizer_required",
            "truncation_strategy",
        )
    }


def _selected_protocol_chunked_embedding(payload: dict[str, Any], embedding: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload or {})
    enabled = bool(data.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    tokenizer_model = data.get("tokenizer_model") or embedding.get("tokenizer_model") or embedding.get("model")
    return {
        "enabled": True,
        "tokenizer_model": tokenizer_model,
        "chunk_tokens": int(data.get("chunk_tokens", 10000)),
        "overlap_tokens": int(data.get("overlap_tokens", 2000)),
        "pooling": str(data.get("pooling", "mean")),
        "add_special_tokens": bool(data.get("add_special_tokens", False)),
        "normalize_after_pooling": bool(data.get("normalize_after_pooling", False)),
        "fail_if_chunk_exceeds_model_limit": bool(data.get("fail_if_chunk_exceeds_model_limit", True)),
    }


def _selected_protocol_hierarchy(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = default_hierarchy_config(payload or {}).to_dict()
    canonical.pop("dynamic_update", None)
    return canonical


def _selected_protocol_analyst(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = ClusterAnalystConfig.__dataclass_fields__
    cleaned = {key: value for key, value in (payload or {}).items() if key in allowed}
    canonical = ClusterAnalystConfig(**cleaned).__dict__
    return {
        key: canonical.get(key)
        for key in (
            "prompt_style",
            "confidence_floor",
            "tokenizer_model",
            "tokenizer_required",
            "allow_regex_tokenizer_fallback",
            "max_prompt_tokens",
            "max_output_tokens",
            "dynamic_max_output_tokens",
            "multi_card_max_level",
            "max_cards_l0",
            "max_cards_higher",
            "higher_level_mode",
            "truncate_higher_level_extra_cards",
        )
    }


def _source_export_filter_status(source_summary: dict[str, Any], source_runtime: dict[str, Any]) -> str:
    export = source_summary.get("skill_export")
    if not isinstance(export, dict):
        export = source_runtime.get("skill_export")
    if not isinstance(export, dict):
        # Legacy full-tree runs predate level-filtered nodebank export metadata.
        # Since those versions could not encode filtered exports, treat absence
        # as an auditable unfiltered source rather than rejecting valid baselines.
        return "legacy_missing_skill_export_assumed_unfiltered"
    is_unfiltered = (
        export.get("min_level") is None
        and export.get("max_level") is None
        and export.get("max_node_count") is None
    )
    return "unfiltered" if is_unfiltered else "filtered"


def _source_export_is_unfiltered(source_summary: dict[str, Any], source_runtime: dict[str, Any]) -> bool:
    return _source_export_filter_status(source_summary, source_runtime) in {
        "unfiltered",
        "legacy_missing_skill_export_assumed_unfiltered",
    }


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_reused_tree_protocol(
    *,
    reuse_tree_dir: Path,
    current_config: dict[str, Any],
    source_summary: dict[str, Any],
    current_records_sha256: str,
) -> None:
    runtime_path = reuse_tree_dir / "analysis" / "runtime_config.json"
    if not runtime_path.is_file():
        raise FileNotFoundError(f"--reuse-tree-dir missing analysis/runtime_config.json: {runtime_path}")
    source_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    def compare(label: str, source: Any, current: Any) -> None:
        if source != current:
            errors.append(f"{label} mismatch: source={source!r}, current={current!r}")

    compare("scenario", source_runtime.get("scenario"), "static_build")
    compare("current scenario", current_config.get("scenario"), "static_build")
    if source_runtime.get("benchmark") is not None:
        compare("benchmark", source_runtime.get("benchmark"), current_config.get("benchmark"))
    compare("dataset_path", _protocol_path(source_runtime.get("dataset_path")), _protocol_path(current_config.get("dataset_path")))
    compare("train_start", source_runtime.get("train_start"), current_config.get("train_start"))
    compare("train_end", source_runtime.get("train_end"), current_config.get("train_end"))
    compare("enforce_dataset_order", source_runtime.get("enforce_dataset_order"), current_config.get("enforce_dataset_order"))
    compare("max_levels", source_runtime.get("max_levels"), current_config.get("max_levels"))
    compare("generation", _selected_protocol_generation(source_runtime.get("generation") or {}), _selected_protocol_generation(current_config.get("generation") or {}))
    compare("embedding", _selected_protocol_embedding(source_runtime.get("embedding") or {}), _selected_protocol_embedding(current_config.get("embedding") or {}))
    compare(
        "chunked_embedding",
        _selected_protocol_chunked_embedding(source_runtime.get("chunked_embedding") or {}, source_runtime.get("embedding") or {}),
        _selected_protocol_chunked_embedding(current_config.get("chunked_embedding") or {}, current_config.get("embedding") or {}),
    )
    compare("hierarchy", _selected_protocol_hierarchy(source_runtime.get("hierarchy") or {}), _selected_protocol_hierarchy(current_config.get("hierarchy") or {}))
    compare("analyst", _selected_protocol_analyst(source_runtime.get("analyst") or {}), _selected_protocol_analyst(current_config.get("analyst") or {}))

    source_records = Path(str(source_runtime.get("records_path") or ""))
    if not source_records.is_file():
        errors.append(f"source records_path is not readable: {source_records}")
    elif file_sha256(source_records) != current_records_sha256:
        errors.append("records_sha256 mismatch between reused tree source records and current records")

    if not _source_export_is_unfiltered(source_summary, source_runtime):
        errors.append("source tree was already level/max-node filtered; retrieval ablations require an unfiltered full static tree")

    if errors:
        raise RuntimeError(
            "Rejected --reuse-tree-dir because its source protocol does not match the current retrieval ablation:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def clear_reuse_tree_outputs(tree_dir: Path, *, skill_output_dir_name: str, usage_logs: list[Path]) -> None:
    for path in [
        tree_dir / "analysis",
        tree_dir / "dynamic_snapshots",
        tree_dir / skill_output_dir_name,
    ]:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
    for path in [
        tree_dir / "summary.json",
        tree_dir / "hierarchy_state.json",
        tree_dir / "hierarchy_layers.json",
    ]:
        if path.exists() and path.is_file():
            path.unlink()
    for path in usage_logs:
        if path.exists() and path.is_file():
            path.unlink()


def materialize_reused_tree_nodebank(
    *,
    reuse_tree_dir: Path,
    tree_dir: Path,
    args: argparse.Namespace,
    current_config: dict[str, Any],
    fingerprint: dict,
    marker_dir: Path,
    log_path: Path,
    usage_logs: list[Path] | None = None,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    started = time.monotonic()
    state_path = reuse_tree_dir / "hierarchy_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"--reuse-tree-dir missing hierarchy_state.json: {state_path}")
    resolved_tree_dir = tree_dir.resolve()
    resolved_reuse_tree_dir = reuse_tree_dir.resolve()
    if (
        resolved_tree_dir == resolved_reuse_tree_dir
        or _path_contains(resolved_tree_dir, resolved_reuse_tree_dir)
        or _path_contains(resolved_reuse_tree_dir, resolved_tree_dir)
    ):
        raise ValueError(
            "--reuse-tree-dir must not overlap the output tree dir: "
            f"reuse={resolved_reuse_tree_dir}, output={resolved_tree_dir}"
        )
    tree_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    source_summary_path = reuse_tree_dir / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.is_file() else {}
    source_export_status = _source_export_filter_status(source_summary, json.loads((reuse_tree_dir / "analysis" / "runtime_config.json").read_text(encoding="utf-8")))
    validate_reused_tree_protocol(
        reuse_tree_dir=reuse_tree_dir,
        current_config=current_config,
        source_summary=source_summary,
        current_records_sha256=str(fingerprint.get("records_sha256") or ""),
    )
    clear_reuse_tree_outputs(tree_dir, skill_output_dir_name=args.skill_output_dir_name, usage_logs=list(usage_logs or []))
    tree_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = tree_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    runtime_payload = {
        **current_config,
        "output_dir": str(tree_dir),
        "reuse_tree_dir": str(reuse_tree_dir),
        "reuse_materialization": True,
        "source_tree_summary": str(source_summary_path) if source_summary_path.is_file() else "",
        "source_export_filter_status": source_export_status,
    }
    (analysis_dir / "runtime_config.json").write_text(json.dumps(runtime_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (analysis_dir / "reuse_tree_audit.json").write_text(json.dumps({
        "reuse_tree_dir": str(reuse_tree_dir.resolve()),
        "output_tree_dir": str(tree_dir.resolve()),
        "source_export_filter_status": source_export_status,
        "source_summary": str(source_summary_path) if source_summary_path.is_file() else "",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (tree_dir / "hierarchy_state.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    export = export_skill_files_from_payload(payload, tree_dir, config=skill_export_config_from_args(args))
    selector = SkillBankSelector(
        skillbank_root=export.output_dir,
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        api_key="EMPTY",
        cache_path=Path(export.output_dir) / ".dynamix_skillbank_index.json",
    )
    selector._load_or_build_index()
    summary = {
        **source_summary,
        "scenario": "static_build",
        "reuse_tree_dir": str(reuse_tree_dir),
        "node_count": export.node_count,
        "node_bank_dir": export.output_dir,
        "node_bank_manifest": export.manifest_path,
        "skillbank_index": str(Path(export.output_dir) / ".dynamix_skillbank_index.json"),
        "skill_export": {
            "output_dir_name": args.skill_output_dir_name,
            **skill_export_payload_from_args(args),
        },
    }
    (tree_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join([
            f"reused_tree_dir={reuse_tree_dir}",
            f"state_path={state_path}",
            f"node_bank_manifest={export.manifest_path}",
            f"node_count={export.node_count}",
        ]) + "\n",
        encoding="utf-8",
    )
    write_done_marker(
        "04_build_tree",
        marker_dir=marker_dir,
        outputs=[tree_dir / "summary.json"],
        fingerprint=fingerprint,
        log_path=log_path,
        started_at=started_at,
        elapsed_seconds=time.monotonic() - started,
    )
    return summary


def dataset_json_path(data_path: str) -> Path:
    path = Path(data_path)
    return path / "dataset.json" if path.is_dir() else path


def stage_fingerprint(contract: str, cmd: list[str], **payload: object) -> dict[str, object]:
    return {"stage_contract": contract, "cmd": cmd, **payload}


def resolved_optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def stage_source_fingerprints(repo: Path) -> dict[str, dict[str, str | bool | int]]:
    return {
        "runner": path_fingerprint(Path(__file__).resolve()),
        "run_spreadsheetbench": path_fingerprint(repo / "run_spreadsheetbench.py"),
        "evaluate_with_official": path_fingerprint(repo / "evaluate_with_official.py"),
        "extract_trace2skill_logs": path_fingerprint(repo / "scripts" / "extract_trace2skill_logs.py"),
        "run_officeqa_benchmark": path_fingerprint(repo / "scripts" / "run_officeqa_benchmark.py"),
        "evaluate_officeqa_results": path_fingerprint(repo / "scripts" / "evaluate_officeqa_results.py"),
        "extract_officeqa_records": path_fingerprint(repo / "scripts" / "extract_officeqa_records.py"),
        "build_dynamix_tree": path_fingerprint(repo / "scripts" / "build_dynamix_tree.py"),
        "spreadsheetbench_support": path_fingerprint(repo / "spreadsheetbench_support.py"),
        "spreadsheet_agent": path_fingerprint(repo / "spreadsheet_agent", source_only=True),
        "react_agent": path_fingerprint(repo / "src" / "react_agent", source_only=True),
        "dynamix_benchmarks": path_fingerprint(repo / "src" / "dynamix_benchmarks", source_only=True),
        "dynamix_core": path_fingerprint(repo / "src" / "dynamix_core", source_only=True),
        "dynamix_trace2skill": path_fingerprint(repo / "src" / "dynamix_trace2skill", source_only=True),
    }


def benchmark_source_fingerprints(source_fp: dict[str, dict[str, str | bool | int]], *, benchmark: str, stage: str) -> dict[str, dict[str, str | bool | int]]:
    base = {"runner": source_fp["runner"]}
    if benchmark == "officeqa":
        by_stage = {
            "rollout": {
                "run_officeqa_benchmark": source_fp["run_officeqa_benchmark"],
                "react_agent": source_fp["react_agent"],
                "dynamix_benchmarks": source_fp["dynamix_benchmarks"],
                "dynamix_trace2skill": source_fp["dynamix_trace2skill"],
            },
            "eval": {
                "evaluate_officeqa_results": source_fp["evaluate_officeqa_results"],
                "dynamix_benchmarks": source_fp["dynamix_benchmarks"],
            },
            "extract": {
                "extract_officeqa_records": source_fp["extract_officeqa_records"],
                "dynamix_benchmarks": source_fp["dynamix_benchmarks"],
            },
        }
        return {**base, **by_stage[stage]}
    by_stage = {
        "rollout": {
            "run_spreadsheetbench": source_fp["run_spreadsheetbench"],
            "spreadsheet_agent": source_fp["spreadsheet_agent"],
            "react_agent": source_fp["react_agent"],
            "dynamix_trace2skill": source_fp["dynamix_trace2skill"],
        },
        "eval": {
            "evaluate_with_official": source_fp["evaluate_with_official"],
            "spreadsheetbench_support": source_fp["spreadsheetbench_support"],
        },
        "extract": {
            "extract_trace2skill_logs": source_fp["extract_trace2skill_logs"],
        },
    }
    return {**base, **by_stage[stage]}


def rollout_protocol(args: argparse.Namespace, *, generation_config: Path) -> dict[str, object]:
    return {
        "benchmark": args.benchmark,
        "model": args.model,
        "openai_base_url": args.openai_base_url,
        "openai_api_key": api_key_fingerprint(args.openai_api_key),
        "thinking": args.thinking,
        "max_turns": int(args.max_turns),
        "workers": int(args.workers),
        "timeout_seconds": float(args.rollout_client_timeout_seconds),
        "retry_wait_seconds": list(args.rollout_client_retry_wait_seconds),
        "llm_client": args.rollout_llm_client,
        "num_random_seeds": int(args.rollout_num_random_seeds),
        "seeds": str(args.rollout_seeds),
        "instance_ids": str(args.rollout_instance_ids),
        "missing_only": bool(args.rollout_missing_only),
        "repeat": int(args.rollout_repeat),
        "shuffle_seed": str(args.rollout_shuffle_seed),
        "sample": int(args.rollout_sample),
        "generation_config": path_fingerprint(generation_config),
        "officeqa": {
            "docs_dir": list(getattr(args, "_officeqa_docs_dirs_resolved", getattr(args, "officeqa_docs_dir", []))),
            "train_split": getattr(args, "officeqa_train_split", ""),
            "heldout_split": getattr(args, "officeqa_heldout_split", ""),
            "evaluator": getattr(args, "officeqa_evaluator", "skillopt"),
            "max_completion_tokens": int(getattr(args, "officeqa_max_completion_tokens", 16384)),
            "reward_path": getattr(args, "_officeqa_reward_path_resolved", getattr(args, "officeqa_reward_path", "")),
            "reward_tolerance": float(getattr(args, "officeqa_reward_tolerance", 0.0)),
            "allow_fallback_evaluator": bool(getattr(args, "officeqa_allow_fallback_evaluator", False)),
            "continue_on_infra_error": bool(getattr(args, "officeqa_continue_on_infra_error", False)),
            "use_oracle_context": bool(getattr(args, "officeqa_use_oracle_context", True)),
        } if getattr(args, "benchmark", "spreadsheetbench") == "officeqa" else {},
    }


def skillbank_retrieval_protocol(args: argparse.Namespace, *, cache_path: Path, selection_log: Path) -> dict[str, object]:
    return {
        "benchmark": args.benchmark,
        "query_policy": "instruction/question + Task type; answer_position/gold answer/source_docs excluded",
        "top_k": int(args.skillbank_top_k),
        "embedding_base_url": args.embedding_base_url,
        "embedding_model": args.embedding_model,
        "embedding_api_key": api_key_fingerprint("EMPTY"),
        "cache_path": str(cache_path),
        "selection_log": str(selection_log),
    }


def expected_dynamic_counts(*, record_count: int, initial_count: int, arrival_count: int) -> dict[str, int]:
    safe_record_count = max(0, int(record_count))
    safe_initial = min(max(1, int(initial_count)), safe_record_count) if safe_record_count else 0
    remaining = max(0, safe_record_count - safe_initial)
    arrival_limit = int(arrival_count)
    arrivals = remaining if arrival_limit <= 0 else min(remaining, arrival_limit)
    return {"initial_count": safe_initial, "arrival_count": arrivals, "insertion_count": arrivals}


def load_record_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("records", "data", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    raise ValueError(f"unsupported records format: {path}")


def load_record_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        for key in ("records", "data", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
    else:
        rows = []
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"unsupported records format: {path}")
    return list(rows)


def load_dataset_rows(data_path: Path) -> list[dict[str, Any]]:
    dataset_path = data_path / "dataset.json" if data_path.is_dir() else data_path
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("results") or payload.get("data") or payload.get("instances") or []
    else:
        rows = []
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"unsupported dataset format: {dataset_path}")
    return list(rows)


def _task_id_from_row(row: dict[str, Any], fallback: object | None = None) -> str:
    value = row.get("task_id", row.get("id", row.get("instance_id", fallback)))
    if value is None:
        raise ValueError(f"row has no task id: {row}")
    return str(value)


def write_dataset_ordered_records(
    *,
    source_records: Path,
    data_path: Path,
    output_path: Path,
    manifest_path: Path,
    train_start: int,
    train_end: int,
) -> dict[str, Any]:
    """Write records in the exact SpreadsheetBench dataset order for the train slice."""
    records = load_record_rows(source_records)
    dataset_rows = load_dataset_rows(data_path)
    expected_ids = [_task_id_from_row(row, fallback=index) for index, row in enumerate(dataset_rows[train_start:train_end], start=train_start)]
    by_task_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        task_id = _task_id_from_row(record)
        if task_id in by_task_id:
            duplicates.append(task_id)
        by_task_id[task_id] = record
    missing = [task_id for task_id in expected_ids if task_id not in by_task_id]
    expected_set = set(expected_ids)
    extra = [task_id for task_id in by_task_id if task_id not in expected_set]
    if duplicates or missing or extra:
        raise RuntimeError(
            "records.json does not match the requested train slice exactly: "
            f"duplicates={duplicates[:10]}, missing={missing[:10]}, extra={extra[:10]}"
        )
    ordered = [by_task_id[task_id] for task_id in expected_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    source_ids = [_task_id_from_row(record) for record in records]
    manifest = {
        "policy": "records are ordered by dataset.json train slice order; no filename sorting or random shuffling",
        "source_records": str(source_records),
        "ordered_records": str(output_path),
        "source_dataset_json": str(dataset_json_path(str(data_path)).resolve()),
        "train_range": [int(train_start), int(train_end)],
        "record_count": len(ordered),
        "source_order_equal_dataset_order": source_ids == expected_ids,
        "first_task_ids": expected_ids[:10],
        "last_task_ids": expected_ids[-10:],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_tree_summary_for_heldout(summary: dict, args: argparse.Namespace) -> None:
    scenario = str(summary.get("scenario", ""))
    if scenario != args.tree_scenario:
        raise RuntimeError(f"DynaMix tree summary scenario mismatch: expected {args.tree_scenario!r}, got {scenario!r}")
    if args.tree_scenario != "dynamic_update":
        return
    if getattr(args, "benchmark", "spreadsheetbench") == "officeqa":
        train_count = int(summary.get("record_count", 0))
    elif hasattr(args, "train_start") and hasattr(args, "train_end"):
        train_count = int(args.train_end) - int(args.train_start)
    else:
        train_count = int(summary.get("record_count", 0))
    expected = expected_dynamic_counts(
        record_count=train_count,
        initial_count=int(args.dynamic_initial_count),
        arrival_count=int(args.dynamic_arrival_count),
    )
    observed = {key: int(summary.get(key, -1)) for key in expected}
    if observed != expected:
        raise RuntimeError(f"DynaMix dynamic summary mismatch before heldout: expected {expected}, got {observed}")
    updated = int(summary.get("updated_count", -1))
    excluded = int(summary.get("excluded_count", 0))
    if updated < 0 or excluded < 0 or updated + excluded != expected["arrival_count"]:
        raise RuntimeError(
            "DynaMix dynamic insertion accounting mismatch before heldout: "
            f"arrival_count={expected['arrival_count']}, updated_count={updated}, excluded_count={excluded}"
        )


def aggregate_usage_jsonl(path: Path) -> dict[str, Any]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    records = 0
    cache_hits = 0
    non_cache_records = 0
    records_with_usage = 0
    records_without_usage = 0
    malformed_records = 0
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "records": 0,
            "cache_hits": 0,
            "non_cache_records": 0,
            "records_with_usage": 0,
            "records_without_usage": 0,
            "malformed_records": 0,
            "usage_available": False,
            "provider_usage_status": "missing",
            "call_source_status": "missing_log",
            "usage_status": "missing",
            "totals": totals,
        }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records += 1
            try:
                payload = json.loads(line)
            except Exception:
                malformed_records += 1
                continue
            if payload.get("cache_hit"):
                cache_hits += 1
                continue
            non_cache_records += 1
            usage = payload.get("usage") if isinstance(payload, dict) else None
            if not isinstance(usage, dict) or not usage:
                records_without_usage += 1
                continue
            records_with_usage += 1
            for key in totals:
                value = usage.get(key)
                if value is None:
                    continue
                try:
                    totals[key] += int(float(value))
                except (TypeError, ValueError):
                    continue
    if records == 0:
        provider_usage_status = "missing"
        call_source_status = "empty_log"
    elif non_cache_records == 0:
        provider_usage_status = "all_cached"
        call_source_status = "all_cached"
    elif records_with_usage == 0:
        provider_usage_status = "missing"
        call_source_status = "uncached_without_usage" if cache_hits == 0 else "mixed_cached_missing_usage"
    elif records_with_usage == non_cache_records:
        provider_usage_status = "complete"
        call_source_status = "uncached_complete" if cache_hits == 0 else "mixed_cached_complete"
    else:
        provider_usage_status = "partial"
        call_source_status = "uncached_partial" if cache_hits == 0 else "mixed_cached_partial"
    return {
        "path": str(path),
        "exists": True,
        "records": records,
        "cache_hits": cache_hits,
        "non_cache_records": non_cache_records,
        "records_with_usage": records_with_usage,
        "records_without_usage": records_without_usage,
        "malformed_records": malformed_records,
        "usage_available": records_with_usage > 0,
        "provider_usage_status": provider_usage_status,
        "call_source_status": call_source_status,
        "usage_status": provider_usage_status,
        "totals": totals,
    }


def read_done_marker(marker_dir: Path, stage: str) -> dict[str, Any]:
    path = marker_dir / f"{stage}.done"
    if not path.exists():
        return {"stage": stage, "done": False, "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"stage": stage, "done": False, "path": str(path), "error": repr(exc)}
    payload["done"] = True
    payload["path"] = str(path)
    return payload


def collect_prompt_token_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events") if isinstance(payload, dict) else []
    if not isinstance(events, list):
        events = []
    prompt_tokens = [int(event.get("prompt_tokens", 0) or 0) for event in events if isinstance(event, dict)]
    max_prompt_tokens = [int(event.get("max_prompt_tokens", 0) or 0) for event in events if isinstance(event, dict)]
    top_events = sorted(
        [
            {
                "community_id": event.get("community_id"),
                "level": event.get("level"),
                "member_count": event.get("member_count"),
                "prompt_tokens": int(event.get("prompt_tokens", 0) or 0),
                "max_prompt_tokens": int(event.get("max_prompt_tokens", 0) or 0),
                "over_budget": bool(event.get("over_budget")),
            }
            for event in events
            if isinstance(event, dict)
        ],
        key=lambda item: item["prompt_tokens"],
        reverse=True,
    )[:10]
    return {
        "path": str(path),
        "exists": True,
        "event_count": len(events),
        "max_prompt_tokens_observed": max(prompt_tokens, default=0),
        "configured_max_prompt_tokens": max(max_prompt_tokens, default=0),
        "near_configured_limit_count": sum(
            1 for value, limit in zip(prompt_tokens, max_prompt_tokens) if limit > 0 and value >= int(limit * 0.95)
        ),
        "over_budget_count": sum(1 for event in events if isinstance(event, dict) and bool(event.get("over_budget"))),
        "top_events": top_events,
    }


def collect_chunked_embedding_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "exists": True,
        "chunk_tokens": payload.get("chunk_tokens"),
        "overlap_tokens": payload.get("overlap_tokens"),
        "pooling": payload.get("pooling"),
        "text_count": payload.get("text_count"),
        "total_chunk_count": payload.get("total_chunk_count") or payload.get("chunk_count"),
        "max_token_count": payload.get("max_token_count"),
        "over_limit_chunk_count": payload.get("over_limit_chunk_count"),
    }


def runtime_dead_corner_findings(args: argparse.Namespace) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    analyst_budget = int(float(args.summary_max_model_tokens) * float(args.summary_budget_ratio))
    evidence_budget = analyst_budget - int(args.summary_prompt_overhead_reserve_tokens)
    if analyst_budget >= int(args.summary_max_model_tokens) * 0.9:
        findings.append({
            "severity": "high",
            "area": "summary_budget",
            "finding": "analyst prompt budget leaves little context-window headroom for chat-template/thinking overhead.",
            "evidence": f"analyst_budget={analyst_budget}, max_model_tokens={args.summary_max_model_tokens}",
        })
    if evidence_budget <= 0:
        findings.append({
            "severity": "blocker",
            "area": "summary_budget",
            "finding": "member evidence budget is non-positive, so build cannot select feasible communities.",
            "evidence": f"analyst_budget={analyst_budget}, overhead={args.summary_prompt_overhead_reserve_tokens}",
        })
    if int(args.analyst_max_prompt_tokens) > 0 and int(args.analyst_max_prompt_tokens) < evidence_budget:
        findings.append({
            "severity": "high",
            "area": "analyst_budget_override",
            "finding": "analyst max prompt override is smaller than the tree-builder evidence budget; build may pass but analyst preflight can fail.",
            "evidence": f"analyst_max_prompt_tokens={args.analyst_max_prompt_tokens}, evidence_budget={evidence_budget}",
        })
    if int(args.budget_refinement_apply_to_level) == 0:
        findings.append({
            "severity": "watch",
            "area": "budget_refinement",
            "finding": "budget refinement only protects L0 raw-trajectory communities; unusually verbose L1+ cards can still trigger analyst over-budget failures.",
            "evidence": "budget_refinement_apply_to_level=0",
        })
    if args.soft_recursive_assignment == "cumulative_mass":
        findings.append({
            "severity": "info",
            "area": "soft_membership",
            "finding": "top_r_memberships is inactive under cumulative_mass assignment; max_membership_gap and cumulative_mass_coverage control fan-out.",
            "evidence": f"recursive_assignment={args.soft_recursive_assignment}, top_r={args.soft_top_r_memberships}",
        })
    if args.soft_recursive_assignment == "cumulative_mass":
        findings.append({
            "severity": "info",
            "area": "soft_membership",
            "finding": "cumulative_mass assignment uses max_membership_gap as the practical tail stop; loosening the gap can enlarge communities and reintroduce over-budget prompts.",
            "evidence": f"coverage={args.soft_cumulative_mass_coverage}, max_gap={args.soft_max_membership_gap}",
        })
    if int(args.workers) > 4 and str(args.thinking) == "true":
        findings.append({
            "severity": "watch",
            "area": "concurrency_timeout",
            "finding": "thinking=true with high rollout/generation concurrency can create long queueing and timeout retries even when the model endpoint is healthy.",
            "evidence": f"workers={args.workers}, generation_timeout={args.generation_timeout_seconds}, rollout_timeout={args.rollout_client_timeout_seconds}",
        })
    if bool(args.chunked_embedding_enabled) and int(args.embedding_batch_size) * int(args.chunked_embedding_chunk_tokens) >= int(args.embedding_max_model_len) * 4:
        findings.append({
            "severity": "watch",
            "area": "embedding_batching",
            "finding": "each embedding item is under the model limit, but a large batch of long chunks can still overload an embedding service by aggregate tokens.",
            "evidence": f"batch_size={args.embedding_batch_size}, chunk_tokens={args.chunked_embedding_chunk_tokens}, max_model_len={args.embedding_max_model_len}",
        })
    train_count = int(getattr(args, "_observed_train_records", int(args.train_end) - int(args.train_start)))
    expected_dynamic = expected_dynamic_counts(
        record_count=train_count,
        initial_count=int(args.dynamic_initial_count),
        arrival_count=int(args.dynamic_arrival_count),
    )
    if args.tree_scenario == "dynamic_update" and expected_dynamic["initial_count"] + expected_dynamic["arrival_count"] != train_count:
        findings.append({
            "severity": "high",
            "area": "dynamic_coverage",
            "finding": "dynamic protocol will not consume the full train split.",
            "evidence": f"train_count={train_count}, expected_dynamic={expected_dynamic}",
        })
    return findings


def write_experiment_stage_report(
    *,
    run_dir: Path,
    marker_dir: Path,
    stages: list[str],
    usage_logs_by_stage: dict[str, list[Path]],
    runtime: dict[str, Any],
    args: argparse.Namespace,
    marker_dirs_by_stage: dict[str, Path] | None = None,
) -> dict[str, Any]:
    stage_reports = []
    for stage in stages:
        marker = read_done_marker((marker_dirs_by_stage or {}).get(stage, marker_dir), stage)
        usage_logs = [aggregate_usage_jsonl(path) for path in usage_logs_by_stage.get(stage, [])]
        token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        for usage_log in usage_logs:
            for key, value in usage_log.get("totals", {}).items():
                if key in token_totals:
                    token_totals[key] += int(value or 0)
        stage_reports.append({
            "stage": stage,
            "done": bool(marker.get("done")),
            "started_at": marker.get("started_at"),
            "ended_at": marker.get("ended_at"),
            "elapsed_seconds": marker.get("elapsed_seconds"),
            "log": marker.get("log"),
            "usage_logs": usage_logs,
            "usage_summary": {
                "records": sum(int(log.get("records", 0) or 0) for log in usage_logs),
                "cache_hits": sum(int(log.get("cache_hits", 0) or 0) for log in usage_logs),
                "non_cache_records": sum(int(log.get("non_cache_records", 0) or 0) for log in usage_logs),
                "records_with_usage": sum(int(log.get("records_with_usage", 0) or 0) for log in usage_logs),
                "records_without_usage": sum(int(log.get("records_without_usage", 0) or 0) for log in usage_logs),
                "provider_usage_statuses": [log.get("provider_usage_status") for log in usage_logs],
                "call_source_statuses": [log.get("call_source_status") for log in usage_logs],
            },
            "token_totals": token_totals,
        })
    prompt_stats = collect_prompt_token_stats(run_dir / "dynamix_tree" / "analysis" / "cluster_prompt_token_report.json")
    chunk_stats = collect_chunked_embedding_stats(run_dir / "dynamix_tree" / "analysis" / "chunked_embedding_report.json")
    report = {
        "format": "dynamix_experiment_stage_report_v1",
        "created_at": utc_now_iso(),
        "run_dir": str(run_dir),
        "runtime": runtime,
        "stages": stage_reports,
        "prompt_token_stats": prompt_stats,
        "chunked_embedding_stats": chunk_stats,
        "runtime_dead_corner_findings": runtime_dead_corner_findings(args),
    }
    json_path = run_dir / "experiment_stage_report.json"
    md_path = run_dir / "experiment_stage_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_experiment_stage_report_md(report), encoding="utf-8")
    return report


def render_experiment_stage_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# DynaMix Experiment Stage Report",
        "",
        f"Run dir: `{report.get('run_dir')}`",
        f"Created at: `{report.get('created_at')}`",
        "",
        "## Stage Time And Token Usage",
        "",
        "| Stage | Done | Elapsed(s) | Prompt/Input Tokens | Completion/Output Tokens | Total Tokens | Usage Status | Calls(cache/non-cache/with/missing) |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for stage in report.get("stages", []):
        totals = stage.get("token_totals", {})
        prompt_like = int(totals.get("prompt_tokens", 0) or 0) + int(totals.get("input_tokens", 0) or 0)
        completion_like = int(totals.get("completion_tokens", 0) or 0) + int(totals.get("output_tokens", 0) or 0)
        total = int(totals.get("total_tokens", 0) or 0)
        usage_logs = stage.get("usage_logs", [])
        available = ", ".join(
            f"{log.get('provider_usage_status', 'missing')}/{log.get('call_source_status', 'missing')}"
            for log in usage_logs
        ) or "none"
        usage_summary = stage.get("usage_summary", {})
        call_counts = (
            f"{int(usage_summary.get('records', 0) or 0)}/"
            f"{int(usage_summary.get('cache_hits', 0) or 0)}/"
            f"{int(usage_summary.get('non_cache_records', 0) or 0)}/"
            f"{int(usage_summary.get('records_with_usage', 0) or 0)}/"
            f"{int(usage_summary.get('records_without_usage', 0) or 0)}"
        )
        elapsed = stage.get("elapsed_seconds")
        elapsed_text = f"{float(elapsed):.1f}" if isinstance(elapsed, (int, float)) else ""
        lines.append(
            f"| `{stage.get('stage')}` | {stage.get('done')} | {elapsed_text} | {prompt_like} | {completion_like} | {total} | {available} | {call_counts} |"
        )
    lines.extend(["", "## Build Token Pressure", ""])
    prompt_stats = report.get("prompt_token_stats", {})
    lines.append(f"- Prompt token report: `{prompt_stats.get('path')}`")
    lines.append(f"- Max observed prompt tokens: `{prompt_stats.get('max_prompt_tokens_observed', 0)}` / configured `{prompt_stats.get('configured_max_prompt_tokens', 0)}`")
    lines.append(f"- Near configured limit count: `{prompt_stats.get('near_configured_limit_count', 0)}`; over budget count: `{prompt_stats.get('over_budget_count', 0)}`")
    for event in prompt_stats.get("top_events", [])[:5]:
        lines.append(
            f"- Top prompt `{event.get('community_id')}` level={event.get('level')} members={event.get('member_count')} tokens={event.get('prompt_tokens')}/{event.get('max_prompt_tokens')}"
        )
    chunk_stats = report.get("chunked_embedding_stats", {})
    lines.extend(["", "## Chunked Embedding", ""])
    lines.append(f"- Chunk report: `{chunk_stats.get('path')}`")
    lines.append(f"- chunk_tokens=`{chunk_stats.get('chunk_tokens')}`, overlap_tokens=`{chunk_stats.get('overlap_tokens')}`, pooling=`{chunk_stats.get('pooling')}`")
    lines.append(f"- max_token_count=`{chunk_stats.get('max_token_count')}`, over_limit_chunk_count=`{chunk_stats.get('over_limit_chunk_count')}`")
    lines.extend(["", "## Runtime Dead-Corner Findings", ""])
    findings = report.get("runtime_dead_corner_findings", [])
    if not findings:
        lines.append("- No runtime dead-corner findings recorded.")
    for finding in findings:
        lines.append(f"- [{finding.get('severity')}] {finding.get('area')}: {finding.get('finding')} Evidence: `{finding.get('evidence')}`")
    lines.append("")
    return "\n".join(lines)


def write_split_manifest(data_path: Path, run_dir: Path, *, train_start: int, train_end: int, heldout_start: int, heldout_end: int) -> dict:
    dataset_path = data_path / "dataset.json" if data_path.is_dir() else data_path
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = rows.get("results") or rows.get("data") or rows.get("instances") or []
    def subset(start: int, end: int) -> list[dict]:
        out = []
        for index, row in enumerate(rows[start:end], start=start):
            out.append({"index": index, "id": str(row.get("id", row.get("task_id", index))), "instruction_type": row.get("instruction_type", ""), "answer_position": row.get("answer_position", "")})
        return out
    manifest = {
        "source_dataset_json": str(dataset_path.resolve()),
        "policy": "Trace2Skill dataset order / natural task id order; runner still uses start/end indices",
        "train_range": [train_start, train_end],
        "heldout_range": [heldout_start, heldout_end],
        "train": subset(train_start, train_end),
        "heldout": subset(heldout_start, heldout_end),
    }
    path = run_dir / "split_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trace2Skill train collection -> nodebank build -> heldout experiment")
    parser.add_argument("--benchmark", choices=["spreadsheetbench", "officeqa"], default="spreadsheetbench", help="Benchmark adapter. spreadsheetbench preserves the original Trace2Skill flow; officeqa uses local OfficeQA document ReAct rollout.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--officeqa-docs-dir", action="append", default=[], help="OfficeQA docs root; repeatable. Defaults to OFFICEQA_DOCS_DIR in run_officeqa_benchmark.py.")
    parser.add_argument("--officeqa-train-split", default="train")
    parser.add_argument("--officeqa-heldout-split", default="test")
    parser.add_argument("--officeqa-evaluator", choices=["skillopt", "official_reward", "fallback"], default="skillopt")
    parser.add_argument("--officeqa-max-completion-tokens", type=int, default=16384)
    parser.add_argument("--officeqa-reward-path", default="")
    parser.add_argument("--officeqa-reward-tolerance", type=float, default=0.0)
    parser.add_argument("--officeqa-allow-fallback-evaluator", action="store_true", help="Debug only: allow simplified OfficeQA numeric/text evaluator when official reward.py is missing.")
    parser.add_argument("--officeqa-continue-on-infra-error", action="store_true", help="Debug only: convert OfficeQA infra/config exceptions into failed rows instead of failing the stage.")
    parser.add_argument("--officeqa-use-oracle-context", type=parse_bool, default=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--records-path", default=None, help="Existing extracted records.json; when set, skip train rollout/eval/extraction and build from this file")
    parser.add_argument("--reuse-train-run-dir", default=None, help="Existing run dir containing records.json; when set, skip train rollout/eval/extraction")
    parser.add_argument("--train-artifact-dir", default=None, help="Directory for train rollout/eval/extraction artifacts; default: --run-dir")
    parser.add_argument("--scenario-output-dir", default=None, help="Directory for tree, nodebank, heldout, and final reports; default: --run-dir")
    parser.add_argument("--reuse-tree-dir", default=None, help="Existing dynamix_tree directory with hierarchy_state.json; skip tree rebuild and re-export a filtered nodebank")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=200)
    parser.add_argument("--heldout-start", type=int, default=200)
    parser.add_argument("--heldout-end", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=os.environ.get("GEN_MODEL", "Qwen3.5-9B"))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18002/v1"))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8017/v1"))
    parser.add_argument("--embedding-model", default=os.environ.get("EMBED_MODEL", "Qwen3-Embedding-0.6B"))
    parser.add_argument("--embedding-tokenizer", default=os.environ.get("EMBED_TOKENIZER", "/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-0___6B"))
    parser.add_argument("--python-executable", default=os.environ.get("DYNAMIX_PYTHON", sys.executable), help="Python executable used for all experiment stages; its bin dir is prepended to PATH so agent bash actions can call bare python")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--thinking", choices=["true", "false", "null"], default="true", help="Qwen thinking setting for Trace2Skill rollout and DynaMix analyst calls")
    parser.add_argument("--skillbank-top-k", type=int, default=10, help="Select top-k DynaMix nodebank nodes by embedding before each heldout task")
    parser.add_argument("--tree-scenario", choices=["dynamic_update", "static_build"], default="dynamic_update", help="DynaMix build mode before heldout; default is the train200 60/40 dynamic protocol")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--tree-policy", default="projected_gmm_bic")
    parser.add_argument("--graph-kind", default="overlapping_experience_hierarchy")
    parser.add_argument("--allow-overlap", type=parse_bool, default=True)
    parser.add_argument("--allow-multi-parent", type=parse_bool, default=True)
    parser.add_argument("--use-support-mass", type=parse_bool, default=True)
    parser.add_argument("--dynamic-initial-count", type=int, default=120, help="Dynamic mode: number of initial train records used for the static seed tree")
    parser.add_argument("--dynamic-arrival-count", type=int, default=80, help="Dynamic mode: number of later train records inserted sequentially; <=0 consumes all remaining train records")
    parser.add_argument("--dynamic-update-batch-size", type=int, default=8, help="Dynamic mode: admit this many arrival trajectories sequentially, then run layer-local LLM summaries concurrently")
    parser.add_argument("--dynamic-shuffle-seed", type=int, default=42, help="Dynamic mode: reproducibly shuffle arrival trajectories before batched admission; use -1 to disable shuffle")
    parser.add_argument("--dynamic-snapshot-include-embeddings", type=parse_bool, default=True, help="Dynamic mode: include item embeddings in per-batch snapshots so resume can continue routing")
    parser.add_argument("--dynamic-resume-from-snapshots", type=parse_bool, default=False, help="Dynamic mode: resume from latest dynamic_snapshots/batch_* snapshot when present; default false until fingerprint validation is enabled")
    parser.add_argument("--max-levels", type=int, default=8)
    parser.add_argument("--skill-output-dir-name", default="skills")
    parser.add_argument("--skill-export-min-level", type=int, default=-1, help="-1 exports all lower levels; 1 exports only ExperienceCards at level >= 1")
    parser.add_argument("--skill-export-max-level", type=int, default=-1, help="-1 exports all higher levels; set 1 for L1-only retrieval nodebank")
    parser.add_argument("--skill-export-max-node-count", type=int, default=-1, help="-1 exports all matching nodes")
    parser.add_argument("--rollout-temperature", type=float, default=0.0)
    parser.add_argument("--rollout-client-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--rollout-client-retry-wait-seconds", type=parse_float_csv, default=[5.0, 10.0, 30.0])
    parser.add_argument("--rollout-llm-client", default="openai")
    parser.add_argument("--rollout-num-random-seeds", type=int, default=1)
    parser.add_argument("--rollout-seeds", default="")
    parser.add_argument("--rollout-instance-ids", default="")
    parser.add_argument("--rollout-missing-only", type=parse_bool, default=False)
    parser.add_argument("--rollout-repeat", type=int, default=1)
    parser.add_argument("--rollout-shuffle-seed", default="")
    parser.add_argument("--rollout-sample", type=int, default=0)
    parser.add_argument("--generation-temperature", type=float, default=0.6)
    parser.add_argument("--generation-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--generation-max-concurrency", type=int, default=None, help="DynaMix analyst generation concurrency; default: --workers")
    parser.add_argument("--generation-retry-wait-seconds", type=parse_float_csv, default=[2.0, 5.0, 15.0])
    parser.add_argument("--embedding-max-model-len", type=int, default=8192)
    parser.add_argument("--embedding-max-input-tokens", type=int, default=8000)
    parser.add_argument("--embedding-truncate-long-texts", type=parse_bool, default=True)
    parser.add_argument("--embedding-truncation-strategy", default="head")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-concurrency", type=int, default=8, help="Embedding API concurrency")
    parser.add_argument("--embedding-tokenizer-required", type=parse_bool, default=True)
    parser.add_argument("--chunked-embedding-enabled", type=parse_bool, default=True)
    parser.add_argument("--chunked-embedding-chunk-tokens", type=int, default=7600)
    parser.add_argument("--chunked-embedding-overlap-tokens", type=int, default=512)
    parser.add_argument("--chunked-embedding-pooling", choices=["mean"], default="mean")
    parser.add_argument("--chunked-embedding-add-special-tokens", type=parse_bool, default=False)
    parser.add_argument("--chunked-embedding-normalize-after-pooling", type=parse_bool, default=False)
    parser.add_argument("--chunked-embedding-fail-if-chunk-exceeds-model-limit", type=parse_bool, default=True)
    parser.add_argument("--projection-method", default="local_pca")
    parser.add_argument("--projection-variance-ratio", type=float, default=0.90)
    parser.add_argument("--projection-max-dim", type=int, default=32)
    parser.add_argument("--projection-min-dim", type=int, default=2)
    parser.add_argument("--projection-whiten", type=parse_bool, default=False)
    parser.add_argument("--gmm-covariance-type", choices=["spherical", "diag", "tied"], default="spherical")
    parser.add_argument("--gmm-num-restarts", type=int, default=5)
    parser.add_argument("--gmm-kmeans-init-iters", type=int, default=15)
    parser.add_argument("--gmm-max-iter", type=int, default=100)
    parser.add_argument("--gmm-tol", type=float, default=1.0e-4)
    parser.add_argument("--gmm-min-covar", type=float, default=1.0e-6)
    parser.add_argument("--gmm-min-split-size", type=int, default=4)
    parser.add_argument("--gmm-min-effective-samples-per-component", type=int, default=2)
    parser.add_argument("--gmm-abs-kmax", type=int, default=64)
    parser.add_argument("--gmm-max-concurrent-candidates", type=int, default=1)
    parser.add_argument("--gmm-max-concurrent-restarts", type=int, default=1)
    parser.add_argument("--soft-save-soft-edges", type=parse_bool, default=True)
    parser.add_argument("--soft-top-r-memberships", type=int, default=2)
    parser.add_argument("--soft-recursive-assignment", choices=["primary_argmax", "top_r_threshold", "cumulative_mass"], default="cumulative_mass")
    parser.add_argument("--soft-min-membership-weight", type=float, default=0.05)
    parser.add_argument("--soft-max-membership-gap", type=float, default=0.25)
    parser.add_argument("--soft-cumulative-mass-coverage", type=float, default=0.90)
    parser.add_argument("--budget-refinement-enabled", type=parse_bool, default=True)
    parser.add_argument("--budget-refinement-apply-to-level", type=int, default=0)
    parser.add_argument("--budget-refinement-selection-policy", default="bic_best_with_token_progress")
    parser.add_argument("--budget-refinement-min-token-reduction-fraction", type=float, default=0.10)
    parser.add_argument("--budget-refinement-fallback", default="gmm_bic_recursive")
    parser.add_argument("--budget-refinement-flatten-leaves-to-l0", type=parse_bool, default=True)
    parser.add_argument("--budget-refinement-skip-oversize-singleton", type=parse_bool, default=True)
    parser.add_argument("--summary-max-model-tokens", type=int, default=100000)
    parser.add_argument("--summary-budget-ratio", type=float, default=0.85)
    parser.add_argument("--summary-prompt-overhead-reserve-tokens", type=int, default=8000)
    parser.add_argument("--summary-token-count-metadata-keys", type=parse_str_csv, default=["analysis_token_count", "prompt_token_count", "token_count", "tokens"])
    parser.add_argument("--dynamic-update-mode", default="budget_constrained_online_gmm")
    parser.add_argument("--dynamic-assignment", choices=["primary_argmax", "top_r_threshold", "cumulative_mass"], default="cumulative_mass")
    parser.add_argument("--dynamic-top-r", type=int, default=2)
    parser.add_argument("--dynamic-min-membership-weight", type=float, default=0.05)
    parser.add_argument("--dynamic-max-membership-gap", type=float, default=0.25)
    parser.add_argument("--dynamic-cumulative-mass-coverage", type=float, default=0.90)
    parser.add_argument("--dynamic-update-routing-model", type=parse_bool, default=True)
    parser.add_argument("--dynamic-clear-stale-after-propagation", type=parse_bool, default=True)
    parser.add_argument("--dynamic-confidence-metadata-key", default="confidence")
    parser.add_argument("--dynamic-max-propagation-rounds", type=int, default=16)
    parser.add_argument("--analyst-prompt-style", default="trace2skill_cluster_level_template_inheritance_v4")
    parser.add_argument("--analyst-confidence-floor", type=float, default=0.05)
    parser.add_argument("--analyst-tokenizer-required", type=parse_bool, default=True)
    parser.add_argument("--analyst-allow-regex-tokenizer-fallback", type=parse_bool, default=False)
    parser.add_argument("--analyst-max-prompt-tokens", type=int, default=-1, help="-1 derives this from summary_max_model_tokens * budget_ratio")
    parser.add_argument("--analyst-max-output-tokens", type=int, default=4096, help="-1 disables explicit output cap for static cluster analyst JSON generation")
    parser.add_argument("--analyst-dynamic-max-output-tokens", type=int, default=8192, help="-1 disables explicit output cap for dynamic patch analyst JSON generation")
    parser.add_argument("--analyst-multi-card-max-level", type=int, default=0)
    parser.add_argument("--analyst-max-cards-l0", type=int, default=0, help="0 means unlimited L0 cards")
    parser.add_argument("--analyst-max-cards-higher", type=int, default=1)
    parser.add_argument("--analyst-higher-level-mode", default="single_abstraction")
    parser.add_argument("--analyst-truncate-higher-level-extra-cards", type=parse_bool, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if args.tree_scenario == "dynamic_update":
        if args.tree_policy != "projected_gmm_bic":
            parser.error("--tree-policy ablations are static_build-only; dynamic_update requires projected_gmm_bic")
        if int(args.skill_export_min_level) >= 0 or int(args.skill_export_max_level) >= 0 or int(args.skill_export_max_node_count) >= 0:
            parser.error("--skill-export-* level/count filters are static_build retrieval ablations only")

    adapter = get_benchmark_adapter(args.benchmark)
    try:
        apply_officeqa_default_ranges(args, adapter, Path(args.data_path), argv)
    except ValueError as exc:
        parser.error(str(exc))
    train_slice = BenchmarkSlice(
        split=args.officeqa_train_split if args.benchmark == "officeqa" else "dataset",
        start=int(args.train_start),
        end=int(args.train_end),
    )
    heldout_slice = BenchmarkSlice(
        split=args.officeqa_heldout_split if args.benchmark == "officeqa" else "dataset",
        start=int(args.heldout_start),
        end=int(args.heldout_end),
    )

    if args.dynamic_update_mode != "budget_constrained_online_gmm":
        parser.error("--dynamic-update-mode is currently fixed to budget_constrained_online_gmm; this is not a tunable protocol knob")
    if not bool(args.dynamic_update_routing_model):
        parser.error("--dynamic-update-routing-model is fixed to true for budget_constrained_online_gmm")
    if not bool(args.budget_refinement_skip_oversize_singleton):
        parser.error("--budget-refinement-skip-oversize-singleton is currently fixed to true; false is not implemented")
    if args.rollout_llm_client != "openai":
        parser.error("--rollout-llm-client is fixed to openai for this handoff protocol")
    if int(args.rollout_num_random_seeds) != 1:
        parser.error("--rollout-num-random-seeds is fixed to 1 for this handoff protocol")
    if str(args.rollout_seeds).strip():
        parser.error("--rollout-seeds must be empty for this handoff protocol")
    if str(args.rollout_instance_ids).strip():
        parser.error("--rollout-instance-ids must be empty for this handoff protocol")
    if bool(args.rollout_missing_only):
        parser.error("--rollout-missing-only is fixed to false for this handoff protocol")
    if int(args.rollout_repeat) != 1:
        parser.error("--rollout-repeat is fixed to 1 for this handoff protocol")
    if str(args.rollout_shuffle_seed).strip():
        parser.error("--rollout-shuffle-seed must be empty for this handoff protocol")
    if int(args.rollout_sample) != 0:
        parser.error("--rollout-sample is fixed to 0/full split for this handoff protocol")

    thinking = None if args.thinking == "null" else args.thinking == "true"
    python_executable = resolve_python_executable(args.python_executable)
    repo = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve()
    records_path_arg = resolved_optional_path(args.records_path)
    reuse_train_run_dir = resolved_optional_path(args.reuse_train_run_dir)
    reuse_tree_dir = resolved_optional_path(args.reuse_tree_dir)
    train_artifact_dir = resolved_optional_path(args.train_artifact_dir) or run_dir
    scenario_dir = resolved_optional_path(args.scenario_output_dir) or run_dir
    if records_path_arg is not None and reuse_train_run_dir is not None:
        parser.error("--records-path and --reuse-train-run-dir are mutually exclusive")
    if reuse_train_run_dir is not None:
        train_artifact_dir = reuse_train_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    train_artifact_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    train_stage_logs = train_artifact_dir / "logs"
    train_markers = train_artifact_dir / "stage_markers"
    logs = scenario_dir / "logs"
    markers = scenario_dir / "stage_markers"
    train_stage_logs.mkdir(parents=True, exist_ok=True)
    train_markers.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    markers.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + str(repo)
    env["PATH"] = str(Path(python_executable).parent) + os.pathsep + env.get("PATH", "")
    env["DYNAMIX_PYTHON"] = python_executable
    env["OPENAI_API_KEY"] = args.openai_api_key
    env["OPENAI_BASE_URL"] = args.openai_base_url
    append_no_proxy_hosts(env, [args.openai_base_url, args.embedding_base_url])

    officeqa_docs_dirs_resolved: list[Path] = []
    officeqa_reward_path_resolved: Path | None = None
    if args.benchmark == "officeqa":
        if args.officeqa_evaluator == "fallback" and not args.officeqa_allow_fallback_evaluator:
            parser.error("--officeqa-evaluator fallback requires --officeqa-allow-fallback-evaluator")
        docs_values = list(args.officeqa_docs_dir)
        if not docs_values:
            docs_values = [part for part in env.get("OFFICEQA_DOCS_DIR", "").split(os.pathsep) if part]
        if not docs_values:
            parser.error("--officeqa-docs-dir or OFFICEQA_DOCS_DIR is required for --benchmark officeqa")
        for value in docs_values:
            docs_dir = Path(value).expanduser().resolve()
            if not docs_dir.is_dir():
                parser.error(f"OfficeQA docs directory does not exist: {docs_dir}")
            officeqa_docs_dirs_resolved.append(docs_dir)
        officeqa_reward_path_resolved = (
            resolve_reward_path(args.data_path, args.officeqa_reward_path or None, required=args.officeqa_evaluator == "official_reward")
            if args.officeqa_evaluator == "official_reward" or args.officeqa_reward_path
            else None
        )
        args._officeqa_docs_dirs_resolved = [str(path) for path in officeqa_docs_dirs_resolved]
        args._officeqa_reward_path_resolved = str(officeqa_reward_path_resolved) if officeqa_reward_path_resolved else ""

    train_gen_config_path = train_artifact_dir / "trace2skill_generation_config.json"
    write_generation_config(train_gen_config_path, thinking=thinking, temperature=args.rollout_temperature)
    scenario_gen_config_path = scenario_dir / "trace2skill_generation_config.json"
    write_generation_config(scenario_gen_config_path, thinking=thinking, temperature=args.rollout_temperature)
    split_manifest = adapter.write_split_manifest(
        data_path=Path(args.data_path),
        run_dir=scenario_dir,
        train_slice=train_slice,
        heldout_slice=heldout_slice,
    )
    dataset_fp = path_fingerprint(dataset_json_path(args.data_path) if args.benchmark == "spreadsheetbench" else Path(args.data_path))
    source_fp = stage_source_fingerprints(repo)
    officeqa_inputs_fp = {
        "docs_dirs": [
            {"path": str(path), "fingerprint": path_fingerprint(path)}
            for path in officeqa_docs_dirs_resolved
        ],
        "reward": (
            {"path": str(officeqa_reward_path_resolved), "fingerprint": path_fingerprint(officeqa_reward_path_resolved)}
            if officeqa_reward_path_resolved is not None
            else {"exists": False, "allow_fallback_evaluator": bool(args.officeqa_allow_fallback_evaluator)}
        ),
    } if args.benchmark == "officeqa" else {}

    runtime = {
        "benchmark": args.benchmark,
        "data_path": str(Path(args.data_path).resolve()),
        "officeqa_docs_dir": [str(path) for path in officeqa_docs_dirs_resolved],
        "officeqa_train_split": args.officeqa_train_split,
        "officeqa_heldout_split": args.officeqa_heldout_split,
        "officeqa_evaluator": args.officeqa_evaluator,
        "officeqa_max_completion_tokens": int(args.officeqa_max_completion_tokens),
        "officeqa_reward_path": str(officeqa_reward_path_resolved) if officeqa_reward_path_resolved else "",
        "officeqa_reward_tolerance": float(args.officeqa_reward_tolerance),
        "officeqa_allow_fallback_evaluator": bool(args.officeqa_allow_fallback_evaluator),
        "officeqa_continue_on_infra_error": bool(args.officeqa_continue_on_infra_error),
        "officeqa_use_oracle_context": bool(args.officeqa_use_oracle_context),
        "run_dir": str(run_dir),
        "train_artifact_dir": str(train_artifact_dir),
        "scenario_output_dir": str(scenario_dir),
        "model": args.model,
        "openai_base_url": args.openai_base_url,
        "embedding_base_url": args.embedding_base_url,
        "embedding_model": args.embedding_model,
        "embedding_tokenizer": args.embedding_tokenizer,
        "train_range": [args.train_start, args.train_end],
        "heldout_range": [args.heldout_start, args.heldout_end],
        "split_manifest": str(scenario_dir / "split_manifest.json"),
        "workers": args.workers,
        "python_executable": python_executable,
        "max_turns": args.max_turns,
        "thinking": args.thinking,
        "trace2skill_generation_config": str(scenario_gen_config_path),
        "skillbank_top_k": int(args.skillbank_top_k),
        "tree_scenario": args.tree_scenario,
        "reuse_tree_dir": str(reuse_tree_dir) if reuse_tree_dir else "",
        "dynamic_initial_count": int(args.dynamic_initial_count),
        "dynamic_arrival_count": int(args.dynamic_arrival_count),
        "dynamic_update_batch_size": int(args.dynamic_update_batch_size),
        "dynamic_shuffle_seed": None if int(args.dynamic_shuffle_seed) < 0 else int(args.dynamic_shuffle_seed),
        "dynamic_snapshot_include_embeddings": bool(args.dynamic_snapshot_include_embeddings),
        "dynamic_resume_from_snapshots": bool(args.dynamic_resume_from_snapshots),
        "max_levels": int(args.max_levels),
        "skill_output_dir_name": args.skill_output_dir_name,
        "skill_export": skill_export_payload_from_args(args),
        "rollout_temperature": float(args.rollout_temperature),
        "rollout_client_timeout_seconds": float(args.rollout_client_timeout_seconds),
        "rollout_client_retry_wait_seconds": list(args.rollout_client_retry_wait_seconds),
        "rollout_llm_client": args.rollout_llm_client,
        "rollout_num_random_seeds": int(args.rollout_num_random_seeds),
        "rollout_seeds": str(args.rollout_seeds),
        "rollout_instance_ids": str(args.rollout_instance_ids),
        "rollout_missing_only": bool(args.rollout_missing_only),
        "rollout_repeat": int(args.rollout_repeat),
        "rollout_shuffle_seed": str(args.rollout_shuffle_seed),
        "rollout_sample": int(args.rollout_sample),
        "resume": bool(args.resume),
    }
    (scenario_dir / "experiment_runtime_config.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    train_usage_dir = train_artifact_dir / "usage"
    usage_dir = scenario_dir / "usage"
    train_usage_dir.mkdir(parents=True, exist_ok=True)
    usage_dir.mkdir(parents=True, exist_ok=True)
    usage_logs_by_stage = {
        "01_train_collect": [train_usage_dir / "01_train_collect.react_usage.jsonl"],
        "02_train_eval": [],
        "03_extract_records": [],
        "04_build_tree": [
            usage_dir / "04_build_tree.generation_usage.jsonl",
            usage_dir / "04_build_tree.embedding_usage.jsonl",
            usage_dir / "04_build_tree.skillbank_usage.jsonl",
        ],
        "06_heldout_collect": [
            usage_dir / "06_heldout_collect.react_usage.jsonl",
            usage_dir / "06_heldout_collect.skillbank_usage.jsonl",
        ],
        "07_heldout_eval": [],
    }

    source_records = records_path_arg or (reuse_train_run_dir / "records.json" if reuse_train_run_dir is not None else train_artifact_dir / "records.json")
    skip_train_stages = records_path_arg is not None or reuse_train_run_dir is not None
    if skip_train_stages and not source_records.is_file():
        raise FileNotFoundError(f"reused records.json not found: {source_records}")
    ordered_records = scenario_dir / "ordered_records.json"
    records_order_manifest = scenario_dir / "records_order_manifest.json"
    records = ordered_records
    runtime["source_records_path"] = str(source_records)
    runtime["records_path"] = str(records)
    runtime["records_order_manifest"] = str(records_order_manifest)
    runtime["skip_train_stages"] = bool(skip_train_stages)
    runtime["records_path_arg"] = str(records_path_arg) if records_path_arg is not None else ""
    runtime["reuse_train_run_dir"] = str(reuse_train_run_dir) if reuse_train_run_dir is not None else ""
    (scenario_dir / "experiment_runtime_config.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    train_out = train_artifact_dir / "trace2skill_train_outputs"
    train_logs = train_artifact_dir / "trace2skill_train_logs"
    train_results = train_artifact_dir / "trace2skill_train_results.json"
    train_collect_cmd = adapter.run_rollout(RolloutCommandSpec(
        python_executable=python_executable,
        data_path=Path(args.data_path),
        data_slice=train_slice,
        output_dir=train_out,
        results_file=train_results,
        log_dir=train_logs,
        generation_config=train_gen_config_path,
        model=args.model,
        openai_base_url=args.openai_base_url,
        rollout_llm_client=args.rollout_llm_client,
        rollout_temperature=float(args.rollout_temperature),
        llm_timeout_seconds=float(args.rollout_client_timeout_seconds),
        llm_retry_wait_seconds=list(args.rollout_client_retry_wait_seconds),
        rollout_num_random_seeds=int(args.rollout_num_random_seeds),
        rollout_repeat=int(args.rollout_repeat),
        max_turns=int(args.max_turns),
        workers=int(args.workers),
        officeqa_docs_dirs=officeqa_docs_dirs_resolved,
        officeqa_evaluator=args.officeqa_evaluator,
        officeqa_max_completion_tokens=int(args.officeqa_max_completion_tokens),
        officeqa_reward_path=officeqa_reward_path_resolved,
        officeqa_reward_tolerance=float(args.officeqa_reward_tolerance),
        officeqa_allow_fallback_evaluator=bool(args.officeqa_allow_fallback_evaluator),
        officeqa_use_oracle_context=bool(args.officeqa_use_oracle_context),
        officeqa_continue_on_infra_error=bool(args.officeqa_continue_on_infra_error),
    ))
    if not skip_train_stages:
        run_stage(
            "01_train_collect",
            train_collect_cmd,
            cwd=repo,
            env={**env, "REACT_AGENT_USAGE_LOG": str(usage_logs_by_stage["01_train_collect"][0])},
            log_path=train_stage_logs / "01_train_collect.log",
            marker_dir=train_markers,
            outputs=[train_results],
            resume=args.resume,
            clear_outputs_before_run=list(usage_logs_by_stage["01_train_collect"]),
            fingerprint=stage_fingerprint(
                "01_train_collect:v2",
                train_collect_cmd,
                dataset=dataset_fp,
                officeqa_inputs=officeqa_inputs_fp,
                rollout_protocol=rollout_protocol(args, generation_config=train_gen_config_path),
                source=benchmark_source_fingerprints(source_fp, benchmark=args.benchmark, stage="rollout"),
                split=[train_slice.split, args.train_start, args.train_end],
            ),
        )

    train_eval = train_artifact_dir / "trace2skill_train_eval.json"
    train_eval_cmd = adapter.evaluate_results(EvalCommandSpec(
        python_executable=python_executable,
        data_path=Path(args.data_path),
        data_slice=train_slice,
        output_dir=train_out,
        results_file=train_results,
        eval_file=train_eval,
        officeqa_evaluator=args.officeqa_evaluator,
        officeqa_reward_path=officeqa_reward_path_resolved,
        officeqa_reward_tolerance=float(args.officeqa_reward_tolerance),
        officeqa_allow_fallback_evaluator=bool(args.officeqa_allow_fallback_evaluator),
    ))
    if not skip_train_stages:
        run_stage(
            "02_train_eval",
            train_eval_cmd,
            cwd=repo,
            env=env,
            log_path=train_stage_logs / "02_train_eval.log",
            marker_dir=train_markers,
            outputs=[train_eval],
            resume=args.resume,
            fingerprint=stage_fingerprint(
                "02_train_eval:v2",
                train_eval_cmd,
                dataset=dataset_fp,
                officeqa_inputs=officeqa_inputs_fp,
                train_outputs=path_fingerprint(train_out),
                train_results=path_fingerprint(train_results),
                source=benchmark_source_fingerprints(source_fp, benchmark=args.benchmark, stage="eval"),
                split=[train_slice.split, args.train_start, args.train_end],
            ),
        )

    extract_records_cmd = adapter.extract_records(ExtractCommandSpec(
        python_executable=python_executable,
        log_dir=train_logs,
        eval_file=train_eval,
        records_file=source_records,
    ))
    if not skip_train_stages:
        run_stage(
            "03_extract_records",
            extract_records_cmd,
            cwd=repo,
            env=env,
            log_path=train_stage_logs / "03_extract_records.log",
            marker_dir=train_markers,
            outputs=[source_records],
            resume=args.resume,
            fingerprint=stage_fingerprint(
                "03_extract_records:v2",
                extract_records_cmd,
                train_logs=path_fingerprint(train_logs),
                train_eval=path_fingerprint(train_eval),
                source=benchmark_source_fingerprints(source_fp, benchmark=args.benchmark, stage="extract"),
            ),
        )

    expected_train_records = len(adapter.load_rows(Path(args.data_path), train_slice))
    observed_train_records = load_record_count(source_records)
    if observed_train_records != expected_train_records:
        raise RuntimeError(
            f"records.json count mismatch: expected {expected_train_records} "
            f"from train range [{args.train_start}, {args.train_end}), got {observed_train_records}"
        )
    order_manifest = adapter.write_ordered_records(
        source_records=source_records,
        output_path=ordered_records,
        manifest_path=records_order_manifest,
        data_path=Path(args.data_path),
        train_slice=train_slice,
    )
    args._observed_train_records = observed_train_records
    if load_record_count(records) != expected_train_records:
        raise RuntimeError(f"ordered_records.json count mismatch after dataset-order rewrite: {records}")
    runtime["records_order_policy"] = order_manifest["policy"]
    runtime["records_source_order_equal_dataset_order"] = bool(order_manifest["source_order_equal_dataset_order"])
    (scenario_dir / "experiment_runtime_config.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    tree_dir = scenario_dir / "dynamix_tree"
    config = {
        "scenario": args.tree_scenario,
        "benchmark": args.benchmark,
        "output_dir": str(tree_dir),
        "records_path": str(records),
        **tree_dataset_order_payload(args),
        "generation": {
            "base_url": args.openai_base_url,
            "model": args.model,
            "api_key": "EMPTY",
            "api_key_env_var": "OPENAI_API_KEY",
            "temperature": float(args.generation_temperature),
            "timeout_seconds": float(args.generation_timeout_seconds),
            "max_concurrency": int(args.generation_max_concurrency or args.workers),
            "thinking_mode": thinking,
            "extra_body": ({"chat_template_kwargs": {"enable_thinking": bool(thinking)}} if thinking is not None else {}),
            "debug_dir": str(tree_dir / "analysis" / "generation_debug"),
            "retry_wait_seconds": list(args.generation_retry_wait_seconds),
        },
        "embedding": {
            "base_url": args.embedding_base_url,
            "model": args.embedding_model,
            "api_key": "EMPTY",
            "max_model_len": int(args.embedding_max_model_len),
            "max_input_tokens": int(args.embedding_max_input_tokens),
            "truncate_long_texts": bool(args.embedding_truncate_long_texts),
            "tokenizer_model": args.embedding_tokenizer,
            "tokenizer_required": bool(args.embedding_tokenizer_required),
            "truncation_strategy": args.embedding_truncation_strategy,
            "batch_size": int(args.embedding_batch_size),
            "max_concurrency": int(args.embedding_max_concurrency or args.workers),
            "cache_path": str(scenario_dir / "cache" / "embedding_cache.sqlite"),
        },
        "chunked_embedding": {
            "enabled": bool(args.chunked_embedding_enabled),
            "chunk_tokens": int(args.chunked_embedding_chunk_tokens),
            "overlap_tokens": int(args.chunked_embedding_overlap_tokens),
            "pooling": args.chunked_embedding_pooling,
            "add_special_tokens": bool(args.chunked_embedding_add_special_tokens),
            "normalize_after_pooling": bool(args.chunked_embedding_normalize_after_pooling),
            "fail_if_chunk_exceeds_model_limit": bool(args.chunked_embedding_fail_if_chunk_exceeds_model_limit),
        },
        "hierarchy": {
            "tree_policy": args.tree_policy,
            "graph_kind": args.graph_kind,
            "allow_overlap": bool(args.allow_overlap),
            "allow_multi_parent": bool(args.allow_multi_parent),
            "use_support_mass": bool(args.use_support_mass),
            "random_seed": int(args.random_seed),
            "projection": {
                "method": args.projection_method,
                "variance_ratio": float(args.projection_variance_ratio),
                "max_dim": int(args.projection_max_dim),
                "min_dim": int(args.projection_min_dim),
                "whiten": bool(args.projection_whiten),
            },
            "gmm_bic": {
                "covariance_type": args.gmm_covariance_type,
                "num_restarts": int(args.gmm_num_restarts),
                "kmeans_init_iters": int(args.gmm_kmeans_init_iters),
                "max_iter": int(args.gmm_max_iter),
                "tol": float(args.gmm_tol),
                "min_covar": float(args.gmm_min_covar),
                "min_split_size": int(args.gmm_min_split_size),
                "min_effective_samples_per_component": int(args.gmm_min_effective_samples_per_component),
                "abs_kmax": int(args.gmm_abs_kmax),
                "max_concurrent_candidates": int(args.gmm_max_concurrent_candidates),
                "max_concurrent_restarts": int(args.gmm_max_concurrent_restarts),
            },
            "soft_membership": {
                "save_soft_edges": bool(args.soft_save_soft_edges),
                "top_r_memberships": int(args.soft_top_r_memberships),
                "recursive_assignment": args.soft_recursive_assignment,
                "min_membership_weight": float(args.soft_min_membership_weight),
                "max_membership_gap": float(args.soft_max_membership_gap),
                "cumulative_mass_coverage": float(args.soft_cumulative_mass_coverage),
            },
            "budget_refinement": {
                "enabled": bool(args.budget_refinement_enabled),
                "apply_to_level": int(args.budget_refinement_apply_to_level),
                "selection_policy": args.budget_refinement_selection_policy,
                "min_token_reduction_fraction": float(args.budget_refinement_min_token_reduction_fraction),
                "fallback": args.budget_refinement_fallback,
                "flatten_refinement_leaves_to_l0": bool(args.budget_refinement_flatten_leaves_to_l0),
                "skip_oversize_singleton": bool(args.budget_refinement_skip_oversize_singleton),
            },
            "summary_budget": {
                "max_model_tokens": int(args.summary_max_model_tokens),
                "budget_ratio": float(args.summary_budget_ratio),
                "prompt_overhead_reserve_tokens": int(args.summary_prompt_overhead_reserve_tokens),
                "token_count_metadata_keys": list(args.summary_token_count_metadata_keys),
            },
            "dynamic_update": {
                "mode": args.dynamic_update_mode,
                "assignment": args.dynamic_assignment,
                "top_r": int(args.dynamic_top_r),
                "min_membership_weight": float(args.dynamic_min_membership_weight),
                "max_membership_gap": float(args.dynamic_max_membership_gap),
                "cumulative_mass_coverage": float(args.dynamic_cumulative_mass_coverage),
                "update_routing_model": bool(args.dynamic_update_routing_model),
                "clear_stale_after_propagation": bool(args.dynamic_clear_stale_after_propagation),
                "confidence_metadata_key": args.dynamic_confidence_metadata_key,
            },
        },
        "dynamic": {
            "initial_count": int(args.dynamic_initial_count),
            "arrival_count": int(args.dynamic_arrival_count),
            "update_batch_size": int(args.dynamic_update_batch_size),
            "shuffle_seed": None if int(args.dynamic_shuffle_seed) < 0 else int(args.dynamic_shuffle_seed),
            "snapshot_include_embeddings": bool(args.dynamic_snapshot_include_embeddings),
            "resume_from_snapshots": bool(args.dynamic_resume_from_snapshots),
            "max_propagation_rounds": int(args.dynamic_max_propagation_rounds),
        },
        "analyst": {
            "prompt_style": args.analyst_prompt_style,
            "confidence_floor": float(args.analyst_confidence_floor),
            "tokenizer_model": args.embedding_tokenizer,
            "tokenizer_required": bool(args.analyst_tokenizer_required),
            "allow_regex_tokenizer_fallback": bool(args.analyst_allow_regex_tokenizer_fallback),
            "max_prompt_tokens": None if int(args.analyst_max_prompt_tokens) <= 0 else int(args.analyst_max_prompt_tokens),
            "max_output_tokens": None if int(args.analyst_max_output_tokens) <= 0 else int(args.analyst_max_output_tokens),
            "dynamic_max_output_tokens": None if int(args.analyst_dynamic_max_output_tokens) <= 0 else int(args.analyst_dynamic_max_output_tokens),
            "multi_card_max_level": int(args.analyst_multi_card_max_level),
            "max_cards_l0": None if int(args.analyst_max_cards_l0) <= 0 else int(args.analyst_max_cards_l0),
            "max_cards_higher": int(args.analyst_max_cards_higher),
            "higher_level_mode": args.analyst_higher_level_mode,
            "truncate_higher_level_extra_cards": bool(args.analyst_truncate_higher_level_extra_cards),
        },
        "max_levels": int(args.max_levels),
        "skill_output_dir_name": args.skill_output_dir_name,
        "skill_export": skill_export_payload_from_args(args),
    }
    config_path = scenario_dir / "dynamix_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    build_tree_cmd = [python_executable, "scripts/build_dynamix_tree.py", "--config", str(config_path)]
    build_tree_fingerprint = {
        "stage_contract": "04_build_tree:v2",
        "cmd": build_tree_cmd,
        "config_sha256": file_sha256(config_path),
        "records_sha256": file_sha256(records),
        "openai_api_key": api_key_fingerprint(args.openai_api_key),
        "tree_scenario": args.tree_scenario,
        "source": {
            "runner": source_fp["runner"],
            "build_dynamix_tree": source_fp["build_dynamix_tree"],
            "dynamix_core": source_fp["dynamix_core"],
            "dynamix_trace2skill": source_fp["dynamix_trace2skill"],
        },
    }
    build_tree_usage_env = {
        **env,
        "DYNAMIX_GENERATION_USAGE_LOG": str(usage_logs_by_stage["04_build_tree"][0]),
        "DYNAMIX_EMBEDDING_USAGE_LOG": str(usage_logs_by_stage["04_build_tree"][1]),
        "DYNAMIX_SKILLBANK_USAGE_LOG": str(usage_logs_by_stage["04_build_tree"][2]),
    }
    if reuse_tree_dir is not None:
        reuse_fingerprint = {
            **build_tree_fingerprint,
            "stage_contract": "04_build_tree:reuse_tree_filtered_nodebank:v1",
            "reuse_tree_dir": path_fingerprint(reuse_tree_dir),
            "skill_export": skill_export_payload_from_args(args),
        }
        if args.resume and stage_done(markers / "04_build_tree.done", [tree_dir / "summary.json"], fingerprint=reuse_fingerprint):
            print("[resume] skip stage 04_build_tree", flush=True)
            summary = json.loads((tree_dir / "summary.json").read_text(encoding="utf-8"))
        else:
            summary = materialize_reused_tree_nodebank(
                reuse_tree_dir=reuse_tree_dir,
                tree_dir=tree_dir,
                args=args,
                current_config=config,
                fingerprint=reuse_fingerprint,
                marker_dir=markers,
                log_path=logs / "04_build_tree.log",
                usage_logs=usage_logs_by_stage["04_build_tree"],
            )
    else:
        run_stage(
            "04_build_tree",
            build_tree_cmd,
            cwd=repo,
            env=build_tree_usage_env,
            log_path=logs / "04_build_tree.log",
            marker_dir=markers,
            outputs=[tree_dir / "summary.json"],
            resume=args.resume,
            clear_outputs_before_run=list(usage_logs_by_stage["04_build_tree"]),
            fingerprint=build_tree_fingerprint,
        )
        summary = json.loads((tree_dir / "summary.json").read_text(encoding="utf-8"))
    validate_tree_summary_for_heldout(summary, args)
    manifest = json.loads(Path(summary["node_bank_manifest"]).read_text(encoding="utf-8"))
    if int(manifest.get("node_count", 0)) <= 0:
        raise RuntimeError("DynaMix produced no retrievable nodebank nodes")
    skillbank_root = Path(manifest.get("output_dir") or Path(summary["node_bank_manifest"]).parent)
    skills_root = skillbank_root
    # Enable per-task top-k nodebank selection during heldout.  The agent injects
    # the selected node snippets directly into the usual preloaded-skill slot.
    env["DYNAMIX_SKILLBANK_ROOT"] = str(skillbank_root)
    env["DYNAMIX_SKILLBANK_TOP_K"] = str(max(0, int(args.skillbank_top_k)))
    env["DYNAMIX_SKILLBANK_EMBED_BASE_URL"] = args.embedding_base_url
    env["DYNAMIX_SKILLBANK_EMBED_MODEL"] = args.embedding_model
    env["DYNAMIX_SKILLBANK_EMBED_API_KEY"] = "EMPTY"
    skillbank_cache_path = Path(summary.get("skillbank_index") or (skillbank_root / ".dynamix_skillbank_index.json"))
    if not skillbank_cache_path.is_file():
        raise RuntimeError(f"DynaMix skillbank index missing before heldout: {skillbank_cache_path}")
    env["DYNAMIX_SKILLBANK_CACHE_PATH"] = str(skillbank_cache_path)
    selection_log = scenario_dir / "raw" / "skill_selection_records.jsonl"
    selection_log.parent.mkdir(parents=True, exist_ok=True)
    env["DYNAMIX_SKILL_SELECTION_LOG"] = str(selection_log)

    heldout_out = scenario_dir / "trace2skill_heldout_outputs"
    heldout_logs = scenario_dir / "trace2skill_heldout_logs"
    heldout_results = scenario_dir / "trace2skill_heldout_results.json"
    heldout_collect_cmd = adapter.run_rollout(RolloutCommandSpec(
        python_executable=python_executable,
        data_path=Path(args.data_path),
        data_slice=heldout_slice,
        output_dir=heldout_out,
        results_file=heldout_results,
        log_dir=heldout_logs,
        generation_config=scenario_gen_config_path,
        model=args.model,
        openai_base_url=args.openai_base_url,
        rollout_llm_client=args.rollout_llm_client,
        rollout_temperature=float(args.rollout_temperature),
        llm_timeout_seconds=float(args.rollout_client_timeout_seconds),
        llm_retry_wait_seconds=list(args.rollout_client_retry_wait_seconds),
        rollout_num_random_seeds=int(args.rollout_num_random_seeds),
        rollout_repeat=int(args.rollout_repeat),
        max_turns=int(args.max_turns),
        workers=int(args.workers),
        skillbank_root=skills_root,
        skillbank_top_k=int(args.skillbank_top_k),
        selection_log=selection_log,
        officeqa_docs_dirs=officeqa_docs_dirs_resolved,
        officeqa_evaluator=args.officeqa_evaluator,
        officeqa_max_completion_tokens=int(args.officeqa_max_completion_tokens),
        officeqa_reward_path=officeqa_reward_path_resolved,
        officeqa_reward_tolerance=float(args.officeqa_reward_tolerance),
        officeqa_allow_fallback_evaluator=bool(args.officeqa_allow_fallback_evaluator),
        officeqa_use_oracle_context=bool(args.officeqa_use_oracle_context),
        officeqa_continue_on_infra_error=bool(args.officeqa_continue_on_infra_error),
    ))
    run_stage(
        "06_heldout_collect",
        heldout_collect_cmd,
        cwd=repo,
        env={
            **env,
            "REACT_AGENT_USAGE_LOG": str(usage_logs_by_stage["06_heldout_collect"][0]),
            "DYNAMIX_SKILLBANK_USAGE_LOG": str(usage_logs_by_stage["06_heldout_collect"][1]),
        },
        log_path=logs / "06_heldout_collect.log",
        marker_dir=markers,
        outputs=[heldout_results, selection_log],
        resume=args.resume,
        clear_outputs_before_run=[selection_log, *usage_logs_by_stage["06_heldout_collect"]],
        fingerprint=stage_fingerprint(
            "06_heldout_collect:v2",
            heldout_collect_cmd,
            dataset=dataset_fp,
            officeqa_inputs=officeqa_inputs_fp,
            generation_config=path_fingerprint(scenario_gen_config_path),
            tree_summary=path_fingerprint(tree_dir / "summary.json"),
            node_bank_manifest=path_fingerprint(Path(summary["node_bank_manifest"])),
            skillbank_root=path_fingerprint(skillbank_root),
            rollout_protocol=rollout_protocol(args, generation_config=scenario_gen_config_path),
            skillbank_retrieval_protocol=skillbank_retrieval_protocol(args, cache_path=skillbank_cache_path, selection_log=selection_log),
            source=benchmark_source_fingerprints(source_fp, benchmark=args.benchmark, stage="rollout"),
            split=[heldout_slice.split, args.heldout_start, args.heldout_end],
        ),
    )

    heldout_eval = scenario_dir / "trace2skill_heldout_eval.json"
    heldout_eval_cmd = adapter.evaluate_results(EvalCommandSpec(
        python_executable=python_executable,
        data_path=Path(args.data_path),
        data_slice=heldout_slice,
        output_dir=heldout_out,
        results_file=heldout_results,
        eval_file=heldout_eval,
        officeqa_evaluator=args.officeqa_evaluator,
        officeqa_reward_path=officeqa_reward_path_resolved,
        officeqa_reward_tolerance=float(args.officeqa_reward_tolerance),
        officeqa_allow_fallback_evaluator=bool(args.officeqa_allow_fallback_evaluator),
    ))
    run_stage(
        "07_heldout_eval",
        heldout_eval_cmd,
        cwd=repo,
        env=env,
        log_path=logs / "07_heldout_eval.log",
        marker_dir=markers,
        outputs=[heldout_eval],
        resume=args.resume,
        fingerprint=stage_fingerprint(
            "07_heldout_eval:v2",
            heldout_eval_cmd,
            dataset=dataset_fp,
            officeqa_inputs=officeqa_inputs_fp,
            heldout_outputs=path_fingerprint(heldout_out),
            heldout_results=path_fingerprint(heldout_results),
            source=benchmark_source_fingerprints(source_fp, benchmark=args.benchmark, stage="eval"),
            split=[heldout_slice.split, args.heldout_start, args.heldout_end],
        ),
    )

    final = {
        **runtime,
        "records_path": str(records),
        "tree_summary": str(tree_dir / "summary.json"),
        "skillbank_root": str(skillbank_root),
        "node_bank_manifest": str(summary["node_bank_manifest"]),
        "skills_root": str(skills_root),
        "skillbank_top_k": int(args.skillbank_top_k),
        "skillbank_index": str(skillbank_cache_path),
        "skill_selection_records": str(selection_log),
        "heldout_eval": str(heldout_eval),
    }
    stage_report = write_experiment_stage_report(
        run_dir=scenario_dir,
        marker_dir=markers,
        stages=(
            ["04_build_tree", "06_heldout_collect", "07_heldout_eval"]
            if skip_train_stages
            else ["01_train_collect", "02_train_eval", "03_extract_records", "04_build_tree", "06_heldout_collect", "07_heldout_eval"]
        ),
        usage_logs_by_stage=usage_logs_by_stage,
        runtime=runtime,
        args=args,
        marker_dirs_by_stage={
            "01_train_collect": train_markers,
            "02_train_eval": train_markers,
            "03_extract_records": train_markers,
            "04_build_tree": markers,
            "06_heldout_collect": markers,
            "07_heldout_eval": markers,
        },
    )
    final["experiment_stage_report"] = str(scenario_dir / "experiment_stage_report.json")
    final["experiment_stage_report_md"] = str(scenario_dir / "experiment_stage_report.md")
    final["runtime_dead_corner_findings"] = stage_report.get("runtime_dead_corner_findings", [])
    (scenario_dir / "experiment_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
