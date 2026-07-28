#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


CERTIFIED_SINGLE_PARENT_TREE_POLICIES = frozenset(
    {
        "certified_dual_view_otd",
        "evidence_balanced_skill_tree",
    }
)


def is_certified_single_parent_tree(tree_policy: str) -> bool:
    return str(tree_policy).strip() in CERTIFIED_SINGLE_PARENT_TREE_POLICIES


def atom_cache_path_for_args(args: argparse.Namespace) -> str | None:
    if args.tree_policy == "evidence_balanced_skill_tree":
        return args.ebst_atom_cache_path
    return args.otd_atom_cache_path


def run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path | None = None,
    append_log: bool = False,
) -> None:
    print("+", " ".join(cmd), flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open(
            "a" if append_log else "w",
            encoding="utf-8",
        ) as log:
            if append_log:
                log.write("\n[resume] continuing interrupted stage\n")
            proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def stage_done(marker: Path, outputs: Iterable[Path], *, fingerprint: dict | None = None) -> bool:
    if not marker.exists():
        return False
    output_paths = list(outputs)
    stage_name = marker.name.removesuffix(".done")
    if (
        marker.with_name(f"{stage_name}.running").exists()
        or marker.with_name(f"{stage_name}.failed.json").exists()
    ):
        return False
    if not all(path.exists() for path in output_paths):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    if fingerprint is not None and payload.get("fingerprint") != fingerprint:
        return False
    expected_outputs = payload.get("output_identities")
    if not isinstance(expected_outputs, dict):
        return False
    actual_outputs = {
        str(path): path_fingerprint(path)
        for path in output_paths
    }
    return expected_outputs == actual_outputs


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
    preserve_partial_outputs_on_resume: bool = False,
) -> None:
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{name}.done"
    if resume and stage_done(marker, outputs, fingerprint=fingerprint):
        print(f"[resume] skip stage {name}", flush=True)
        return
    running = marker_dir / f"{name}.running"
    failed = marker_dir / f"{name}.failed.json"
    resume_partial = False
    if resume and preserve_partial_outputs_on_resume and running.is_file():
        try:
            running_payload = json.loads(
                running.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            running_payload = {}
        resume_partial = (
            running_payload.get("fingerprint") == fingerprint
        )
        if resume_partial:
            print(
                f"[resume] continue partial stage {name}",
                flush=True,
            )
    marker.unlink(missing_ok=True)
    running.unlink(missing_ok=True)
    failed.unlink(missing_ok=True)
    if not resume_partial:
        for path in clear_outputs_before_run or []:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    started_at = utc_now_iso()
    started_monotonic = time.monotonic()
    running_payload = {"stage": name, "cmd": cmd, "fingerprint": fingerprint, "started_at": started_at, "log": str(log_path)}
    write_json_atomic(running, running_payload)
    try:
        run(
            cmd,
            cwd=cwd,
            env=env,
            log_path=log_path,
            append_log=resume_partial,
        )
    except Exception as exc:
        write_json_atomic(failed, {
            "stage": name,
            "cmd": cmd,
            "error": repr(exc),
            "log": str(log_path),
            "started_at": started_at,
            "ended_at": utc_now_iso(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
        })
        raise
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        error = f"stage completed but required outputs are missing: {missing}"
        write_json_atomic(failed, {
            "stage": name,
            "cmd": cmd,
            "error": error,
            "log": str(log_path),
            "started_at": started_at,
            "ended_at": utc_now_iso(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
        })
        raise RuntimeError(error)
    ended_at = utc_now_iso()
    elapsed_seconds = time.monotonic() - started_monotonic
    write_json_atomic(marker, {
        "stage": name,
        "outputs": [str(p) for p in outputs],
        "output_identities": {
            str(path): path_fingerprint(path)
            for path in outputs
        },
        "fingerprint": fingerprint,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed_seconds,
        "log": str(log_path),
    })
    running.unlink(missing_ok=True)


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


def validate_cdost_vector_cache(
    *,
    cache_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    source_root = Path(__file__).resolve().parents[1] / "src"
    source_root_text = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)
    from dynamix_trace2skill.clients import (
        validate_embedding_cache_manifest,
    )

    return validate_embedding_cache_manifest(
        cache_path=cache_path,
        manifest_path=manifest_path,
    )


def resolve_embedding_cache_path(
    *,
    explicit_path: str | None,
    scenario_dir: Path,
    tree_policy: str,
    tree_scenario: str,
    atom_cache_path: str | None,
) -> Path:
    explicit = resolved_optional_path(explicit_path)
    if (
        not is_certified_single_parent_tree(tree_policy)
        or tree_scenario != "dynamic_update"
    ):
        return explicit or (scenario_dir / "cache" / "embedding_cache.sqlite")

    atom_cache = resolved_optional_path(atom_cache_path)
    if atom_cache is None:
        raise ValueError(
            f"controlled {tree_policy} dynamic runs require a "
            "frozen atom cache"
        )
    if not atom_cache.is_file():
        raise FileNotFoundError(f"frozen atom cache is missing: {atom_cache}")
    source_scenario_dir = atom_cache.parent.parent
    validate_source_build_output(
        marker_path=(
            source_scenario_dir
            / "stage_markers"
            / "04_build_tree.done"
        ),
        output_path=atom_cache,
    )
    source_config_path = source_scenario_dir / "dynamix_config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(
            "matching static dynamix_config.json is required beside the "
            f"frozen atom cache: {source_config_path}"
        )
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_output = Path(str(source_config.get("output_dir") or "")).resolve()
    if (
        source_config.get("scenario") != "static_build"
        or source_config.get("hierarchy", {}).get("tree_policy")
        != tree_policy
        or atom_cache != (source_output / "experience_atoms.json").resolve()
    ):
        raise ValueError(
            "frozen atom cache is not bound to a matching static tree run"
        )
    source_cache = resolved_optional_path(
        source_config.get("embedding", {}).get("cache_path")
    )
    if source_cache is None or not source_cache.is_file():
        raise FileNotFoundError(
            "matching static embedding vector cache is missing: "
            f"{source_cache}"
        )
    if explicit is not None and explicit != source_cache:
        raise ValueError(
            "controlled static/dynamic tree runs must share the same "
            "content-addressed embedding vector cache"
        )
    source_vector_manifest = (
        source_output / "embedding_vector_cache_manifest.json"
    )
    validate_source_build_output(
        marker_path=(
            source_scenario_dir
            / "stage_markers"
            / "04_build_tree.done"
        ),
        output_path=source_vector_manifest,
    )
    validate_cdost_vector_cache(
        cache_path=source_cache,
        manifest_path=source_vector_manifest,
    )
    return source_cache


def stage_source_fingerprints(repo: Path) -> dict[str, dict[str, str | bool | int]]:
    return {
        "runner": path_fingerprint(Path(__file__).resolve()),
        "run_spreadsheetbench": path_fingerprint(repo / "run_spreadsheetbench.py"),
        "evaluate_with_official": path_fingerprint(repo / "evaluate_with_official.py"),
        "extract_trace2skill_logs": path_fingerprint(repo / "scripts" / "extract_trace2skill_logs.py"),
        "build_dynamix_tree": path_fingerprint(repo / "scripts" / "build_dynamix_tree.py"),
        "audit_cdost_query_vector_cache": path_fingerprint(
            repo / "scripts" / "audit_cdost_query_vector_cache.py"
        ),
        "spreadsheetbench_support": path_fingerprint(repo / "spreadsheetbench_support.py"),
        "spreadsheet_agent": path_fingerprint(repo / "spreadsheet_agent", source_only=True),
        "react_agent": path_fingerprint(repo / "src" / "react_agent", source_only=True),
        "skillbank": path_fingerprint(
            repo / "src" / "dynamix_trace2skill" / "skillbank.py"
        ),
        "antichain_retrieval": path_fingerprint(
            repo / "src" / "dynamix_core" / "certified_otd.py"
        ),
        "dynamix_core": path_fingerprint(repo / "src" / "dynamix_core", source_only=True),
        "dynamix_trace2skill": path_fingerprint(repo / "src" / "dynamix_trace2skill", source_only=True),
    }


def rollout_protocol(args: argparse.Namespace, *, generation_config: Path) -> dict[str, object]:
    return {
        "model": args.model,
        "openai_base_url": args.openai_base_url,
        "openai_api_key": api_key_fingerprint(args.openai_api_key),
        "thinking": args.thinking,
        "max_turns": int(args.max_turns),
        "workers": int(args.workers),
        "timeout_seconds": float(args.rollout_client_timeout_seconds),
        "retry_wait_seconds": list(args.rollout_client_retry_wait_seconds),
        "sdk_max_retries": 0,
        "response_cache_enabled": not bool(
            getattr(args, "rollout_disable_response_cache", False)
        ),
        "llm_client": args.rollout_llm_client,
        "num_random_seeds": int(args.rollout_num_random_seeds),
        "seeds": str(args.rollout_seeds),
        "instance_ids": str(args.rollout_instance_ids),
        "missing_only": bool(args.rollout_missing_only),
        "repeat": int(args.rollout_repeat),
        "shuffle_seed": str(args.rollout_shuffle_seed),
        "sample": int(args.rollout_sample),
        "generation_config": path_fingerprint(generation_config),
    }


def build_evaluation_command(
    *,
    python_executable: str,
    data_path: str,
    output_dir: Path,
    recalc_dir: Path,
    start_idx: int,
    end_idx: int,
    results_file: Path,
    evaluator_backend: str,
) -> list[str]:
    return [
        python_executable,
        "evaluate_with_official.py",
        "--data_path",
        data_path,
        "--output_dir",
        str(output_dir),
        "--recalc_dir",
        str(recalc_dir),
        "--start_idx",
        str(start_idx),
        "--end_idx",
        str(end_idx),
        "--results_file",
        str(results_file),
        "--evaluator-backend",
        evaluator_backend,
    ]


def evaluation_runtime_identity(
    repo: Path,
    *,
    evaluator_backend: str,
) -> dict[str, object]:
    repo_text = str(repo.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    evaluator = importlib.import_module("evaluate_with_official")
    return evaluator.evaluation_runtime_identity(evaluator_backend)


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def cdost_control_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    records_path: Path,
    dataset_fingerprint: Mapping[str, Any],
    generation_config_path: Path,
    evaluator_identity: Mapping[str, Any],
    source_fingerprints: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    hierarchy = dict(config.get("hierarchy") or {})
    otd = dict(hierarchy.get("otd") or {})
    otd.pop("atom_cache_path", None)
    generation = dict(config.get("generation") or {})
    generation.pop("debug_dir", None)
    generation["api_key_fingerprint"] = api_key_fingerprint(
        str(generation.pop("api_key", ""))
    )
    embedding = dict(config.get("embedding") or {})
    embedding.pop("cache_path", None)
    embedding["api_key_fingerprint"] = api_key_fingerprint(
        str(embedding.pop("api_key", ""))
    )
    analyst = dict(config.get("analyst") or {})
    analyst.pop("prompt_token_report_path", None)
    retrieval = skillbank_retrieval_protocol(
        args,
        cache_path=Path("<run-local-index>"),
        vector_cache_path=Path("<shared-vector-cache>"),
        selection_log=Path("<run-local-selection-log>"),
    )
    for key in (
        "cache_path",
        "vector_cache_path",
        "selection_log",
        "require_vector_cache_match",
    ):
        retrieval.pop(key, None)
    retrieval["embedding_api_key_fingerprint"] = api_key_fingerprint(
        str(retrieval.pop("embedding_api_key", ""))
    )
    rollout = rollout_protocol(
        args,
        generation_config=generation_config_path,
    )
    rollout["generation_config"] = json.loads(
        generation_config_path.read_text(encoding="utf-8")
    )
    rollout["openai_api_key_fingerprint"] = str(
        rollout.pop("openai_api_key", "")
    )
    dynamic_counts = expected_dynamic_counts(
        record_count=load_record_count(records_path),
        initial_count=int(args.dynamic_initial_count),
        arrival_count=int(args.dynamic_arrival_count),
    )
    paired_dynamic_schedule = {
        **dynamic_counts,
        "arrival_order": "dataset",
        "shuffle_seed": (
            None
            if int(args.dynamic_shuffle_seed) < 0
            else int(args.dynamic_shuffle_seed)
        ),
        "snapshot_interval": max(
            1,
            int(args.dynamic_update_batch_size),
        ),
        "snapshot_include_embeddings": bool(
            args.dynamic_snapshot_include_embeddings
        ),
        "resume_from_snapshots": bool(
            args.dynamic_resume_from_snapshots
        ),
    }
    return {
        "format": "cdost_control_contract_v2",
        "dataset": dataset_fingerprint,
        "records_sha256": file_sha256(records_path),
        "train_split": [int(args.train_start), int(args.train_end)],
        "heldout_split": [int(args.heldout_start), int(args.heldout_end)],
        "tree": {
            "tree_policy": "certified_dual_view_otd",
            "otd": otd,
            "generation": generation,
            "embedding": embedding,
            "analyst": analyst,
        },
        "paired_dynamic_schedule": paired_dynamic_schedule,
        "retrieval": retrieval,
        "rollout": rollout,
        "evaluator": evaluator_identity,
        "source": {
            key: source_fingerprints[key]
            for key in (
                "runner",
                "run_spreadsheetbench",
                "evaluate_with_official",
                "spreadsheetbench_support",
                "spreadsheet_agent",
                "react_agent",
                "skillbank",
                "antichain_retrieval",
                "dynamix_core",
                "dynamix_trace2skill",
            )
        },
    }


def write_cdost_control_manifest(
    path: Path,
    contract: dict[str, object],
) -> dict[str, object]:
    payload = {
        "format": "cdost_control_manifest_v1",
        "contract_sha256": _canonical_sha256(contract),
        "contract": contract,
    }
    write_json_atomic(path, payload)
    return payload


def ebst_control_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    records_path: Path,
    dataset_fingerprint: Mapping[str, Any],
    generation_config_path: Path,
    evaluator_identity: Mapping[str, Any],
    source_fingerprints: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    contract = cdost_control_contract(
        args=args,
        config=config,
        records_path=records_path,
        dataset_fingerprint=dataset_fingerprint,
        generation_config_path=generation_config_path,
        evaluator_identity=evaluator_identity,
        source_fingerprints=source_fingerprints,
    )
    hierarchy = config.get("hierarchy")
    if not isinstance(hierarchy, Mapping):
        raise ValueError("hierarchy config must be a mapping")
    raw_ebst = hierarchy.get("ebst")
    if not isinstance(raw_ebst, Mapping):
        raise ValueError("hierarchy.ebst config must be a mapping")
    ebst = dict(raw_ebst)
    ebst.pop("atom_cache_path", None)
    tree_contract = contract.get("tree")
    if not isinstance(tree_contract, Mapping):
        raise RuntimeError("CDOST control contract is missing tree settings")
    contract["format"] = "ebst_control_contract_v1"
    contract["tree"] = {
        **dict(tree_contract),
        "tree_policy": "evidence_balanced_skill_tree",
        "ebst": ebst,
    }
    contract["tree"].pop("otd", None)
    return contract


def write_ebst_control_manifest(
    path: Path,
    contract: dict[str, object],
) -> dict[str, object]:
    payload = {
        "format": "ebst_control_manifest_v1",
        "contract_sha256": _canonical_sha256(contract),
        "contract": contract,
    }
    write_json_atomic(path, payload)
    return payload


def validate_matching_ebst_control_manifest(
    *,
    current_manifest: Path,
    source_manifest: Path,
) -> None:
    current = json.loads(current_manifest.read_text(encoding="utf-8"))
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    for path, payload in (
        (current_manifest, current),
        (source_manifest, source),
    ):
        if (
            payload.get("format") != "ebst_control_manifest_v1"
            or payload.get("contract_sha256")
            != _canonical_sha256(payload.get("contract"))
        ):
            raise ValueError(
                f"invalid evidence-balanced control manifest: {path}"
            )
    if current["contract"] != source["contract"]:
        raise ValueError(
            "dynamic evidence-balanced control contract differs from the "
            "source static run"
        )


def _without_keys(
    payload: Mapping[str, Any],
    *keys: str,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in set(keys)
    }


def _ebst_baseline_invariants(
    contract: Mapping[str, Any],
    *,
    tree_policy: str,
) -> dict[str, Any]:
    tree = dict(contract.get("tree") or {})
    source = dict(contract.get("source") or {})
    rollout_evaluator_source = {
        key: source.get(key)
        for key in (
            "run_spreadsheetbench",
            "evaluate_with_official",
            "spreadsheetbench_support",
            "spreadsheet_agent",
            "react_agent",
            "skillbank",
            "antichain_retrieval",
        )
    }
    generation = _without_keys(
        dict(tree.get("generation") or {}),
        "base_url",
        "api_key_fingerprint",
    )
    embedding = _without_keys(
        dict(tree.get("embedding") or {}),
        "base_url",
        "api_key_fingerprint",
    )
    retrieval = _without_keys(
        dict(contract.get("retrieval") or {}),
        "embedding_base_url",
        "embedding_api_key_fingerprint",
    )
    rollout = _without_keys(
        dict(contract.get("rollout") or {}),
        "openai_base_url",
        "openai_api_key_fingerprint",
        "instance_ids",
        "missing_only",
        "sample",
    )
    if tree_policy == "certified_dual_view_otd":
        method = dict(tree.get("otd") or {})
        method_invariants = {
            "dual_view_lambda": method.get("dual_view_lambda"),
            "atom_temperature": method.get("atom_temperature"),
            "skill_temperature": method.get("parent_temperature"),
            "retrieval_token_budget": method.get(
                "retrieval_token_budget"
            ),
            "retrieval_token_unit": method.get("retrieval_token_unit"),
            "retrieval_exact_search_max_states": method.get(
                "retrieval_exact_search_max_states"
            ),
            "validation_mode": method.get("validation_mode"),
        }
    elif tree_policy == "evidence_balanced_skill_tree":
        method = dict(tree.get("ebst") or {})
        method_invariants = {
            "dual_view_lambda": method.get("dual_view_lambda"),
            "atom_temperature": method.get("atom_temperature"),
            "skill_temperature": method.get("capsule_temperature"),
            "retrieval_token_budget": method.get(
                "retrieval_token_budget"
            ),
            "retrieval_token_unit": method.get("retrieval_token_unit"),
            "retrieval_exact_search_max_states": method.get(
                "retrieval_exact_search_max_states"
            ),
            "validation_mode": method.get("validation_mode"),
        }
    else:
        raise ValueError(f"unsupported control tree policy: {tree_policy}")
    return {
        "dataset": contract.get("dataset"),
        "records_sha256": contract.get("records_sha256"),
        "train_split": contract.get("train_split"),
        "heldout_split": contract.get("heldout_split"),
        "generation": generation,
        "embedding": embedding,
        "analyst": tree.get("analyst"),
        "method_invariants": method_invariants,
        "paired_dynamic_schedule": contract.get(
            "paired_dynamic_schedule"
        ),
        "retrieval": retrieval,
        "rollout": rollout,
        "evaluator": contract.get("evaluator"),
        "rollout_evaluator_source": rollout_evaluator_source,
    }


def validate_ebst_against_cdost_baseline(
    *,
    current_contract: Mapping[str, Any],
    baseline_manifest: Path,
) -> dict[str, Any]:
    baseline = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    if (
        baseline.get("format") != "cdost_control_manifest_v1"
        or baseline.get("contract_sha256")
        != _canonical_sha256(baseline.get("contract"))
    ):
        raise ValueError(
            f"invalid baseline CDOST control manifest: {baseline_manifest}"
        )
    expected = _ebst_baseline_invariants(
        dict(baseline["contract"]),
        tree_policy="certified_dual_view_otd",
    )
    observed = _ebst_baseline_invariants(
        current_contract,
        tree_policy="evidence_balanced_skill_tree",
    )
    if observed != expected:
        differing = sorted(
            key
            for key in expected
            if observed.get(key) != expected.get(key)
        )
        raise ValueError(
            "evidence-balanced treatment is not protocol-compatible with "
            f"its CDOST baseline; differing sections: {differing}"
        )
    return {
        "format": "ebst_cdost_baseline_compatibility_v1",
        "compatible": True,
        "baseline_manifest": path_fingerprint(baseline_manifest),
        "baseline_contract_sha256": baseline["contract_sha256"],
        "invariants_sha256": _canonical_sha256(observed),
    }


def validate_matching_cdost_control_manifest(
    *,
    current_manifest: Path,
    source_manifest: Path,
) -> None:
    current = json.loads(current_manifest.read_text(encoding="utf-8"))
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    for path, payload in (
        (current_manifest, current),
        (source_manifest, source),
    ):
        if (
            payload.get("format") != "cdost_control_manifest_v1"
            or payload.get("contract_sha256")
            != _canonical_sha256(payload.get("contract"))
        ):
            raise ValueError(f"invalid CDOST control manifest: {path}")
    if current["contract"] != source["contract"]:
        raise ValueError(
            "dynamic CDOST control contract differs from the source static run"
        )


def validate_source_build_output(
    *,
    marker_path: Path,
    output_path: Path,
) -> None:
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"source build marker is missing: {marker_path}"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    identities = marker.get("output_identities")
    expected = (
        identities.get(str(output_path))
        if isinstance(identities, dict)
        else None
    )
    if not isinstance(expected, dict) or expected != path_fingerprint(
        output_path
    ):
        raise ValueError(
            "source output no longer matches its completed stage marker: "
            f"{output_path}"
        )


def skillbank_retrieval_protocol(
    args: argparse.Namespace,
    *,
    cache_path: Path,
    vector_cache_path: Path,
    selection_log: Path,
) -> dict[str, object]:
    strict_tree = is_certified_single_parent_tree(args.tree_policy)
    protocol: dict[str, object] = {
        "query_policy": "instruction + Task type; answer_position excluded",
        "top_k": int(args.skillbank_top_k),
        "embedding_base_url": args.embedding_base_url,
        "embedding_model": args.embedding_model,
        "embedding_api_key": api_key_fingerprint("EMPTY"),
        "embedding_max_model_len": int(args.embedding_max_model_len),
        "embedding_max_input_tokens": int(args.embedding_max_input_tokens),
        "embedding_batch_size": int(args.embedding_batch_size),
        "embedding_tokenizer": args.embedding_tokenizer,
        "vector_cache_path": str(vector_cache_path),
        "require_vector_cache_match": (
            strict_tree
            and args.tree_scenario == "dynamic_update"
        ),
        "require_cache_match": strict_tree,
        "cache_path": str(cache_path),
        "selection_log": str(selection_log),
    }
    if strict_tree:
        protocol.update(
            {
                "embedding_input_policy": (
                    "single_vector_fail_if_over_limit"
                ),
                "chunked_embedding_active": False,
                "vector_cache_policy": (
                    "content_addressed_first_success_frozen"
                ),
            }
        )
    else:
        protocol.update(
            {
                "embedding_input_policy": "legacy_single_vector",
                "chunked_embedding_active": False,
                "chunked_embedding_configured": bool(
                    args.chunked_embedding_enabled
                ),
                "chunk_tokens": int(
                    args.chunked_embedding_chunk_tokens
                ),
                "chunk_overlap_tokens": int(
                    args.chunked_embedding_overlap_tokens
                ),
                "chunk_pooling": args.chunked_embedding_pooling,
                "chunk_add_special_tokens": bool(
                    args.chunked_embedding_add_special_tokens
                ),
                "chunk_normalize_after_pooling": bool(
                    args.chunked_embedding_normalize_after_pooling
                ),
                "chunk_fail_if_exceeds_model_limit": bool(
                    args.chunked_embedding_fail_if_chunk_exceeds_model_limit
                ),
                "vector_cache_policy": "disabled",
            }
        )
    return protocol


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
    if getattr(args, "tree_policy", "") == "certified_dual_view_otd":
        if summary.get("tree_policy") != "certified_dual_view_otd":
            raise RuntimeError("heldout requires a certified_dual_view_otd tree")
        if int(summary.get("parent_generation_error_count", -1)) != 0:
            raise RuntimeError(
                "heldout is blocked because parent skill generation was incomplete"
            )
        if int(summary.get("excluded_count", -1)) != 0:
            raise RuntimeError(
                "heldout is blocked because the certified tree excluded input records"
            )
        if int(summary.get("atom_count", -1)) != int(
            summary.get("record_count", -2)
        ):
            raise RuntimeError(
                "heldout is blocked because atom_count does not match record_count"
            )
        if (
            scenario == "dynamic_update"
            and summary.get("atom_source") != "frozen_cache"
        ):
            raise RuntimeError(
                "controlled CDOST dynamic heldout requires frozen_cache atoms"
            )
    if getattr(args, "tree_policy", "") == "evidence_balanced_skill_tree":
        if summary.get("tree_policy") != "evidence_balanced_skill_tree":
            raise RuntimeError(
                "heldout requires an evidence_balanced_skill_tree"
            )
        if int(summary.get("excluded_count", -1)) != 0:
            raise RuntimeError(
                "heldout is blocked because the evidence-balanced tree "
                "excluded input records"
            )
        if int(summary.get("atom_count", -1)) != int(
            summary.get("record_count", -2)
        ):
            raise RuntimeError(
                "heldout is blocked because atom_count does not match "
                "record_count"
            )
        if int(summary.get("retrievable_atom_count", -1)) != 0:
            raise RuntimeError(
                "heldout is blocked because raw evidence atoms are retrievable"
            )
        if int(summary.get("active_capsule_count", 0)) <= 0:
            raise RuntimeError(
                "heldout is blocked because no active skill capsule exists"
            )
        if int(summary.get("runtime_generation_error_count", -1)) != 0:
            raise RuntimeError(
                "heldout is blocked because skill capsule generation had "
                "runtime errors"
            )
        if int(summary.get("prompt_budget_error_count", -1)) != 0:
            raise RuntimeError(
                "heldout is blocked because skill capsule generation exceeded "
                "the configured prompt budget"
            )
        if (
            scenario == "dynamic_update"
            and summary.get("atom_source") != "frozen_cache"
        ):
            raise RuntimeError(
                "controlled evidence-balanced dynamic heldout requires "
                "frozen_cache atoms"
            )
    if args.tree_scenario != "dynamic_update":
        return
    if hasattr(args, "train_start") and hasattr(args, "train_end"):
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
        key=lambda item: int(item["prompt_tokens"] or 0),
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
    strict_tree = is_certified_single_parent_tree(
        str(getattr(args, "tree_policy", "")).strip()
    )
    analyst_budget = int(float(args.summary_max_model_tokens) * float(args.summary_budget_ratio))
    evidence_budget = analyst_budget - int(args.summary_prompt_overhead_reserve_tokens)
    if analyst_budget >= int(args.summary_max_model_tokens) * 0.9:
        findings.append({
            "severity": "high",
            "area": "summary_budget",
            "finding": "analyst prompt budget leaves little context-window headroom for chat-template/thinking overhead.",
            "evidence": f"analyst_budget={analyst_budget}, max_model_tokens={args.summary_max_model_tokens}",
        })
    if not strict_tree and evidence_budget <= 0:
        findings.append({
            "severity": "blocker",
            "area": "summary_budget",
            "finding": "member evidence budget is non-positive, so build cannot select feasible communities.",
            "evidence": f"analyst_budget={analyst_budget}, overhead={args.summary_prompt_overhead_reserve_tokens}",
        })
    if (
        not strict_tree
        and int(args.analyst_max_prompt_tokens) > 0
        and int(args.analyst_max_prompt_tokens) < evidence_budget
    ):
        findings.append({
            "severity": "high",
            "area": "analyst_budget_override",
            "finding": "analyst max prompt override is smaller than the tree-builder evidence budget; build may pass but analyst preflight can fail.",
            "evidence": f"analyst_max_prompt_tokens={args.analyst_max_prompt_tokens}, evidence_budget={evidence_budget}",
        })
    if not strict_tree and int(args.budget_refinement_apply_to_level) == 0:
        findings.append({
            "severity": "watch",
            "area": "budget_refinement",
            "finding": "budget refinement only protects L0 raw-trajectory communities; unusually verbose L1+ cards can still trigger analyst over-budget failures.",
            "evidence": "budget_refinement_apply_to_level=0",
        })
    if not strict_tree and args.soft_recursive_assignment == "cumulative_mass":
        findings.append({
            "severity": "info",
            "area": "soft_membership",
            "finding": "top_r_memberships is inactive under cumulative_mass assignment; max_membership_gap and cumulative_mass_coverage control fan-out.",
            "evidence": f"recursive_assignment={args.soft_recursive_assignment}, top_r={args.soft_top_r_memberships}",
        })
    if not strict_tree and args.soft_recursive_assignment == "cumulative_mass":
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
    train_count = int(args.train_end) - int(args.train_start)
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


def validate_nodebank_manifest_for_heldout(
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if not is_certified_single_parent_tree(args.tree_policy):
        return
    if manifest.get("tree_policy") != args.tree_policy:
        raise RuntimeError(
            f"{args.tree_policy} nodebank tree_policy identity is missing"
        )
    export_policy = dict(manifest.get("export_policy", {}))
    if export_policy.get("heldout_retrieval") != "tree_antichain_knapsack":
        raise RuntimeError(
            f"{args.tree_policy} nodebank antichain retrieval policy is missing"
        )
    if args.tree_policy == "evidence_balanced_skill_tree":
        if export_policy.get("experience_atoms_exported") is not False:
            raise RuntimeError(
                "evidence-balanced nodebank must exclude experience atoms"
            )
        if export_policy.get("retrieval_unit") != "validated_skill_capsule":
            raise RuntimeError(
                "evidence-balanced nodebank must retrieve skill capsules"
            )
        source_root = Path(__file__).resolve().parents[1] / "src"
        source_root_text = str(source_root)
        if source_root_text not in sys.path:
            sys.path.insert(0, source_root_text)
        from dynamix_trace2skill.skillbank import (
            validate_ebst_nodebank_manifest,
        )

        validate_ebst_nodebank_manifest(manifest)


def active_hierarchy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tree_policy = str(payload.get("tree_policy") or "").strip()
    if tree_policy == "certified_dual_view_otd":
        return {
            key: payload[key]
            for key in ("tree_policy", "otd", "summary_budget")
        }
    if tree_policy == "evidence_balanced_skill_tree":
        return {
            key: payload[key]
            for key in ("tree_policy", "ebst", "summary_budget")
        }
    return {
        key: value
        for key, value in payload.items()
        if key not in {"otd", "ebst"}
    }


def active_dynamic_payload(
    payload: dict[str, Any],
    *,
    tree_policy: str,
) -> dict[str, Any]:
    if not is_certified_single_parent_tree(tree_policy):
        return dict(payload)
    return {
        key: payload[key]
        for key in (
            "initial_count",
            "arrival_count",
            "update_batch_size",
            "shuffle_seed",
            "snapshot_include_embeddings",
            "resume_from_snapshots",
        )
    }


def method_runtime_identity(args: argparse.Namespace) -> dict[str, Any]:
    if not is_certified_single_parent_tree(args.tree_policy):
        return {
            "tree_policy": args.tree_policy,
            "structural_graph_kind": args.graph_kind,
            "allow_overlap": bool(args.allow_overlap),
            "allow_multi_parent": bool(args.allow_multi_parent),
        }
    dynamic = args.tree_scenario == "dynamic_update"
    if args.tree_policy == "evidence_balanced_skill_tree":
        return {
            "tree_policy": "evidence_balanced_skill_tree",
            "structural_graph_kind": "single_parent_balanced_metric_tree",
            "allow_overlap": False,
            "allow_multi_parent": False,
            "arrival_update_semantics": (
                "sequential_structural_insert"
                if dynamic
                else "static_dataset_order_same_insert_operation"
            ),
            "parent_refresh_semantics": (
                "changed_path_batched_bottom_up"
                if dynamic
                else "all_nodes_bottom_up_after_build"
            ),
            "snapshot_interval": (
                max(1, int(args.dynamic_update_batch_size))
                if dynamic
                else None
            ),
            "nodebank_scope": "active_skill_capsules_only",
            "retrieval_policy": "tree_antichain_knapsack",
            "embedding_vector_control": (
                "shared_content_addressed_cache_required"
                if dynamic
                else "content_addressed_cache_populates_control"
            ),
        }
    return {
        "tree_policy": "certified_dual_view_otd",
        "structural_graph_kind": "single_parent_binary_tree",
        "allow_overlap": False,
        "allow_multi_parent": False,
        "arrival_update_semantics": (
            "sequential_per_atom" if dynamic else "static_dataset_order"
        ),
        "parent_refresh_semantics": (
            "changed_path_bottom_up_per_atom"
            if dynamic
            else "all_internal_nodes_bottom_up_after_build"
        ),
        "snapshot_interval": (
            max(1, int(args.dynamic_update_batch_size))
            if dynamic
            else None
        ),
        "nodebank_scope": "complete_tree",
        "retrieval_policy": "tree_antichain_knapsack",
        "embedding_vector_control": (
            "shared_content_addressed_cache_required"
            if dynamic
            else "content_addressed_cache_populates_control"
        ),
    }


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
    if not prompt_stats.get("exists"):
        lines.append("- Artifact status: unavailable; this report was not produced for the run.")
    else:
        lines.append(f"- Max observed prompt tokens: `{prompt_stats.get('max_prompt_tokens_observed', 0)}` / configured `{prompt_stats.get('configured_max_prompt_tokens', 0)}`")
        lines.append(f"- Near configured limit count: `{prompt_stats.get('near_configured_limit_count', 0)}`; over budget count: `{prompt_stats.get('over_budget_count', 0)}`")
        for event in prompt_stats.get("top_events", [])[:5]:
            lines.append(
                f"- Top prompt `{event.get('community_id')}` level={event.get('level')} members={event.get('member_count')} tokens={event.get('prompt_tokens')}/{event.get('max_prompt_tokens')}"
            )
    chunk_stats = report.get("chunked_embedding_stats", {})
    lines.extend(["", "## Chunked Embedding", ""])
    lines.append(f"- Chunk report: `{chunk_stats.get('path')}`")
    if not chunk_stats.get("exists"):
        lines.append("- Artifact status: unavailable; this report was not produced for the run.")
    else:
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
    dataset_size = len(rows)
    if not 0 <= train_start < train_end <= heldout_start < heldout_end:
        raise ValueError(
            "train and heldout ranges must be ordered, non-empty, and "
            "non-overlapping"
        )
    if heldout_end > dataset_size:
        raise ValueError(
            f"heldout_end={heldout_end} exceeds dataset size {dataset_size}"
        )

    def subset(start: int, end: int) -> list[dict]:
        out = []
        for index, row in enumerate(rows[start:end], start=start):
            out.append({"index": index, "id": str(row.get("id", row.get("task_id", index))), "instruction_type": row.get("instruction_type", ""), "answer_position": row.get("answer_position", "")})
        return out

    train = subset(train_start, train_end)
    heldout = subset(heldout_start, heldout_end)
    if len(train) != train_end - train_start:
        raise ValueError("train slice count does not match the requested range")
    if len(heldout) != heldout_end - heldout_start:
        raise ValueError("heldout slice count does not match the requested range")
    overlap = {row["id"] for row in train} & {row["id"] for row in heldout}
    if overlap:
        raise ValueError(
            "train and heldout task IDs must be disjoint; overlap="
            f"{sorted(overlap)[:10]}"
        )
    manifest = {
        "source_dataset_json": str(dataset_path.resolve()),
        "policy": "Trace2Skill dataset order / natural task id order; runner still uses start/end indices",
        "dataset_size": dataset_size,
        "train_range": [train_start, train_end],
        "heldout_range": [heldout_start, heldout_end],
        "train_expected_count": train_end - train_start,
        "heldout_expected_count": heldout_end - heldout_start,
        "train": train,
        "heldout": heldout,
    }
    path = run_dir / "split_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_heldout_eval_coverage(
    eval_path: Path,
    split_manifest: dict[str, Any],
) -> None:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    expected_ids = [
        str(row["id"])
        for row in split_manifest["heldout"]
    ]
    expected_count = int(
        split_manifest.get("heldout_expected_count", len(expected_ids))
    )
    actual_count = int(payload.get("summary", {}).get("total_instances", -1))
    if actual_count != expected_count:
        raise RuntimeError(
            "heldout evaluator denominator mismatch: "
            f"expected={expected_count}, actual={actual_count}"
        )
    actual_ids = [
        str(row.get("id", ""))
        for row in payload.get("results", [])
    ]
    if actual_ids != expected_ids:
        raise RuntimeError(
            "heldout evaluator identity/order mismatch: "
            f"expected={expected_ids[:10]}, actual={actual_ids[:10]}"
        )


def validate_evaluation_runtime_identity(
    eval_path: Path,
    expected_identity: Mapping[str, Any],
) -> None:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    summary = dict(payload.get("summary", {}))
    actual_identity = {
        "workbook_comparator": summary.get("workbook_comparator"),
        "libreoffice": summary.get("libreoffice"),
    }
    if actual_identity != dict(expected_identity):
        raise RuntimeError(
            "evaluator runtime identity changed after preflight: "
            f"expected={dict(expected_identity)}, actual={actual_identity}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trace2Skill train collection -> nodebank build -> heldout experiment")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--records-path", default=None, help="Existing extracted records.json; when set, skip train rollout/eval/extraction and build from this file")
    parser.add_argument("--reuse-train-run-dir", default=None, help="Existing run dir containing records.json; when set, skip train rollout/eval/extraction")
    parser.add_argument("--reuse-tree-dir", default=None, help="Existing dynamix_tree dir to reuse; exports a fresh nodebank with this run's skill_export filter")
    parser.add_argument("--train-artifact-dir", default=None, help="Directory for train rollout/eval/extraction artifacts; default: --run-dir")
    parser.add_argument("--scenario-output-dir", default=None, help="Directory for tree, nodebank, heldout, and final reports; default: --run-dir")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=200)
    parser.add_argument("--heldout-start", type=int, default=200)
    parser.add_argument("--heldout-end", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=os.environ.get("GEN_MODEL", "Qwen3.5-9B"))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18002/v1"))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("EMBED_BASE_URL", "http://10.26.1.184:18007/v1"))
    parser.add_argument("--embedding-model", default=os.environ.get("EMBED_MODEL", "Qwen3-Embedding-8B"))
    parser.add_argument("--embedding-tokenizer", default=os.environ.get("EMBED_TOKENIZER", "/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-8B"))
    parser.add_argument("--python-executable", default=os.environ.get("DYNAMIX_PYTHON", sys.executable), help="Python executable used for all experiment stages; its bin dir is prepended to PATH so agent bash actions can call bare python")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--thinking", choices=["true", "false", "null"], default="true", help="Qwen thinking setting for Trace2Skill rollout and DynaMix analyst calls")
    parser.add_argument("--skillbank-top-k", type=int, default=10, help="Select top-k DynaMix nodebank nodes by embedding before each heldout task")
    parser.add_argument("--tree-scenario", choices=["dynamic_update", "static_build"], default="dynamic_update", help="DynaMix build mode before heldout; default is the train200 60/40 dynamic protocol")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--tree-policy", default="projected_gmm_bic")
    parser.add_argument("--otd-dual-view-lambda", type=float, default=0.5)
    parser.add_argument("--otd-tie-epsilon", type=float, default=0.0)
    parser.add_argument("--otd-atom-temperature", type=float, default=0.0)
    parser.add_argument("--otd-parent-temperature", type=float, default=0.0)
    parser.add_argument("--otd-atom-cache-path", default=None, help="Frozen experience_atoms.json used to compare static and dynamic OTD builds")
    parser.add_argument("--otd-retrieval-token-budget", type=int, default=24000)
    parser.add_argument("--otd-retrieval-token-unit", type=int, default=128)
    parser.add_argument(
        "--otd-retrieval-exact-search-max-states",
        type=int,
        default=250_000,
        help=(
            "Fail-closed work bound used only when exact non-additive "
            "antichain retrieval cannot use its unique-optimum fast path."
        ),
    )
    parser.add_argument("--ebst-max-entries", type=int, default=8)
    parser.add_argument("--ebst-dual-view-lambda", type=float, default=0.5)
    parser.add_argument("--ebst-atom-temperature", type=float, default=0.0)
    parser.add_argument("--ebst-capsule-temperature", type=float, default=0.0)
    parser.add_argument(
        "--ebst-validator-temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--ebst-atom-cache-path",
        default=None,
        help=(
            "Frozen experience_atoms.json used by the paired dynamic "
            "evidence-balanced tree run"
        ),
    )
    parser.add_argument(
        "--ebst-retrieval-token-budget",
        type=int,
        default=24000,
    )
    parser.add_argument(
        "--ebst-retrieval-token-unit",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--ebst-retrieval-exact-search-max-states",
        type=int,
        default=250_000,
    )
    parser.add_argument("--graph-kind", default="overlapping_experience_hierarchy")
    parser.add_argument("--allow-overlap", type=parse_bool, default=True)
    parser.add_argument("--allow-multi-parent", type=parse_bool, default=True)
    parser.add_argument("--use-support-mass", type=parse_bool, default=True)
    parser.add_argument("--dynamic-initial-count", type=int, default=120, help="Dynamic mode: number of initial train records used for the static seed tree")
    parser.add_argument("--dynamic-arrival-count", type=int, default=80, help="Dynamic mode: number of later train records inserted sequentially; <=0 consumes all remaining train records")
    parser.add_argument(
        "--dynamic-update-batch-size",
        type=int,
        default=8,
        help=(
            "Legacy dynamic policies: sequential admissions per layer-local "
            "summary batch. certified_dual_view_otd: snapshot interval only; "
            "every atom is inserted and its changed parent path refreshed "
            "before the next arrival. evidence_balanced_skill_tree: structural "
            "insertions remain sequential; this controls how many arrivals "
            "share one parallel local Skill Capsule refresh."
        ),
    )
    parser.add_argument(
        "--dynamic-shuffle-seed",
        type=int,
        default=42,
        help=(
            "Legacy dynamic policies: reproducible arrival shuffle; -1 "
            "disables it. certified_dual_view_otd rejects shuffle and "
            "therefore requires -1."
        ),
    )
    parser.add_argument(
        "--dynamic-snapshot-include-embeddings",
        type=parse_bool,
        default=True,
        help=(
            "Legacy dynamic snapshot embedding control. "
            "Both certified single-parent policies snapshot their complete "
            "immutable atom state and require this to remain true."
        ),
    )
    parser.add_argument(
        "--dynamic-resume-from-snapshots",
        type=parse_bool,
        default=False,
        help=(
            "Legacy dynamic snapshot resume control. "
            "Certified single-parent policies reject snapshot resume until "
            "a validated resume protocol is implemented."
        ),
    )
    parser.add_argument("--max-levels", type=int, default=8)
    parser.add_argument("--skill-output-dir-name", default="skills")
    parser.add_argument("--skill-export-min-level", type=int, default=-1, help="-1 exports all lower levels; 1 exports L1+")
    parser.add_argument("--skill-export-max-level", type=int, default=-1, help="-1 exports all upper levels; 1 exports L1 only")
    parser.add_argument("--rollout-temperature", type=float, default=0.0)
    parser.add_argument(
        "--evaluator-backend",
        choices=["official", "local"],
        default="local",
        help=(
            "Explicit SpreadsheetBench workbook comparator. The experiment "
            "runner never uses environment-dependent auto discovery."
        ),
    )
    parser.add_argument("--rollout-client-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--rollout-client-retry-wait-seconds", type=parse_float_csv, default=[5.0, 10.0, 30.0])
    parser.add_argument(
        "--rollout-disable-response-cache",
        type=parse_bool,
        default=False,
    )
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
    parser.add_argument("--embedding-max-model-len", type=int, default=32000)
    parser.add_argument("--embedding-max-input-tokens", type=int, default=32000)
    parser.add_argument("--embedding-truncate-long-texts", type=parse_bool, default=True)
    parser.add_argument("--embedding-truncation-strategy", default="head")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-concurrency", type=int, default=8, help="Embedding API concurrency")
    parser.add_argument(
        "--embedding-cache-path",
        default="",
        help=(
            "Content-addressed embedding vector cache. Controlled CDOST "
            "dynamic runs must reuse the matching static run cache."
        ),
    )
    parser.add_argument("--embedding-tokenizer-required", type=parse_bool, default=True)
    parser.add_argument("--chunked-embedding-enabled", type=parse_bool, default=True)
    parser.add_argument("--chunked-embedding-chunk-tokens", type=int, default=8000)
    parser.add_argument("--chunked-embedding-overlap-tokens", type=int, default=1000)
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
    parser.add_argument("--gmm-min-split-size", type=int, default=2)
    parser.add_argument("--gmm-min-effective-samples-per-component", type=int, default=2)
    parser.add_argument("--gmm-abs-kmax", type=int, default=64)
    parser.add_argument("--gmm-max-concurrent-candidates", type=int, default=1)
    parser.add_argument("--gmm-max-concurrent-restarts", type=int, default=1)
    parser.add_argument("--kmeans-fixed-k", type=int, default=8)
    parser.add_argument("--kmeans-min-k", type=int, default=1)
    parser.add_argument("--kmeans-num-restarts", type=int, default=5)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--kmeans-tol", type=float, default=1.0e-4)
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
    parser.add_argument("--analyst-tokenizer", default="")
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
    parser.add_argument("--analysis-bundle-max-chars", type=int, default=60000, help="0 disables compact analyst evidence bundles")
    parser.add_argument("--analysis-bundle-max-steps", type=int, default=12)
    parser.add_argument("--analysis-bundle-max-step-chars", type=int, default=6000)
    parser.add_argument("--analysis-bundle-max-final-response-chars", type=int, default=12000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if (
        is_certified_single_parent_tree(args.tree_policy)
        and int(args.dynamic_shuffle_seed) >= 0
    ):
        parser.error(
            f"{args.tree_policy} paired runs require dataset order; "
            "pass --dynamic-shuffle-seed -1"
        )
    if (
        is_certified_single_parent_tree(args.tree_policy)
        and args.tree_scenario == "dynamic_update"
        and not str(atom_cache_path_for_args(args) or "").strip()
    ):
        parser.error(
            f"controlled {args.tree_policy} dynamic runs require the "
            "matching --otd-atom-cache-path or --ebst-atom-cache-path"
        )
    if (
        is_certified_single_parent_tree(args.tree_policy)
        and bool(args.dynamic_resume_from_snapshots)
    ):
        parser.error(
            f"{args.tree_policy} snapshot resume is not implemented with "
            "fingerprint validation; pass --dynamic-resume-from-snapshots false"
        )
    if (
        is_certified_single_parent_tree(args.tree_policy)
        and not bool(args.dynamic_snapshot_include_embeddings)
    ):
        parser.error(
            f"{args.tree_policy} requires "
            "--dynamic-snapshot-include-embeddings true"
        )
    if is_certified_single_parent_tree(args.tree_policy):
        if int(args.max_levels) != 8:
            parser.error(
                f"{args.tree_policy} builds its complete structural tree; "
                "--max-levels is inactive and must remain 8"
            )
        if (
            int(args.skill_export_min_level) >= 0
            or int(args.skill_export_max_level) >= 0
        ):
            parser.error(
                f"{args.tree_policy} antichain retrieval requires the "
                "complete tree; skill-export level filters are not allowed"
            )
    if args.tree_policy == "evidence_balanced_skill_tree":
        if int(args.ebst_max_entries) < 4:
            parser.error("--ebst-max-entries must be at least 4")
        if not 0.0 <= float(args.ebst_dual_view_lambda) <= 1.0:
            parser.error("--ebst-dual-view-lambda must be in [0, 1]")
    if not is_certified_single_parent_tree(args.tree_policy):
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
    if int(args.rollout_repeat) != 1:
        parser.error("--rollout-repeat is fixed to 1 for this handoff protocol")
    if str(args.rollout_shuffle_seed).strip():
        parser.error("--rollout-shuffle-seed must be empty for this handoff protocol")
    if int(args.rollout_sample) != 0:
        parser.error("--rollout-sample is fixed to 0/full split for this handoff protocol")
    if (
        not str(args.openai_base_url or "").startswith("mock://")
        and bool(args.analyst_tokenizer_required)
        and not bool(args.analyst_allow_regex_tokenizer_fallback)
        and not str(args.analyst_tokenizer or "").strip()
    ):
        parser.error(
            "--analyst-tokenizer is required for real DynaMix tree builds when "
            "--analyst-tokenizer-required=true and regex fallback is disabled. "
            "Use the generation model tokenizer, not the embedding tokenizer."
        )

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
    reuse_tree_state_name = (
        "balanced_tree_state.json"
        if args.tree_policy == "evidence_balanced_skill_tree"
        else "hierarchy_state.json"
    )
    if (
        reuse_tree_dir is not None
        and not (reuse_tree_dir / reuse_tree_state_name).is_file()
    ):
        raise FileNotFoundError(
            f"--reuse-tree-dir must contain {reuse_tree_state_name}: "
            f"{reuse_tree_dir}"
        )
    if reuse_tree_dir is not None and records_path_arg is None and reuse_train_run_dir is None:
        raise RuntimeError("--reuse-tree-dir requires --records-path or --reuse-train-run-dir so train stages are not re-run for retrieval-only ablations")
    if (
        reuse_tree_dir is not None
        and args.tree_policy == "evidence_balanced_skill_tree"
    ):
        raise RuntimeError(
            "--reuse-tree-dir export is not implemented for "
            "evidence_balanced_skill_tree; use the original completed run "
            "directly"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    train_artifact_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_path = resolve_embedding_cache_path(
        explicit_path=args.embedding_cache_path,
        scenario_dir=scenario_dir,
        tree_policy=args.tree_policy,
        tree_scenario=args.tree_scenario,
        atom_cache_path=atom_cache_path_for_args(args),
    )
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

    train_gen_config_path = train_artifact_dir / "trace2skill_generation_config.json"
    write_generation_config(train_gen_config_path, thinking=thinking, temperature=args.rollout_temperature)
    scenario_gen_config_path = scenario_dir / "trace2skill_generation_config.json"
    write_generation_config(scenario_gen_config_path, thinking=thinking, temperature=args.rollout_temperature)
    split_manifest = write_split_manifest(Path(args.data_path), scenario_dir, train_start=args.train_start, train_end=args.train_end, heldout_start=args.heldout_start, heldout_end=args.heldout_end)
    dataset_fp = path_fingerprint(dataset_json_path(args.data_path))
    source_fp = stage_source_fingerprints(repo)
    evaluator_identity = evaluation_runtime_identity(
        repo,
        evaluator_backend=args.evaluator_backend,
    )

    runtime = {
        "data_path": str(Path(args.data_path).resolve()),
        "run_dir": str(run_dir),
        "train_artifact_dir": str(train_artifact_dir),
        "scenario_output_dir": str(scenario_dir),
        "model": args.model,
        "openai_base_url": args.openai_base_url,
        "embedding_base_url": args.embedding_base_url,
        "embedding_model": args.embedding_model,
        "embedding_tokenizer": args.embedding_tokenizer,
        "embedding_vector_cache_path": str(embedding_cache_path),
        "evaluator_backend": args.evaluator_backend,
        "evaluator_identity": evaluator_identity,
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
        "method_identity": method_runtime_identity(args),
        "dynamic_initial_count": (
            int(args.dynamic_initial_count)
            if not is_certified_single_parent_tree(args.tree_policy)
            or args.tree_scenario == "dynamic_update"
            else None
        ),
        "dynamic_arrival_count": (
            int(args.dynamic_arrival_count)
            if not is_certified_single_parent_tree(args.tree_policy)
            or args.tree_scenario == "dynamic_update"
            else None
        ),
        "dynamic_snapshot_interval": (
            max(1, int(args.dynamic_update_batch_size))
            if is_certified_single_parent_tree(args.tree_policy)
            and args.tree_scenario == "dynamic_update"
            else None
        ),
        "dynamic_update_batch_size": (
            int(args.dynamic_update_batch_size)
            if args.tree_policy != "certified_dual_view_otd"
            else None
        ),
        "dynamic_shuffle_seed": None if int(args.dynamic_shuffle_seed) < 0 else int(args.dynamic_shuffle_seed),
        "dynamic_snapshot_include_embeddings": (
            bool(args.dynamic_snapshot_include_embeddings)
            if not is_certified_single_parent_tree(args.tree_policy)
            or args.tree_scenario == "dynamic_update"
            else None
        ),
        "dynamic_resume_from_snapshots": (
            bool(args.dynamic_resume_from_snapshots)
            if not is_certified_single_parent_tree(args.tree_policy)
            or args.tree_scenario == "dynamic_update"
            else None
        ),
        "max_levels": int(args.max_levels),
        "skill_output_dir_name": args.skill_output_dir_name,
        "skill_export_min_level": None if int(args.skill_export_min_level) < 0 else int(args.skill_export_min_level),
        "skill_export_max_level": None if int(args.skill_export_max_level) < 0 else int(args.skill_export_max_level),
        "reuse_tree_dir": str(reuse_tree_dir) if reuse_tree_dir is not None else "",
        "rollout_temperature": float(args.rollout_temperature),
        "rollout_client_timeout_seconds": float(args.rollout_client_timeout_seconds),
        "rollout_client_retry_wait_seconds": list(args.rollout_client_retry_wait_seconds),
        "rollout_disable_response_cache": bool(
            args.rollout_disable_response_cache
        ),
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
        "06b_query_vector_audit": [],
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
    train_results_jsonl = train_results.with_suffix(".jsonl")
    train_results_ledger_manifest = train_results_jsonl.with_suffix(
        train_results_jsonl.suffix + ".manifest.json"
    )
    train_collect_cmd = [
        python_executable, "run_spreadsheetbench.py",
        "--data_path", args.data_path,
        "--output_dir", str(train_out),
        "--agent", "cli_only",
        "--model", args.model,
        "--llm_client", args.rollout_llm_client,
        "--temperature", str(args.rollout_temperature),
        "--generation_config", str(train_gen_config_path),
        "--llm_timeout_seconds", str(args.rollout_client_timeout_seconds),
        "--llm_retry_wait_seconds", ",".join(str(value) for value in args.rollout_client_retry_wait_seconds),
        "--num_random_seeds", str(args.rollout_num_random_seeds),
        "--repeat", str(args.rollout_repeat),
        "--max_turns", str(args.max_turns),
        "--start_idx", str(args.train_start),
        "--end_idx", str(args.train_end),
        "--workers", str(args.workers),
        "--results_file", str(train_results),
        "--log_dir", str(train_logs),
        "--log_format", "markdown",
    ]
    if args.rollout_disable_response_cache:
        train_collect_cmd.append("--disable_response_cache")
    if args.rollout_missing_only:
        train_collect_cmd.append("--missing_only")
    if not skip_train_stages:
        run_stage(
            "01_train_collect",
            train_collect_cmd,
            cwd=repo,
            env={**env, "REACT_AGENT_USAGE_LOG": str(usage_logs_by_stage["01_train_collect"][0])},
            log_path=train_stage_logs / "01_train_collect.log",
            marker_dir=train_markers,
            outputs=[
                train_results,
                train_results_jsonl,
                train_results_ledger_manifest,
                train_out,
                train_logs,
            ],
            resume=args.resume,
            clear_outputs_before_run=[
                train_results,
                train_results_jsonl,
                train_results_ledger_manifest,
                train_out,
                train_logs,
                *usage_logs_by_stage["01_train_collect"],
            ],
            preserve_partial_outputs_on_resume=bool(
                args.rollout_missing_only
            ),
            fingerprint=stage_fingerprint(
                "01_train_collect:v3",
                train_collect_cmd,
                dataset=dataset_fp,
                rollout_protocol=rollout_protocol(args, generation_config=train_gen_config_path),
                source={
                    "runner": source_fp["runner"],
                    "run_spreadsheetbench": source_fp["run_spreadsheetbench"],
                    "spreadsheet_agent": source_fp["spreadsheet_agent"],
                    "react_agent": source_fp["react_agent"],
                },
                split=[args.train_start, args.train_end],
            ),
        )
    train_eval = train_artifact_dir / "trace2skill_train_eval.json"
    train_recalc_dir = train_artifact_dir / "trace2skill_train_recalculated_outputs"
    train_eval_cmd = build_evaluation_command(
        python_executable=python_executable,
        data_path=args.data_path,
        output_dir=train_out,
        recalc_dir=train_recalc_dir,
        start_idx=args.train_start,
        end_idx=args.train_end,
        results_file=train_eval,
        evaluator_backend=args.evaluator_backend,
    )
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
            clear_outputs_before_run=[train_eval, train_recalc_dir],
            fingerprint=stage_fingerprint(
                "02_train_eval:v2",
                train_eval_cmd,
                dataset=dataset_fp,
                train_outputs=path_fingerprint(train_out),
                evaluator_identity=evaluator_identity,
                source={
                    "runner": source_fp["runner"],
                    "evaluate_with_official": source_fp["evaluate_with_official"],
                    "spreadsheetbench_support": source_fp["spreadsheetbench_support"],
                },
                split=[args.train_start, args.train_end],
            ),
        )
        validate_evaluation_runtime_identity(
            train_eval,
            evaluator_identity,
        )

    extract_records_cmd = [
        python_executable, "scripts/extract_trace2skill_logs.py",
        "--log-dir", str(train_logs),
        "--results-file", str(train_eval),
        "--output", str(source_records),
    ]
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
                source={
                    "runner": source_fp["runner"],
                    "extract_trace2skill_logs": source_fp["extract_trace2skill_logs"],
                },
            ),
        )

    expected_train_records = int(args.train_end) - int(args.train_start)
    observed_train_records = load_record_count(source_records)
    if observed_train_records != expected_train_records:
        raise RuntimeError(
            f"records.json count mismatch: expected {expected_train_records} "
            f"from train range [{args.train_start}, {args.train_end}), got {observed_train_records}"
        )
    order_manifest = write_dataset_ordered_records(
        source_records=source_records,
        data_path=Path(args.data_path),
        output_path=ordered_records,
        manifest_path=records_order_manifest,
        train_start=int(args.train_start),
        train_end=int(args.train_end),
    )
    if load_record_count(records) != expected_train_records:
        raise RuntimeError(f"ordered_records.json count mismatch after dataset-order rewrite: {records}")
    runtime["records_order_policy"] = order_manifest["policy"]
    runtime["records_source_order_equal_dataset_order"] = bool(order_manifest["source_order_equal_dataset_order"])
    (scenario_dir / "experiment_runtime_config.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    tree_dir = scenario_dir / "dynamix_tree"
    config = {
        "scenario": args.tree_scenario,
        "output_dir": str(tree_dir),
        "records_path": str(records),
        "dataset_path": str(Path(args.data_path).resolve()),
        "train_start": int(args.train_start),
        "train_end": int(args.train_end),
        "enforce_dataset_order": True,
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
            "cache_path": str(embedding_cache_path),
            "cache_write_policy": (
                "first_write_wins"
                if is_certified_single_parent_tree(args.tree_policy)
                else "replace"
            ),
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
        "hierarchy": active_hierarchy_payload({
            "tree_policy": args.tree_policy,
            "otd": {
                "dual_view_lambda": float(args.otd_dual_view_lambda),
                "tie_epsilon": float(args.otd_tie_epsilon),
                "atom_temperature": float(args.otd_atom_temperature),
                "parent_temperature": float(args.otd_parent_temperature),
                "atom_cache_path": args.otd_atom_cache_path,
                "retrieval_token_budget": int(args.otd_retrieval_token_budget),
                "retrieval_token_unit": int(args.otd_retrieval_token_unit),
                "retrieval_exact_search_max_states": int(
                    args.otd_retrieval_exact_search_max_states
                ),
                "validation_mode": "structural_only",
            },
            "ebst": {
                "max_entries": int(args.ebst_max_entries),
                "dual_view_lambda": float(args.ebst_dual_view_lambda),
                "atom_temperature": float(args.ebst_atom_temperature),
                "capsule_temperature": float(
                    args.ebst_capsule_temperature
                ),
                "validator_temperature": float(
                    args.ebst_validator_temperature
                ),
                "atom_cache_path": args.ebst_atom_cache_path,
                "retrieval_token_budget": int(
                    args.ebst_retrieval_token_budget
                ),
                "retrieval_token_unit": int(
                    args.ebst_retrieval_token_unit
                ),
                "retrieval_exact_search_max_states": int(
                    args.ebst_retrieval_exact_search_max_states
                ),
                "validation_mode": "structural_only",
            },
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
            "kmeans": {
                "fixed_k": int(args.kmeans_fixed_k),
                "min_k": int(args.kmeans_min_k),
                "num_restarts": int(args.kmeans_num_restarts),
                "max_iter": int(args.kmeans_max_iter),
                "tol": float(args.kmeans_tol),
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
        }),
        "dynamic": active_dynamic_payload({
            "initial_count": int(args.dynamic_initial_count),
            "arrival_count": int(args.dynamic_arrival_count),
            "update_batch_size": int(args.dynamic_update_batch_size),
            "shuffle_seed": None if int(args.dynamic_shuffle_seed) < 0 else int(args.dynamic_shuffle_seed),
            "snapshot_include_embeddings": bool(args.dynamic_snapshot_include_embeddings),
            "resume_from_snapshots": bool(args.dynamic_resume_from_snapshots),
            "max_propagation_rounds": int(args.dynamic_max_propagation_rounds),
        }, tree_policy=args.tree_policy),
        "analyst": {
            "prompt_style": args.analyst_prompt_style,
            "confidence_floor": float(args.analyst_confidence_floor),
            "tokenizer_model": args.analyst_tokenizer or None,
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
            "analysis_bundle_max_chars": None if int(args.analysis_bundle_max_chars) <= 0 else int(args.analysis_bundle_max_chars),
            "analysis_bundle_max_steps": int(args.analysis_bundle_max_steps),
            "analysis_bundle_max_step_chars": int(args.analysis_bundle_max_step_chars),
            "analysis_bundle_max_final_response_chars": int(args.analysis_bundle_max_final_response_chars),
        },
        "max_levels": int(args.max_levels),
        "skill_output_dir_name": args.skill_output_dir_name,
        "skill_export": {
            "min_level": None if int(args.skill_export_min_level) < 0 else int(args.skill_export_min_level),
            "max_level": None if int(args.skill_export_max_level) < 0 else int(args.skill_export_max_level),
        },
    }
    config_path = scenario_dir / "dynamix_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    cdost_control_manifest_path = (
        scenario_dir / "analysis" / "cdost_control_manifest.json"
    )
    ebst_control_manifest_path = (
        scenario_dir / "analysis" / "ebst_control_manifest.json"
    )
    source_cdost_control_manifest: Path | None = None
    source_ebst_control_manifest: Path | None = None
    if args.tree_policy == "certified_dual_view_otd":
        contract = cdost_control_contract(
            args=args,
            config=config,
            records_path=records,
            dataset_fingerprint=dataset_fp,
            generation_config_path=scenario_gen_config_path,
            evaluator_identity=evaluator_identity,
            source_fingerprints=source_fp,
        )
        write_cdost_control_manifest(
            cdost_control_manifest_path,
            contract,
        )
        if args.tree_scenario == "dynamic_update":
            atom_cache_for_control = resolved_optional_path(
                args.otd_atom_cache_path
            )
            if atom_cache_for_control is None:
                raise ValueError(
                    "dynamic CDOST requires a source atom cache"
                )
            source_cdost_control_manifest = (
                atom_cache_for_control.parent.parent
                / "analysis"
                / "cdost_control_manifest.json"
            )
            validate_source_build_output(
                marker_path=(
                    atom_cache_for_control.parent.parent
                    / "stage_markers"
                    / "04_build_tree.done"
                ),
                output_path=source_cdost_control_manifest,
            )
            validate_matching_cdost_control_manifest(
                current_manifest=cdost_control_manifest_path,
                source_manifest=source_cdost_control_manifest,
            )
    if args.tree_policy == "evidence_balanced_skill_tree":
        contract = ebst_control_contract(
            args=args,
            config=config,
            records_path=records,
            dataset_fingerprint=dataset_fp,
            generation_config_path=scenario_gen_config_path,
            evaluator_identity=evaluator_identity,
            source_fingerprints=source_fp,
        )
        baseline_compatibility: dict[str, Any] | None = None
        source_ebst_control_manifest: Path | None = None
        atom_cache_for_control = resolved_optional_path(
            args.ebst_atom_cache_path
        )
        if args.tree_scenario == "static_build":
            if atom_cache_for_control is None:
                raise ValueError(
                    "controlled evidence-balanced static runs require the "
                    "baseline CDOST experience_atoms.json"
                )
            baseline_cdost_manifest = (
                atom_cache_for_control.parent.parent
                / "analysis"
                / "cdost_control_manifest.json"
            )
            validate_source_build_output(
                marker_path=(
                    atom_cache_for_control.parent.parent
                    / "stage_markers"
                    / "04_build_tree.done"
                ),
                output_path=baseline_cdost_manifest,
            )
            baseline_compatibility = validate_ebst_against_cdost_baseline(
                current_contract=contract,
                baseline_manifest=baseline_cdost_manifest,
            )
            contract["baseline_cdost"] = baseline_compatibility
        else:
            if atom_cache_for_control is None:
                raise ValueError(
                    "dynamic evidence-balanced tree requires a source atom "
                    "cache"
                )
            source_ebst_control_manifest = (
                atom_cache_for_control.parent.parent
                / "analysis"
                / "ebst_control_manifest.json"
            )
            validate_source_build_output(
                marker_path=(
                    atom_cache_for_control.parent.parent
                    / "stage_markers"
                    / "04_build_tree.done"
                ),
                output_path=source_ebst_control_manifest,
            )
            source_payload = json.loads(
                source_ebst_control_manifest.read_text(encoding="utf-8")
            )
            if (
                source_payload.get("format")
                != "ebst_control_manifest_v1"
                or source_payload.get("contract_sha256")
                != _canonical_sha256(source_payload.get("contract"))
            ):
                raise ValueError(
                    "invalid source evidence-balanced control manifest"
                )
            baseline_compatibility = dict(
                source_payload["contract"].get("baseline_cdost") or {}
            )
            if not baseline_compatibility.get("compatible"):
                raise ValueError(
                    "source static evidence-balanced run has no validated "
                    "CDOST baseline binding"
                )
            contract["baseline_cdost"] = baseline_compatibility
        write_ebst_control_manifest(
            ebst_control_manifest_path,
            contract,
        )
        write_json_atomic(
            scenario_dir / "analysis" / "ebst_baseline_compatibility.json",
            baseline_compatibility,
        )
        if source_ebst_control_manifest is not None:
            validate_matching_ebst_control_manifest(
                current_manifest=ebst_control_manifest_path,
                source_manifest=source_ebst_control_manifest,
            )
    if reuse_tree_dir is not None:
        if tree_dir.resolve() == reuse_tree_dir.resolve():
            raise RuntimeError("--reuse-tree-dir cannot be the same as this run's output dynamix_tree dir")
        build_tree_cmd = [
            python_executable,
            "scripts/export_dynamix_nodebank.py",
            "--source-tree-dir",
            str(reuse_tree_dir),
            "--output-tree-dir",
            str(tree_dir),
            "--config",
            str(config_path),
        ]
        build_tree_contract = "04_reuse_tree_export_nodebank:v1"
        build_tree_source = {
            "runner": source_fp["runner"],
            "export_dynamix_nodebank": path_fingerprint(repo / "scripts" / "export_dynamix_nodebank.py"),
            "dynamix_core": source_fp["dynamix_core"],
            "dynamix_trace2skill": source_fp["dynamix_trace2skill"],
        }
        reuse_config_path = reuse_tree_dir / "analysis" / "runtime_config.json"
        if not reuse_config_path.is_file():
            reuse_config_path = reuse_tree_dir.parent / "dynamix_config.json"
        reuse_tree_fingerprint = {
            "hierarchy_state": path_fingerprint(reuse_tree_dir / "hierarchy_state.json"),
            "hierarchy_layers": path_fingerprint(reuse_tree_dir / "hierarchy_layers.json"),
            "summary": path_fingerprint(reuse_tree_dir / "summary.json"),
            "runtime_config": path_fingerprint(reuse_config_path),
        }
    else:
        build_tree_cmd = [python_executable, "scripts/build_dynamix_tree.py", "--config", str(config_path)]
        build_tree_contract = "04_build_tree:v2"
        build_tree_source = {
            "runner": source_fp["runner"],
            "build_dynamix_tree": source_fp["build_dynamix_tree"],
            "dynamix_core": source_fp["dynamix_core"],
            "dynamix_trace2skill": source_fp["dynamix_trace2skill"],
        }
        reuse_tree_fingerprint = {"exists": False}
    atom_cache_path = resolved_optional_path(atom_cache_path_for_args(args))
    source_vector_cache_manifest = (
        atom_cache_path.parent / "embedding_vector_cache_manifest.json"
        if atom_cache_path is not None
        else None
    )
    build_tree_fingerprint = {
        "stage_contract": build_tree_contract,
        "cmd": build_tree_cmd,
        "config_sha256": file_sha256(config_path),
        "records_sha256": file_sha256(records),
        "atom_cache": (
            path_fingerprint(atom_cache_path)
            if atom_cache_path is not None
            else {"exists": False}
        ),
        "source_embedding_vector_cache_manifest": (
            path_fingerprint(source_vector_cache_manifest)
            if source_vector_cache_manifest is not None
            else {"exists": False}
        ),
        "cdost_control_manifest": (
            path_fingerprint(cdost_control_manifest_path)
            if args.tree_policy == "certified_dual_view_otd"
            else {"exists": False}
        ),
        "source_cdost_control_manifest": (
            path_fingerprint(source_cdost_control_manifest)
            if source_cdost_control_manifest is not None
            else {"exists": False}
        ),
        "ebst_control_manifest": (
            path_fingerprint(ebst_control_manifest_path)
            if args.tree_policy == "evidence_balanced_skill_tree"
            else {"exists": False}
        ),
        "source_ebst_control_manifest": (
            path_fingerprint(source_ebst_control_manifest)
            if source_ebst_control_manifest is not None
            else {"exists": False}
        ),
        "openai_api_key": api_key_fingerprint(args.openai_api_key),
        "tree_scenario": args.tree_scenario,
        "reuse_tree_dir": str(reuse_tree_dir) if reuse_tree_dir is not None else "",
        "reuse_tree_state": reuse_tree_fingerprint,
        "source": build_tree_source,
    }
    build_tree_usage_env = {
        **env,
        "DYNAMIX_GENERATION_USAGE_LOG": str(usage_logs_by_stage["04_build_tree"][0]),
        "DYNAMIX_EMBEDDING_USAGE_LOG": str(usage_logs_by_stage["04_build_tree"][1]),
        "DYNAMIX_SKILLBANK_USAGE_LOG": str(usage_logs_by_stage["04_build_tree"][2]),
    }
    build_outputs = [
        tree_dir / "summary.json",
        tree_dir / args.skill_output_dir_name / "node_bank_manifest.json",
        tree_dir / args.skill_output_dir_name / ".dynamix_skillbank_index.json",
    ]
    if args.tree_policy == "certified_dual_view_otd":
        build_outputs.extend(
            [
                tree_dir / "experience_atoms.json",
                tree_dir / "otd_tree_state.json",
                tree_dir / "otd_tree_structure.json",
                tree_dir / "otd_insertions.jsonl",
                tree_dir / "otd_parent_updates.jsonl",
                tree_dir / "otd_structural_diagnostics.json",
                tree_dir / "otd_observed_beta_separation.json",
                tree_dir / "analysis" / "runtime_config.json",
                tree_dir / "analysis" / "run_manifest.json",
                tree_dir / "embedding_vector_cache_manifest.json",
                cdost_control_manifest_path,
            ]
        )
    if args.tree_policy == "evidence_balanced_skill_tree":
        build_outputs.extend(
            [
                tree_dir / "experience_atoms.json",
                tree_dir / "balanced_tree_state.json",
                tree_dir / "balanced_tree_insertions.jsonl",
                tree_dir / "skill_capsules.json",
                tree_dir / "skill_capsules.jsonl",
                tree_dir / "skill_lifecycle_events.jsonl",
                tree_dir / "tree_quality_audit.json",
                tree_dir / "analysis" / "runtime_config.json",
                tree_dir / "analysis" / "run_manifest.json",
                tree_dir / "embedding_vector_cache_manifest.json",
                ebst_control_manifest_path,
                scenario_dir
                / "analysis"
                / "ebst_baseline_compatibility.json",
            ]
        )
    run_stage(
        "04_build_tree",
        build_tree_cmd,
        cwd=repo,
        env=build_tree_usage_env,
        log_path=logs / "04_build_tree.log",
        marker_dir=markers,
        outputs=build_outputs,
        resume=args.resume,
        clear_outputs_before_run=[
            tree_dir,
            *usage_logs_by_stage["04_build_tree"],
        ],
        fingerprint=build_tree_fingerprint,
    )
    if is_certified_single_parent_tree(args.tree_policy):
        validate_cdost_vector_cache(
            cache_path=embedding_cache_path,
            manifest_path=(
                tree_dir / "embedding_vector_cache_manifest.json"
            ),
        )

    summary = json.loads((tree_dir / "summary.json").read_text(encoding="utf-8"))
    validate_tree_summary_for_heldout(summary, args)
    manifest = json.loads(Path(summary["node_bank_manifest"]).read_text(encoding="utf-8"))
    validate_nodebank_manifest_for_heldout(manifest, args)
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
    env["DYNAMIX_SKILLBANK_EMBED_MAX_MODEL_LEN"] = str(
        int(args.embedding_max_model_len)
    )
    env["DYNAMIX_SKILLBANK_EMBED_MAX_INPUT_TOKENS"] = str(
        int(args.embedding_max_input_tokens)
    )
    env["DYNAMIX_SKILLBANK_EMBED_BATCH_SIZE"] = str(
        int(args.embedding_batch_size)
    )
    env["DYNAMIX_SKILLBANK_EMBED_TOKENIZER"] = str(
        args.embedding_tokenizer or ""
    )
    env["DYNAMIX_SKILLBANK_CHUNK_TOKENS"] = (
        ""
        if is_certified_single_parent_tree(args.tree_policy)
        or not args.chunked_embedding_enabled
        else str(args.chunked_embedding_chunk_tokens)
    )
    env["DYNAMIX_SKILLBANK_CHUNK_OVERLAP_TOKENS"] = (
        ""
        if is_certified_single_parent_tree(args.tree_policy)
        or not args.chunked_embedding_enabled
        else str(args.chunked_embedding_overlap_tokens)
    )
    env["DYNAMIX_SKILLBANK_REQUIRE_CACHE_MATCH"] = (
        "true"
        if is_certified_single_parent_tree(args.tree_policy)
        else "false"
    )
    if is_certified_single_parent_tree(args.tree_policy):
        env["DYNAMIX_SKILLBANK_EXPECT_TREE_POLICY"] = args.tree_policy
    skillbank_cache_path = Path(summary.get("skillbank_index") or (skillbank_root / ".dynamix_skillbank_index.json"))
    if not skillbank_cache_path.is_file():
        raise RuntimeError(f"DynaMix skillbank index missing before heldout: {skillbank_cache_path}")
    env["DYNAMIX_SKILLBANK_CACHE_PATH"] = str(skillbank_cache_path)
    if is_certified_single_parent_tree(args.tree_policy):
        env["DYNAMIX_SKILLBANK_VECTOR_CACHE_PATH"] = str(
            embedding_cache_path
        )
    else:
        env.pop("DYNAMIX_SKILLBANK_VECTOR_CACHE_PATH", None)
    env["DYNAMIX_SKILLBANK_REQUIRE_VECTOR_CACHE_MATCH"] = (
        "true"
        if is_certified_single_parent_tree(args.tree_policy)
        and args.tree_scenario == "dynamic_update"
        else "false"
    )
    selection_log = scenario_dir / "raw" / "skill_selection_records.jsonl"
    selection_log.parent.mkdir(parents=True, exist_ok=True)
    env["DYNAMIX_SKILL_SELECTION_LOG"] = str(selection_log)
    query_vector_manifest = (
        scenario_dir
        / "raw"
        / "heldout_query_embedding_cache_manifest.json"
    )
    source_query_vector_manifest: Path | None = None
    if (
        is_certified_single_parent_tree(args.tree_policy)
        and args.tree_scenario == "dynamic_update"
    ):
        if atom_cache_path is None:
            raise ValueError(
                f"dynamic {args.tree_policy} requires a source atom cache"
            )
        source_query_vector_manifest = (
            atom_cache_path.parent.parent
            / "raw"
            / "heldout_query_embedding_cache_manifest.json"
        )
        validate_source_build_output(
            marker_path=(
                atom_cache_path.parent.parent
                / "stage_markers"
                / "06b_query_vector_audit.done"
            ),
            output_path=source_query_vector_manifest,
        )
        validate_cdost_vector_cache(
            cache_path=embedding_cache_path,
            manifest_path=source_query_vector_manifest,
        )

    heldout_out = scenario_dir / "trace2skill_heldout_outputs"
    heldout_logs = scenario_dir / "trace2skill_heldout_logs"
    heldout_results = scenario_dir / "trace2skill_heldout_results.json"
    heldout_results_jsonl = heldout_results.with_suffix(".jsonl")
    heldout_results_ledger_manifest = heldout_results_jsonl.with_suffix(
        heldout_results_jsonl.suffix + ".manifest.json"
    )
    heldout_collect_cmd = [
        python_executable, "run_spreadsheetbench.py",
        "--data_path", args.data_path,
        "--output_dir", str(heldout_out),
        "--agent", "cli_skill_preloaded",
        "--skills_dir", str(skills_root),
        "--model", args.model,
        "--llm_client", args.rollout_llm_client,
        "--temperature", str(args.rollout_temperature),
        "--generation_config", str(scenario_gen_config_path),
        "--llm_timeout_seconds", str(args.rollout_client_timeout_seconds),
        "--llm_retry_wait_seconds", ",".join(str(value) for value in args.rollout_client_retry_wait_seconds),
        "--num_random_seeds", str(args.rollout_num_random_seeds),
        "--repeat", str(args.rollout_repeat),
        "--max_turns", str(args.max_turns),
        "--start_idx", str(args.heldout_start),
        "--end_idx", str(args.heldout_end),
        "--workers", str(args.workers),
        "--results_file", str(heldout_results),
        "--log_dir", str(heldout_logs),
        "--log_format", "markdown",
    ]
    if args.rollout_disable_response_cache:
        heldout_collect_cmd.append("--disable_response_cache")
    if args.rollout_missing_only:
        heldout_collect_cmd.append("--missing_only")
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
        outputs=[
            heldout_results,
            heldout_results_jsonl,
            heldout_results_ledger_manifest,
            selection_log,
            heldout_out,
            heldout_logs,
        ],
        resume=args.resume,
        clear_outputs_before_run=[
            heldout_results,
            heldout_results_jsonl,
            heldout_results_ledger_manifest,
            selection_log,
            heldout_out,
            heldout_logs,
            *usage_logs_by_stage["06_heldout_collect"],
        ],
        preserve_partial_outputs_on_resume=bool(
            args.rollout_missing_only
        ),
        fingerprint=stage_fingerprint(
            "06_heldout_collect:v3",
            heldout_collect_cmd,
            dataset=dataset_fp,
            generation_config=path_fingerprint(scenario_gen_config_path),
            tree_summary=path_fingerprint(tree_dir / "summary.json"),
            node_bank_manifest=path_fingerprint(Path(summary["node_bank_manifest"])),
            embedding_vector_cache_manifest=path_fingerprint(
                tree_dir / "embedding_vector_cache_manifest.json"
            ),
            source_query_vector_manifest=(
                path_fingerprint(source_query_vector_manifest)
                if source_query_vector_manifest is not None
                else {"exists": False}
            ),
            skillbank_root=path_fingerprint(skillbank_root),
            rollout_protocol=rollout_protocol(args, generation_config=scenario_gen_config_path),
            skillbank_retrieval_protocol=skillbank_retrieval_protocol(
                args,
                cache_path=skillbank_cache_path,
                vector_cache_path=embedding_cache_path,
                selection_log=selection_log,
            ),
            source={
                "runner": source_fp["runner"],
                "run_spreadsheetbench": source_fp["run_spreadsheetbench"],
                "spreadsheet_agent": source_fp["spreadsheet_agent"],
                "react_agent": source_fp["react_agent"],
                "dynamix_trace2skill": source_fp["dynamix_trace2skill"],
            },
            split=[args.heldout_start, args.heldout_end],
        ),
    )

    if is_certified_single_parent_tree(args.tree_policy):
        query_audit_cmd = [
            python_executable,
            "scripts/audit_cdost_query_vector_cache.py",
            "--selection-log",
            str(selection_log),
            "--cache-path",
            str(embedding_cache_path),
            "--output-path",
            str(query_vector_manifest),
        ]
        if source_query_vector_manifest is not None:
            query_audit_cmd.extend(
                [
                    "--reference-manifest",
                    str(source_query_vector_manifest),
                ]
            )
        run_stage(
            "06b_query_vector_audit",
            query_audit_cmd,
            cwd=repo,
            env=env,
            log_path=logs / "06b_query_vector_audit.log",
            marker_dir=markers,
            outputs=[query_vector_manifest],
            resume=args.resume,
            clear_outputs_before_run=[query_vector_manifest],
            fingerprint=stage_fingerprint(
                "06b_query_vector_audit:v1",
                query_audit_cmd,
                selection_log=path_fingerprint(selection_log),
                build_vector_manifest=path_fingerprint(
                    tree_dir / "embedding_vector_cache_manifest.json"
                ),
                source_query_vector_manifest=(
                    path_fingerprint(source_query_vector_manifest)
                    if source_query_vector_manifest is not None
                    else {"exists": False}
                ),
                source={
                    "runner": source_fp["runner"],
                    "audit_cdost_query_vector_cache": source_fp[
                        "audit_cdost_query_vector_cache"
                    ],
                    "dynamix_trace2skill": source_fp[
                        "dynamix_trace2skill"
                    ],
                },
            ),
        )
        validate_cdost_vector_cache(
            cache_path=embedding_cache_path,
            manifest_path=query_vector_manifest,
        )

    heldout_eval = scenario_dir / "trace2skill_heldout_eval.json"
    heldout_recalc_dir = scenario_dir / "trace2skill_heldout_recalculated_outputs"
    heldout_eval_cmd = build_evaluation_command(
        python_executable=python_executable,
        data_path=args.data_path,
        output_dir=heldout_out,
        recalc_dir=heldout_recalc_dir,
        start_idx=args.heldout_start,
        end_idx=args.heldout_end,
        results_file=heldout_eval,
        evaluator_backend=args.evaluator_backend,
    )
    run_stage(
        "07_heldout_eval",
        heldout_eval_cmd,
        cwd=repo,
        env=env,
        log_path=logs / "07_heldout_eval.log",
        marker_dir=markers,
        outputs=[heldout_eval],
        resume=args.resume,
        clear_outputs_before_run=[heldout_eval, heldout_recalc_dir],
        fingerprint=stage_fingerprint(
            "07_heldout_eval:v2",
            heldout_eval_cmd,
            dataset=dataset_fp,
            heldout_outputs=path_fingerprint(heldout_out),
            heldout_results=path_fingerprint(heldout_results),
            query_vector_manifest=(
                path_fingerprint(query_vector_manifest)
                if is_certified_single_parent_tree(args.tree_policy)
                else {"exists": False}
            ),
            evaluator_identity=evaluator_identity,
            source={
                "runner": source_fp["runner"],
                "evaluate_with_official": source_fp["evaluate_with_official"],
                "spreadsheetbench_support": source_fp["spreadsheetbench_support"],
            },
            split=[args.heldout_start, args.heldout_end],
        ),
    )
    validate_heldout_eval_coverage(heldout_eval, split_manifest)
    validate_evaluation_runtime_identity(
        heldout_eval,
        evaluator_identity,
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
        "heldout_query_embedding_cache_manifest": (
            str(query_vector_manifest)
            if is_certified_single_parent_tree(args.tree_policy)
            else None
        ),
        "heldout_eval": str(heldout_eval),
    }
    scenario_stages = ["04_build_tree", "06_heldout_collect"]
    if is_certified_single_parent_tree(args.tree_policy):
        scenario_stages.append("06b_query_vector_audit")
    scenario_stages.append("07_heldout_eval")
    stage_report = write_experiment_stage_report(
        run_dir=scenario_dir,
        marker_dir=markers,
        stages=(
            scenario_stages
            if skip_train_stages
            else [
                "01_train_collect",
                "02_train_eval",
                "03_extract_records",
                *scenario_stages,
            ]
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
            "06b_query_vector_audit": markers,
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
