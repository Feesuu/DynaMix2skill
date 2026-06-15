#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


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


def stage_done(marker: Path, outputs: Iterable[Path]) -> bool:
    if not marker.exists():
        return False
    return all(path.exists() for path in outputs)


def run_stage(name: str, cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, marker_dir: Path, outputs: list[Path], resume: bool) -> None:
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{name}.done"
    if resume and stage_done(marker, outputs):
        print(f"[resume] skip stage {name}", flush=True)
        return
    (marker_dir / f"{name}.running").write_text(json.dumps({"cmd": cmd}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        run(cmd, cwd=cwd, env=env, log_path=log_path)
    except Exception as exc:
        fail = marker_dir / f"{name}.failed.json"
        fail.write_text(json.dumps({"stage": name, "cmd": cmd, "error": repr(exc), "log": str(log_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    marker.write_text(json.dumps({"stage": name, "outputs": [str(p) for p in outputs]}, ensure_ascii=False, indent=2), encoding="utf-8")
    running = marker_dir / f"{name}.running"
    if running.exists():
        running.unlink()


def write_generation_config(path: Path, *, thinking: bool | None, temperature: float = 0.0) -> None:
    payload: dict = {"temperature": temperature}
    if thinking is not None:
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=200)
    parser.add_argument("--heldout-start", type=int, default=200)
    parser.add_argument("--heldout-end", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=os.environ.get("GEN_MODEL", "Qwen3.5-9B"))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18002/v1"))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:18000/v1"))
    parser.add_argument("--embedding-model", default=os.environ.get("EMBED_MODEL", "Qwen3-Embedding-8B"))
    parser.add_argument("--embedding-tokenizer", default=os.environ.get("EMBED_TOKENIZER", "Qwen3-Embedding-8B"))
    parser.add_argument("--python-executable", default=os.environ.get("DYNAMIX_PYTHON", sys.executable), help="Python executable used for all experiment stages; its bin dir is prepended to PATH so agent bash actions can call bare python")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--thinking", choices=["true", "false", "null"], default="true", help="Unified Qwen thinking setting passed to Trace2Skill rollout and DynaMix analyst")
    parser.add_argument("--skillbank-top-k", type=int, default=10, help="Select top-k DynaMix nodebank nodes by embedding before each heldout task")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    thinking = None if args.thinking == "null" else args.thinking == "true"
    python_executable = resolve_python_executable(args.python_executable)
    repo = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logs = run_dir / "logs"
    markers = run_dir / "stage_markers"
    logs.mkdir(parents=True, exist_ok=True)
    markers.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + str(repo)
    env["PATH"] = str(Path(python_executable).parent) + os.pathsep + env.get("PATH", "")
    env["DYNAMIX_PYTHON"] = python_executable
    env["OPENAI_API_KEY"] = args.openai_api_key
    env["OPENAI_BASE_URL"] = args.openai_base_url

    gen_config_path = run_dir / "trace2skill_generation_config.json"
    write_generation_config(gen_config_path, thinking=thinking, temperature=0.0)
    split_manifest = write_split_manifest(Path(args.data_path), run_dir, train_start=args.train_start, train_end=args.train_end, heldout_start=args.heldout_start, heldout_end=args.heldout_end)

    runtime = {
        "data_path": str(Path(args.data_path).resolve()),
        "run_dir": str(run_dir),
        "model": args.model,
        "openai_base_url": args.openai_base_url,
        "embedding_base_url": args.embedding_base_url,
        "embedding_model": args.embedding_model,
        "embedding_tokenizer": args.embedding_tokenizer,
        "train_range": [args.train_start, args.train_end],
        "heldout_range": [args.heldout_start, args.heldout_end],
        "split_manifest": str(run_dir / "split_manifest.json"),
        "workers": args.workers,
        "python_executable": python_executable,
        "max_turns": args.max_turns,
        "thinking": args.thinking,
        "trace2skill_generation_config": str(gen_config_path),
        "skillbank_top_k": int(args.skillbank_top_k),
        "resume": bool(args.resume),
    }
    (run_dir / "experiment_runtime_config.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    train_out = run_dir / "trace2skill_train_outputs"
    train_logs = run_dir / "trace2skill_train_logs"
    train_results = run_dir / "trace2skill_train_results.json"
    run_stage("01_train_collect", [
        python_executable, "run_spreadsheetbench.py",
        "--data_path", args.data_path,
        "--output_dir", str(train_out),
        "--agent", "cli_only",
        "--model", args.model,
        "--generation_config", str(gen_config_path),
        "--max_turns", str(args.max_turns),
        "--start_idx", str(args.train_start),
        "--end_idx", str(args.train_end),
        "--workers", str(args.workers),
        "--results_file", str(train_results),
        "--log_dir", str(train_logs),
        "--log_format", "markdown",
    ], cwd=repo, env=env, log_path=logs / "01_train_collect.log", marker_dir=markers, outputs=[train_results], resume=args.resume)

    train_eval = run_dir / "trace2skill_train_eval.json"
    run_stage("02_train_eval", [
        python_executable, "evaluate_with_official.py",
        "--data_path", args.data_path,
        "--output_dir", str(train_out),
        "--start_idx", str(args.train_start),
        "--end_idx", str(args.train_end),
        "--results_file", str(train_eval),
    ], cwd=repo, env=env, log_path=logs / "02_train_eval.log", marker_dir=markers, outputs=[train_eval], resume=args.resume)

    records = run_dir / "records.json"
    run_stage("03_extract_records", [
        python_executable, "scripts/extract_trace2skill_logs.py",
        "--log-dir", str(train_logs),
        "--results-file", str(train_eval),
        "--output", str(records),
    ], cwd=repo, env=env, log_path=logs / "03_extract_records.log", marker_dir=markers, outputs=[records], resume=args.resume)

    tree_dir = run_dir / "dynamix_tree"
    config = {
        "scenario": "static_build",
        "output_dir": str(tree_dir),
        "records_path": str(records),
        "generation": {
            "base_url": args.openai_base_url,
            "model": args.model,
            "api_key": args.openai_api_key,
            "temperature": 0.6,
            "timeout_seconds": 600,
            "max_concurrency": args.workers,
            "thinking_mode": thinking,
            "extra_body": ({"chat_template_kwargs": {"enable_thinking": bool(thinking)}} if thinking is not None else {}),
        },
        "embedding": {
            "base_url": args.embedding_base_url,
            "model": args.embedding_model,
            "api_key": "EMPTY",
            "max_model_len": 32000,
            "max_input_tokens": 32000,
            "truncate_long_texts": True,
            "tokenizer_model": args.embedding_tokenizer,
            "tokenizer_required": True,
            "truncation_strategy": "head",
            "batch_size": 8,
            "max_concurrency": args.workers,
            "cache_path": str(run_dir / "cache" / "embedding_cache.sqlite"),
        },
        "hierarchy": {
            "gmm_bic": {
                "min_split_size": 8,
            },
        },
        "analyst": {
            "prompt_style": "trace2skill_cluster_level_template_inheritance_v4",
            "confidence_floor": 0.05,
            "tokenizer_model": args.embedding_tokenizer,
            "tokenizer_required": True,
            "max_prompt_tokens": None,
        },
    }
    config_path = run_dir / "dynamix_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    run_stage("04_build_tree", [python_executable, "scripts/build_dynamix_tree.py", "--config", str(config_path)], cwd=repo, env=env, log_path=logs / "04_build_tree.log", marker_dir=markers, outputs=[tree_dir / "summary.json"], resume=args.resume)

    summary = json.loads((tree_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path(summary["node_bank_manifest"]).read_text(encoding="utf-8"))
    if int(manifest.get("node_count", 0)) <= 0:
        raise RuntimeError("DynaMix produced no retrievable nodebank nodes")
    skillbank_root = Path(manifest.get("output_dir") or Path(summary["node_bank_manifest"]).parent)
    skills_root = run_dir / "trace2skill_empty_skills_dir"
    skills_root.mkdir(parents=True, exist_ok=True)
    # Enable per-task top-k nodebank selection during heldout.  The agent injects
    # the selected node snippets directly into the usual preloaded-skill slot.
    env["DYNAMIX_SKILLBANK_ROOT"] = str(skillbank_root)
    env["DYNAMIX_SKILLBANK_TOP_K"] = str(max(0, int(args.skillbank_top_k)))
    env["DYNAMIX_SKILLBANK_EMBED_BASE_URL"] = args.embedding_base_url
    env["DYNAMIX_SKILLBANK_EMBED_MODEL"] = args.embedding_model
    env["DYNAMIX_SKILLBANK_EMBED_API_KEY"] = "EMPTY"
    env["DYNAMIX_SKILLBANK_CACHE_PATH"] = str(run_dir / "cache" / "skillbank_index.json")
    selection_log = run_dir / "raw" / "skill_selection_records.jsonl"
    selection_log.parent.mkdir(parents=True, exist_ok=True)
    env["DYNAMIX_SKILL_SELECTION_LOG"] = str(selection_log)

    heldout_out = run_dir / "trace2skill_heldout_outputs"
    heldout_logs = run_dir / "trace2skill_heldout_logs"
    heldout_results = run_dir / "trace2skill_heldout_results.json"
    run_stage("06_heldout_collect", [
        python_executable, "run_spreadsheetbench.py",
        "--data_path", args.data_path,
        "--output_dir", str(heldout_out),
        "--agent", "cli_skill_preloaded",
        "--skills_dir", str(skills_root),
        "--model", args.model,
        "--generation_config", str(gen_config_path),
        "--max_turns", str(args.max_turns),
        "--start_idx", str(args.heldout_start),
        "--end_idx", str(args.heldout_end),
        "--workers", str(args.workers),
        "--results_file", str(heldout_results),
        "--log_dir", str(heldout_logs),
        "--log_format", "markdown",
    ], cwd=repo, env=env, log_path=logs / "06_heldout_collect.log", marker_dir=markers, outputs=[heldout_results], resume=args.resume)

    heldout_eval = run_dir / "trace2skill_heldout_eval.json"
    run_stage("07_heldout_eval", [
        python_executable, "evaluate_with_official.py",
        "--data_path", args.data_path,
        "--output_dir", str(heldout_out),
        "--start_idx", str(args.heldout_start),
        "--end_idx", str(args.heldout_end),
        "--results_file", str(heldout_eval),
    ], cwd=repo, env=env, log_path=logs / "07_heldout_eval.log", marker_dir=markers, outputs=[heldout_eval], resume=args.resume)

    final = {
        **runtime,
        "records_path": str(records),
        "tree_summary": str(tree_dir / "summary.json"),
        "skillbank_root": str(skillbank_root),
        "node_bank_manifest": str(summary["node_bank_manifest"]),
        "skills_root": str(skills_root),
        "skillbank_top_k": int(args.skillbank_top_k),
        "skillbank_index": str(run_dir / "cache" / "skillbank_index.json"),
        "skill_selection_records": str(selection_log),
        "heldout_eval": str(heldout_eval),
    }
    (run_dir / "experiment_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
