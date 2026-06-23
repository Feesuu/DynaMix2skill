#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamix_benchmarks.officeqa import (  # noqa: E402
    OfficeQAInfrastructureError,
    OfficeQAOfficialRewardError,
    load_officeqa_items,
    record_from_officeqa_result,
    resolve_reward_path,
    run_officeqa_item,
)


SKILLOPT_QWEN_DEFAULT_TEMPERATURE = 0.7
SKILLOPT_QWEN_DEFAULT_ENABLE_THINKING = False


def parse_float_csv(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return tuple(float(part) for part in parts)


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_generation_config(value: str | None) -> dict:
    if not value:
        return {}
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--generation_config must be a JSON object or a path to one")
    return parsed


def apply_skillopt_qwen_generation_defaults(generation_config: dict, *, max_completion_tokens: int) -> dict:
    generation_config.setdefault("max_tokens", int(max_completion_tokens))
    generation_config.setdefault("temperature", SKILLOPT_QWEN_DEFAULT_TEMPERATURE)
    extra_body = generation_config.setdefault("extra_body", {})
    if isinstance(extra_body, dict):
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        if isinstance(chat_template_kwargs, dict):
            chat_template_kwargs.setdefault("enable_thinking", SKILLOPT_QWEN_DEFAULT_ENABLE_THINKING)
    return generation_config


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def api_key_fingerprint(value: str) -> str:
    if value == "EMPTY":
        return "EMPTY"
    if not value:
        return ""
    return f"sha256:{text_sha256(value)}"


def path_identity(path: str | Path | None) -> dict:
    if path is None:
        return {"exists": False}
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return {"exists": False, "path": str(candidate)}
    stat = candidate.stat()
    return {
        "exists": True,
        "path": str(candidate.resolve()),
        "kind": "dir" if candidate.is_dir() else "file",
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _update_digest_with_file(digest: "hashlib._Hash", root: Path, child: Path) -> None:
    rel = child.relative_to(root).as_posix()
    stat = child.stat()
    digest.update(rel.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(int(stat.st_size)).encode("ascii"))
    digest.update(b"\0")
    with child.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\0")


def path_tree_identity(path: str | Path | None) -> dict:
    base = path_identity(path)
    if not base.get("exists"):
        return base
    if base.get("kind") == "file":
        file_path = Path(str(base["path"]))
        digest = hashlib.sha256()
        _update_digest_with_file(digest, file_path.parent, file_path)
        return {**base, "content_sha256": digest.hexdigest()}
    if base.get("kind") != "dir":
        return base
    root = Path(str(base["path"]))
    digest = hashlib.sha256()
    file_count = 0
    for child in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        if child.suffix == ".pyc" or "__pycache__" in child.parts:
            continue
        _update_digest_with_file(digest, root, child)
        file_count += 1
    return {**base, "file_count": file_count, "tree_content_sha256": digest.hexdigest()}


def source_protocol_identity() -> dict:
    return {
        "run_officeqa_benchmark": path_tree_identity(Path(__file__)),
        "evaluate_officeqa_results": path_tree_identity(ROOT / "scripts" / "evaluate_officeqa_results.py"),
        "extract_officeqa_records": path_tree_identity(ROOT / "scripts" / "extract_officeqa_records.py"),
        "officeqa_module": path_tree_identity(ROOT / "src" / "dynamix_benchmarks" / "officeqa.py"),
        "adapters_module": path_tree_identity(ROOT / "src" / "dynamix_benchmarks" / "adapters.py"),
        "react_agent": path_tree_identity(ROOT / "src" / "react_agent"),
    }


def officeqa_context_variant(use_oracle_context: bool) -> str:
    return "skillopt_oracle_source_pages" if use_oracle_context else "local_docs_without_oracle_context"


def build_item_results_fingerprint(args: argparse.Namespace, docs_dirs: list[str], generation_config: dict, reward_path: Path | None) -> dict:
    return {
        "contract": "dynamix_officeqa_item_results_jsonl_v1",
        "split_dir": str(Path(args.split_dir).expanduser().resolve()),
        "split": args.split,
        "range": [int(args.start_idx), None if args.end_idx is None else int(args.end_idx)],
        "docs_dirs": [path_tree_identity(path) for path in docs_dirs],
        "model": args.model,
        "openai_base_url": args.openai_base_url,
        "openai_api_key": api_key_fingerprint(args.openai_api_key),
        "generation_config": generation_config,
        "max_turns": int(args.max_turns),
        "evaluator": args.evaluator,
        "reward_path": path_identity(reward_path) if reward_path is not None else {"exists": False},
        "reward_tolerance": float(args.reward_tolerance),
        "allow_fallback_evaluator": bool(args.allow_fallback_evaluator),
        "continue_on_infra_error": bool(args.continue_on_infra_error),
        "skillbank_root": path_tree_identity(args.skillbank_root) if args.skillbank_root else {"exists": False},
        "skillbank_top_k": int(args.skillbank_top_k),
        "use_oracle_context": bool(args.use_oracle_context),
        "officeqa_context_variant": officeqa_context_variant(bool(args.use_oracle_context)),
        "source_protocol": source_protocol_identity(),
    }


def rotate_stale_item_results(jsonl_path: Path, manifest_path: Path, *, reason: str, selection_log_path: Path | None = None) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = [jsonl_path, manifest_path]
    if selection_log_path is not None:
        paths.append(selection_log_path)
    for path in paths:
        if not path.exists():
            continue
        stale = path.with_name(f"{path.name}.stale_{timestamp}")
        suffix = 0
        while stale.exists():
            suffix += 1
            stale = path.with_name(f"{path.name}.stale_{timestamp}_{suffix}")
        path.rename(stale)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"rotated_at": timestamp, "reason": reason}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def selection_log_task_ids(path: Path) -> set[str]:
    task_ids: set[str] = set()
    if not path.is_file():
        return task_ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = str(row.get("task_id") or row.get("id") or "")
            if task_id:
                task_ids.add(task_id)
    return task_ids


def selection_log_covers(path: Path, expected_task_ids: set[str]) -> bool:
    return expected_task_ids.issubset(selection_log_task_ids(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DynaMix OfficeQA ReAct rollouts.")
    parser.add_argument("--split-dir", required=True, help="Materialized OfficeQA split directory")
    parser.add_argument("--docs-dir", action="append", default=[], help="OfficeQA docs root. May be repeated. Defaults to OFFICEQA_DOCS_DIR.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--item-results-jsonl", default="", help="Per-item append/resume log. Defaults to <results_file>.jsonl.")
    parser.add_argument("--resume-item-results", type=parse_bool, default=True)
    parser.add_argument("--records_file", default="")
    parser.add_argument("--log_dir", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18002/v1"))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--generation_config", default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=16384, help="SkillOpt rollout completion-token cap; sent to this OpenAI-compatible/vLLM client as max_tokens.")
    parser.add_argument("--max_turns", type=int, default=24)
    parser.add_argument("--llm_timeout_seconds", type=float, default=1200.0)
    parser.add_argument("--llm_retry_wait_seconds", type=parse_float_csv, default=(5.0, 10.0, 30.0))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--evaluator", choices=["skillopt", "official_reward", "fallback"], default="skillopt")
    parser.add_argument("--reward-path", default="")
    parser.add_argument("--reward-tolerance", type=float, default=0.0)
    parser.add_argument("--allow-fallback-evaluator", action="store_true", help="Debug only: allow simplified numeric/text matching when official reward.py is unavailable.")
    parser.add_argument("--continue-on-infra-error", action="store_true", help="Debug only: convert config/API/doc exceptions into failed rows instead of failing the stage.")
    parser.add_argument("--skillbank-root", default="")
    parser.add_argument("--skillbank-top-k", type=int, default=0)
    parser.add_argument("--selection-log", default="")
    parser.add_argument("--use-oracle-context", type=parse_bool, default=True, help="SkillOpt-style oracle source-page context variant; set false for local-docs-only diagnostic runs.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    docs_dirs = args.docs_dir or [part for part in os.environ.get("OFFICEQA_DOCS_DIR", "").split(os.pathsep) if part]
    if not docs_dirs:
        raise FileNotFoundError("--docs-dir or OFFICEQA_DOCS_DIR is required for OfficeQA local-document rollout")
    if not any(Path(value).expanduser().is_dir() for value in docs_dirs):
        raise FileNotFoundError(f"No OfficeQA docs directory exists: {docs_dirs}")
    items = load_officeqa_items(args.split_dir, split=args.split, start=args.start_idx, end=args.end_idx)
    generation_config = parse_generation_config(args.generation_config)
    apply_skillopt_qwen_generation_defaults(generation_config, max_completion_tokens=int(args.max_completion_tokens))
    reward_path = (
        resolve_reward_path(args.split_dir, args.reward_path or None, required=args.evaluator == "official_reward")
        if args.evaluator == "official_reward" or args.reward_path
        else None
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    selection_log = Path(args.selection_log) if args.selection_log else None
    if selection_log is not None:
        selection_log.parent.mkdir(parents=True, exist_ok=True)
    item_results_jsonl = Path(args.item_results_jsonl) if args.item_results_jsonl else Path(args.results_file).with_suffix(".jsonl")
    item_manifest = item_results_jsonl.with_suffix(item_results_jsonl.suffix + ".manifest.json")
    item_fingerprint = build_item_results_fingerprint(args, docs_dirs, generation_config, reward_path)
    if item_results_jsonl.exists():
        if not args.resume_item_results:
            rotate_stale_item_results(item_results_jsonl, item_manifest, reason="resume disabled for this run", selection_log_path=selection_log)
        elif not item_manifest.is_file():
            rotate_stale_item_results(item_results_jsonl, item_manifest, reason="missing item-results manifest", selection_log_path=selection_log)
        else:
            try:
                prior_manifest = json.loads(item_manifest.read_text(encoding="utf-8"))
            except Exception:
                prior_manifest = {}
            if prior_manifest.get("fingerprint") != item_fingerprint:
                rotate_stale_item_results(item_results_jsonl, item_manifest, reason="item-results fingerprint mismatch", selection_log_path=selection_log)

    def write_item_manifest() -> None:
        item_manifest.parent.mkdir(parents=True, exist_ok=True)
        item_manifest.write_text(
            json.dumps({
                "format": "dynamix_officeqa_item_results_manifest_v1",
                "fingerprint": item_fingerprint,
                "results_jsonl": str(item_results_jsonl),
                "selection_log": str(selection_log) if selection_log is not None else "",
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    write_item_manifest()

    def run_one(item):
        return run_officeqa_item(
            item,
            docs_dirs=docs_dirs,
            model=args.model,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            generation_config=generation_config,
            max_turns=args.max_turns,
            llm_timeout_seconds=args.llm_timeout_seconds,
            llm_retry_wait_seconds=args.llm_retry_wait_seconds,
            reward_path=reward_path,
            reward_tolerance=args.reward_tolerance,
            evaluator=args.evaluator,
            allow_fallback_evaluator=args.allow_fallback_evaluator,
            output_dir=output_dir,
            log_dir=log_dir,
            skillbank_root=args.skillbank_root or None,
            skillbank_top_k=args.skillbank_top_k,
            selection_log=args.selection_log or None,
            use_oracle_context=args.use_oracle_context,
            verbose=args.verbose,
        )

    def failed_row(item, exc: BaseException) -> dict:
        return {
            "id": item.uid,
            "trajectory_id": f"officeqa_{item.uid}",
            "benchmark": "officeqa",
            "item": item.to_public_dict(),
            "question": item.question,
            "category": item.category,
            "final_response": "",
            "predicted_answer": "",
            "score": 0.0,
            "hard": 0,
            "success": False,
            "fail_reason": f"runner_exception:{exc}",
            "evaluator": args.evaluator,
            "agent_success": False,
            "total_turns": 0,
            "steps": [],
        }

    def safe_run_one(item) -> dict:
        try:
            return run_one(item)
        except OfficeQAOfficialRewardError:
            raise
        except OfficeQAInfrastructureError as exc:
            if args.continue_on_infra_error:
                return failed_row(item, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            if args.continue_on_infra_error:
                return failed_row(item, exc)
            raise

    results_by_id: dict[str, dict] = {}
    if args.resume_item_results and item_results_jsonl.is_file():
        with item_results_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = str(row.get("id") or "")
                if uid:
                    results_by_id[uid] = row
    if (
        results_by_id
        and selection_log is not None
        and args.skillbank_root
        and int(args.skillbank_top_k) > 0
        and not selection_log_covers(selection_log, set(results_by_id))
    ):
        rotate_stale_item_results(
            item_results_jsonl,
            item_manifest,
            reason="selection log missing or incomplete for resumed item results",
            selection_log_path=selection_log,
        )
        write_item_manifest()
        results_by_id = {}
    if selection_log is not None and not results_by_id:
        selection_log.write_text("", encoding="utf-8")
    pending_items = [item for item in items if item.uid not in results_by_id]
    item_results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.workers <= 1:
        with item_results_jsonl.open("a", encoding="utf-8") as handle:
            for item in tqdm(pending_items, desc=f"OfficeQA {args.split}", unit="task"):
                row = safe_run_one(item)
                results_by_id[item.uid] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    else:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(items) or 1)) as executor:
            future_to_item = {executor.submit(safe_run_one, item): item for item in pending_items}
            with item_results_jsonl.open("a", encoding="utf-8") as handle:
                for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc=f"OfficeQA {args.split}", unit="task"):
                    item = future_to_item[future]
                    row = future.result()
                    results_by_id[item.uid] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
    ordered = [results_by_id[item.uid] for item in items if item.uid in results_by_id]
    success_count = sum(1 for row in ordered if row.get("success"))
    payload = {
        "format": "dynamix_officeqa_results_v1",
        "split": args.split,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "count": len(ordered),
        "success_count": success_count,
        "accuracy": success_count / len(ordered) if ordered else 0.0,
        "evaluator": args.evaluator,
        "officeqa_context_variant": officeqa_context_variant(bool(args.use_oracle_context)),
        "use_oracle_context": bool(args.use_oracle_context),
        "item_results_jsonl": str(item_results_jsonl),
        "item_results_manifest": str(item_manifest),
        "selection_log": str(selection_log) if selection_log is not None else "",
        "results": ordered,
    }
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_file).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.records_file:
        records = [record_from_officeqa_result(row).to_dict() for row in ordered]
        Path(args.records_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.records_file).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"results_file": args.results_file, "count": len(ordered), "success_count": success_count, "accuracy": payload["accuracy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
