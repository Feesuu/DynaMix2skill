#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OFFICEQA_PROTOCOL = "skillopt_compatible_officeqa_oracle_pages_v2"
DEFAULT_EMBEDDING_TOKENIZER = "/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-8B"
sys.path.insert(0, str(ROOT / "src"))

from dynamix_benchmarks.officeqa import (  # noqa: E402
    OfficeQARolloutConfig,
    load_officeqa_split,
    load_officeqa_splits,
    officeqa_results_to_records,
    run_officeqa_batch,
)
from dynamix_benchmarks.officeqa.tools import resolve_docs_roots  # noqa: E402
from dynamix_trace2skill.pipeline import DynaMixRunConfig, run_config  # noqa: E402
from dynamix_trace2skill.skillbank import SkillBankSelector  # noqa: E402


def main() -> None:
    args = parse_args()
    _validate_generation_endpoint(args)
    if not args.docs_dir:
        args.docs_dir = ["/mnt/data/yaodong/officeqa/hf/treasury_bulletins_parsed"]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    report: dict[str, Any] = {"started_at": _utc_timestamp(), "args": _redact_args(vars(args))}

    docs_roots = resolve_docs_roots(args.docs_dir)
    train_splits = _resolve_train_splits(args)
    train_items = load_officeqa_splits(args.split_dir, train_splits, start=args.train_start, end=args.train_end)
    heldout_items = load_officeqa_split(args.split_dir, args.heldout_split, start=args.heldout_start, end=args.heldout_end)
    train_resume_fingerprint = _rollout_fingerprint(args, docs_roots, stage="train")
    report["preflight"] = {
        "docs_roots": docs_roots,
        "train_splits": train_splits,
        "train_count": len(train_items),
        "heldout_split": args.heldout_split,
        "heldout_count": len(heldout_items),
        "protocol": OFFICEQA_PROTOCOL,
        "primary_reward": "skillopt_em_f1",
        "official_reward_audit": bool(args.reward_path),
        "train_resume_fingerprint": train_resume_fingerprint,
        "heldout_retrieval_query": "question + task_type",
        "heldout_retrieval_query_note": "task_type is OfficeQA category metadata; gold answers, source answers, and source documents are not used for retrieval.",
    }

    rollout_cfg = OfficeQARolloutConfig(
        base_url=args.openai_base_url,
        model=args.model,
        api_key=_resolve_api_key(args.openai_api_key, args.openai_api_key_env),
        temperature=args.generation_temperature,
        timeout_seconds=args.generation_timeout,
        max_tool_turns=args.max_tool_turns,
        max_completion_tokens=args.max_completion_tokens,
        workers=args.workers,
        thinking=_parse_bool(args.thinking),
        docs_dirs=tuple(args.docs_dir),
        reward_path=args.reward_path,
        resume=args.resume,
        resume_fingerprint=train_resume_fingerprint,
    )

    records_path = Path(args.records_path) if args.records_path else run_dir / "records.json"
    if not args.skip_train_rollout and not args.records_path:
        train_results = _timed(
            report,
            "01_train_rollout",
            lambda: run_officeqa_batch(train_items, run_dir / "train_rollout", rollout_cfg),
        )
        records = officeqa_results_to_records(train_results)
        records_path.write_text(json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2), encoding="utf-8")
        report["records"] = {"path": str(records_path), "count": len(records)}
    elif not records_path.is_file():
        raise FileNotFoundError(f"records_path not found: {records_path}")
    else:
        records_validation = _validate_records_match_items(records_path, train_items)
        report["records"] = {
            "path": str(records_path),
            "count": _json_list_len(records_path),
            "reused": True,
            **records_validation,
        }

    tree_summary: dict[str, Any] | None = None
    tree_dir = run_dir / "tree"
    build_config_path = run_dir / "officeqa_dynamix_config.json"
    if not args.skip_build_tree:
        build_config = _build_tree_config(args, records_path=records_path, output_dir=tree_dir)
        build_config_path.write_text(json.dumps(build_config, ensure_ascii=False, indent=2), encoding="utf-8")
        report["build_config"] = str(build_config_path)
        tree_summary = _timed(
            report,
            "02_build_tree",
            lambda: asyncio.run(run_config(DynaMixRunConfig.from_json(build_config_path))),
        )
    elif (tree_dir / "summary.json").is_file():
        tree_summary = json.loads((tree_dir / "summary.json").read_text(encoding="utf-8"))
    elif args.run_heldout:
        raise FileNotFoundError(f"tree summary not found for heldout: {tree_dir / 'summary.json'}")

    if tree_summary is not None:
        report["nodebank_diagnostic_audit"] = _audit_nodebank_for_train_diagnostic_leakage(
            Path(tree_summary["node_bank_dir"]),
            records_path,
            run_dir,
        )

    if args.run_heldout and not args.skip_heldout:
        if tree_summary is None:
            tree_summary = json.loads((tree_dir / "summary.json").read_text(encoding="utf-8"))
        skillbank_root = str(tree_summary["node_bank_dir"])
        node_count = int(tree_summary.get("node_count", 0) or 0)
        heldout_resume_fingerprint = _rollout_fingerprint(
            args,
            docs_roots,
            stage="heldout",
            extra={
                "skillbank_root": skillbank_root,
                "node_count": node_count,
                "skillbank_manifest_sha256": _file_sha256(Path(skillbank_root) / "node_bank_manifest.json"),
                "skillbank_index_sha256": _file_sha256(Path(skillbank_root) / ".dynamix_skillbank_index.json"),
                "embedding_protocol": _embedding_protocol_payload(args),
            },
        )
        report["preflight"]["heldout_resume_fingerprint"] = heldout_resume_fingerprint
        report["preflight"]["heldout_skillbank_empty"] = node_count <= 0
        heldout_cfg = replace(rollout_cfg, resume_fingerprint=heldout_resume_fingerprint)

        if node_count <= 0:
            def skill_content(item) -> str:
                return ""
        else:
            selector = SkillBankSelector(
                skillbank_root=skillbank_root,
                base_url=args.embedding_base_url,
                model=args.embedding_model,
                api_key=_resolve_api_key(args.embedding_api_key, args.embedding_api_key_env),
                max_model_len=args.embedding_max_model_len,
                max_input_tokens=args.embedding_max_model_len,
                batch_size=args.embedding_batch_size,
                tokenizer_model=args.embedding_tokenizer or None,
                chunk_tokens=args.chunk_tokens,
                chunk_overlap_tokens=args.chunk_overlap_tokens,
            )

            def skill_content(item) -> str:
                query = f"{item.question}\n\nTask type: {item.task_type}"
                selections = selector.select(query, top_k=args.skillbank_top_k)
                return _render_retrieved_experience(selections)

        heldout_results = _timed(
            report,
            "03_heldout_rollout",
            lambda: run_officeqa_batch(heldout_items, run_dir / "heldout_rollout", heldout_cfg, skill_content_fn=skill_content),
        )
        report["heldout"] = _score_summary(heldout_results)

    report["finished_at"] = _utc_timestamp()
    (run_dir / "officeqa_experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "officeqa_experiment_report.md").write_text(_render_report_md(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SkillOpt-compatible OfficeQA rollout -> DynaMix nodebank experiment")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-dir", default="/mnt/data/yaodong/officeqa/splits")
    parser.add_argument("--docs-dir", action="append", default=None)
    parser.add_argument("--reward-path", default="/mnt/data/yaodong/officeqa/reward.py")
    parser.add_argument("--train-splits", default="train,val")
    parser.add_argument("--train-split", default="", help="Legacy single training split override; prefer --train-splits.")
    parser.add_argument("--heldout-split", default="test")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-end", type=int, default=None)
    parser.add_argument("--heldout-start", type=int, default=0)
    parser.add_argument("--heldout-end", type=int, default=None)
    parser.add_argument("--records-path", default="")
    parser.add_argument("--model", default="Qwen3.5-9B-AWQ")
    parser.add_argument("--openai-base-url", default="https://asmiatbrqksz.10.27.127.9.nip.io/v1")
    parser.add_argument("--openai-api-key", default="EMPTY")
    parser.add_argument("--openai-api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--generation-temperature", type=float, default=0.6)
    parser.add_argument("--generation-timeout", type=float, default=1200.0)
    parser.add_argument("--thinking", default="true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tool-turns", type=int, default=30)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--embedding-base-url", default="http://10.26.1.184:18007/v1")
    parser.add_argument("--embedding-model", default="Qwen3-Embedding-8B")
    parser.add_argument("--embedding-api-key", default="EMPTY")
    parser.add_argument("--embedding-api-key-env", default="")
    parser.add_argument("--embedding-tokenizer", default=DEFAULT_EMBEDDING_TOKENIZER)
    parser.add_argument("--embedding-max-model-len", type=int, default=32000)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--chunk-tokens", type=int, default=28000)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=1000)
    parser.add_argument("--analyst-tokenizer", default="")
    parser.add_argument("--analyst-tokenizer-required", default="true")
    parser.add_argument("--analyst-allow-regex-tokenizer-fallback", default="false")
    parser.add_argument("--skillbank-top-k", type=int, default=10)
    parser.add_argument("--max-levels", type=int, default=8)
    parser.add_argument("--summary-max-model-tokens", type=int, default=100000)
    parser.add_argument("--summary-budget-ratio", type=float, default=0.85)
    parser.add_argument("--gmm-min-split-size", type=int, default=4)
    parser.add_argument("--gmm-min-effective-samples-per-component", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-train-rollout", action="store_true")
    parser.add_argument("--skip-build-tree", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--run-heldout", action="store_true")
    return parser.parse_args(argv)


def _build_tree_config(args: argparse.Namespace, *, records_path: Path, output_dir: Path) -> dict[str, Any]:
    thinking = _parse_bool(args.thinking)
    chunked_embedding_enabled = bool(args.embedding_tokenizer) or not str(args.embedding_base_url).startswith("mock://")
    generation_api_key, generation_api_key_env_var = _generation_api_key_config(args)
    embedding_api_key, embedding_api_key_env_var = _embedding_api_key_config(args)
    return {
        "scenario": "static_build",
        "output_dir": str(output_dir),
        "records_path": str(records_path),
        "enforce_dataset_order": False,
        "generation": {
            "base_url": args.openai_base_url,
            "model": args.model,
            "api_key": generation_api_key,
            "api_key_env_var": generation_api_key_env_var,
            "temperature": args.generation_temperature,
            "timeout_seconds": args.generation_timeout,
            "max_concurrency": args.workers,
            "thinking_mode": thinking,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
        },
        "embedding": {
            "base_url": args.embedding_base_url,
            "model": args.embedding_model,
            "api_key": embedding_api_key,
            "api_key_env_var": embedding_api_key_env_var,
            "max_model_len": args.embedding_max_model_len,
            "max_input_tokens": args.embedding_max_model_len,
            "truncate_long_texts": True,
            "batch_size": args.embedding_batch_size,
            "max_concurrency": args.embedding_batch_size,
            "cache_path": str(Path(args.run_dir) / "embedding_cache.sqlite"),
            "tokenizer_model": args.embedding_tokenizer or None,
            "tokenizer_required": bool(args.embedding_tokenizer),
            "truncation_strategy": "head",
        },
        "chunked_embedding": {
            "enabled": chunked_embedding_enabled,
            "chunk_tokens": args.chunk_tokens,
            "overlap_tokens": args.chunk_overlap_tokens,
            "pooling": "mean",
            "add_special_tokens": False,
            "normalize_after_pooling": False,
            "fail_if_chunk_exceeds_model_limit": True,
        },
        "hierarchy": {
            "projection": {"method": "local_pca", "variance_ratio": 0.9, "max_dim": 32, "min_dim": 2, "whiten": False},
            "gmm_bic": {
                "covariance_type": "spherical",
                "num_restarts": 5,
                "kmeans_init_iters": 15,
                "max_iter": 100,
                "tol": 0.0001,
                "min_covar": 1e-6,
                "min_split_size": args.gmm_min_split_size,
                "min_effective_samples_per_component": args.gmm_min_effective_samples_per_component,
                "abs_kmax": 64,
                "max_concurrent_candidates": 1,
                "max_concurrent_restarts": 1,
            },
            "soft_membership": {
                "save_soft_edges": True,
                "recursive_assignment": "cumulative_mass",
                "cumulative_mass_coverage": 0.9,
            },
            "budget_refinement": {
                "enabled": True,
                "apply_to_level": 0,
                "selection_policy": "bic_best_with_token_progress",
                "min_token_reduction_fraction": 0.1,
                "fallback": "gmm_bic_recursive",
                "flatten_refinement_leaves_to_l0": True,
                "skip_oversize_singleton": True,
            },
            "summary_budget": {
                "max_model_tokens": args.summary_max_model_tokens,
                "budget_ratio": args.summary_budget_ratio,
                "prompt_overhead_reserve_tokens": 8000,
            },
            "random_seed": 42,
        },
        "analyst": {
            "prompt_style": "officeqa_cluster_level_v1",
            "task_profile": "officeqa",
            "confidence_floor": 0.05,
            "tokenizer_model": args.analyst_tokenizer or None,
            "tokenizer_required": _parse_bool(args.analyst_tokenizer_required),
            "allow_regex_tokenizer_fallback": _parse_bool(args.analyst_allow_regex_tokenizer_fallback),
        },
        "max_levels": args.max_levels,
        "skill_output_dir_name": "skills",
    }


def _render_retrieved_experience(selections) -> str:
    lines = ["# Retrieved Experience", ""]
    for rank, selection in enumerate(selections, start=1):
        node = selection.skill
        lines.extend([
            f"## Experience {rank}: {node.name}",
            "",
            f"Trigger: {node.trigger}",
            "",
            "Guidance:",
            node.content.strip(),
            "",
        ])
    return "\n".join(lines).strip()


def _audit_nodebank_for_train_diagnostic_leakage(skillbank_root: Path, records_path: Path, run_dir: Path) -> dict[str, Any]:
    manifest_path = skillbank_root / "node_bank_manifest.json"
    if not manifest_path.is_file():
        return {"status": "skipped", "reason": f"node bank manifest not found: {manifest_path}"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    forbidden_labels = {
        "ground_truth",
        "gold_answer",
        "predicted_answer",
        "official_reward_audit",
        "verifier_score",
        "verifier_feedback",
        "answer_mismatch",
    }
    forbidden_terms = _officeqa_train_diagnostic_terms(records)
    nodes = manifest.get("nodes") or manifest.get("entries") or []
    violations: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or node.get("id") or "")
        text = "\n".join(str(node.get(field) or "") for field in ("name", "trigger", "content")).lower()
        for label in sorted(forbidden_labels):
            if label in text:
                violations.append({"node_id": node_id, "kind": "diagnostic_label", "term": label})
        for term in forbidden_terms:
            if term in text:
                violations.append({"node_id": node_id, "kind": "train_answer_value", "term": term})
    audit = {
        "status": "pass" if not violations else "fail",
        "manifest": str(manifest_path),
        "records": str(records_path),
        "node_count": len(nodes),
        "forbidden_term_count": len(forbidden_terms),
        "violation_count": len(violations),
        "violations": violations[:100],
    }
    audit_path = run_dir / "officeqa_nodebank_diagnostic_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    audit["path"] = str(audit_path)
    if violations:
        raise ValueError(f"OfficeQA nodebank leaked train diagnostics; see {audit_path}")
    return audit


def _officeqa_train_diagnostic_terms(records: list[Any]) -> list[str]:
    terms: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
        payloads = [
            record,
            extra.get("officeqa_result"),
            extra.get("primary_eval"),
            extra.get("official_reward_audit"),
        ]
        for payload in payloads:
            _collect_answer_terms(payload, terms)
    return sorted(terms)


def _collect_answer_terms(value: Any, terms: set[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in {"ground_truth", "gold_answer", "predicted_answer"}:
                _add_forbidden_term(nested, terms)
            elif isinstance(nested, (dict, list)):
                _collect_answer_terms(nested, terms)
    elif isinstance(value, list):
        for item in value:
            _collect_answer_terms(item, terms)


def _add_forbidden_term(value: Any, terms: set[str]) -> None:
    text = str(value or "").strip().lower()
    if len(text) < 4:
        return
    if not any(ch.isalnum() for ch in text):
        return
    terms.add(text)


def _timed(report: dict[str, Any], stage: str, fn):
    start = time.time()
    result = fn()
    report.setdefault("stages", {})[stage] = {"seconds": round(time.time() - start, 3)}
    return result


def _score_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    hard = sum(1 for row in results if int(row.get("hard", 0) or 0))
    mean_f1 = sum(float(row.get("soft", 0.0) or 0.0) for row in results) / count if count else 0.0
    return {
        "count": count,
        "hard": hard,
        "skillopt_em": hard / count if count else 0.0,
        "skillopt_f1": mean_f1,
        "accuracy": hard / count if count else 0.0,
        "mean_f1": mean_f1,
    }


def _render_report_md(report: dict[str, Any]) -> str:
    heldout = report.get("heldout", {})
    return "\n".join([
        "# OfficeQA DynaMix Experiment Report",
        "",
        f"- Protocol: {report.get('preflight', {}).get('protocol', '')}",
        f"- Primary reward: {report.get('preflight', {}).get('primary_reward', '')}",
        f"- Train splits: {','.join(report.get('preflight', {}).get('train_splits', []))}",
        f"- Heldout split: {report.get('preflight', {}).get('heldout_split', '')}",
        f"- Train count: {report.get('preflight', {}).get('train_count', 0)}",
        f"- Heldout count: {report.get('preflight', {}).get('heldout_count', 0)}",
        f"- Records: {report.get('records', {}).get('path', '')}",
        f"- Heldout SkillOpt EM: {heldout.get('hard', 0)}/{heldout.get('count', 0)} ({heldout.get('skillopt_em', 0.0):.4f})",
        f"- Heldout SkillOpt F1: {heldout.get('skillopt_f1', 0.0):.4f}",
        "",
    ])


def _json_list_len(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload) if isinstance(payload, list) else 0


def _validate_records_match_items(records_path: Path, train_items: list[Any]) -> dict[str, Any]:
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"records_path must contain a JSON list: {records_path}")
    expected = [str(item.uid) for item in train_items]
    actual = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f"records_path contains a non-object row: {records_path}")
        task_id = str(row.get("task_id") or "")
        if not task_id:
            trajectory_id = str(row.get("trajectory_id") or "")
            task_id = trajectory_id.removeprefix("officeqa:")
        actual.append(task_id)
    if Counter(actual) != Counter(expected):
        raise ValueError(
            "Reused OfficeQA records do not match the requested training split: "
            f"records_count={len(actual)} expected_count={len(expected)} "
            f"first_records={actual[:5]} first_expected={expected[:5]}"
        )
    return {
        "order_matches": actual == expected,
        "first_record_ids": actual[:5],
        "first_expected_ids": expected[:5],
    }


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_train_splits(args: argparse.Namespace) -> list[str]:
    if str(getattr(args, "train_split", "") or "").strip():
        return [str(args.train_split).strip()]
    return [part.strip() for part in str(args.train_splits or "").split(",") if part.strip()]


def _validate_generation_endpoint(args: argparse.Namespace) -> None:
    base_url = str(args.openai_base_url or "").rstrip("/")
    forbidden_local = {"http://127.0.0.1:18002/v1", "http://localhost:18002/v1"}
    if base_url in forbidden_local and os.environ.get("ALLOW_OFFICEQA_LOCAL_18002") != "1":
        raise ValueError(
            "OfficeQA runner refuses local A5000 18002 by default. "
            "Use the external AWQ endpoint/tunnel, or set ALLOW_OFFICEQA_LOCAL_18002=1 for an explicit debug run."
        )


def _embedding_protocol_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "base_url": args.embedding_base_url,
        "model": args.embedding_model,
        "max_model_len": args.embedding_max_model_len,
        "batch_size": args.embedding_batch_size,
        "tokenizer": args.embedding_tokenizer or "",
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap_tokens": args.chunk_overlap_tokens,
    }


def _resolve_api_key(value: str, env_var: str = "") -> str:
    if env_var:
        return os.environ.get(env_var, value or "EMPTY")
    return value or "EMPTY"


def _generation_api_key_config(args: argparse.Namespace) -> tuple[str, str | None]:
    return _api_key_config(
        args.openai_api_key,
        args.openai_api_key_env,
        internal_env_var="DYNAMIX_OFFICEQA_OPENAI_API_KEY",
    )


def _embedding_api_key_config(args: argparse.Namespace) -> tuple[str, str | None]:
    return _api_key_config(
        args.embedding_api_key,
        args.embedding_api_key_env,
        internal_env_var="DYNAMIX_OFFICEQA_EMBEDDING_API_KEY",
    )


def _api_key_config(value: str, env_var: str, *, internal_env_var: str) -> tuple[str, str | None]:
    if env_var:
        if os.environ.get(env_var):
            return "EMPTY", env_var
        if value and value != "EMPTY":
            os.environ[internal_env_var] = value
            return "EMPTY", internal_env_var
        return "EMPTY", env_var
    if value and value != "EMPTY":
        os.environ[internal_env_var] = value
        return "EMPTY", internal_env_var
    return "EMPTY", None


def _redact_args(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    for field in ("openai_api_key", "embedding_api_key"):
        key = str(data.get(field) or "")
        if key and key != "EMPTY":
            data[field] = "sha256:redacted"
    return data


def _rollout_fingerprint(
    args: argparse.Namespace,
    docs_roots: list[str],
    *,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> str:
    if stage == "heldout":
        split = args.heldout_split
        start = args.heldout_start
        end = args.heldout_end
    else:
        split = _resolve_train_splits(args)
        start = args.train_start
        end = args.train_end
    split_list = split if isinstance(split, list) else [split]
    payload = {
        "protocol": OFFICEQA_PROTOCOL,
        "stage": stage,
        "split_dir": str(Path(args.split_dir).expanduser()),
        "split": split,
        "split_files": _split_fingerprints(args.split_dir, split_list),
        "range": [start, end],
        "docs_roots": docs_roots,
        "reward_path": str(Path(args.reward_path).expanduser()) if args.reward_path else "",
        "reward_sha256": _file_sha256(Path(args.reward_path).expanduser()) if args.reward_path else "",
        "officeqa_code_sha256": _officeqa_code_fingerprint(),
        "model": args.model,
        "base_url": args.openai_base_url,
        "temperature": args.generation_temperature,
        "timeout_seconds": args.generation_timeout,
        "thinking": _parse_bool(args.thinking),
        "max_tool_turns": args.max_tool_turns,
        "max_completion_tokens": args.max_completion_tokens,
        "skillbank_top_k": args.skillbank_top_k if stage == "heldout" else None,
        "extra": extra or {},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_fingerprints(split_dir: str | Path, splits: list[str]) -> list[dict[str, str]]:
    return [
        {"split": split, "path": str(path), "sha256": _file_sha256(path)}
        for split in splits
        for path in [_resolve_split_file(split_dir, split)]
    ]


def _resolve_split_file(split_dir: str | Path, split: str) -> Path:
    root = Path(split_dir).expanduser()
    candidates = [
        root / split / "items.json",
        root / split / "items.csv",
        root / "items.json",
        root / "items.csv",
        root,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (root / split / "items.json").resolve()


def _officeqa_code_fingerprint() -> str:
    files = [
        ROOT / "src" / "dynamix_benchmarks" / "officeqa" / "data.py",
        ROOT / "src" / "dynamix_benchmarks" / "officeqa" / "rollout.py",
        ROOT / "src" / "dynamix_benchmarks" / "officeqa" / "tools.py",
        ROOT / "src" / "dynamix_benchmarks" / "officeqa" / "records.py",
        ROOT / "src" / "dynamix_benchmarks" / "officeqa" / "reward.py",
        ROOT / "src" / "dynamix_trace2skill" / "skillbank.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"")
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        if not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    main()
