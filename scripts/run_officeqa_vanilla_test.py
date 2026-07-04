#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dynamix_benchmarks.officeqa import OfficeQARolloutConfig, load_officeqa_split, run_officeqa_batch  # noqa: E402
from dynamix_benchmarks.officeqa.tools import resolve_docs_roots  # noqa: E402
from officeqa_runner_common import (  # noqa: E402
    file_sha256,
    parse_bool,
    redact_args,
    resolve_api_key,
    score_summary,
    split_fingerprints,
    timed,
    utc_timestamp,
    validate_generation_endpoint,
)

VANILLA_PROTOCOL = "skillopt_compatible_officeqa_vanilla_oracle_pages_v1"
DYNAMIX_COMPARISON_RUN = "officeqa_full_skillopt_no_max_20260703_2351"
DYNAMIX_COMPARISON_PROTOCOL = "skillopt_compatible_officeqa_oracle_pages_v1"


def main() -> None:
    args = parse_args()
    validate_generation_endpoint(args)
    if not args.docs_dir:
        args.docs_dir = ["/mnt/data/yaodong/officeqa/hf/treasury_bulletins_parsed"]

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    docs_roots = resolve_docs_roots(args.docs_dir)
    items = load_officeqa_split(args.split_dir, args.split, start=args.start, end=args.end)
    expected_count = int(args.expected_count)
    if expected_count > 0 and len(items) != expected_count:
        raise ValueError(
            f"OfficeQA vanilla expected {expected_count} items for a paper-grade run, got {len(items)}. "
            "Set --expected-count 0 only for smoke/debug subsets."
        )
    resume_fingerprint = _vanilla_fingerprint(args, docs_roots)
    report: dict[str, Any] = {
        "started_at": utc_timestamp(),
        "args": redact_args(vars(args)),
        "preflight": {
            "docs_roots": docs_roots,
            "split": args.split,
            "count": len(items),
            "expected_count": expected_count,
            "subset_debug": expected_count <= 0 or len(items) != 172,
            "protocol": VANILLA_PROTOCOL,
            "dynamix_comparison_run": DYNAMIX_COMPARISON_RUN,
            "dynamix_comparison_protocol": DYNAMIX_COMPARISON_PROTOCOL,
            "primary_reward": "skillopt_em_f1",
            "official_reward_audit": bool(args.reward_path),
            "nodebank_used": False,
            "retrieved_experience_injected": False,
            "model": args.model,
            "base_url": args.openai_base_url,
            "temperature": args.generation_temperature,
            "timeout_seconds": args.generation_timeout,
            "thinking": parse_bool(args.thinking),
            "workers": args.workers,
            "max_tool_turns": args.max_tool_turns,
            "max_completion_tokens": args.max_completion_tokens,
            "resume_fingerprint": resume_fingerprint,
        },
    }

    rollout_cfg = OfficeQARolloutConfig(
        base_url=args.openai_base_url,
        model=args.model,
        api_key=resolve_api_key(args.openai_api_key, args.openai_api_key_env),
        temperature=args.generation_temperature,
        timeout_seconds=args.generation_timeout,
        max_tool_turns=args.max_tool_turns,
        max_completion_tokens=args.max_completion_tokens,
        workers=args.workers,
        thinking=parse_bool(args.thinking),
        docs_dirs=tuple(args.docs_dir),
        reward_path=args.reward_path,
        resume=args.resume,
        resume_fingerprint=resume_fingerprint,
    )

    results = timed(
        report,
        "01_vanilla_rollout",
        lambda: run_officeqa_batch(items, run_dir / "vanilla_rollout", rollout_cfg),
    )
    report["vanilla"] = score_summary(results)
    report["finished_at"] = utc_timestamp()
    (run_dir / "officeqa_vanilla_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "officeqa_vanilla_report.md").write_text(_render_vanilla_report_md(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SkillOpt-compatible OfficeQA vanilla no-skill baseline on one split")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-dir", default="/mnt/data/yaodong/officeqa/splits")
    parser.add_argument("--docs-dir", action="append", default=None)
    parser.add_argument("--reward-path", default="/mnt/data/yaodong/officeqa/reward.py")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--expected-count", type=int, default=172)
    parser.add_argument("--model", default="Qwen3.5-9B-AWQ")
    parser.add_argument("--openai-base-url", default="http://asmiatbrqksz.10.27.127.9.nip.io/v1")
    parser.add_argument("--openai-api-key", default="EMPTY")
    parser.add_argument("--openai-api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--generation-temperature", type=float, default=0.6)
    parser.add_argument("--generation-timeout", type=float, default=1200.0)
    parser.add_argument("--thinking", default="true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tool-turns", type=int, default=30)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _vanilla_fingerprint(args: argparse.Namespace, docs_roots: list[str]) -> str:
    payload = {
        "protocol": VANILLA_PROTOCOL,
        "split_dir": str(Path(args.split_dir).expanduser()),
        "split": args.split,
        "split_files": split_fingerprints(args.split_dir, [args.split]),
        "range": [args.start, args.end],
        "expected_count": args.expected_count,
        "docs_roots": docs_roots,
        "reward_path": str(Path(args.reward_path).expanduser()) if args.reward_path else "",
        "reward_sha256": file_sha256(Path(args.reward_path).expanduser()) if args.reward_path else "",
        "officeqa_vanilla_code_sha256": _vanilla_code_fingerprint(),
        "model": args.model,
        "base_url": args.openai_base_url,
        "temperature": args.generation_temperature,
        "timeout_seconds": args.generation_timeout,
        "thinking": parse_bool(args.thinking),
        "workers": args.workers,
        "max_tool_turns": args.max_tool_turns,
        "max_completion_tokens": args.max_completion_tokens,
        "nodebank_used": False,
        "retrieved_experience_injected": False,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _vanilla_code_fingerprint() -> str:
    files = [
        ROOT / "scripts" / "run_officeqa_vanilla_test.py",
        ROOT / "scripts" / "officeqa_runner_common.py",
        ROOT / "src" / "dynamix_benchmarks" / "officeqa" / "data.py",
        ROOT / "src" / "dynamix_benchmarks" / "officeqa" / "rollout.py",
        ROOT / "src" / "dynamix_benchmarks" / "officeqa" / "tools.py",
        ROOT / "src" / "dynamix_benchmarks" / "officeqa" / "reward.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"")
        digest.update(b"\0")
    return digest.hexdigest()


def _render_vanilla_report_md(report: dict[str, Any]) -> str:
    result = report.get("vanilla", {})
    preflight = report.get("preflight", {})
    return "\n".join([
        "# OfficeQA Vanilla Test Report",
        "",
        f"- Protocol: {preflight.get('protocol', '')}",
        "- Method: no DynaMix tree, no nodebank, no Retrieved Experience injection",
        f"- DynaMix comparison run: {preflight.get('dynamix_comparison_run', '')}",
        f"- Split: {preflight.get('split', '')}",
        f"- Count: {preflight.get('count', 0)} (expected: {preflight.get('expected_count', 0)})",
        f"- Subset/debug run: {preflight.get('subset_debug', False)}",
        f"- Model: {preflight.get('model', '')}",
        f"- Base URL: {preflight.get('base_url', '')}",
        f"- Decoding: temperature={preflight.get('temperature', '')}, thinking={preflight.get('thinking', '')}, max_tool_turns={preflight.get('max_tool_turns', '')}, max_completion_tokens={preflight.get('max_completion_tokens', None)}",
        f"- Primary reward: {preflight.get('primary_reward', '')}",
        f"- SkillOpt EM: {result.get('hard', 0)}/{result.get('count', 0)} ({result.get('skillopt_em', 0.0):.4f})",
        f"- SkillOpt F1: {result.get('skillopt_f1', 0.0):.4f}",
        "",
    ])


if __name__ == "__main__":
    main()
