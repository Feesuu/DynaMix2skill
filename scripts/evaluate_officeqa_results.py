#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamix_benchmarks.officeqa import evaluate_officeqa_prediction, load_officeqa_items, resolve_reward_path  # noqa: E402


def _sanitize_result_row(row: dict) -> dict:
    sanitized = dict(row)
    sanitized.pop("gold_answer", None)
    item = dict(sanitized.get("item") or {})
    for key in ("ground_truth", "answer", "answers"):
        item.pop(key, None)
    sanitized["item"] = item
    return sanitized


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OfficeQA rollout results.")
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reward-path", default="")
    parser.add_argument("--reward-tolerance", type=float, default=0.0)
    parser.add_argument("--evaluator", choices=["skillopt", "official_reward", "fallback"], default="skillopt")
    parser.add_argument("--split", default="", help="OfficeQA split name. Defaults to the split recorded in the results payload.")
    parser.add_argument("--allow-fallback-evaluator", action="store_true", help="Debug only: allow simplified numeric/text matching when official reward.py is unavailable.")
    args = parser.parse_args()
    payload = json.loads(Path(args.results_file).read_text(encoding="utf-8"))
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported OfficeQA results file: {args.results_file}")
    split = args.split or str(payload.get("split") or "")
    if not split:
        raise ValueError("--split is required when the results payload does not record a split")
    reward_path = (
        resolve_reward_path(args.split_dir, args.reward_path or None, required=args.evaluator == "official_reward")
        if args.evaluator == "official_reward" or args.reward_path
        else None
    )
    gold_by_uid = {item.uid: item.ground_truth for item in load_officeqa_items(args.split_dir, split=split)}
    evaluated = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = row.get("item") if isinstance(row.get("item"), dict) else {}
        uid = str(row.get("id") or item.get("uid") or item.get("id") or "")
        if uid not in gold_by_uid:
            raise KeyError(f"OfficeQA result uid={uid!r} not found in split={split!r}")
        eval_result = evaluate_officeqa_prediction(
            prediction_text=str(row.get("final_response") or row.get("predicted_answer") or ""),
            gold_answer=gold_by_uid[uid],
            reward_path=reward_path,
            tolerance=args.reward_tolerance,
            evaluator=args.evaluator,
            allow_fallback=args.allow_fallback_evaluator,
            raise_official_errors=not args.allow_fallback_evaluator,
        )
        evaluated.append({
            **_sanitize_result_row(row),
            "score": eval_result.score,
            "f1": eval_result.f1,
            "hard": eval_result.hard,
            "success": bool(eval_result.hard),
            "predicted_answer": eval_result.predicted_answer,
            "fail_reason": eval_result.fail_reason,
            "evaluator": eval_result.evaluator,
        })
    success_count = sum(1 for row in evaluated if row.get("success"))
    output = {
        "format": "dynamix_officeqa_eval_v1",
        "results_file": args.results_file,
        "split": split,
        "evaluator": args.evaluator,
        "count": len(evaluated),
        "success_count": success_count,
        "accuracy": success_count / len(evaluated) if evaluated else 0.0,
        "results": evaluated,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "count": output["count"], "success_count": success_count, "accuracy": output["accuracy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
