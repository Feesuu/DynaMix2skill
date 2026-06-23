#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamix_benchmarks.officeqa import record_from_officeqa_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OfficeQA rollout results into DynaMix RawTrajectoryRecord JSON.")
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.results_file).read_text(encoding="utf-8"))
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported OfficeQA results file: {args.results_file}")
    records = [record_from_officeqa_result(row).to_dict() for row in rows if isinstance(row, dict)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "record_count": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
