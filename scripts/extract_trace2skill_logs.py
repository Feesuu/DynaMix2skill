#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamix_trace2skill.log_parser import parse_trace2skill_logs, save_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Trace2Skill chat logs into RawTrajectoryRecord JSON")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--results-file", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = parse_trace2skill_logs(args.log_dir, results_file=args.results_file)
    save_records(records, args.output)
    print(f"extracted_records={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
