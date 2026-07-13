#!/usr/bin/env python3
"""
Evaluation script that uses the official SpreadsheetBench evaluation logic.

This script imports and calls the official SpreadsheetBench evaluation functions
to ensure 100% compatibility with their evaluation methodology.

Usage:
    python evaluate_with_official.py --data_path data/sample_data_200 --output_dir outputs/spreadsheetbench
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from spreadsheetbench_support import (
    compare_workbooks as local_compare_workbooks,
    find_output_dir,
    find_spreadsheet_dir,
    load_dataset,
)

try:
    from evaluation_official import compare_workbooks as official_compare_workbooks
except ImportError:
    official_compare_workbooks = None


def compare_workbooks(gt_path, output_path, instruction_type, answer_position):
    if official_compare_workbooks is not None:
        return official_compare_workbooks(gt_path, output_path, instruction_type, answer_position)
    return local_compare_workbooks(gt_path, output_path, answer_position)


def _soffice_executable() -> str:
    soffice = shutil.which("soffice")
    if not soffice:
        raise RuntimeError("LibreOffice executable `soffice` not found on PATH")
    return soffice


def _safe_instance_dir_name(instance_id: str) -> str:
    raw = str(instance_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not safe:
        safe = "instance"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:80]}_{digest}"


def _ensure_within(parent: Path, child: Path) -> None:
    try:
        child.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(f"path escapes recalc audit root: {child}") from exc


def _preflight_libreoffice(timeout_seconds: int = 30) -> str:
    soffice = _soffice_executable()
    try:
        proc = subprocess.run(
            [soffice, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"LibreOffice preflight timed out after {timeout_seconds}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"LibreOffice preflight failed with exit {proc.returncode}: {proc.stdout.strip()}")
    return soffice


def _recalculate_workbook(
    input_path: str,
    recalc_dir: str,
    instance_id: str,
    *,
    soffice: str | None = None,
    timeout_seconds: int = 180,
) -> str:
    """Recalculate a workbook via LibreOffice and return the copied output path."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    soffice = soffice or _soffice_executable()
    input_file = Path(input_path).resolve()
    recalc_root = Path(recalc_dir)
    recalc_root.mkdir(parents=True, exist_ok=True)
    recalc_root = recalc_root.resolve()
    try:
        input_file.relative_to(recalc_root)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            f"recalc_dir must not contain source workbook; recalc_dir={recalc_root}, source={input_file}"
        )

    instance_root = recalc_root / _safe_instance_dir_name(instance_id)
    instance_root.mkdir(parents=True, exist_ok=True)
    if instance_root.is_symlink():
        raise RuntimeError(f"refusing symlinked recalc instance directory: {instance_root}")
    instance_root = instance_root.resolve()
    _ensure_within(recalc_root, instance_root)

    out_dir = Path(tempfile.mkdtemp(prefix="recalc_", dir=str(instance_root))).resolve()
    _ensure_within(recalc_root, out_dir)
    output_file = out_dir / input_file.name

    with tempfile.TemporaryDirectory(prefix="dynamix_recalc_") as tmp:
        profile = Path(tmp) / "profile"
        cmd = [
            soffice,
            f"-env:UserInstallation=file://{profile}",
            "--headless",
            "--invisible",
            "--norestore",
            "--nodefault",
            "--nolockcheck",
            "--nofirststartwizard",
            "--convert-to",
            "xlsx",
            "--outdir",
            str(out_dir),
            str(input_file),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"LibreOffice recalc timed out after {timeout_seconds}s for {input_file}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"LibreOffice recalc failed with exit {proc.returncode}: {proc.stdout.strip()}")
    if not output_file.exists():
        raise RuntimeError(f"LibreOffice did not produce recalculated workbook: {output_file}; output={proc.stdout.strip()}")
    return str(output_file)


def evaluate(data_path, output_dir, start_idx=0, end_idx=None, verbose=False, recalc_dir=None):
    """
    Evaluate outputs against ground truth using official SpreadsheetBench logic.

    Returns:
        dict with evaluation results
    """
    dataset = load_dataset(data_path)

    if end_idx is None:
        end_idx = len(dataset)
    dataset = dataset[start_idx:end_idx]

    if recalc_dir is None:
        recalc_dir = os.path.join(output_dir, "eval_artifacts", "libreoffice_recalculated_outputs")
    soffice = _preflight_libreoffice()
    print(f"Evaluating {len(dataset)} instances using official SpreadsheetBench evaluation with LibreOffice recalc...")
    print(f"LibreOffice executable: {soffice}")

    results = []
    total_test_cases = 0
    passed_test_cases = 0
    fully_correct = 0
    raw_passed_test_cases = 0
    raw_fully_correct = 0

    # Track by instruction type (like official eval)
    type_results = defaultdict(lambda: {"soft": [], "hard": []})
    raw_type_results = defaultdict(lambda: {"soft": [], "hard": []})

    for instance in tqdm(dataset):
        instance_id = str(instance["id"])
        spreadsheet_path = str(instance.get("spreadsheet_path", instance_id))
        instruction_type = instance.get("instruction_type", "")
        answer_position = instance.get("answer_position", "")

        if not answer_position:
            if verbose:
                print(f"Warning: No answer_position for {instance_id}, skipping")
            continue

        # Find spreadsheet directory (contains ground truth)
        spreadsheet_dir = find_spreadsheet_dir(data_path, instance)
        if spreadsheet_dir is None:
            results.append({
                "id": instance_id,
                "success": False,
                "error": "Spreadsheet directory not found",
                "test_cases": [],
            })
            continue

        # Find output directory for this instance
        output_instance_dir = find_output_dir(output_dir, instance)

        # Find all test cases (ground truth files)
        # Standard format: *_answer.xlsx, Verified format: *_golden.xlsx
        try:
            all_files = os.listdir(spreadsheet_dir)
        except FileNotFoundError:
            results.append({
                "id": instance_id,
                "success": False,
                "error": f"Cannot list spreadsheet directory: {spreadsheet_dir}",
                "test_cases": [],
            })
            continue

        gt_files = sorted([f for f in all_files if f.endswith("_answer.xlsx")])

        if not gt_files:
            # Try verified dataset format
            gt_files = sorted([f for f in all_files if f.endswith("_golden.xlsx")])

        if not gt_files:
            # Try exact match for simple naming: golden.xlsx
            if "golden.xlsx" in all_files:
                gt_files = ["golden.xlsx"]

        if not gt_files:
            results.append({
                "id": instance_id,
                "success": False,
                "error": "No ground truth files found (expected *_answer.xlsx or *_golden.xlsx)",
                "test_cases": [],
            })
            continue

        test_case_results = []

        for gt_file in gt_files:
            # Derive output filename from ground truth filename
            if gt_file.endswith("_answer.xlsx"):
                output_file = gt_file.replace("_answer.xlsx", "_output.xlsx")
            elif gt_file == "golden.xlsx":
                # Simple naming: golden.xlsx -> initial_output.xlsx
                output_file = "initial_output.xlsx"
            else:  # _golden.xlsx
                output_file = gt_file.replace("_golden.xlsx", "_output.xlsx")

            gt_path = os.path.join(spreadsheet_dir, gt_file)
            output_path = os.path.join(output_instance_dir, output_file)

            total_test_cases += 1

            raw_result = False
            raw_msg = ""
            try:
                raw_result, raw_msg = compare_workbooks(
                    gt_path, output_path, instruction_type, answer_position
                )
            except Exception as e:
                raw_msg = str(e)

            if raw_result:
                raw_passed_test_cases += 1

            recalculated_output_path = ""
            recalc_error = ""
            try:
                recalculated_output_path = _recalculate_workbook(output_path, recalc_dir, instance_id, soffice=soffice)
                result, msg = compare_workbooks(
                    gt_path, recalculated_output_path, instruction_type, answer_position
                )
            except FileNotFoundError as e:
                result = False
                recalc_error = f"Output file not found for LibreOffice recalc: {e}"
                msg = raw_msg or recalc_error
            except Exception as e:
                result = False
                recalc_error = f"LibreOffice recalc/eval failed: {e}"
                msg = recalc_error

            test_case_results.append({
                "gt_file": gt_file,
                "output_file": output_file,
                "output_path": output_path,
                "recalculated_output_path": recalculated_output_path,
                "evaluation_mode": "libreoffice_recalc",
                "raw_evaluation_mode": "audit_only_no_recalc",
                "raw_passed": raw_result,
                "raw_message": raw_msg,
                "recalc_error": recalc_error,
                "passed": result,
                "message": msg,
            })

            if result:
                passed_test_cases += 1
            elif verbose:
                print(f"  {instance_id}/{output_file}: {msg}")

        # Calculate metrics for this instance (matching official eval)
        passed_count = sum(1 for tc in test_case_results if tc["passed"])
        raw_passed_count = sum(1 for tc in test_case_results if tc.get("raw_passed"))
        total_count = len(test_case_results)
        soft_score = passed_count / total_count if total_count > 0 else 0
        hard_score = 1 if passed_count == total_count else 0
        raw_soft_score = raw_passed_count / total_count if total_count > 0 else 0
        raw_hard_score = 1 if raw_passed_count == total_count else 0

        if hard_score == 1:
            fully_correct += 1
        if raw_hard_score == 1:
            raw_fully_correct += 1

        # Track by instruction type
        type_results[instruction_type]["soft"].append(soft_score)
        type_results[instruction_type]["hard"].append(hard_score)
        raw_type_results[instruction_type]["soft"].append(raw_soft_score)
        raw_type_results[instruction_type]["hard"].append(raw_hard_score)

        results.append({
            "id": instance_id,
            "instruction_type": instruction_type,
            "success": hard_score == 1,
            "test_cases": test_case_results,
            "passed_count": passed_count,
            "raw_passed_count": raw_passed_count,
            "total_count": total_count,
            "soft_score": soft_score,
            "hard_score": hard_score,
            "raw_soft_score": raw_soft_score,
            "raw_hard_score": raw_hard_score,
        })

    # Calculate overall metrics
    total_instances = len(results)

    soft_scores = [r.get("soft_score", 0) for r in results if "soft_score" in r]
    hard_scores = [r.get("hard_score", 0) for r in results if "hard_score" in r]
    raw_soft_scores = [r.get("raw_soft_score", 0) for r in results if "raw_soft_score" in r]
    raw_hard_scores = [r.get("raw_hard_score", 0) for r in results if "raw_hard_score" in r]

    avg_soft_score = sum(soft_scores) / len(soft_scores) if soft_scores else 0
    avg_hard_score = sum(hard_scores) / len(hard_scores) if hard_scores else 0
    raw_avg_soft_score = sum(raw_soft_scores) / len(raw_soft_scores) if raw_soft_scores else 0
    raw_avg_hard_score = sum(raw_hard_scores) / len(raw_hard_scores) if raw_hard_scores else 0

    # Calculate per-type metrics
    type_metrics = {}
    for inst_type, scores in type_results.items():
        type_metrics[inst_type] = {
            "count": len(scores["soft"]),
            "avg_soft_score": sum(scores["soft"]) / len(scores["soft"]) if scores["soft"] else 0,
            "avg_hard_score": sum(scores["hard"]) / len(scores["hard"]) if scores["hard"] else 0,
        }
    raw_type_metrics = {}
    for inst_type, scores in raw_type_results.items():
        raw_type_metrics[inst_type] = {
            "count": len(scores["soft"]),
            "avg_soft_score": sum(scores["soft"]) / len(scores["soft"]) if scores["soft"] else 0,
            "avg_hard_score": sum(scores["hard"]) / len(scores["hard"]) if scores["hard"] else 0,
        }

    summary = {
        "total_instances": total_instances,
        "fully_correct_instances": fully_correct,
        "instance_accuracy": fully_correct / total_instances if total_instances > 0 else 0,
        "total_test_cases": total_test_cases,
        "passed_test_cases": passed_test_cases,
        "test_case_accuracy": passed_test_cases / total_test_cases if total_test_cases > 0 else 0,
        "avg_soft_score": avg_soft_score,
        "avg_hard_score": avg_hard_score,
        "by_instruction_type": type_metrics,
        "evaluation_mode": "libreoffice_recalc",
        "recalculated_output_dir": recalc_dir,
        "trace2skill_compatible_no_recalc": {
            "total_instances": total_instances,
            "fully_correct_instances": raw_fully_correct,
            "instance_accuracy": raw_fully_correct / total_instances if total_instances > 0 else 0,
            "total_test_cases": total_test_cases,
            "passed_test_cases": raw_passed_test_cases,
            "test_case_accuracy": raw_passed_test_cases / total_test_cases if total_test_cases > 0 else 0,
            "avg_soft_score": raw_avg_soft_score,
            "avg_hard_score": raw_avg_hard_score,
            "by_instruction_type": raw_type_metrics,
            "evaluation_mode": "trace2skill_compatible_no_recalc",
        },
    }

    return {
        "summary": summary,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SpreadsheetBench outputs using official evaluation logic"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to SpreadsheetBench data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing agent outputs",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Path to save evaluation results JSON (default: output_dir/eval_official_results.json)",
    )
    parser.add_argument(
        "--recalc_dir",
        type=str,
        default=None,
        help="Directory for LibreOffice-recalculated workbook audit copies",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for evaluation",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for evaluation (exclusive)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed error messages",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of seed runs to evaluate. When > 1, scans output_dir for seed_*/ "
             "subdirectories and evaluates each independently (default: 1).",
    )
    args = parser.parse_args()

    if args.repeat > 1:
        _run_repeat_evaluation(args)
        return

    # Run evaluation
    eval_result = evaluate(
        data_path=args.data_path,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        verbose=args.verbose,
        recalc_dir=args.recalc_dir,
    )

    # Print summary
    _print_summary(eval_result["summary"])

    # Save results
    results_file = args.results_file or os.path.join(args.output_dir, "eval_official_results.json")
    with open(results_file, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"Results saved to: {results_file}")


def _print_summary(summary: dict, label: str = "") -> None:
    """Print a formatted evaluation summary."""
    header = f"EVALUATION RESULTS{' (' + label + ')' if label else ''} (Official SpreadsheetBench Logic)"
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    print(f"Total Instances:        {summary['total_instances']}")
    print(f"Fully Correct:          {summary['fully_correct_instances']}")
    print(f"Instance Accuracy:      {summary['instance_accuracy']*100:.1f}%")
    print(f"Total Test Cases:       {summary['total_test_cases']}")
    print(f"Passed Test Cases:      {summary['passed_test_cases']}")
    print(f"Test Case Accuracy:     {summary['test_case_accuracy']*100:.1f}%")
    print(f"Avg Soft Score:         {summary['avg_soft_score']*100:.1f}%")
    print(f"Avg Hard Score:         {summary['avg_hard_score']*100:.1f}%")
    print(f"Evaluation Mode:        {summary.get('evaluation_mode', 'unknown')}")
    raw_summary = summary.get("trace2skill_compatible_no_recalc", {})
    if raw_summary:
        print(
            "Trace2Skill-Compatible Raw Soft/Hard: "
            f"{raw_summary.get('avg_soft_score', 0)*100:.1f}% / "
            f"{raw_summary.get('avg_hard_score', 0)*100:.1f}%"
        )
    if summary.get("recalculated_output_dir"):
        print(f"Recalculated Outputs:   {summary['recalculated_output_dir']}")

    if summary["by_instruction_type"]:
        print("-" * 60)
        print("By Instruction Type:")
        for inst_type, metrics in sorted(summary["by_instruction_type"].items()):
            print(f"  {inst_type or '(unknown)'}:")
            print(f"    Count: {metrics['count']}")
            print(f"    Soft:  {metrics['avg_soft_score']*100:.1f}%")
            print(f"    Hard:  {metrics['avg_hard_score']*100:.1f}%")

    print("=" * 60)


def _run_repeat_evaluation(args) -> None:
    """Evaluate all seed_* subdirectories under args.output_dir."""
    seed_dirs = sorted(
        d for d in os.scandir(args.output_dir)
        if d.is_dir() and d.name.startswith("seed_")
    )
    if not seed_dirs:
        print(f"No seed_* subdirectories found in {args.output_dir}", file=__import__("sys").stderr)
        __import__("sys").exit(1)

    print(f"Found {len(seed_dirs)} seed run(s): {[d.name for d in seed_dirs]}")
    all_seed_results = {}

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        print(f"\nEvaluating {seed_name} ...")
        result = evaluate(
            data_path=args.data_path,
            output_dir=seed_dir.path,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            verbose=args.verbose,
            recalc_dir=os.path.join(args.recalc_dir, seed_name) if args.recalc_dir else None,
        )
        all_seed_results[seed_name] = result
        per_seed_file = os.path.join(seed_dir.path, "eval_official_results.json")
        with open(per_seed_file, "w") as f:
            json.dump(result, f, indent=2)
        _print_summary(result["summary"], label=seed_name)
        print(f"Results saved to: {per_seed_file}")

    # Aggregate summary: pass@k = fraction of seeds where instance passed
    _print_aggregate_summary(all_seed_results)


def _print_aggregate_summary(all_seed_results: dict) -> None:
    """Print aggregate pass@k statistics across all seeds."""
    if not all_seed_results:
        return

    # Collect per-instance pass/fail across seeds
    instance_seeds: dict[str, list[bool]] = {}
    for seed_name, result in all_seed_results.items():
        for r in result.get("results", []):
            iid = str(r["id"])
            instance_seeds.setdefault(iid, []).append(r.get("success", False))

    total_instances = len(instance_seeds)
    # pass@1: fraction that passed in at least one seed
    passed_any = sum(1 for v in instance_seeds.values() if any(v))
    # pass@all: fraction that passed in all seeds
    passed_all = sum(1 for v in instance_seeds.values() if all(v))

    n_seeds = len(all_seed_results)
    print("\n" + "=" * 60)
    print(f"AGGREGATE SUMMARY ({n_seeds} seeds)")
    print("=" * 60)
    print(f"Unique instances:       {total_instances}")
    print(f"pass@any (>=1 seed):    {passed_any}/{total_instances} "
          f"({passed_any/total_instances*100:.1f}%)" if total_instances else "N/A")
    print(f"pass@all (all seeds):   {passed_all}/{total_instances} "
          f"({passed_all/total_instances*100:.1f}%)" if total_instances else "N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()
