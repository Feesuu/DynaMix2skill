#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def bool_flag(name: str, default: str = "false") -> str:
    value = env(name, default).strip().lower()
    return "true" if value in {"1", "true", "yes", "y", "on"} else "false"


def optional_arg(cmd: list[str], flag: str, value: str) -> None:
    if str(value).strip():
        cmd.extend([flag, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one static DynaMix ablation variant")
    parser.add_argument("--variant-json", required=True)
    args = parser.parse_args()

    variant_path = Path(args.variant_json).resolve()
    variant = json.loads(variant_path.read_text(encoding="utf-8"))
    name = str(variant["variant_name"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo = Path(env("REPO_ROOT")).resolve()
    run_dir = Path(env("RUN_ROOT")).resolve() / name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    tree = dict(variant.get("tree", {}))
    skill_export = dict(variant.get("skill_export", {}))
    retrieval_only = bool(variant.get("reuse_full_tree", False))
    tree_policy = str(tree.get("tree_policy", "projected_gmm_bic"))
    soft_assignment = str(tree.get("soft_recursive_assignment", env("SOFT_RECURSIVE_ASSIGNMENT", "cumulative_mass")))
    max_levels = int(tree.get("max_levels", env("MAX_LEVELS", "8")))
    analyst_max_cards_l0 = int(tree.get("analyst_max_cards_l0", env("ANALYST_MAX_CARDS_L0", "0")))
    if retrieval_only and not (env("RECORDS_PATH").strip() or env("REUSE_TRAIN_RUN_DIR").strip()):
        raise SystemExit("retrieval-only variants require RECORDS_PATH or REUSE_TRAIN_RUN_DIR so stages 01-03 are not re-run")

    cmd = [
        env("DYNAMIX_PYTHON"),
        "scripts/run_dynamix_trace2skill_experiment.py",
        "--data-path", env("DATA_PATH"),
        "--run-dir", str(run_dir),
        "--scenario-output-dir", str(run_dir),
        "--tree-scenario", "static_build",
        "--train-start", env("TRAIN_START", "0"),
        "--train-end", env("TRAIN_END", "200"),
        "--heldout-start", env("HELDOUT_START", "200"),
        "--heldout-end", env("HELDOUT_END", "400"),
        "--workers", env("WORKERS", "8"),
        "--model", env("MODEL"),
        "--openai-base-url", env("OPENAI_BASE_URL"),
        "--openai-api-key", env("OPENAI_API_KEY", "EMPTY"),
        "--embedding-base-url", env("EMBEDDING_BASE_URL"),
        "--embedding-model", env("EMBEDDING_MODEL"),
        "--embedding-tokenizer", env("EMBEDDING_TOKENIZER"),
        "--python-executable", env("DYNAMIX_PYTHON"),
        "--max-turns", env("MAX_TURNS", "30"),
        "--thinking", env("THINKING", "true"),
        "--skillbank-top-k", env("SKILLBANK_TOP_K", "10"),
        "--random-seed", env("RANDOM_SEED", "42"),
        "--tree-policy", tree_policy,
        "--soft-recursive-assignment", soft_assignment,
        "--soft-cumulative-mass-coverage", env("SOFT_CUMULATIVE_MASS_COVERAGE", "0.90"),
        "--soft-max-membership-gap", env("SOFT_MAX_MEMBERSHIP_GAP", "0.25"),
        "--soft-min-membership-weight", env("SOFT_MIN_MEMBERSHIP_WEIGHT", "0.05"),
        "--max-levels", str(max_levels),
        "--analyst-max-cards-l0", str(analyst_max_cards_l0),
        "--generation-temperature", env("GENERATION_TEMPERATURE", "0.6"),
        "--generation-timeout-seconds", env("GENERATION_TIMEOUT_SECONDS", "1200"),
        "--generation-max-concurrency", env("GENERATION_MAX_CONCURRENCY", env("WORKERS", "8")),
        "--embedding-max-model-len", env("EMBEDDING_MAX_MODEL_LEN", "32000"),
        "--embedding-max-input-tokens", env("EMBEDDING_MAX_INPUT_TOKENS", "32000"),
        "--embedding-batch-size", env("EMBEDDING_BATCH_SIZE", "8"),
        "--embedding-max-concurrency", env("EMBEDDING_MAX_CONCURRENCY", "8"),
        "--chunked-embedding-chunk-tokens", env("CHUNKED_EMBEDDING_CHUNK_TOKENS", "28000"),
        "--chunked-embedding-overlap-tokens", env("CHUNKED_EMBEDDING_OVERLAP_TOKENS", "1000"),
        "--summary-max-model-tokens", env("SUMMARY_MAX_MODEL_TOKENS", "100000"),
        "--summary-budget-ratio", env("SUMMARY_BUDGET_RATIO", "0.85"),
        "--summary-prompt-overhead-reserve-tokens", env("SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS", "8000"),
        "--gmm-min-split-size", env("GMM_MIN_SPLIT_SIZE", "2"),
        "--gmm-min-effective-samples-per-component", env("GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT", "2"),
        "--gmm-abs-kmax", env("GMM_ABS_KMAX", "64"),
        "--kmeans-fixed-k", str(tree.get("kmeans_fixed_k", env("KMEANS_FIXED_K", "8"))),
        "--kmeans-min-k", str(tree.get("kmeans_min_k", env("KMEANS_MIN_K", "1"))),
        "--kmeans-num-restarts", str(tree.get("kmeans_num_restarts", env("KMEANS_NUM_RESTARTS", "5"))),
        "--kmeans-max-iter", str(tree.get("kmeans_max_iter", env("KMEANS_MAX_ITER", "100"))),
        "--kmeans-tol", str(tree.get("kmeans_tol", env("KMEANS_TOL", "1e-4"))),
        "--analyst-tokenizer", env("ANALYST_TOKENIZER"),
        "--analyst-max-output-tokens", env("ANALYST_MAX_OUTPUT_TOKENS", "4096"),
        "--analyst-dynamic-max-output-tokens", env("ANALYST_DYNAMIC_MAX_OUTPUT_TOKENS", "8192"),
        "--rollout-client-timeout-seconds", env("ROLLOUT_CLIENT_TIMEOUT_SECONDS", "600"),
        "--skill-export-min-level", str(skill_export.get("min_level", -1) if skill_export.get("min_level") is not None else -1),
        "--skill-export-max-level", str(skill_export.get("max_level", -1) if skill_export.get("max_level") is not None else -1),
    ]
    optional_arg(cmd, "--records-path", env("RECORDS_PATH"))
    optional_arg(cmd, "--reuse-train-run-dir", env("REUSE_TRAIN_RUN_DIR"))
    if retrieval_only:
        baseline_tree = env("BASELINE_TREE_DIR")
        if not baseline_tree:
            raise SystemExit("retrieval-only variants require BASELINE_TREE_DIR")
        optional_arg(cmd, "--reuse-tree-dir", baseline_tree)
    if bool_flag("RESUME", "true") == "false":
        cmd.append("--no-resume")
    else:
        cmd.append("--resume")

    manifest = {
        "variant_json": str(variant_path),
        "variant": variant,
        "run_dir": str(run_dir),
        "command": cmd,
    }
    (run_dir / "ablation_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ablation] variant=", name)
    print("[ablation] run_dir=", run_dir)
    print("[ablation] command=", " ".join(cmd))
    raise SystemExit(subprocess.run(cmd, cwd=str(repo), env=os.environ.copy()).returncode)


if __name__ == "__main__":
    main()
