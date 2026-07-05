#!/usr/bin/env bash
set -euo pipefail

# Copy this file to base_env.local.sh and edit values there.

export REPO_ROOT="/mnt/data/yaodong/codes/DynaMix2skill"
export DATA_PATH="/mnt/data/yaodong/codes/Trace2Skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400"
export RUN_ROOT="/mnt/data/yaodong/codes/DynaMix2skill/runs/ablations/static"
export DYNAMIX_PYTHON="/home/yaodong/miniconda3/envs/stableskill-skillrl/bin/python"

# Reuse existing train trajectories when possible. Set exactly one of these.
export RECORDS_PATH=""
export REUSE_TRAIN_RUN_DIR=""

# Required only for retrieve_* variants. Retrieval-only variants also require
# RECORDS_PATH or REUSE_TRAIN_RUN_DIR above, so they do not re-run train stages.
export BASELINE_TREE_DIR=""

export MODEL="Qwen3.5-9B-AWQ"
export OPENAI_BASE_URL="http://127.0.0.1:18002/v1"
export OPENAI_API_KEY="EMPTY"

export EMBEDDING_BASE_URL="http://10.26.1.184:18007/v1"
export EMBEDDING_MODEL="Qwen3-Embedding-8B"
export EMBEDDING_TOKENIZER="/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-8B"

export TRAIN_START="0"
export TRAIN_END="200"
export HELDOUT_START="200"
export HELDOUT_END="400"
export WORKERS="8"
export RANDOM_SEED="42"

export MAX_TURNS="30"
export THINKING="true"
export SKILLBANK_TOP_K="10"
export RESUME="true"

export GENERATION_TEMPERATURE="0.6"
export GENERATION_TIMEOUT_SECONDS="1200"
export GENERATION_MAX_CONCURRENCY="$WORKERS"

export EMBEDDING_MAX_MODEL_LEN="32000"
export EMBEDDING_MAX_INPUT_TOKENS="32000"
export EMBEDDING_BATCH_SIZE="8"
export EMBEDDING_MAX_CONCURRENCY="8"
export CHUNKED_EMBEDDING_CHUNK_TOKENS="28000"
export CHUNKED_EMBEDDING_OVERLAP_TOKENS="1000"

export SUMMARY_MAX_MODEL_TOKENS="100000"
export SUMMARY_BUDGET_RATIO="0.85"
export SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS="8000"

export GMM_MIN_SPLIT_SIZE="2"
export GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT="2"
export GMM_ABS_KMAX="64"

export KMEANS_FIXED_K="8"
export KMEANS_MIN_K="1"
export KMEANS_NUM_RESTARTS="5"
export KMEANS_MAX_ITER="100"
export KMEANS_TOL="1.0e-4"

export SOFT_CUMULATIVE_MASS_COVERAGE="0.90"
export SOFT_MAX_MEMBERSHIP_GAP="0.25"
export SOFT_MIN_MEMBERSHIP_WEIGHT="0.05"

export ANALYST_TOKENIZER="$MODEL"
export ANALYST_MAX_OUTPUT_TOKENS="4096"
export ANALYST_DYNAMIC_MAX_OUTPUT_TOKENS="8192"

export ROLLOUT_CLIENT_TIMEOUT_SECONDS="600"
