#!/usr/bin/env bash
# Copy this file outside the repo and edit local paths/endpoints before running
# static ablation variants.

export REPO_ROOT="${REPO_ROOT:-/mnt/data/yaodong/codes/DynaMix2skill}"
export DYNAMIX_PYTHON="${DYNAMIX_PYTHON:-/home/yaodong/miniconda3/envs/stableskill-skillrl/bin/python}"
export DATA_PATH="${DATA_PATH:-/mnt/data/yaodong/codes/Trace2Skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400}"

# Optional: reuse already collected train records to avoid rerunning train rollout.
export RECORDS_PATH="${RECORDS_PATH:-}"
export REUSE_TRAIN_RUN_DIR="${REUSE_TRAIN_RUN_DIR:-}"

# Optional: for retrieve_l1_only / retrieve_l2plus_only, reuse a full static tree.
export REUSE_TREE_DIR="${REUSE_TREE_DIR:-}"

export RUN_GROUP="${RUN_GROUP:-$REPO_ROOT/runs/ablations/static}"
export TRAIN_START="${TRAIN_START:-0}"
export TRAIN_END="${TRAIN_END:-200}"
export HELDOUT_START="${HELDOUT_START:-200}"
export HELDOUT_END="${HELDOUT_END:-400}"
export WORKERS="${WORKERS:-4}"

export MODEL="${MODEL:-Qwen3.5-9B-AWQ}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:18002/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:8017/v1}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen3-Embedding-0.6B}"
export EMBEDDING_TOKENIZER="${EMBEDDING_TOKENIZER:-/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-0___6B}"

export MAX_TURNS="${MAX_TURNS:-30}"
export THINKING="${THINKING:-true}"
export SKILLBANK_TOP_K="${SKILLBANK_TOP_K:-10}"

export SUMMARY_MAX_MODEL_TOKENS="${SUMMARY_MAX_MODEL_TOKENS:-100000}"
export SUMMARY_BUDGET_RATIO="${SUMMARY_BUDGET_RATIO:-0.85}"
export SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS="${SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS:-8000}"

export EMBEDDING_MAX_MODEL_LEN="${EMBEDDING_MAX_MODEL_LEN:-8192}"
export EMBEDDING_MAX_INPUT_TOKENS="${EMBEDDING_MAX_INPUT_TOKENS:-8000}"
export CHUNKED_EMBEDDING_ENABLED="${CHUNKED_EMBEDDING_ENABLED:-true}"
export CHUNKED_EMBEDDING_CHUNK_TOKENS="${CHUNKED_EMBEDDING_CHUNK_TOKENS:-7600}"
export CHUNKED_EMBEDDING_OVERLAP_TOKENS="${CHUNKED_EMBEDDING_OVERLAP_TOKENS:-512}"
