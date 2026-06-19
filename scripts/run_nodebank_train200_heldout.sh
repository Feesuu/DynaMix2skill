#!/usr/bin/env bash
set -euo pipefail

# DynaMix nodebank full-flow experiment wrapper.
#
# This script is intentionally verbose: remote agents should be able to run the
# experiment without guessing what each parameter means.  The current retrieval
# query used by cli_skill_preloaded is:
#
#   instruction
#   Task type: <instruction_type>
#
# It does NOT use answer_position for retrieval.  answer_position is still shown
# to the spreadsheet-solving agent in the normal SpreadsheetBench task prompt.

# Repository root containing run_spreadsheetbench.py and scripts/.
REPO_ROOT="${REPO_ROOT:-/mnt/data/yaodong/codes/DynaMix2skill}"

# Conda environment used for every Python process, including agent bash actions.
CONDA_ENV="${CONDA_ENV:-/home/yaodong/miniconda3/envs/stableskill-skillrl}"

# Exact Python executable.  The runner also prepends its bin dir to PATH so bare
# `python` inside Trace2Skill agent actions resolves to this same environment.
DYNAMIX_PYTHON="${DYNAMIX_PYTHON:-$CONDA_ENV/bin/python}"

# SpreadsheetBench dataset root.  It must contain dataset.json and spreadsheet/.
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/spreadsheetbench_verified/spreadsheetbench_verified_400}"

# Human-readable run name.  Override RUN_DIR directly if you want a fixed path.
RUN_NAME="${RUN_NAME:-qwen_train200_nodebank_tasktype_$(date +%Y%m%d_%H%M%S)}"

# Output directory for all logs, records, tree files, nodebank index, and heldout
# outputs.  Keep each experiment in a fresh directory unless intentionally resuming.
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/$RUN_NAME}"

# Training split start index, inclusive, in dataset.json order.
TRAIN_START="${TRAIN_START:-0}"

# Training split end index, exclusive.  0..200 means 200 train records.
TRAIN_END="${TRAIN_END:-200}"

# Heldout split start index, inclusive.  For verified_400, 200..400 is heldout.
HELDOUT_START="${HELDOUT_START:-200}"

# Heldout split end index, exclusive.
HELDOUT_END="${HELDOUT_END:-400}"

# Global worker count for train rollout, DynaMix analyst/embedding concurrency,
# and heldout rollout.  Keep this aligned with the serving capacity.
WORKERS="${WORKERS:-4}"

# Generation model name passed to OpenAI-compatible server.
MODEL="${MODEL:-Qwen3.5-9B}"

# OpenAI-compatible chat/completion endpoint for the generation model.
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:18002/v1}"

# API key for the generation server.  Local vLLM-compatible servers usually use EMPTY.
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# OpenAI-compatible embedding endpoint used for trajectory/node embeddings and
# heldout nodebank retrieval.
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:8017/v1}"

# Embedding model name served by EMBEDDING_BASE_URL.
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen3-Embedding-0.6B}"

# Local tokenizer path/name used for token counting and long-trace chunking.
EMBEDDING_TOKENIZER="${EMBEDDING_TOKENIZER:-/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-0___6B}"

# Maximum ReAct turns per train/heldout spreadsheet task.
MAX_TURNS="${MAX_TURNS:-30}"

# Unified Qwen thinking flag for Trace2Skill rollout and DynaMix analyst.
# Allowed values: true, false, null.  Use false for faster non-thinking tests.
THINKING="${THINKING:-false}"

# Number of retrieved nodebank experience nodes injected into each heldout system prompt.
SKILLBANK_TOP_K="${SKILLBANK_TOP_K:-10}"

# Whether to reuse completed stage markers in RUN_DIR.  Set RESUME=false for a
# clean rerun in an existing directory.
RESUME="${RESUME:-true}"

cd "$REPO_ROOT"
mkdir -p "$RUN_DIR/logs"

if [[ ! -x "$DYNAMIX_PYTHON" ]]; then
  echo "ERROR: DYNAMIX_PYTHON is not executable: $DYNAMIX_PYTHON" >&2
  exit 2
fi

export CONDA_ENV
export DYNAMIX_PYTHON
export PATH="$(dirname "$DYNAMIX_PYTHON"):$PATH"

echo "[preflight] repo=$REPO_ROOT"
echo "[preflight] run_dir=$RUN_DIR"
echo "[preflight] python=$(command -v python)"
"$DYNAMIX_PYTHON" -c 'import sys; print("[preflight] sys.executable=" + sys.executable)'

cmd=(
  "$DYNAMIX_PYTHON" "scripts/run_dynamix_trace2skill_experiment.py"

  # Dataset root containing dataset.json and spreadsheet/.
  "--data-path" "$DATA_PATH"

  # Directory where this run writes all artifacts.
  "--run-dir" "$RUN_DIR"

  # Train records: [TRAIN_START, TRAIN_END).
  "--train-start" "$TRAIN_START"
  "--train-end" "$TRAIN_END"

  # Heldout records: [HELDOUT_START, HELDOUT_END).
  "--heldout-start" "$HELDOUT_START"
  "--heldout-end" "$HELDOUT_END"

  # Shared concurrency level.
  "--workers" "$WORKERS"

  # Generation model and OpenAI-compatible endpoint.
  "--model" "$MODEL"
  "--openai-base-url" "$OPENAI_BASE_URL"
  "--openai-api-key" "$OPENAI_API_KEY"

  # Embedding endpoint/model/tokenizer.
  "--embedding-base-url" "$EMBEDDING_BASE_URL"
  "--embedding-model" "$EMBEDDING_MODEL"
  "--embedding-tokenizer" "$EMBEDDING_TOKENIZER"

  # Python executable propagated to all experiment stages.
  "--python-executable" "$DYNAMIX_PYTHON"

  # Max ReAct turns per spreadsheet task.
  "--max-turns" "$MAX_TURNS"

  # Unified Qwen thinking setting.
  "--thinking" "$THINKING"

  # Heldout nodebank retrieval top-k.
  "--skillbank-top-k" "$SKILLBANK_TOP_K"
)

if [[ "$RESUME" == "true" ]]; then
  # Reuse completed stage markers and outputs in RUN_DIR.
  cmd+=("--resume")
elif [[ "$RESUME" == "false" ]]; then
  # Do not reuse completed stage markers in RUN_DIR.
  cmd+=("--no-resume")
else
  echo "ERROR: RESUME must be true or false, got: $RESUME" >&2
  exit 2
fi

printf '[run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$RUN_DIR/logs/experiment_wrapper.log"
