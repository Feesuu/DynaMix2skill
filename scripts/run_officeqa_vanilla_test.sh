#!/usr/bin/env bash
set -euo pipefail

# SkillOpt-compatible OfficeQA vanilla no-skill baseline.
# This intentionally does not build/read DynaMix tree or nodebank.

REPO_ROOT="${REPO_ROOT:-/mnt/data/yaodong/codes/DynaMix2skill}"
PYTHON="${PYTHON:-/home/yaodong/miniconda3/envs/stableskill-skillrl/bin/python}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/officeqa_vanilla_test_$(date +%Y%m%d_%H%M%S)}"

SPLIT_DIR="${SPLIT_DIR:-/mnt/data/yaodong/officeqa/splits}"
DOCS_DIR="${DOCS_DIR:-/mnt/data/yaodong/officeqa/hf/treasury_bulletins_parsed}"
REWARD_PATH="${REWARD_PATH:-/mnt/data/yaodong/officeqa/reward.py}"
SPLIT="${SPLIT:-test}"
START="${START:-0}"
END="${END:-}"
EXPECTED_COUNT="${EXPECTED_COUNT:-172}"

MODEL="${MODEL:-Qwen3.5-9B-AWQ}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://asmiatbrqksz.10.27.127.9.nip.io/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-0.6}"
GENERATION_TIMEOUT="${GENERATION_TIMEOUT:-1200}"
THINKING="${THINKING:-true}"
WORKERS="${WORKERS:-8}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-30}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-}"

export no_proxy="${no_proxy:-*}"
export NO_PROXY="${NO_PROXY:-*}"

mkdir -p "$RUN_DIR/logs"
cd "$REPO_ROOT"

OPENAI_API_KEY_ENV_ARG=()
if [[ "$OPENAI_API_KEY" != "EMPTY" && -n "$OPENAI_API_KEY" ]]; then
  export OFFICEQA_OPENAI_API_KEY="$OPENAI_API_KEY"
  OPENAI_API_KEY_ENV_ARG=(--openai-api-key-env OFFICEQA_OPENAI_API_KEY)
  OPENAI_API_KEY="EMPTY"
fi

cmd=(
  "$PYTHON" scripts/run_officeqa_vanilla_test.py
  --run-dir "$RUN_DIR"
  --split-dir "$SPLIT_DIR"
  --docs-dir "$DOCS_DIR"
  --reward-path "$REWARD_PATH"
  --split "$SPLIT"
  --start "$START"
  --expected-count "$EXPECTED_COUNT"
  --model "$MODEL"
  --openai-base-url "$OPENAI_BASE_URL"
  --openai-api-key "$OPENAI_API_KEY"
  "${OPENAI_API_KEY_ENV_ARG[@]}"
  --generation-temperature "$GENERATION_TEMPERATURE"
  --generation-timeout "$GENERATION_TIMEOUT"
  --thinking "$THINKING"
  --workers "$WORKERS"
  --max-tool-turns "$MAX_TOOL_TURNS"
  --resume
)

if [[ -n "$END" ]]; then
  cmd+=(--end "$END")
fi
if [[ -n "$MAX_COMPLETION_TOKENS" ]]; then
  cmd+=(--max-completion-tokens "$MAX_COMPLETION_TOKENS")
fi

printf '[officeqa-vanilla]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$RUN_DIR/logs/officeqa_vanilla.log"
