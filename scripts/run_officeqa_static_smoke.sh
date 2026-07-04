#!/usr/bin/env bash
set -euo pipefail

# SkillOpt-compatible OfficeQA -> DynaMix static nodebank smoke/full runner.
# Override variables from the shell; this file intentionally exposes the main
# protocol knobs so another agent does not have to guess hidden defaults.

REPO_ROOT="${REPO_ROOT:-/mnt/data/yaodong/codes/DynaMix2skill}"
PYTHON="${PYTHON:-/home/yaodong/miniconda3/envs/stableskill-skillrl/bin/python}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/officeqa_static_smoke_$(date +%Y%m%d_%H%M%S)}"
FULL="${FULL:-false}"

SPLIT_DIR="${SPLIT_DIR:-/mnt/data/yaodong/officeqa/splits}"
DOCS_DIR="${DOCS_DIR:-/mnt/data/yaodong/officeqa/hf/treasury_bulletins_parsed}"
REWARD_PATH="${REWARD_PATH:-/mnt/data/yaodong/officeqa/reward.py}"

TRAIN_SPLITS="${TRAIN_SPLITS:-train,val}"
HELDOUT_SPLIT="${HELDOUT_SPLIT:-test}"
TRAIN_START="${TRAIN_START:-0}"
HELDOUT_START="${HELDOUT_START:-0}"
if [[ "$FULL" == "1" || "$FULL" == "true" || "$FULL" == "yes" ]]; then
  TRAIN_END="${TRAIN_END:-}"
  HELDOUT_END="${HELDOUT_END:-}"
else
  TRAIN_END="${TRAIN_END:-5}"
  HELDOUT_END="${HELDOUT_END:-3}"
fi

MODEL="${MODEL:-Qwen3.5-9B-AWQ}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://asmiatbrqksz.10.27.127.9.nip.io/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-0.6}"
GENERATION_TIMEOUT="${GENERATION_TIMEOUT:-1200}"
THINKING="${THINKING:-true}"
WORKERS="${WORKERS:-8}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-30}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-}"

EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://10.26.1.184:18007/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen3-Embedding-8B}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-EMPTY}"
EMBEDDING_TOKENIZER="${EMBEDDING_TOKENIZER:-/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-8B}"
EMBEDDING_MAX_MODEL_LEN="${EMBEDDING_MAX_MODEL_LEN:-32000}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
CHUNK_TOKENS="${CHUNK_TOKENS:-28000}"
CHUNK_OVERLAP_TOKENS="${CHUNK_OVERLAP_TOKENS:-1000}"
ANALYST_TOKENIZER="${ANALYST_TOKENIZER:-}"
ANALYST_TOKENIZER_REQUIRED="${ANALYST_TOKENIZER_REQUIRED:-true}"
ANALYST_ALLOW_REGEX_TOKENIZER_FALLBACK="${ANALYST_ALLOW_REGEX_TOKENIZER_FALLBACK:-false}"
ANALYSIS_BUNDLE_MAX_CHARS="${ANALYSIS_BUNDLE_MAX_CHARS:-60000}"
ANALYSIS_BUNDLE_MAX_STEPS="${ANALYSIS_BUNDLE_MAX_STEPS:-12}"
ANALYSIS_BUNDLE_MAX_STEP_CHARS="${ANALYSIS_BUNDLE_MAX_STEP_CHARS:-6000}"
ANALYSIS_BUNDLE_MAX_FINAL_RESPONSE_CHARS="${ANALYSIS_BUNDLE_MAX_FINAL_RESPONSE_CHARS:-12000}"

SKILLBANK_TOP_K="${SKILLBANK_TOP_K:-10}"
MAX_LEVELS="${MAX_LEVELS:-8}"
SUMMARY_MAX_MODEL_TOKENS="${SUMMARY_MAX_MODEL_TOKENS:-100000}"
SUMMARY_BUDGET_RATIO="${SUMMARY_BUDGET_RATIO:-0.85}"
GMM_MIN_SPLIT_SIZE="${GMM_MIN_SPLIT_SIZE:-2}"
GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT="${GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT:-2}"

mkdir -p "$RUN_DIR/logs"
cd "$REPO_ROOT"

if [[ "$OPENAI_BASE_URL" == "http://127.0.0.1:18002/v1" || "$OPENAI_BASE_URL" == "http://localhost:18002/v1" ]]; then
  if [[ "${ALLOW_OFFICEQA_LOCAL_18002:-}" != "1" ]]; then
    echo "Refusing local A5000 18002 for OfficeQA. Use the external AWQ endpoint/tunnel, or set ALLOW_OFFICEQA_LOCAL_18002=1 for debug." >&2
    exit 2
  fi
fi

OPENAI_API_KEY_ENV_ARG=()
if [[ "$OPENAI_API_KEY" != "EMPTY" && -n "$OPENAI_API_KEY" ]]; then
  export OFFICEQA_OPENAI_API_KEY="$OPENAI_API_KEY"
  OPENAI_API_KEY_ENV_ARG=(--openai-api-key-env OFFICEQA_OPENAI_API_KEY)
  OPENAI_API_KEY="EMPTY"
fi
EMBEDDING_API_KEY_ENV_ARG=()
if [[ "$EMBEDDING_API_KEY" != "EMPTY" && -n "$EMBEDDING_API_KEY" ]]; then
  export OFFICEQA_EMBEDDING_API_KEY="$EMBEDDING_API_KEY"
  EMBEDDING_API_KEY_ENV_ARG=(--embedding-api-key-env OFFICEQA_EMBEDDING_API_KEY)
  EMBEDDING_API_KEY="EMPTY"
fi
if [[ "${ANALYST_TOKENIZER_REQUIRED,,}" == "true" && "${ANALYST_ALLOW_REGEX_TOKENIZER_FALLBACK,,}" != "true" && -z "$ANALYST_TOKENIZER" ]]; then
  echo "[preflight:error] ANALYST_TOKENIZER is required for real tree builds. Set it to the generation model tokenizer; do not use EMBEDDING_TOKENIZER for analyst prompt budgeting." >&2
  exit 2
fi

cmd=(
  "$PYTHON" scripts/run_officeqa_dynamix_experiment.py
  --run-dir "$RUN_DIR"
  --split-dir "$SPLIT_DIR"
  --docs-dir "$DOCS_DIR"
  --reward-path "$REWARD_PATH"
  --train-splits "$TRAIN_SPLITS"
  --heldout-split "$HELDOUT_SPLIT"
  --train-start "$TRAIN_START"
  --heldout-start "$HELDOUT_START"
  --model "$MODEL"
  --openai-base-url "$OPENAI_BASE_URL"
  --openai-api-key "$OPENAI_API_KEY"
  "${OPENAI_API_KEY_ENV_ARG[@]}"
  --generation-temperature "$GENERATION_TEMPERATURE"
  --generation-timeout "$GENERATION_TIMEOUT"
  --thinking "$THINKING"
  --workers "$WORKERS"
  --max-tool-turns "$MAX_TOOL_TURNS"
  --embedding-base-url "$EMBEDDING_BASE_URL"
  --embedding-model "$EMBEDDING_MODEL"
  --embedding-api-key "$EMBEDDING_API_KEY"
  "${EMBEDDING_API_KEY_ENV_ARG[@]}"
  --embedding-max-model-len "$EMBEDDING_MAX_MODEL_LEN"
  --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
  --chunk-tokens "$CHUNK_TOKENS"
  --chunk-overlap-tokens "$CHUNK_OVERLAP_TOKENS"
  --analyst-tokenizer-required "$ANALYST_TOKENIZER_REQUIRED"
  --analyst-allow-regex-tokenizer-fallback "$ANALYST_ALLOW_REGEX_TOKENIZER_FALLBACK"
  --analysis-bundle-max-chars "$ANALYSIS_BUNDLE_MAX_CHARS"
  --analysis-bundle-max-steps "$ANALYSIS_BUNDLE_MAX_STEPS"
  --analysis-bundle-max-step-chars "$ANALYSIS_BUNDLE_MAX_STEP_CHARS"
  --analysis-bundle-max-final-response-chars "$ANALYSIS_BUNDLE_MAX_FINAL_RESPONSE_CHARS"
  --skillbank-top-k "$SKILLBANK_TOP_K"
  --max-levels "$MAX_LEVELS"
  --summary-max-model-tokens "$SUMMARY_MAX_MODEL_TOKENS"
  --summary-budget-ratio "$SUMMARY_BUDGET_RATIO"
  --gmm-min-split-size "$GMM_MIN_SPLIT_SIZE"
  --gmm-min-effective-samples-per-component "$GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT"
  --resume
  --run-heldout
)

if [[ -n "$TRAIN_END" ]]; then
  cmd+=(--train-end "$TRAIN_END")
fi
if [[ -n "$HELDOUT_END" ]]; then
  cmd+=(--heldout-end "$HELDOUT_END")
fi
if [[ -n "$MAX_COMPLETION_TOKENS" ]]; then
  cmd+=(--max-completion-tokens "$MAX_COMPLETION_TOKENS")
fi
if [[ -n "$EMBEDDING_TOKENIZER" ]]; then
  cmd+=(--embedding-tokenizer "$EMBEDDING_TOKENIZER")
fi
if [[ -n "$ANALYST_TOKENIZER" ]]; then
  cmd+=(--analyst-tokenizer "$ANALYST_TOKENIZER")
fi

printf '[officeqa-run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" 2>&1 | tee "$RUN_DIR/logs/officeqa_experiment.log"
