#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yaodong/codes/DynaMix2skill_contract_cut_ebst}"
PYTHON="${PYTHON:-/home/yaodong/miniconda3/envs/stableskill-skillrl/bin/python}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/contract_cut_ebst_$(date +%Y%m%d_%H%M%S)}"
RECORDS="${RECORDS:-/mnt/data/yaodong/codes/DynaMix2skill/runs/spreadsheet_splitthink_rolloutfalse_analysttrue_best52_20260720_180410/ordered_records.json}"
DATASET="${DATASET:-/mnt/data/yaodong/codes/DynaMix2skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://10.26.1.184:18085/v1}"
MODEL="${MODEL:-Qwen3.5-9B-AWQ}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://10.26.1.184:18007/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen3-Embedding-8B}"
EMBEDDING_TOKENIZER="${EMBEDDING_TOKENIZER:-/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-8B}"
API_KEY_ENV_VAR="${API_KEY_ENV_VAR:-YD5_API_KEY}"

if [[ -z "${!API_KEY_ENV_VAR:-}" ]]; then
  echo "Required API key environment variable is empty: ${API_KEY_ENV_VAR}" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PATH="$(dirname "${PYTHON}"):${PATH}"
export SAL_DISABLE_OPENCL=1

mkdir -p "${RUN_DIR}/logs"
exec 9>"${RUN_DIR}/.contract_cut_wrapper.lock"
if ! flock -n 9; then
  echo "Another wrapper already owns this run directory: ${RUN_DIR}" >&2
  exit 3
fi
cd "${REPO_ROOT}"

cmd=(
  "${PYTHON}" scripts/run_contract_cut_ebst_experiment.py
  --stage all
  --resume
  --run-dir "${RUN_DIR}"
  --records "${RECORDS}"
  --dataset-path "${DATASET}"
  --train-start 0
  --train-end 200
  --heldout-start 200
  --heldout-end 400
  --openai-base-url "${OPENAI_BASE_URL}"
  --model "${MODEL}"
  --api-key-env-var "${API_KEY_ENV_VAR}"
  --workers 8
  --batch-size 8
  --llm-timeout-seconds 1200
  --max-turns 30
  --embedding-base-url "${EMBEDDING_BASE_URL}"
  --embedding-model "${EMBEDDING_MODEL}"
  --embedding-tokenizer "${EMBEDDING_TOKENIZER}"
  --embedding-max-model-len 32000
  --embedding-batch-size 8
  --embedding-workers 8
  --max-entries 8
  --beta 1.0
  --analyst-prompt-tokens 92000
  --compiler-prompt-tokens 92000
  --python-executable "${PYTHON}"
)

printf '[run]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}" 2>&1 | tee "${RUN_DIR}/logs/experiment_wrapper.log"
