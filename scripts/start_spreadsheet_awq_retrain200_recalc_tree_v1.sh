#!/usr/bin/env bash
set -euo pipefail

# Fresh SpreadsheetBench DynaMix run.
# This script intentionally re-runs train 0:200 under the current model/harness,
# evaluates train with LibreOffice recalc, extracts fresh records, builds a new
# static DynaMix tree, exports nodebank, and evaluates heldout 200:400.
#
# Do not add --records-path, --reuse-train-run-dir, or --reuse-tree-dir here:
# those would silently turn this back into the old mixed-protocol experiment.

REPO_ROOT="${REPO_ROOT:-/mnt/data/yaodong/codes/DynaMix2skill}"
CONDA_ENV="${CONDA_ENV:-/home/yaodong/miniconda3/envs/stableskill-skillrl}"
DYNAMIX_PYTHON="${DYNAMIX_PYTHON:-$CONDA_ENV/bin/python}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/spreadsheetbench_verified/spreadsheetbench_verified_400}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-spreadsheet_awq_retrain200_recalc_tree_v1_$TIMESTAMP}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/$RUN_NAME}"
TMUX_SESSION="${TMUX_SESSION:-$RUN_NAME}"

MODEL="${MODEL:-Qwen3.5-9B-AWQ}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://evirdwimyrmm.10.27.127.9.nip.io/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://10.26.1.184:18007/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen3-Embedding-8B}"
EMBEDDING_TOKENIZER="${EMBEDDING_TOKENIZER:-/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-8B}"
ANALYST_TOKENIZER="${ANALYST_TOKENIZER:-/mnt/data/grouph_share/models/transformer_llm/Qwen3.5-9B-AWQ}"

TRAIN_START="${TRAIN_START:-0}"
TRAIN_END="${TRAIN_END:-200}"
HELDOUT_START="${HELDOUT_START:-200}"
HELDOUT_END="${HELDOUT_END:-400}"

# LLM rollout/tree/heldout concurrency. Embedding stays separate because it runs
# on the local A5000 embedding server.
WORKERS="${WORKERS:-32}"
GENERATION_MAX_CONCURRENCY="${GENERATION_MAX_CONCURRENCY:-32}"
EMBEDDING_MAX_CONCURRENCY="${EMBEDDING_MAX_CONCURRENCY:-8}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"

MAX_TURNS="${MAX_TURNS:-30}"
THINKING="${THINKING:-true}"
SKILLBANK_TOP_K="${SKILLBANK_TOP_K:-10}"

ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.0}"
ROLLOUT_TIMEOUT_SECONDS="${ROLLOUT_TIMEOUT_SECONDS:-1200}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-0.6}"
GENERATION_TIMEOUT_SECONDS="${GENERATION_TIMEOUT_SECONDS:-1200}"

EMBEDDING_MAX_MODEL_LEN="${EMBEDDING_MAX_MODEL_LEN:-32000}"
EMBEDDING_MAX_INPUT_TOKENS="${EMBEDDING_MAX_INPUT_TOKENS:-32000}"
CHUNK_TOKENS="${CHUNK_TOKENS:-28000}"
CHUNK_OVERLAP_TOKENS="${CHUNK_OVERLAP_TOKENS:-1000}"

GMM_MIN_SPLIT_SIZE="${GMM_MIN_SPLIT_SIZE:-4}"
GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT="${GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT:-2}"
SOFT_CUMULATIVE_MASS_COVERAGE="${SOFT_CUMULATIVE_MASS_COVERAGE:-0.90}"
SOFT_MAX_MEMBERSHIP_GAP="${SOFT_MAX_MEMBERSHIP_GAP:-0.25}"
SUMMARY_BUDGET_RATIO="${SUMMARY_BUDGET_RATIO:-0.85}"

export SAL_DISABLE_OPENCL=1
export DYNAMIX_OPENAI_SSL_VERIFY=false
export DYNAMIX_OPENAI_TRUST_ENV=false
export MODEL="$MODEL"
export OPENAI_BASE_URL="$OPENAI_BASE_URL"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export EMBED_BASE_URL="$EMBEDDING_BASE_URL"
export EMBED_MODEL="$EMBEDDING_MODEL"
export EMBED_TOKENIZER="$EMBEDDING_TOKENIZER"
export PATH="$CONDA_ENV/bin:$PATH"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

cd "$REPO_ROOT"

if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to run fresh protocol in non-empty RUN_DIR: $RUN_DIR" >&2
  echo "Use a new RUN_DIR or remove stale artifacts after manual review." >&2
  exit 1
elif [[ -e "$RUN_DIR" && ! -d "$RUN_DIR" ]]; then
  echo "Refusing to run fresh protocol because RUN_DIR exists but is not a directory: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/logs"

cat > "$RUN_DIR/RUN_PROTOCOL.md" <<EOF
# $RUN_NAME

- scenario: fresh static SpreadsheetBench DynaMix build
- train: [$TRAIN_START, $TRAIN_END)
- heldout: [$HELDOUT_START, $HELDOUT_END)
- model: $MODEL
- base_url: $OPENAI_BASE_URL
- api_key: dummy/redacted
- embedding: $EMBEDDING_MODEL at $EMBEDDING_BASE_URL
- analyst_tokenizer: $ANALYST_TOKENIZER
- workers: $WORKERS
- generation_max_concurrency: $GENERATION_MAX_CONCURRENCY
- embedding_max_concurrency: $EMBEDDING_MAX_CONCURRENCY
- max_turns: $MAX_TURNS
- thinking: $THINKING
- rollout_temperature: $ROLLOUT_TEMPERATURE
- generation_temperature: $GENERATION_TEMPERATURE
- evaluator: evaluate_with_official.py, LibreOffice recalc primary
- forbidden reuse flags: --records-path, --reuse-train-run-dir, --reuse-tree-dir
- chunked_embedding: enabled, chunk_tokens=$CHUNK_TOKENS, overlap_tokens=$CHUNK_OVERLAP_TOKENS, pooling=mean
- gmm_min_split_size: $GMM_MIN_SPLIT_SIZE
- gmm_min_effective_samples_per_component: $GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT
- soft_assignment: cumulative_mass, coverage=$SOFT_CUMULATIVE_MASS_COVERAGE, max_gap=$SOFT_MAX_MEMBERSHIP_GAP
- summary_budget_ratio: $SUMMARY_BUDGET_RATIO
EOF

"$DYNAMIX_PYTHON" - <<'PY' | tee "$RUN_DIR/logs/preflight.log"
from openai import OpenAI
import httpx
import os

base_url = os.environ["OPENAI_BASE_URL"]
api_key = os.environ["OPENAI_API_KEY"]
model = os.environ.get("MODEL", "Qwen3.5-9B-AWQ")
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    http_client=httpx.Client(verify=False, trust_env=False, timeout=60.0),
)
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Reply with exactly: ready"}],
    temperature=0,
    max_tokens=32,
)
print("llm_preflight:", resp.choices[0].message.content[:200])

embed = OpenAI(base_url=os.environ["EMBED_BASE_URL"], api_key="EMPTY", timeout=60.0)
vec = embed.embeddings.create(model=os.environ["EMBED_MODEL"], input=["embedding preflight"]).data[0].embedding
print("embedding_preflight_dim:", len(vec))
PY

cat > "$RUN_DIR/launch_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export SAL_DISABLE_OPENCL=1
export DYNAMIX_OPENAI_SSL_VERIFY=false
export DYNAMIX_OPENAI_TRUST_ENV=false
export OPENAI_BASE_URL="$OPENAI_BASE_URL"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export EMBED_BASE_URL="$EMBEDDING_BASE_URL"
export EMBED_MODEL="$EMBEDDING_MODEL"
export EMBED_TOKENIZER="$EMBEDDING_TOKENIZER"
export PATH="$CONDA_ENV/bin:\$PATH"
cd "$REPO_ROOT"

"$DYNAMIX_PYTHON" scripts/run_dynamix_trace2skill_experiment.py \\
  --data-path "$DATA_PATH" \\
  --run-dir "$RUN_DIR" \\
  --train-start "$TRAIN_START" \\
  --train-end "$TRAIN_END" \\
  --heldout-start "$HELDOUT_START" \\
  --heldout-end "$HELDOUT_END" \\
  --workers "$WORKERS" \\
  --model "$MODEL" \\
  --openai-base-url "$OPENAI_BASE_URL" \\
  --openai-api-key "$OPENAI_API_KEY" \\
  --embedding-base-url "$EMBEDDING_BASE_URL" \\
  --embedding-model "$EMBEDDING_MODEL" \\
  --embedding-tokenizer "$EMBEDDING_TOKENIZER" \\
  --python-executable "$DYNAMIX_PYTHON" \\
  --max-turns "$MAX_TURNS" \\
  --thinking "$THINKING" \\
  --skillbank-top-k "$SKILLBANK_TOP_K" \\
  --tree-scenario static_build \\
  --rollout-temperature "$ROLLOUT_TEMPERATURE" \\
  --rollout-client-timeout-seconds "$ROLLOUT_TIMEOUT_SECONDS" \\
  --generation-temperature "$GENERATION_TEMPERATURE" \\
  --generation-timeout-seconds "$GENERATION_TIMEOUT_SECONDS" \\
  --generation-max-concurrency "$GENERATION_MAX_CONCURRENCY" \\
  --analyst-tokenizer "$ANALYST_TOKENIZER" \\
  --embedding-max-model-len "$EMBEDDING_MAX_MODEL_LEN" \\
  --embedding-max-input-tokens "$EMBEDDING_MAX_INPUT_TOKENS" \\
  --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \\
  --embedding-max-concurrency "$EMBEDDING_MAX_CONCURRENCY" \\
  --chunked-embedding-enabled true \\
  --chunked-embedding-chunk-tokens "$CHUNK_TOKENS" \\
  --chunked-embedding-overlap-tokens "$CHUNK_OVERLAP_TOKENS" \\
  --chunked-embedding-pooling mean \\
  --gmm-min-split-size "$GMM_MIN_SPLIT_SIZE" \\
  --gmm-min-effective-samples-per-component "$GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT" \\
  --soft-recursive-assignment cumulative_mass \\
  --soft-cumulative-mass-coverage "$SOFT_CUMULATIVE_MASS_COVERAGE" \\
  --soft-max-membership-gap "$SOFT_MAX_MEMBERSHIP_GAP" \\
  --summary-budget-ratio "$SUMMARY_BUDGET_RATIO" \\
  --no-resume 2>&1 | tee "$RUN_DIR/logs/experiment.log"
EOF

chmod +x "$RUN_DIR/launch_command.sh"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session already exists: $TMUX_SESSION" >&2
  echo "Attach with: tmux attach -t $TMUX_SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$TMUX_SESSION" "$RUN_DIR/launch_command.sh"
sleep 5
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session exited during startup: $TMUX_SESSION" >&2
  echo "Last log lines from $RUN_DIR/logs/experiment.log:" >&2
  tail -80 "$RUN_DIR/logs/experiment.log" >&2 || true
  exit 1
fi

echo "started_tmux_session=$TMUX_SESSION"
echo "run_dir=$RUN_DIR"
echo "log=$RUN_DIR/logs/experiment.log"
echo "attach: tmux attach -t $TMUX_SESSION"
