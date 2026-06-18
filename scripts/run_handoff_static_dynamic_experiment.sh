#!/usr/bin/env bash
set -euo pipefail

# DynaMix2skill handoff launcher.
#
# Purpose:
#   Run the complete SpreadsheetBench flow with the current nodebank pipeline:
#   train rollout -> train eval -> trace extraction -> DynaMix tree build ->
#   nodebank export -> heldout rollout with top-k retrieved experiences ->
#   heldout eval.
#
# Runnable example 1: static rebuild from train[0:200].
#   cd /mnt/data/yaodong/codes/DynaMix2skill
#   export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
#   export RUN_GROUP="$PWD/runs/qwen35_awq_train200_nodebank_$(date +%Y%m%d_%H%M%S)"
#   TREE_SCENARIO=static_build \
#   RUN_DIR="$RUN_GROUP" \
#   WORKERS=8 \
#   MODEL=Qwen3.5-9B-AWQ \
#   OPENAI_BASE_URL=http://asmiatbrqksz.10.27.127.9.nip.io/v1 \
#   EMBEDDING_BASE_URL=http://127.0.0.1:8017/v1 \
#   EMBEDDING_MODEL=Qwen3-Embedding-0.6B \
#   EMBEDDING_TOKENIZER=/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-0___6B \
#   bash scripts/run_handoff_static_dynamic_experiment.sh
#
# Runnable example 2: dynamic update with the same train artifacts.
# Run this after example 1.  It reuses "$RUN_GROUP/train_artifacts" and writes
# dynamic outputs to "$RUN_GROUP/scenarios/dynamic_update".
#   cd /mnt/data/yaodong/codes/DynaMix2skill
#   export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
#   TREE_SCENARIO=dynamic_update \
#   RUN_DIR="$RUN_GROUP" \
#   WORKERS=8 \
#   MODEL=Qwen3.5-9B-AWQ \
#   OPENAI_BASE_URL=http://asmiatbrqksz.10.27.127.9.nip.io/v1 \
#   EMBEDDING_BASE_URL=http://127.0.0.1:8017/v1 \
#   EMBEDDING_MODEL=Qwen3-Embedding-0.6B \
#   EMBEDDING_TOKENIZER=/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-0___6B \
#   bash scripts/run_handoff_static_dynamic_experiment.sh
#
# If train[0:200] records were already produced elsewhere, add either:
#   RECORDS_PATH=/absolute/path/to/records.json
# or:
#   REUSE_TRAIN_RUN_DIR=/absolute/path/to/run_dir_with_records_json
# to either example.  The script will then skip train rollout/eval/extraction
# and start directly from DynaMix tree build.
#
# Optional tmux wrapper:
#   tmux new-session -d -s dynamix_static 'cd /mnt/data/yaodong/codes/DynaMix2skill && TREE_SCENARIO=static_build RUN_DIR=/mnt/data/yaodong/codes/DynaMix2skill/runs/qwen35_awq_train200_nodebank bash scripts/run_handoff_static_dynamic_experiment.sh'
#
# API key policy:
#   This handoff targets self-hosted vLLM endpoints.  The default API key is the
#   dummy value EMPTY.  Non-empty keys are exported through OPENAI_API_KEY, not
#   printed in the command line or written raw into DynaMix runtime artifacts.

# ---------------------------------------------------------------------------
# Repository, Python, data, and run location.
# ---------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-/mnt/data/yaodong/codes/DynaMix2skill}"
CONDA_ENV="${CONDA_ENV:-/home/yaodong/miniconda3/envs/stableskill-skillrl}"
DYNAMIX_PYTHON="${DYNAMIX_PYTHON:-$CONDA_ENV/bin/python}"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/spreadsheetbench_verified/spreadsheetbench_verified_400}"

# TREE_SCENARIO must be one of:
#   static_build    : build the full tree from train[0:200] at once.
#   dynamic_update  : seed from train[0:120], then insert train[120:200] in 10 batches of 8.
TREE_SCENARIO="${TREE_SCENARIO:-static_build}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-qwen35_awq_nodebank_${RUN_STAMP}}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/$RUN_NAME}"

# Separate immutable train artifacts from per-scenario outputs.  Static and
# dynamic runs can share the same RUN_DIR/train_artifacts while writing their
# tree, nodebank, heldout, logs, and reports into different scenario dirs.
TRAIN_ARTIFACT_DIR="${TRAIN_ARTIFACT_DIR:-$RUN_DIR/train_artifacts}"
SCENARIO_OUTPUT_DIR="${SCENARIO_OUTPUT_DIR:-$RUN_DIR/scenarios/$TREE_SCENARIO}"

# Optional reuse entry points.  Set exactly one when train0-200 already exists
# and the current run should start from records.json.
RECORDS_PATH="${RECORDS_PATH:-}"
REUSE_TRAIN_RUN_DIR="${REUSE_TRAIN_RUN_DIR:-}"

# ---------------------------------------------------------------------------
# Dataset split and execution scale.
# ---------------------------------------------------------------------------

TRAIN_START="${TRAIN_START:-0}"
TRAIN_END="${TRAIN_END:-200}"
HELDOUT_START="${HELDOUT_START:-200}"
HELDOUT_END="${HELDOUT_END:-400}"

# Shared worker count for train rollout, DynaMix generation/embedding, and heldout rollout.
WORKERS="${WORKERS:-8}"

# Trace2Skill ReAct turn limit.
MAX_TURNS="${MAX_TURNS:-30}"

# Resume completed stages in RUN_DIR when fingerprints match.
RESUME="${RESUME:-true}"

# Print the fully expanded command and exit without launching the experiment.
DRY_RUN="${DRY_RUN:-false}"

# ---------------------------------------------------------------------------
# Generation model and rollout generation settings.
# ---------------------------------------------------------------------------

MODEL="${MODEL:-Qwen3.5-9B-AWQ}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://asmiatbrqksz.10.27.127.9.nip.io/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Unified thinking mode for train rollout, heldout rollout, and DynaMix analyst.
# Allowed: true, false, null.
THINKING="${THINKING:-true}"

# Spreadsheet task rollout sampling temperature.
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.0}"
ROLLOUT_CLIENT_TIMEOUT_SECONDS="${ROLLOUT_CLIENT_TIMEOUT_SECONDS:-600}"
ROLLOUT_CLIENT_RETRY_WAIT_SECONDS="${ROLLOUT_CLIENT_RETRY_WAIT_SECONDS:-5,10,30}"
ROLLOUT_LLM_CLIENT="${ROLLOUT_LLM_CLIENT:-openai}"
ROLLOUT_NUM_RANDOM_SEEDS="${ROLLOUT_NUM_RANDOM_SEEDS:-1}"
ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-}"
ROLLOUT_INSTANCE_IDS="${ROLLOUT_INSTANCE_IDS:-}"
ROLLOUT_MISSING_ONLY="${ROLLOUT_MISSING_ONLY:-false}"
ROLLOUT_REPEAT="${ROLLOUT_REPEAT:-1}"
ROLLOUT_SHUFFLE_SEED="${ROLLOUT_SHUFFLE_SEED:-}"
ROLLOUT_SAMPLE="${ROLLOUT_SAMPLE:-0}"

# DynaMix analyst LLM call settings.
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-0.6}"
GENERATION_TIMEOUT_SECONDS="${GENERATION_TIMEOUT_SECONDS:-600}"
GENERATION_MAX_CONCURRENCY="${GENERATION_MAX_CONCURRENCY:-$WORKERS}"
GENERATION_RETRY_WAIT_SECONDS="${GENERATION_RETRY_WAIT_SECONDS:-2,5,15}"

# ---------------------------------------------------------------------------
# Embedding model, tokenizer, long-trace chunking, and retrieval.
# ---------------------------------------------------------------------------

EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:8017/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen3-Embedding-0.6B}"
EMBEDDING_TOKENIZER="${EMBEDDING_TOKENIZER:-/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-0___6B}"

EMBEDDING_MAX_MODEL_LEN="${EMBEDDING_MAX_MODEL_LEN:-32000}"
EMBEDDING_MAX_INPUT_TOKENS="${EMBEDDING_MAX_INPUT_TOKENS:-32000}"
EMBEDDING_TRUNCATE_LONG_TEXTS="${EMBEDDING_TRUNCATE_LONG_TEXTS:-true}"
EMBEDDING_TRUNCATION_STRATEGY="${EMBEDDING_TRUNCATION_STRATEGY:-head}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
EMBEDDING_MAX_CONCURRENCY="${EMBEDDING_MAX_CONCURRENCY:-$WORKERS}"
EMBEDDING_TOKENIZER_REQUIRED="${EMBEDDING_TOKENIZER_REQUIRED:-true}"

# Long trajectory embedding uses overlapping chunks and mean pooling.
CHUNKED_EMBEDDING_ENABLED="${CHUNKED_EMBEDDING_ENABLED:-true}"
CHUNKED_EMBEDDING_CHUNK_TOKENS="${CHUNKED_EMBEDDING_CHUNK_TOKENS:-28000}"
CHUNKED_EMBEDDING_OVERLAP_TOKENS="${CHUNKED_EMBEDDING_OVERLAP_TOKENS:-1000}"
CHUNKED_EMBEDDING_POOLING="${CHUNKED_EMBEDDING_POOLING:-mean}"
CHUNKED_EMBEDDING_ADD_SPECIAL_TOKENS="${CHUNKED_EMBEDDING_ADD_SPECIAL_TOKENS:-false}"
CHUNKED_EMBEDDING_NORMALIZE_AFTER_POOLING="${CHUNKED_EMBEDDING_NORMALIZE_AFTER_POOLING:-false}"
CHUNKED_EMBEDDING_FAIL_IF_CHUNK_EXCEEDS_MODEL_LIMIT="${CHUNKED_EMBEDDING_FAIL_IF_CHUNK_EXCEEDS_MODEL_LIMIT:-true}"

# Heldout retrieval query is:
#   instruction
#   Task type: <instruction_type>
# answer_position is not used for retrieval.
SKILLBANK_TOP_K="${SKILLBANK_TOP_K:-10}"

# ---------------------------------------------------------------------------
# DynaMix tree hyperparameters.
# ---------------------------------------------------------------------------

MAX_LEVELS="${MAX_LEVELS:-8}"
SKILL_OUTPUT_DIR_NAME="${SKILL_OUTPUT_DIR_NAME:-skills}"
RANDOM_SEED="${RANDOM_SEED:-42}"

TREE_POLICY="${TREE_POLICY:-projected_gmm_bic}"
GRAPH_KIND="${GRAPH_KIND:-overlapping_experience_hierarchy}"
ALLOW_OVERLAP="${ALLOW_OVERLAP:-true}"
ALLOW_MULTI_PARENT="${ALLOW_MULTI_PARENT:-true}"
USE_SUPPORT_MASS="${USE_SUPPORT_MASS:-true}"

PROJECTION_METHOD="${PROJECTION_METHOD:-local_pca}"
PROJECTION_VARIANCE_RATIO="${PROJECTION_VARIANCE_RATIO:-0.90}"
PROJECTION_MAX_DIM="${PROJECTION_MAX_DIM:-32}"
PROJECTION_MIN_DIM="${PROJECTION_MIN_DIM:-2}"
PROJECTION_WHITEN="${PROJECTION_WHITEN:-false}"

GMM_COVARIANCE_TYPE="${GMM_COVARIANCE_TYPE:-spherical}"
GMM_NUM_RESTARTS="${GMM_NUM_RESTARTS:-5}"
GMM_KMEANS_INIT_ITERS="${GMM_KMEANS_INIT_ITERS:-15}"
GMM_MAX_ITER="${GMM_MAX_ITER:-100}"
GMM_TOL="${GMM_TOL:-0.0001}"
GMM_MIN_COVAR="${GMM_MIN_COVAR:-0.000001}"
GMM_MIN_SPLIT_SIZE="${GMM_MIN_SPLIT_SIZE:-4}"
GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT="${GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT:-2}"
GMM_ABS_KMAX="${GMM_ABS_KMAX:-64}"
GMM_MAX_CONCURRENT_CANDIDATES="${GMM_MAX_CONCURRENT_CANDIDATES:-1}"
GMM_MAX_CONCURRENT_RESTARTS="${GMM_MAX_CONCURRENT_RESTARTS:-1}"

# Soft overlapping assignment: sort memberships descending, keep entries above
# min weight and gap constraints until cumulative mass coverage is met.
SOFT_SAVE_SOFT_EDGES="${SOFT_SAVE_SOFT_EDGES:-true}"
SOFT_TOP_R_MEMBERSHIPS="${SOFT_TOP_R_MEMBERSHIPS:-2}"
SOFT_RECURSIVE_ASSIGNMENT="${SOFT_RECURSIVE_ASSIGNMENT:-cumulative_mass}"
SOFT_MIN_MEMBERSHIP_WEIGHT="${SOFT_MIN_MEMBERSHIP_WEIGHT:-0.05}"
SOFT_MAX_MEMBERSHIP_GAP="${SOFT_MAX_MEMBERSHIP_GAP:-0.25}"
SOFT_CUMULATIVE_MASS_COVERAGE="${SOFT_CUMULATIVE_MASS_COVERAGE:-0.90}"

# L0 over-budget communities are recursively refined by local GMM-BIC, then
# feasible leaves are flattened back to L0.  Only true oversize singleton leaves
# are excluded.
BUDGET_REFINEMENT_ENABLED="${BUDGET_REFINEMENT_ENABLED:-true}"
BUDGET_REFINEMENT_APPLY_TO_LEVEL="${BUDGET_REFINEMENT_APPLY_TO_LEVEL:-0}"
BUDGET_REFINEMENT_SELECTION_POLICY="${BUDGET_REFINEMENT_SELECTION_POLICY:-bic_best_with_token_progress}"
BUDGET_REFINEMENT_MIN_TOKEN_REDUCTION_FRACTION="${BUDGET_REFINEMENT_MIN_TOKEN_REDUCTION_FRACTION:-0.10}"
BUDGET_REFINEMENT_FALLBACK="${BUDGET_REFINEMENT_FALLBACK:-gmm_bic_recursive}"
BUDGET_REFINEMENT_FLATTEN_LEAVES_TO_L0="${BUDGET_REFINEMENT_FLATTEN_LEAVES_TO_L0:-true}"
BUDGET_REFINEMENT_SKIP_OVERSIZE_SINGLETON="${BUDGET_REFINEMENT_SKIP_OVERSIZE_SINGLETON:-true}"

# Effective evidence budget = SUMMARY_MAX_MODEL_TOKENS * SUMMARY_BUDGET_RATIO
# minus SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS.
SUMMARY_MAX_MODEL_TOKENS="${SUMMARY_MAX_MODEL_TOKENS:-100000}"
SUMMARY_BUDGET_RATIO="${SUMMARY_BUDGET_RATIO:-0.85}"
SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS="${SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS:-8000}"
SUMMARY_TOKEN_COUNT_METADATA_KEYS="${SUMMARY_TOKEN_COUNT_METADATA_KEYS:-analysis_token_count,prompt_token_count,token_count,tokens}"

# ---------------------------------------------------------------------------
# Dynamic-update scenario hyperparameters.
# These are ignored by static_build except for being recorded in the config.
# ---------------------------------------------------------------------------

DYNAMIC_INITIAL_COUNT="${DYNAMIC_INITIAL_COUNT:-120}"
DYNAMIC_UPDATE_BATCH_SIZE="${DYNAMIC_UPDATE_BATCH_SIZE:-8}"
DYNAMIC_UPDATE_BATCH_COUNT="${DYNAMIC_UPDATE_BATCH_COUNT:-10}"
DYNAMIC_MAX_PROPAGATION_ROUNDS="${DYNAMIC_MAX_PROPAGATION_ROUNDS:-16}"

DYNAMIC_UPDATE_MODE="${DYNAMIC_UPDATE_MODE:-fixed_k_online_em}"
DYNAMIC_ASSIGNMENT="${DYNAMIC_ASSIGNMENT:-cumulative_mass}"
DYNAMIC_TOP_R="${DYNAMIC_TOP_R:-2}"
DYNAMIC_MIN_MEMBERSHIP_WEIGHT="${DYNAMIC_MIN_MEMBERSHIP_WEIGHT:-0.05}"
DYNAMIC_MAX_MEMBERSHIP_GAP="${DYNAMIC_MAX_MEMBERSHIP_GAP:-0.25}"
DYNAMIC_CUMULATIVE_MASS_COVERAGE="${DYNAMIC_CUMULATIVE_MASS_COVERAGE:-0.90}"
DYNAMIC_UPDATE_ROUTING_MODEL="${DYNAMIC_UPDATE_ROUTING_MODEL:-true}"
DYNAMIC_CLEAR_STALE_AFTER_PROPAGATION="${DYNAMIC_CLEAR_STALE_AFTER_PROPAGATION:-true}"
DYNAMIC_CONFIDENCE_METADATA_KEY="${DYNAMIC_CONFIDENCE_METADATA_KEY:-confidence}"

# ---------------------------------------------------------------------------
# Cluster analyst prompt/cardinality settings.
# ---------------------------------------------------------------------------

ANALYST_PROMPT_STYLE="${ANALYST_PROMPT_STYLE:-trace2skill_cluster_level_template_inheritance_v4}"
ANALYST_CONFIDENCE_FLOOR="${ANALYST_CONFIDENCE_FLOOR:-0.05}"
ANALYST_TOKENIZER_REQUIRED="${ANALYST_TOKENIZER_REQUIRED:-true}"
ANALYST_ALLOW_REGEX_TOKENIZER_FALLBACK="${ANALYST_ALLOW_REGEX_TOKENIZER_FALLBACK:-false}"

# -1 means derive analyst max prompt tokens from summary budget.
ANALYST_MAX_PROMPT_TOKENS="${ANALYST_MAX_PROMPT_TOKENS:--1}"

# L0 raw trajectory communities may generate multiple cards.  L1+ communities
# are constrained to one higher-level abstraction card.
ANALYST_MULTI_CARD_MAX_LEVEL="${ANALYST_MULTI_CARD_MAX_LEVEL:-0}"
ANALYST_MAX_CARDS_L0="${ANALYST_MAX_CARDS_L0:-0}"
ANALYST_MAX_CARDS_HIGHER="${ANALYST_MAX_CARDS_HIGHER:-1}"
ANALYST_HIGHER_LEVEL_MODE="${ANALYST_HIGHER_LEVEL_MODE:-single_abstraction}"
ANALYST_TRUNCATE_HIGHER_LEVEL_EXTRA_CARDS="${ANALYST_TRUNCATE_HIGHER_LEVEL_EXTRA_CARDS:-true}"

# ---------------------------------------------------------------------------
# Preflight and command assembly.
# ---------------------------------------------------------------------------

if [[ "$TREE_SCENARIO" != "static_build" && "$TREE_SCENARIO" != "dynamic_update" ]]; then
  echo "ERROR: TREE_SCENARIO must be static_build or dynamic_update, got: $TREE_SCENARIO" >&2
  exit 2
fi

if [[ "$RESUME" != "true" && "$RESUME" != "false" ]]; then
  echo "ERROR: RESUME must be true or false, got: $RESUME" >&2
  exit 2
fi

if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "false" ]]; then
  echo "ERROR: DRY_RUN must be true or false, got: $DRY_RUN" >&2
  exit 2
fi

if [[ -n "$RECORDS_PATH" && -n "$REUSE_TRAIN_RUN_DIR" ]]; then
  echo "ERROR: set only one of RECORDS_PATH or REUSE_TRAIN_RUN_DIR" >&2
  exit 2
fi

if [[ ! -x "$DYNAMIX_PYTHON" ]]; then
  echo "ERROR: DYNAMIX_PYTHON is not executable: $DYNAMIX_PYTHON" >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p "$RUN_DIR/logs" "$TRAIN_ARTIFACT_DIR/logs" "$SCENARIO_OUTPUT_DIR/logs"

export CONDA_ENV
export DYNAMIX_PYTHON
export OPENAI_API_KEY
export PATH="$(dirname "$DYNAMIX_PYTHON"):$PATH"
export SAL_DISABLE_OPENCL="${SAL_DISABLE_OPENCL:-1}"

echo "[preflight] repo=$REPO_ROOT"
echo "[preflight] run_dir=$RUN_DIR"
echo "[preflight] train_artifact_dir=$TRAIN_ARTIFACT_DIR"
echo "[preflight] scenario_output_dir=$SCENARIO_OUTPUT_DIR"
echo "[preflight] tree_scenario=$TREE_SCENARIO"
echo "[preflight] python=$(command -v python)"
"$DYNAMIX_PYTHON" -c 'import sys; print("[preflight] sys.executable=" + sys.executable)'
echo "[preflight] generation=$MODEL $OPENAI_BASE_URL thinking=$THINKING workers=$WORKERS"
echo "[preflight] embedding=$EMBEDDING_MODEL $EMBEDDING_BASE_URL tokenizer=$EMBEDDING_TOKENIZER"
echo "[preflight] heldout_query='instruction + Task type'; answer_position is not used for retrieval"
command -v soffice || true
soffice --version || true

cmd=(
  "$DYNAMIX_PYTHON" "scripts/run_dynamix_trace2skill_experiment.py"
  "--data-path" "$DATA_PATH"
  "--run-dir" "$RUN_DIR"
  "--train-artifact-dir" "$TRAIN_ARTIFACT_DIR"
  "--scenario-output-dir" "$SCENARIO_OUTPUT_DIR"
  "--train-start" "$TRAIN_START"
  "--train-end" "$TRAIN_END"
  "--heldout-start" "$HELDOUT_START"
  "--heldout-end" "$HELDOUT_END"
  "--workers" "$WORKERS"
  "--model" "$MODEL"
  "--openai-base-url" "$OPENAI_BASE_URL"
  "--embedding-base-url" "$EMBEDDING_BASE_URL"
  "--embedding-model" "$EMBEDDING_MODEL"
  "--embedding-tokenizer" "$EMBEDDING_TOKENIZER"
  "--python-executable" "$DYNAMIX_PYTHON"
  "--max-turns" "$MAX_TURNS"
  "--thinking" "$THINKING"
  "--skillbank-top-k" "$SKILLBANK_TOP_K"
  "--tree-scenario" "$TREE_SCENARIO"
  "--random-seed" "$RANDOM_SEED"
  "--tree-policy" "$TREE_POLICY"
  "--graph-kind" "$GRAPH_KIND"
  "--allow-overlap" "$ALLOW_OVERLAP"
  "--allow-multi-parent" "$ALLOW_MULTI_PARENT"
  "--use-support-mass" "$USE_SUPPORT_MASS"
  "--dynamic-initial-count" "$DYNAMIC_INITIAL_COUNT"
  "--dynamic-update-batch-size" "$DYNAMIC_UPDATE_BATCH_SIZE"
  "--dynamic-update-batch-count" "$DYNAMIC_UPDATE_BATCH_COUNT"
  "--max-levels" "$MAX_LEVELS"
  "--skill-output-dir-name" "$SKILL_OUTPUT_DIR_NAME"
  "--rollout-temperature" "$ROLLOUT_TEMPERATURE"
  "--rollout-client-timeout-seconds" "$ROLLOUT_CLIENT_TIMEOUT_SECONDS"
  "--rollout-client-retry-wait-seconds" "$ROLLOUT_CLIENT_RETRY_WAIT_SECONDS"
  "--rollout-llm-client" "$ROLLOUT_LLM_CLIENT"
  "--rollout-num-random-seeds" "$ROLLOUT_NUM_RANDOM_SEEDS"
  "--rollout-seeds" "$ROLLOUT_SEEDS"
  "--rollout-instance-ids" "$ROLLOUT_INSTANCE_IDS"
  "--rollout-missing-only" "$ROLLOUT_MISSING_ONLY"
  "--rollout-repeat" "$ROLLOUT_REPEAT"
  "--rollout-shuffle-seed" "$ROLLOUT_SHUFFLE_SEED"
  "--rollout-sample" "$ROLLOUT_SAMPLE"
  "--generation-temperature" "$GENERATION_TEMPERATURE"
  "--generation-timeout-seconds" "$GENERATION_TIMEOUT_SECONDS"
  "--generation-max-concurrency" "$GENERATION_MAX_CONCURRENCY"
  "--generation-retry-wait-seconds" "$GENERATION_RETRY_WAIT_SECONDS"
  "--embedding-max-model-len" "$EMBEDDING_MAX_MODEL_LEN"
  "--embedding-max-input-tokens" "$EMBEDDING_MAX_INPUT_TOKENS"
  "--embedding-truncate-long-texts" "$EMBEDDING_TRUNCATE_LONG_TEXTS"
  "--embedding-truncation-strategy" "$EMBEDDING_TRUNCATION_STRATEGY"
  "--embedding-batch-size" "$EMBEDDING_BATCH_SIZE"
  "--embedding-max-concurrency" "$EMBEDDING_MAX_CONCURRENCY"
  "--embedding-tokenizer-required" "$EMBEDDING_TOKENIZER_REQUIRED"
  "--chunked-embedding-enabled" "$CHUNKED_EMBEDDING_ENABLED"
  "--chunked-embedding-chunk-tokens" "$CHUNKED_EMBEDDING_CHUNK_TOKENS"
  "--chunked-embedding-overlap-tokens" "$CHUNKED_EMBEDDING_OVERLAP_TOKENS"
  "--chunked-embedding-pooling" "$CHUNKED_EMBEDDING_POOLING"
  "--chunked-embedding-add-special-tokens" "$CHUNKED_EMBEDDING_ADD_SPECIAL_TOKENS"
  "--chunked-embedding-normalize-after-pooling" "$CHUNKED_EMBEDDING_NORMALIZE_AFTER_POOLING"
  "--chunked-embedding-fail-if-chunk-exceeds-model-limit" "$CHUNKED_EMBEDDING_FAIL_IF_CHUNK_EXCEEDS_MODEL_LIMIT"
  "--projection-method" "$PROJECTION_METHOD"
  "--projection-variance-ratio" "$PROJECTION_VARIANCE_RATIO"
  "--projection-max-dim" "$PROJECTION_MAX_DIM"
  "--projection-min-dim" "$PROJECTION_MIN_DIM"
  "--projection-whiten" "$PROJECTION_WHITEN"
  "--gmm-covariance-type" "$GMM_COVARIANCE_TYPE"
  "--gmm-num-restarts" "$GMM_NUM_RESTARTS"
  "--gmm-kmeans-init-iters" "$GMM_KMEANS_INIT_ITERS"
  "--gmm-max-iter" "$GMM_MAX_ITER"
  "--gmm-tol" "$GMM_TOL"
  "--gmm-min-covar" "$GMM_MIN_COVAR"
  "--gmm-min-split-size" "$GMM_MIN_SPLIT_SIZE"
  "--gmm-min-effective-samples-per-component" "$GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT"
  "--gmm-abs-kmax" "$GMM_ABS_KMAX"
  "--gmm-max-concurrent-candidates" "$GMM_MAX_CONCURRENT_CANDIDATES"
  "--gmm-max-concurrent-restarts" "$GMM_MAX_CONCURRENT_RESTARTS"
  "--soft-save-soft-edges" "$SOFT_SAVE_SOFT_EDGES"
  "--soft-top-r-memberships" "$SOFT_TOP_R_MEMBERSHIPS"
  "--soft-recursive-assignment" "$SOFT_RECURSIVE_ASSIGNMENT"
  "--soft-min-membership-weight" "$SOFT_MIN_MEMBERSHIP_WEIGHT"
  "--soft-max-membership-gap" "$SOFT_MAX_MEMBERSHIP_GAP"
  "--soft-cumulative-mass-coverage" "$SOFT_CUMULATIVE_MASS_COVERAGE"
  "--budget-refinement-enabled" "$BUDGET_REFINEMENT_ENABLED"
  "--budget-refinement-apply-to-level" "$BUDGET_REFINEMENT_APPLY_TO_LEVEL"
  "--budget-refinement-selection-policy" "$BUDGET_REFINEMENT_SELECTION_POLICY"
  "--budget-refinement-min-token-reduction-fraction" "$BUDGET_REFINEMENT_MIN_TOKEN_REDUCTION_FRACTION"
  "--budget-refinement-fallback" "$BUDGET_REFINEMENT_FALLBACK"
  "--budget-refinement-flatten-leaves-to-l0" "$BUDGET_REFINEMENT_FLATTEN_LEAVES_TO_L0"
  "--budget-refinement-skip-oversize-singleton" "$BUDGET_REFINEMENT_SKIP_OVERSIZE_SINGLETON"
  "--summary-max-model-tokens" "$SUMMARY_MAX_MODEL_TOKENS"
  "--summary-budget-ratio" "$SUMMARY_BUDGET_RATIO"
  "--summary-prompt-overhead-reserve-tokens" "$SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS"
  "--summary-token-count-metadata-keys" "$SUMMARY_TOKEN_COUNT_METADATA_KEYS"
  "--dynamic-update-mode" "$DYNAMIC_UPDATE_MODE"
  "--dynamic-assignment" "$DYNAMIC_ASSIGNMENT"
  "--dynamic-top-r" "$DYNAMIC_TOP_R"
  "--dynamic-min-membership-weight" "$DYNAMIC_MIN_MEMBERSHIP_WEIGHT"
  "--dynamic-max-membership-gap" "$DYNAMIC_MAX_MEMBERSHIP_GAP"
  "--dynamic-cumulative-mass-coverage" "$DYNAMIC_CUMULATIVE_MASS_COVERAGE"
  "--dynamic-update-routing-model" "$DYNAMIC_UPDATE_ROUTING_MODEL"
  "--dynamic-clear-stale-after-propagation" "$DYNAMIC_CLEAR_STALE_AFTER_PROPAGATION"
  "--dynamic-confidence-metadata-key" "$DYNAMIC_CONFIDENCE_METADATA_KEY"
  "--dynamic-max-propagation-rounds" "$DYNAMIC_MAX_PROPAGATION_ROUNDS"
  "--analyst-prompt-style" "$ANALYST_PROMPT_STYLE"
  "--analyst-confidence-floor" "$ANALYST_CONFIDENCE_FLOOR"
  "--analyst-tokenizer-required" "$ANALYST_TOKENIZER_REQUIRED"
  "--analyst-allow-regex-tokenizer-fallback" "$ANALYST_ALLOW_REGEX_TOKENIZER_FALLBACK"
  "--analyst-max-prompt-tokens" "$ANALYST_MAX_PROMPT_TOKENS"
  "--analyst-multi-card-max-level" "$ANALYST_MULTI_CARD_MAX_LEVEL"
  "--analyst-max-cards-l0" "$ANALYST_MAX_CARDS_L0"
  "--analyst-max-cards-higher" "$ANALYST_MAX_CARDS_HIGHER"
  "--analyst-higher-level-mode" "$ANALYST_HIGHER_LEVEL_MODE"
  "--analyst-truncate-higher-level-extra-cards" "$ANALYST_TRUNCATE_HIGHER_LEVEL_EXTRA_CARDS"
)

if [[ -n "$RECORDS_PATH" ]]; then
  cmd+=("--records-path" "$RECORDS_PATH")
fi

if [[ -n "$REUSE_TRAIN_RUN_DIR" ]]; then
  cmd+=("--reuse-train-run-dir" "$REUSE_TRAIN_RUN_DIR")
fi

if [[ "$RESUME" == "true" ]]; then
  cmd+=("--resume")
else
  cmd+=("--no-resume")
fi

printf '[run]'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] command printed; experiment not started"
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "$SCENARIO_OUTPUT_DIR/logs/handoff_wrapper.log"
