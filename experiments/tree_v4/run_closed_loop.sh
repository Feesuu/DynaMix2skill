#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DYNAMIX_PYTHON="${DYNAMIX_PYTHON:?DYNAMIX_PYTHON is required}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
OPEN_LOOP_SCENARIO_DIR="${OPEN_LOOP_SCENARIO_DIR:?OPEN_LOOP_SCENARIO_DIR is required}"
CLOSED_LOOP_RUN_DIR="${CLOSED_LOOP_RUN_DIR:?CLOSED_LOOP_RUN_DIR is required}"
MODEL="${MODEL:-Qwen3.5-9B-AWQ}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:?OPENAI_BASE_URL is required}"
OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is required}"

BOOTSTRAP_COUNT="${BOOTSTRAP_COUNT:-120}"
TRAIN_END="${TRAIN_END:-200}"
HELDOUT_START="${HELDOUT_START:-200}"
HELDOUT_END="${HELDOUT_END:-400}"
WORKERS="${WORKERS:-16}"
MAX_TURNS="${MAX_TURNS:-30}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.0}"
ROLLOUT_THINKING="${ROLLOUT_THINKING:-${THINKING:-false}}"
ROLLOUT_TIMEOUT_SECONDS="${ROLLOUT_TIMEOUT_SECONDS:-${ROLLOUT_CLIENT_TIMEOUT_SECONDS:-600}}"
ROLLOUT_RETRY_WAIT_SECONDS="${ROLLOUT_RETRY_WAIT_SECONDS:-5,10,30}"
SKILLBANK_TOP_K="${SKILLBANK_TOP_K:-10}"
EVALUATOR_BACKEND="${EVALUATOR_BACKEND:-auto}"

BASE_CONFIG="${BASE_CONFIG:-$OPEN_LOOP_SCENARIO_DIR/dynamix_config.json}"
BOOTSTRAP_RECORDS="${BOOTSTRAP_RECORDS:-$OPEN_LOOP_SCENARIO_DIR/ordered_records.json}"
BOOTSTRAP_CHECKPOINT="${BOOTSTRAP_CHECKPOINT:-$OPEN_LOOP_SCENARIO_DIR/dynamix_tree/dynamic_snapshots/arrival_$(printf '%04d' "$BOOTSTRAP_COUNT")}"
SOURCE_EBST_CONTROL_MANIFEST="${SOURCE_EBST_CONTROL_MANIFEST:-$OPEN_LOOP_SCENARIO_DIR/analysis/ebst_control_manifest.json}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$CLOSED_LOOP_RUN_DIR/logs"
exec 9>"$CLOSED_LOOP_RUN_DIR/.closed_loop_launcher.lock"
if ! flock -n 9; then
  echo "ERROR: another closed-loop launcher owns $CLOSED_LOOP_RUN_DIR" >&2
  exit 2
fi

for path in "$BASE_CONFIG" "$SOURCE_EBST_CONTROL_MANIFEST"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required closed-loop input is missing: $path" >&2
    exit 2
  fi
done
"$DYNAMIX_PYTHON" -c '
import json
import sys
from evaluate_with_official import evaluation_runtime_identity

payload = json.load(open(sys.argv[1]))
contract = payload["contract"]
rollout = contract["rollout"]
retrieval = contract["retrieval"]
observed = {
    "model": sys.argv[2],
    "thinking": sys.argv[3].casefold() == "true",
    "max_turns": int(sys.argv[4]),
    "workers": int(sys.argv[5]),
    "timeout_seconds": float(sys.argv[6]),
    "retry_wait_seconds": [float(x) for x in sys.argv[7].split(",")],
    "top_k": int(sys.argv[8]),
    "temperature": float(sys.argv[9]),
    "llm_client": sys.argv[10],
    "response_cache_enabled": sys.argv[11].casefold() == "true",
    "openai_base_url": sys.argv[13],
}
expected = {
    "model": rollout["model"],
    "thinking": bool(rollout["thinking"]),
    "max_turns": int(rollout["max_turns"]),
    "workers": int(rollout["workers"]),
    "timeout_seconds": float(rollout["timeout_seconds"]),
    "retry_wait_seconds": [
        float(x) for x in rollout["retry_wait_seconds"]
    ],
    "top_k": int(retrieval["top_k"]),
    "temperature": float(rollout["generation_config"]["temperature"]),
    "llm_client": rollout["llm_client"],
    "response_cache_enabled": bool(
        rollout["response_cache_enabled"]
    ),
    "openai_base_url": rollout["openai_base_url"],
}
if observed != expected:
    raise SystemExit(
        "closed-loop protocol differs from its paired open-loop control: "
        f"expected={expected!r} observed={observed!r}"
    )
if evaluation_runtime_identity(sys.argv[12]) != contract["evaluator"]:
    raise SystemExit(
        "closed-loop evaluator differs from its paired open-loop control"
    )
' \
  "$SOURCE_EBST_CONTROL_MANIFEST" \
  "$MODEL" \
  "$ROLLOUT_THINKING" \
  "$MAX_TURNS" \
  "$WORKERS" \
  "$ROLLOUT_TIMEOUT_SECONDS" \
  "$ROLLOUT_RETRY_WAIT_SECONDS" \
  "$SKILLBANK_TOP_K" \
  "$ROLLOUT_TEMPERATURE" \
  "openai" \
  "false" \
  "$EVALUATOR_BACKEND" \
  "$OPENAI_BASE_URL"
bootstrap_args=()
if (( BOOTSTRAP_COUNT > 0 )); then
  for path in \
    "$BOOTSTRAP_RECORDS" \
    "$BOOTSTRAP_CHECKPOINT/checkpoint.complete.json"; do
    if [[ ! -f "$path" ]]; then
      echo "ERROR: required closed-loop bootstrap input is missing: $path" >&2
      exit 2
    fi
  done
  bootstrap_args=(
    "--bootstrap-checkpoint" "$BOOTSTRAP_CHECKPOINT"
    "--bootstrap-records" "$BOOTSTRAP_RECORDS"
  )
fi

export OPENAI_BASE_URL
export OPENAI_API_KEY

closed_loop_cmd=(
  "$DYNAMIX_PYTHON"
  "$REPO_ROOT/scripts/run_ebst_closed_loop_experiment.py"
  "--config" "$BASE_CONFIG"
  "--run-dir" "$CLOSED_LOOP_RUN_DIR/train"
  "--data-path" "$DATA_PATH"
  "--python-executable" "$DYNAMIX_PYTHON"
  "${bootstrap_args[@]}"
  "--arrival-start" "$BOOTSTRAP_COUNT"
  "--arrival-end" "$TRAIN_END"
  "--model" "$MODEL"
  "--openai-base-url" "$OPENAI_BASE_URL"
  "--openai-api-key-env" "OPENAI_API_KEY"
  "--max-turns" "$MAX_TURNS"
  "--rollout-temperature" "$ROLLOUT_TEMPERATURE"
  "--rollout-thinking" "$ROLLOUT_THINKING"
  "--rollout-timeout-seconds" "$ROLLOUT_TIMEOUT_SECONDS"
  "--rollout-retry-wait-seconds" "$ROLLOUT_RETRY_WAIT_SECONDS"
  "--skillbank-top-k" "$SKILLBANK_TOP_K"
  "--evaluator-backend" "$EVALUATOR_BACKEND"
)
if [[ "${RESUME:-true}" == "true" ]]; then
  closed_loop_cmd+=("--resume")
fi

printf '[closed-loop train]'
printf ' %q' "${closed_loop_cmd[@]}"
printf '\n'
"${closed_loop_cmd[@]}" \
  2>&1 | tee "$CLOSED_LOOP_RUN_DIR/logs/01_closed_loop_train.log"

NODEBANK_DIR="$CLOSED_LOOP_RUN_DIR/train/current_nodebank"
GENERATION_CONFIG="$CLOSED_LOOP_RUN_DIR/train/rollout_generation_config.json"
mapfile -t EMBEDDING_CONFIG < <(
  "$DYNAMIX_PYTHON" -c '
import json
import sys
embedding = json.load(open(sys.argv[1]))["embedding"]
for value in (
    embedding["cache_path"],
    embedding["base_url"],
    embedding["model"],
    embedding["max_model_len"],
    embedding.get("max_input_tokens") or embedding["max_model_len"],
    embedding["batch_size"],
    embedding.get("tokenizer_model") or "",
):
    print(value)
' "$BASE_CONFIG"
)
if (( ${#EMBEDDING_CONFIG[@]} != 7 )); then
  echo "ERROR: failed to read the complete embedding config" >&2
  exit 2
fi
EMBEDDING_CACHE_PATH="${EMBEDDING_CONFIG[0]}"
EMBEDDING_BASE_URL="${EMBEDDING_CONFIG[1]}"
EMBEDDING_MODEL="${EMBEDDING_CONFIG[2]}"
EMBEDDING_MAX_MODEL_LEN="${EMBEDDING_CONFIG[3]}"
EMBEDDING_MAX_INPUT_TOKENS="${EMBEDDING_CONFIG[4]}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_CONFIG[5]}"
EMBEDDING_TOKENIZER="${EMBEDDING_CONFIG[6]}"
SOURCE_QUERY_VECTOR_MANIFEST="${SOURCE_QUERY_VECTOR_MANIFEST:-$OPEN_LOOP_SCENARIO_DIR/raw/heldout_query_embedding_cache_manifest.json}"
if [[ ! -f "$SOURCE_QUERY_VECTOR_MANIFEST" ]]; then
  echo "ERROR: source heldout query-vector manifest is missing: $SOURCE_QUERY_VECTOR_MANIFEST" >&2
  exit 2
fi
"$DYNAMIX_PYTHON" -c '
import sys
from dynamix_trace2skill.clients import validate_embedding_cache_manifest
validate_embedding_cache_manifest(
    cache_path=sys.argv[1],
    manifest_path=sys.argv[2],
)
' "$EMBEDDING_CACHE_PATH" "$SOURCE_QUERY_VECTOR_MANIFEST"
export DYNAMIX_SKILLBANK_ROOT="$NODEBANK_DIR"
export DYNAMIX_SKILLBANK_TOP_K="$SKILLBANK_TOP_K"
export DYNAMIX_SKILLBANK_EMBED_BASE_URL="$EMBEDDING_BASE_URL"
export DYNAMIX_SKILLBANK_EMBED_MODEL="$EMBEDDING_MODEL"
export DYNAMIX_SKILLBANK_EMBED_API_KEY="EMPTY"
export DYNAMIX_SKILLBANK_EMBED_MAX_MODEL_LEN="$EMBEDDING_MAX_MODEL_LEN"
export DYNAMIX_SKILLBANK_EMBED_MAX_INPUT_TOKENS="$EMBEDDING_MAX_INPUT_TOKENS"
export DYNAMIX_SKILLBANK_EMBED_BATCH_SIZE="$EMBEDDING_BATCH_SIZE"
export DYNAMIX_SKILLBANK_EMBED_TOKENIZER="$EMBEDDING_TOKENIZER"
unset DYNAMIX_SKILLBANK_CHUNK_TOKENS
unset DYNAMIX_SKILLBANK_CHUNK_OVERLAP_TOKENS
export DYNAMIX_SKILLBANK_CACHE_PATH="$NODEBANK_DIR/.dynamix_skillbank_index.json"
export DYNAMIX_SKILLBANK_VECTOR_CACHE_PATH="$EMBEDDING_CACHE_PATH"
export DYNAMIX_SKILLBANK_VECTOR_CACHE_MANIFEST="$SOURCE_QUERY_VECTOR_MANIFEST"
export DYNAMIX_SKILLBANK_REQUIRE_CACHE_MATCH="true"
export DYNAMIX_SKILLBANK_REQUIRE_VECTOR_CACHE_MATCH="true"
export DYNAMIX_SKILLBANK_EXPECT_TREE_POLICY="evidence_balanced_skill_tree"
export DYNAMIX_SKILL_SELECTION_LOG="$CLOSED_LOOP_RUN_DIR/heldout/skill_selection_records.jsonl"
export DYNAMIX_SKILLBANK_USAGE_LOG="$CLOSED_LOOP_RUN_DIR/heldout/skillbank_usage.jsonl"
export REACT_AGENT_USAGE_LOG="$CLOSED_LOOP_RUN_DIR/heldout/react_usage.jsonl"

heldout_cmd=(
  "$DYNAMIX_PYTHON"
  "$REPO_ROOT/run_spreadsheetbench.py"
  "--data_path" "$DATA_PATH"
  "--output_dir" "$CLOSED_LOOP_RUN_DIR/heldout/outputs"
  "--working_dir" "$CLOSED_LOOP_RUN_DIR/heldout/work"
  "--agent" "cli_skill_preloaded"
  "--skills_dir" "$NODEBANK_DIR"
  "--model" "$MODEL"
  "--llm_client" "openai"
  "--max_turns" "$MAX_TURNS"
  "--temperature" "$ROLLOUT_TEMPERATURE"
  "--generation_config" "$GENERATION_CONFIG"
  "--llm_timeout_seconds" "$ROLLOUT_TIMEOUT_SECONDS"
  "--llm_retry_wait_seconds" "$ROLLOUT_RETRY_WAIT_SECONDS"
  "--disable_response_cache"
  "--start_idx" "$HELDOUT_START"
  "--end_idx" "$HELDOUT_END"
  "--results_file" "$CLOSED_LOOP_RUN_DIR/heldout/results.json"
  "--log_dir" "$CLOSED_LOOP_RUN_DIR/heldout/logs"
  "--log_format" "markdown"
  "--workers" "$WORKERS"
)
if [[ "${RESUME:-true}" == "true" ]]; then
  heldout_cmd+=("--missing_only")
fi

printf '[closed-loop heldout]'
printf ' %q' "${heldout_cmd[@]}"
printf '\n'
"${heldout_cmd[@]}" \
  2>&1 | tee "$CLOSED_LOOP_RUN_DIR/logs/02_heldout_rollout.log"

"$DYNAMIX_PYTHON" "$REPO_ROOT/scripts/audit_cdost_query_vector_cache.py" \
  --selection-log "$DYNAMIX_SKILL_SELECTION_LOG" \
  --cache-path "$EMBEDDING_CACHE_PATH" \
  --output-path "$CLOSED_LOOP_RUN_DIR/heldout/query_embedding_cache_manifest.json" \
  --reference-manifest "$SOURCE_QUERY_VECTOR_MANIFEST" \
  2>&1 | tee "$CLOSED_LOOP_RUN_DIR/logs/02b_query_vector_audit.log"

"$DYNAMIX_PYTHON" "$REPO_ROOT/evaluate_with_official.py" \
  --data_path "$DATA_PATH" \
  --output_dir "$CLOSED_LOOP_RUN_DIR/heldout/outputs" \
  --start_idx "$HELDOUT_START" \
  --end_idx "$HELDOUT_END" \
  --recalc_dir "$CLOSED_LOOP_RUN_DIR/heldout/libreoffice_recalculated" \
  --evaluator-backend "$EVALUATOR_BACKEND" \
  --results_file "$CLOSED_LOOP_RUN_DIR/heldout/evaluation.json" \
  2>&1 | tee "$CLOSED_LOOP_RUN_DIR/logs/03_heldout_eval.log"
