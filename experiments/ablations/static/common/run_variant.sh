#!/usr/bin/env bash
set -euo pipefail

VARIANT_DIR="${1:?usage: run_variant.sh <variant_dir> [base_env.sh]}"
BASE_ENV="${2:-${DYNAMIX_ABLATION_BASE_ENV:-}}"
if [[ -n "$BASE_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$BASE_ENV"
fi

REPO_ROOT="${REPO_ROOT:-/mnt/data/yaodong/codes/DynaMix2skill}"
DYNAMIX_PYTHON="${DYNAMIX_PYTHON:-python}"
VARIANT_JSON="$VARIANT_DIR/variant.json"
if [[ ! -f "$VARIANT_JSON" ]]; then
  echo "variant.json not found: $VARIANT_JSON" >&2
  exit 2
fi

mapfile -t VARIANT_ARGS < <("$DYNAMIX_PYTHON" - "$VARIANT_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

def emit(flag, value):
    if value is None:
        return
    print(flag)
    print(str(value))

emit("--tree-policy", payload.get("tree_policy"))
emit("--max-levels", payload.get("max_levels"))
soft = (payload.get("hierarchy_override") or {}).get("soft_membership") or {}
emit("--soft-recursive-assignment", soft.get("recursive_assignment"))
skill_export = payload.get("skill_export") or {}
emit("--skill-export-min-level", -1 if skill_export.get("min_level") is None else skill_export.get("min_level"))
emit("--skill-export-max-level", -1 if skill_export.get("max_level") is None else skill_export.get("max_level"))
emit("--skill-export-max-node-count", -1 if skill_export.get("max_node_count") is None else skill_export.get("max_node_count"))
PY
)

VARIANT_NAME="$("$DYNAMIX_PYTHON" - "$VARIANT_JSON" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["variant_name"])
PY
)"
REQUIRES_REUSE_TREE="$("$DYNAMIX_PYTHON" - "$VARIANT_JSON" <<'PY'
import json
import sys
from pathlib import Path
print("1" if json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("reuse_tree_required") else "0")
PY
)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_GROUP="${RUN_GROUP:-$REPO_ROOT/runs/ablations/static}"
RUN_DIR="$RUN_GROUP/$VARIANT_NAME/$TIMESTAMP"
mkdir -p "$RUN_DIR/logs"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

if [[ "$REQUIRES_REUSE_TREE" == "1" && -z "${REUSE_TREE_DIR:-}" ]]; then
  echo "$VARIANT_NAME requires REUSE_TREE_DIR=/path/to/full/static/dynamix_tree" >&2
  exit 2
fi

cmd=(
  "$DYNAMIX_PYTHON" "$REPO_ROOT/scripts/run_dynamix_trace2skill_experiment.py"
  --benchmark spreadsheetbench
  --data-path "${DATA_PATH:?DATA_PATH is required}"
  --run-dir "$RUN_DIR"
  --scenario-output-dir "$RUN_DIR"
  --tree-scenario static_build
  --train-start "${TRAIN_START:-0}"
  --train-end "${TRAIN_END:-200}"
  --heldout-start "${HELDOUT_START:-200}"
  --heldout-end "${HELDOUT_END:-400}"
  --workers "${WORKERS:-4}"
  --model "${MODEL:?MODEL is required}"
  --openai-base-url "${OPENAI_BASE_URL:?OPENAI_BASE_URL is required}"
  --embedding-base-url "${EMBEDDING_BASE_URL:?EMBEDDING_BASE_URL is required}"
  --embedding-model "${EMBEDDING_MODEL:?EMBEDDING_MODEL is required}"
  --embedding-tokenizer "${EMBEDDING_TOKENIZER:?EMBEDDING_TOKENIZER is required}"
  --python-executable "$DYNAMIX_PYTHON"
  --max-turns "${MAX_TURNS:-30}"
  --thinking "${THINKING:-true}"
  --skillbank-top-k "${SKILLBANK_TOP_K:-10}"
  --summary-max-model-tokens "${SUMMARY_MAX_MODEL_TOKENS:-100000}"
  --summary-budget-ratio "${SUMMARY_BUDGET_RATIO:-0.85}"
  --summary-prompt-overhead-reserve-tokens "${SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS:-8000}"
  --embedding-max-model-len "${EMBEDDING_MAX_MODEL_LEN:-8192}"
  --embedding-max-input-tokens "${EMBEDDING_MAX_INPUT_TOKENS:-8000}"
  --chunked-embedding-enabled "${CHUNKED_EMBEDDING_ENABLED:-true}"
  --chunked-embedding-chunk-tokens "${CHUNKED_EMBEDDING_CHUNK_TOKENS:-7600}"
  --chunked-embedding-overlap-tokens "${CHUNKED_EMBEDDING_OVERLAP_TOKENS:-512}"
  "${VARIANT_ARGS[@]}"
)

if [[ -n "${RECORDS_PATH:-}" ]]; then
  cmd+=(--records-path "$RECORDS_PATH")
elif [[ -n "${REUSE_TRAIN_RUN_DIR:-}" ]]; then
  cmd+=(--reuse-train-run-dir "$REUSE_TRAIN_RUN_DIR")
fi
if [[ "$REQUIRES_REUSE_TREE" == "1" && -n "${REUSE_TREE_DIR:-}" ]]; then
  cmd+=(--reuse-tree-dir "$REUSE_TREE_DIR")
elif [[ "$REQUIRES_REUSE_TREE" != "1" && -n "${REUSE_TREE_DIR:-}" ]]; then
  printf '[warn] ignoring REUSE_TREE_DIR for non-retrieval variant %s\n' "$VARIANT_NAME" >&2
fi

printf '[variant] %s\n' "$VARIANT_NAME"
printf '[run_dir] %s\n' "$RUN_DIR"
printf '[cmd]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/logs/ablation_wrapper.log"
