#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ -z "${BASELINE_CDOST_RUN_DIR:-}" ]]; then
  echo "ERROR: BASELINE_CDOST_RUN_DIR is required" >&2
  exit 2
fi
BASELINE_ATOMS="$BASELINE_CDOST_RUN_DIR/dynamix_tree/experience_atoms.json"
BASELINE_CONTROL="$BASELINE_CDOST_RUN_DIR/analysis/cdost_control_manifest.json"
BASELINE_CONFIG="$BASELINE_CDOST_RUN_DIR/dynamix_config.json"
if [[ ! -f "$BASELINE_ATOMS" || ! -f "$BASELINE_CONTROL" || ! -f "$BASELINE_CONFIG" ]]; then
  echo "ERROR: baseline CDOST atoms/control manifest are incomplete" >&2
  exit 2
fi
SOURCE_EMBEDDING_CACHE="$(
  "${DYNAMIX_PYTHON:-python3}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["embedding"]["cache_path"])' \
    "$BASELINE_CONFIG"
)"
if [[ -z "$SOURCE_EMBEDDING_CACHE" ]]; then
  echo "ERROR: baseline CDOST embedding cache path is empty" >&2
  exit 2
fi
if [[ -n "${EMBEDDING_CACHE_PATH:-}" && "$EMBEDDING_CACHE_PATH" != "$SOURCE_EMBEDDING_CACHE" ]]; then
  echo "ERROR: EMBEDDING_CACHE_PATH does not match the baseline CDOST cache" >&2
  exit 2
fi

export REPO_ROOT
export EMBEDDING_CACHE_PATH="$SOURCE_EMBEDDING_CACHE"
export TREE_SCENARIO="static_build"
export TREE_POLICY="evidence_balanced_skill_tree"
export GRAPH_KIND="single_parent_balanced_metric_tree"
export ALLOW_OVERLAP="false"
export ALLOW_MULTI_PARENT="false"
export DYNAMIC_INITIAL_COUNT="0"
export DYNAMIC_ARRIVAL_COUNT="200"
export DYNAMIC_UPDATE_BATCH_SIZE="1"
export DYNAMIC_TRAJECTORY_SOURCE="open_loop_replay"
export DYNAMIC_SHUFFLE_SEED="-1"
export DYNAMIC_SNAPSHOT_INCLUDE_EMBEDDINGS="true"
export DYNAMIC_RESUME_FROM_SNAPSHOTS="false"
export EBST_ATOM_CACHE_PATH="$BASELINE_ATOMS"

export WORKERS="${WORKERS:-16}"
export THINKING="${THINKING:-false}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.0}"
export GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-0.0}"
export GENERATION_MAX_CONCURRENCY="${GENERATION_MAX_CONCURRENCY:-16}"
export EMBEDDING_MAX_MODEL_LEN="${EMBEDDING_MAX_MODEL_LEN:-32000}"
export EMBEDDING_MAX_INPUT_TOKENS="${EMBEDDING_MAX_INPUT_TOKENS:-32000}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
export EMBEDDING_MAX_CONCURRENCY="${EMBEDDING_MAX_CONCURRENCY:-8}"
export CHUNKED_EMBEDDING_CHUNK_TOKENS="${CHUNKED_EMBEDDING_CHUNK_TOKENS:-8000}"
export CHUNKED_EMBEDDING_OVERLAP_TOKENS="${CHUNKED_EMBEDDING_OVERLAP_TOKENS:-1000}"
export CHUNKED_EMBEDDING_POOLING="${CHUNKED_EMBEDDING_POOLING:-mean}"

export EBST_MAX_ENTRIES="${EBST_MAX_ENTRIES:-8}"
export EBST_DUAL_VIEW_LAMBDA="${EBST_DUAL_VIEW_LAMBDA:-0.5}"
export EBST_ATOM_TEMPERATURE="${EBST_ATOM_TEMPERATURE:-0.0}"
export EBST_CAPSULE_TEMPERATURE="${EBST_CAPSULE_TEMPERATURE:-0.0}"
export EBST_VALIDATOR_TEMPERATURE="${EBST_VALIDATOR_TEMPERATURE:-0.0}"
export EBST_RETRIEVAL_TOKEN_BUDGET="${EBST_RETRIEVAL_TOKEN_BUDGET:-24000}"

exec bash "$REPO_ROOT/scripts/run_handoff_static_dynamic_experiment.sh"
