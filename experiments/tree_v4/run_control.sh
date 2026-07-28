#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ -z "${SOURCE_CDOST_RUN_DIR:-}" ]]; then
  echo "ERROR: SOURCE_CDOST_RUN_DIR is required for its frozen Atom cache" >&2
  exit 2
fi
SOURCE_ATOMS="$SOURCE_CDOST_RUN_DIR/dynamix_tree/experience_atoms.json"
if [[ ! -f "$SOURCE_ATOMS" ]]; then
  echo "ERROR: source CDOST experience_atoms.json is missing" >&2
  exit 2
fi

export REPO_ROOT
export TREE_SCENARIO="static_build"
export TREE_POLICY="certified_dual_view_otd"
export GRAPH_KIND="single_parent_binary_tree"
export ALLOW_OVERLAP="false"
export ALLOW_MULTI_PARENT="false"
export OTD_ATOM_CACHE_PATH="$SOURCE_ATOMS"
export DYNAMIC_SHUFFLE_SEED="-1"
export DYNAMIC_SNAPSHOT_INCLUDE_EMBEDDINGS="true"
export DYNAMIC_RESUME_FROM_SNAPSHOTS="false"

export WORKERS="${WORKERS:-16}"
export THINKING="${THINKING:-false}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.0}"
export GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-0.0}"
export GENERATION_MAX_CONCURRENCY="${GENERATION_MAX_CONCURRENCY:-16}"
export EMBEDDING_MAX_MODEL_LEN="${EMBEDDING_MAX_MODEL_LEN:-32000}"
export EMBEDDING_MAX_INPUT_TOKENS="${EMBEDDING_MAX_INPUT_TOKENS:-32000}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
export EMBEDDING_MAX_CONCURRENCY="${EMBEDDING_MAX_CONCURRENCY:-8}"

exec bash "$REPO_ROOT/scripts/run_handoff_static_dynamic_experiment.sh"
