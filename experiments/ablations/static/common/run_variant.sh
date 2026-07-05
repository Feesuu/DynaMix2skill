#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DYNAMIX_ABLATION_ENV:-$SCRIPT_DIR/base_env.local.sh}"
VARIANT_JSON="${1:?usage: run_variant.sh <variant.json>}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  echo "Copy $SCRIPT_DIR/base_env.example.sh to base_env.local.sh, or set DYNAMIX_ABLATION_ENV." >&2
  exit 2
fi

source "$ENV_FILE"

"$DYNAMIX_PYTHON" "$SCRIPT_DIR/run_variant.py" --variant-json "$VARIANT_JSON"
