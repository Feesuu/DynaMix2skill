#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../../common/run_variant.sh" "$SCRIPT_DIR" "${1:-${DYNAMIX_ABLATION_BASE_ENV:-}}"
