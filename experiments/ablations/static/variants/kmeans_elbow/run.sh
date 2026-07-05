#!/usr/bin/env bash
set -euo pipefail

VARIANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$VARIANT_DIR/../../common/run_variant.sh" "$VARIANT_DIR/variant.json"
