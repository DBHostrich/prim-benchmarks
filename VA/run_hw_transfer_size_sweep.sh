#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VA_TRANSFER_COLLECTION_MODE=sweep
exec bash "$SCRIPT_DIR/run_hw_transfer_breakdown.sh"
