#!/usr/bin/env bash
# Convenience entry point for the current native + long-context Meno profile.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/restart_meno2_split_context.sh" "$@"
