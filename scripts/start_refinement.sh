#!/usr/bin/env bash
# Portable foreground launcher. Activate the pinned environment first.
# For a detached process, use nohup and redirect output explicitly.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" -u "$ROOT_DIR/scripts/run_refinement.py" "$@"
