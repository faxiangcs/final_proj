#!/usr/bin/env bash
# Build the course submission .zip from the current git commit.
#
#   bash scripts/make_submission_zip.sh
#   bash scripts/make_submission_zip.sh /tmp/iota5201.zip
#
# Only tracked files are included. Local data, model weights, checkpoints,
# logs, and follow-up experiments are therefore excluded automatically.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="${1:-$ROOT_DIR/dist/yanfaxiang_50051798_iota5201.zip}"
PREFIX="yanfaxiang_50051798_iota5201/"

cd "$ROOT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[ERROR] not inside a git repository: $ROOT_DIR" >&2
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[WARN] working tree has uncommitted changes; the zip contains HEAD only." >&2
fi

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
git archive --format=zip --prefix="$PREFIX" --output="$OUT" HEAD

echo "written: $OUT"
echo "size:    $(du -h "$OUT" | cut -f1)"
