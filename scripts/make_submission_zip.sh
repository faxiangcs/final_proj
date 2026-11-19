#!/usr/bin/env bash
# Build the course submission .zip without data, weights, checkpoints, or logs.
#
#   bash scripts/make_submission_zip.sh
#   bash scripts/make_submission_zip.sh /tmp/iota5201.zip
#
# The default output is dist/yanfaxiang_50051798_iota5201.zip.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="${1:-$ROOT_DIR/dist/yanfaxiang_50051798_iota5201.zip}"

if ! command -v zip >/dev/null 2>&1; then
    echo "[ERROR] zip not found; install zip or use Python's zipfile module." >&2
    exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

FILES=(
    README.md
    requirements.txt
    .gitignore
    .gitattributes
    inference.sh
    medical_reward.py
    eval_all_metric.py
    PRESENTATION.md
    src
    scripts
    tests
    paper
)

zip -r "$OUT" "${FILES[@]}" \
    -x '*/__pycache__/*' '*.pyc' '*.pyo' \
       'paper/review/*' 'paper/*.aux' 'paper/*.bbl' 'paper/*.blg' \
       'paper/*.log' 'paper/*.out' 'paper/*.toc' 'paper/*.synctex.gz' \
       'scripts/__pycache__/*' 'tests/__pycache__/*'

echo "written: $OUT"
echo "size:    $(du -h "$OUT" | cut -f1)"
