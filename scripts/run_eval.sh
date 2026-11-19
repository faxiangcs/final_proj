#!/usr/bin/env bash
# Unified evaluation entry.
#
#   PRED=infer_result/answer.jsonl bash scripts/run_eval.sh
#   PRED=result/qwen35_baseline/pred.jsonl \
#   GOLD=data/nova_sft/test.jsonl COORD=norm1000 bash scripts/run_eval.sh
#
# Defaults:
#   PRED=data/nova_sft/test.jsonl   (only useful as a parser self-check)
#   GOLD=<empty>                    (read labels from PRED)
#   COORD=auto
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/nova_sft}"
PRED="${PRED:-$DATA_DIR/test.jsonl}"
GOLD="${GOLD:-}"
COORD="${COORD:-auto}"
IMAGE_SIZE="${IMAGE_SIZE:-480}"
OUT="${OUT:-}"
EXTRA="${EVAL_EXTRA:-}"

if [[ ! -f "$PRED" ]]; then
    echo "[ERROR] missing prediction file: $PRED" >&2
    exit 1
fi

CMD=(python3 "$ROOT_DIR/src/eval_nova.py" --pred "$PRED" --coord "$COORD" --image_size "$IMAGE_SIZE")
if [[ -n "$GOLD" ]]; then
    CMD+=(--gold "$GOLD")
fi
if [[ -n "$OUT" ]]; then
    CMD+=(--out "$OUT")
fi
if [[ -n "$EXTRA" ]]; then
    read -r -a EXTRA_ARRAY <<< "$EXTRA"
    CMD+=(${EXTRA_ARRAY[@]+"${EXTRA_ARRAY[@]}"})
fi

echo "PRED       = $PRED"
echo "GOLD       = ${GOLD:-<from PRED>}"
echo "COORD      = $COORD"
echo "IMAGE_SIZE = $IMAGE_SIZE"
echo "OUT        = ${OUT:-<stdout only>}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
