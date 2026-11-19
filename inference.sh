#!/usr/bin/env bash
# Unified inference command for Nova.
#
# Default: run the merged SFT model on data/nova_sft/test.jsonl.
# For a LoRA adapter:
#   MODEL=$HOME/models/llava-1.5-7b-hf \
#   ADAPTERS=output/nova_sft/checkpoint-50 \
#   bash inference.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/nova_sft}"
MODEL="${MODEL:-$ROOT_DIR/nova_sft_merged}"
ADAPTERS="${ADAPTERS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

TEST_FILE="${TEST_FILE:-$DATA_DIR/test.jsonl}"
if [[ ! -f "$TEST_FILE" ]]; then
    echo "[ERROR] missing test file: $TEST_FILE" >&2
    echo "        run: python3 src/build_dataset.py --nova_root Nova --image_root Nova/images --out_dir data/nova_sft --seed 42" >&2
    exit 1
fi

if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v swift >/dev/null 2>&1; then
    echo "[ERROR] swift not found in PATH." >&2
    echo '        export PATH="$HOME/venvs/ms-swift/bin:$PATH"' >&2
    exit 1
fi

CMD=(env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" swift infer --model "$MODEL")
if [[ -n "$ADAPTERS" ]]; then
    CMD+=(--adapters "$ADAPTERS")
fi
CMD+=(
    --val_dataset "$TEST_FILE"
    --infer_backend pt
    --temperature 0
    --max_new_tokens "$MAX_NEW_TOKENS"
)
if [[ -n "${ENABLE_THINKING:-}" ]]; then
    CMD+=(--enable_thinking "$ENABLE_THINKING")
fi

echo "MODEL=$MODEL" >&2
echo "ADAPTERS=${ADAPTERS:-<none>}" >&2
echo "TEST_FILE=$TEST_FILE" >&2
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS" >&2

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
