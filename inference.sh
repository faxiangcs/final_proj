#!/usr/bin/env bash
# Greedy inference with an explicit output file; supports an SFT/GRPO adapter.
# MODEL=/path/to/Qwen3.5-9B ADAPTERS=/path/to/checkpoint-201 \
# RESULT_PATH=result/sft_test/pred.jsonl bash inference.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
ADAPTERS="${ADAPTERS:-}"
TEST_FILE="${TEST_FILE:-${DATA_DIR:-$ROOT_DIR/data/nova_sft}/test.jsonl}"
RESULT_PATH="${RESULT_PATH:-$ROOT_DIR/result/manual/pred.jsonl}"
CMD=(env -u NPROC_PER_NODE -u RANK -u LOCAL_RANK -u WORLD_SIZE
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" swift infer --model "$MODEL")
if [[ -n "$ADAPTERS" ]]; then
    CMD+=(--adapters "$ADAPTERS")
fi
CMD+=(--val_dataset "$TEST_FILE" --infer_backend pt --temperature 0
    --max_new_tokens "${MAX_NEW_TOKENS:-512}" --enable_thinking false
    --result_path "$RESULT_PATH")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi
[[ -f "$TEST_FILE" ]] || { echo "Missing data: $TEST_FILE" >&2; exit 1; }
[[ ! -e "$RESULT_PATH" ]] || { echo "Result already exists; choose a fresh RESULT_PATH: $RESULT_PATH" >&2; exit 1; }
command -v swift >/dev/null || { echo 'Activate the pinned ms-swift environment first.' >&2; exit 1; }
mkdir -p "$(dirname "$RESULT_PATH")"
"${CMD[@]}"
