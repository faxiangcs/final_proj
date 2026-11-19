#!/usr/bin/env bash
# Evaluate one or more GRPO LoRA checkpoints on the frozen test split.
#
# Usage:
#   bash scripts/eval_grpo_checkpoints.sh \
#     output/qwen_grpo_v1/v0-XXXX/checkpoint-50 \
#     output/qwen_grpo_v1/v0-XXXX/checkpoint-100 \
#     output/qwen_grpo_v1/v0-XXXX/checkpoint-266
#
# Each checkpoint runs `swift infer` on its own GPU (CUDA_VISIBLE_DEVICES=0..N-1)
# and writes result/<name>/pred.jsonl + report.json. Inference settings match the
# SFT baseline: pt backend, greedy, max_new_tokens=512, no enable_thinking.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-$HOME/models/Qwen3.5-9B}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/test.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/result}"
NAME_PREFIX="${NAME_PREFIX:-}"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <checkpoint-dir> [checkpoint-dir ...]" >&2
    exit 1
fi
require_file "$TEST_FILE"
require_swift

pids=()
names=()
gpu="${GPU_START:-0}"
for ckpt in "$@"; do
    if [[ ! -d "$ckpt" ]]; then
        echo "[ERROR] checkpoint not found: $ckpt" >&2
        exit 1
    fi
    step="$(basename "$ckpt")"
    parent="$(basename "$(dirname "$ckpt")")"
    name="${NAME_PREFIX:-$parent}_${step}"
    out="$RESULTS_DIR/$name"
    mkdir -p "$out"
    echo "[GPU $gpu] $ckpt -> $out"
    CUDA_VISIBLE_DEVICES="$gpu" swift infer \
        --model "$MODEL" \
        --adapters "$ckpt" \
        --val_dataset "$TEST_FILE" \
        --infer_backend pt \
        --temperature 0 \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --result_path "$out/pred.jsonl" \
        > "$out/infer.log" 2>&1 &
    pids+=($!)
    names+=("$name")
    gpu=$((gpu + 1))
done

status=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "[ERROR] inference failed: ${names[$i]} (see $RESULTS_DIR/${names[$i]}/infer.log)" >&2
        status=1
    fi
done
[[ "$status" -eq 0 ]] || exit "$status"

for name in "${names[@]}"; do
    out="$RESULTS_DIR/$name"
    echo "=== eval $name"
    python3 "$ROOT_DIR/src/eval_nova.py" \
        --pred "$out/pred.jsonl" \
        --gold "$TEST_FILE" \
        --out "$out/report.json" \
        | grep -E "Top-1|Top-5|case-level|IoU>=|mean IoU|mAP|Token"
done
