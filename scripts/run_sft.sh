#!/usr/bin/env bash
# Unified SFT command for Nova.
#
# Defaults match the final paper; use four GPUs for effective batch size 8.
#
# Common overrides:
#   MODEL=...              base model (default: Qwen/Qwen3.5-9B)
#   DATA_DIR=...           default: <repo>/data/nova_sft
#   OUTPUT_DIR=...         default: <repo>/output/nova_sft
#   SFT_LR=5e-5            learning rate
#   SFT_EPOCHS=3           three epochs, 201 updates with the paper split
#   LORA_RANK=16           larger LoRA capacity
#   FREEZE_ALIGNER=false   unfreeze aligner if memory allows
#   SFT_EVAL_STRATEGY=no   disable inline eval
#   DRY_RUN=1              print the command without running it
set -euo pipefail
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-$BASE_MODEL}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.jsonl}"
TRAIN_AUX_FILE="${TRAIN_AUX_FILE:-}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/nova_sft}"

DATASET_ARGS=("$TRAIN_FILE")
if [[ -f "$TRAIN_AUX_FILE" ]]; then
    DATASET_ARGS+=("$TRAIN_AUX_FILE")
fi

SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_LR="${SFT_LR:-5e-5}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
SFT_EVAL_STRATEGY="${SFT_EVAL_STRATEGY:-steps}"

require_swift
require_file "$TRAIN_FILE"
require_file "$VAL_FILE"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "================ SFT ================"
show_common
echo "MODEL         = $MODEL"
echo "TRAIN_FILE    = $TRAIN_FILE"
echo "TRAIN_AUX_FILE= ${TRAIN_AUX_FILE}$([[ -f "$TRAIN_AUX_FILE" ]] && echo " (enabled)" || echo " (absent)")"
echo "VAL_FILE      = $VAL_FILE"
echo "OUTPUT_DIR    = $OUTPUT_DIR"
echo "SFT_LR        = $SFT_LR"
echo "SFT_EPOCHS    = $SFT_EPOCHS"
echo "NPROC_PER_NODE= ${NPROC_PER_NODE:-1}"
echo "====================================="

THINK_ARGS=()
if [[ -n "${ENABLE_THINKING:-}" ]]; then
    THINK_ARGS=(--enable_thinking "$ENABLE_THINKING")
fi

run_cmd "$SWIFT_BIN" sft \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset "${DATASET_ARGS[@]}" \
    --val_dataset "$VAL_FILE" \
    --split_dataset_ratio 0 \
    --max_length "$MAX_LENGTH" \
    --num_train_epochs "$SFT_EPOCHS" \
    --per_device_train_batch_size "$SFT_BATCH_SIZE" \
    --gradient_accumulation_steps "$SFT_GRAD_ACCUM" \
    --learning_rate "$SFT_LR" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --freeze_vit "$FREEZE_VIT" \
    --freeze_aligner "$FREEZE_ALIGNER" \
    --gradient_checkpointing true \
    --torch_dtype "$TORCH_DTYPE" \
    --eval_strategy "$SFT_EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --dataloader_num_workers "$DATALOADER_WORKERS" \
    --dataset_num_proc 1 \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    ${THINK_ARGS[@]+"${THINK_ARGS[@]}"}
