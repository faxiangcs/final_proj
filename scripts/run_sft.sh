#!/usr/bin/env bash
# Unified SFT command for Nova.
#
# Defaults match the previous run (LoRA r=8, lr 1e-4, 1 epoch, frozen ViT/aligner)
# but now use the canonical data split and max_length=2048.
#
# Common overrides:
#   MODEL=...              base model (default: llava-hf/llava-1.5-7b-hf)
#   DATA_DIR=...           default: <repo>/data/nova_sft
#   OUTPUT_DIR=...         default: <repo>/output/nova_sft
#   SFT_LR=5e-5            lower LR if you see degeneration (plan 6.2)
#   SFT_EPOCHS=2           more epochs with val/early stopping
#   LORA_RANK=16           larger LoRA capacity
#   FREEZE_ALIGNER=false   unfreeze aligner if memory allows
#   SFT_EVAL_STRATEGY=no   disable inline eval
#   DRY_RUN=1              print the command without running it
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-$BASE_MODEL}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.jsonl}"
TRAIN_AUX_FILE="${TRAIN_AUX_FILE:-$DATA_DIR/train_aux_boxes.jsonl}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/nova_sft}"

DATASET_ARGS=("$TRAIN_FILE")
if [[ -f "$TRAIN_AUX_FILE" ]]; then
    DATASET_ARGS+=("$TRAIN_AUX_FILE")
fi

SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_LR="${SFT_LR:-1e-4}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
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

run_cmd swift sft \
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
