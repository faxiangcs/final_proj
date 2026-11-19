#!/usr/bin/env bash
# Unified GRPO command for Nova.
#
# Defaults:
#   - SFT merged model: <repo>/nova_sft_merged
#   - two currently registered rewards: localization_reward + diagnosis_reward
#   - max_length=2048, max_completion_length=512
#
# Common overrides:
#   MODEL=...              merged SFT model, or base model when ADAPTERS is set
#   ADAPTERS=...           optional LoRA adapter path, e.g. output/nova_sft/checkpoint-50
#   DATA_DIR=...           default: <repo>/data/nova_sft
#   OUTPUT_DIR=...         default: <repo>/output/nova_grpo
#   NUM_GENERATIONS=8      more generations if memory allows
#   TEMPERATURE=1.0        higher exploration
#   TOP_P=0.95             nucleus sampling
#   GRPO_LR=1e-5           lower LR
#   REWARD_WEIGHTS="1 1"   explicit per-reward weights (localization diagnosis)
#   GRPO_MAX_STEPS=...     limit optimizer steps (smoke test)
#   ENABLE_THINKING=false  disable Qwen thinking for a fair output budget
#   GRPO_EVAL_STRATEGY=steps  enable periodic validation generation
#   DRY_RUN=1              print the command without running it
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-}"
ADAPTERS="${ADAPTERS:-}"
REF_ADAPTERS="${REF_ADAPTERS:-}"
GRPO_CALLBACKS="${GRPO_CALLBACKS:-}"
GRPO_RESUME="${GRPO_RESUME:-}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/grpo_train.jsonl}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/grpo_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/nova_grpo}"
EXTERNAL_PLUGIN="${EXTERNAL_PLUGIN:-$ROOT_DIR/medical_reward.py}"

# With ADAPTERS, MODEL must be the base model; without, default to merged SFT.
if [[ -n "$ADAPTERS" ]]; then
    MODEL="${MODEL:-$BASE_MODEL}"
else
    MODEL="${MODEL:-$ROOT_DIR/nova_sft_merged}"
fi

GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
GRPO_LR="${GRPO_LR:-2e-5}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
BETA="${BETA:-0.001}"
USE_VLLM="${USE_VLLM:-false}"
GRPO_EVAL_STRATEGY="${GRPO_EVAL_STRATEGY:-no}"
REWARD_FUNCS="${REWARD_FUNCS:-localization_reward diagnosis_reward}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-}"

require_swift
require_file "$TRAIN_FILE"
require_file "$VAL_FILE"
require_file "$EXTERNAL_PLUGIN"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

read -r -a REWARD_ARRAY <<< "$REWARD_FUNCS"

EXTRA_ARGS=()
if [[ -n "$ADAPTERS" ]]; then
    EXTRA_ARGS+=(--adapters "$ADAPTERS")
fi
if [[ -n "$REF_ADAPTERS" ]]; then
    require_file "$REF_ADAPTERS/adapter_config.json"
    EXTRA_ARGS+=(--ref_adapters "$REF_ADAPTERS")
fi
if [[ -n "$GRPO_RESUME" ]]; then
    require_file "$GRPO_RESUME/trainer_state.json"
    EXTRA_ARGS+=(--resume_from_checkpoint "$GRPO_RESUME")
fi
if [[ -n "$GRPO_CALLBACKS" ]]; then
    read -r -a CALLBACK_ARRAY <<< "$GRPO_CALLBACKS"
    EXTRA_ARGS+=(--callbacks "${CALLBACK_ARRAY[@]}")
fi
if [[ -n "$REWARD_WEIGHTS" ]]; then
    read -r -a REWARD_WEIGHT_ARRAY <<< "$REWARD_WEIGHTS"
    EXTRA_ARGS+=(--reward_weights "${REWARD_WEIGHT_ARRAY[@]}")
fi
if [[ -n "$GRPO_MAX_STEPS" ]]; then
    EXTRA_ARGS+=(--max_steps "$GRPO_MAX_STEPS")
fi
if [[ -n "${ENABLE_THINKING:-}" ]]; then
    EXTRA_ARGS+=(--enable_thinking "$ENABLE_THINKING")
fi

echo "================ GRPO ==============="
show_common
echo "MODEL         = $MODEL"
echo "ADAPTERS      = ${ADAPTERS:-<none>}"
echo "REF_ADAPTERS  = ${REF_ADAPTERS:-<adapter-disabled base>}"
echo "TRAIN_FILE    = $TRAIN_FILE"
echo "VAL_FILE      = $VAL_FILE"
echo "OUTPUT_DIR    = $OUTPUT_DIR"
echo "REWARD_FUNCS  = $REWARD_FUNCS"
echo "NUM_GENERATIONS = $NUM_GENERATIONS"
echo "TEMPERATURE   = $TEMPERATURE"
echo "TOP_P         = $TOP_P"
echo "NPROC_PER_NODE= ${NPROC_PER_NODE:-1}"
echo "====================================="

run_cmd swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --tuner_type lora \
    --dataset "$TRAIN_FILE" \
    --val_dataset "$VAL_FILE" \
    --split_dataset_ratio 0 \
    --max_length "$MAX_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --num_train_epochs "$GRPO_EPOCHS" \
    --per_device_train_batch_size "$GRPO_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRPO_GRAD_ACCUM" \
    --learning_rate "$GRPO_LR" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --freeze_vit "$FREEZE_VIT" \
    --freeze_aligner "$FREEZE_ALIGNER" \
    --gradient_checkpointing true \
    --torch_dtype "$TORCH_DTYPE" \
    --num_generations "$NUM_GENERATIONS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --beta "$BETA" \
    --reward_funcs "${REWARD_ARRAY[@]}" \
    --external_plugins "$EXTERNAL_PLUGIN" \
    --use_vllm "$USE_VLLM" \
    --eval_strategy "$GRPO_EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --dataset_num_proc 1 \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
