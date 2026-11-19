#!/usr/bin/env bash
# Single entry point for Nova training.
#
#   bash scripts/train.sh sft
#   bash scripts/train.sh grpo
#
# Environment overrides are the same as the underlying scripts; see
# scripts/README.md. Examples:
#   SFT_LR=5e-5 bash scripts/train.sh sft
#   ADAPTERS=output/nova_sft/checkpoint-50 bash scripts/train.sh grpo
#   DRY_RUN=1 bash scripts/train.sh sft
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
usage: bash scripts/train.sh {sft|grpo}

environment overrides:
  DATA_DIR, OUTPUT_DIR, MODEL, ADAPTERS, MAX_LENGTH, MAX_COMPLETION_LENGTH,
  LORA_RANK, LORA_ALPHA, FREEZE_VIT, FREEZE_ALIGNER,
  SFT_LR, SFT_EPOCHS, GRPO_LR, GRPO_EPOCHS,
  NUM_GENERATIONS, TEMPERATURE, BETA, USE_VLLM, DRY_RUN,
  NPROC_PER_NODE (multi-GPU DDP, e.g. NPROC_PER_NODE=4)

examples:
  SFT_LR=5e-5 SFT_EPOCHS=2 bash scripts/train.sh sft
  NPROC_PER_NODE=4 bash scripts/train.sh sft
  ADAPTERS=output/nova_sft/checkpoint-50 bash scripts/train.sh grpo
  DRY_RUN=1 bash scripts/train.sh grpo
EOF
}

stage="${1:-}"
case "$stage" in
    sft)
        exec bash "$SCRIPT_DIR/run_sft.sh"
        ;;
    grpo)
        exec bash "$SCRIPT_DIR/run_grpo.sh"
        ;;
    -h|--help)
        usage
        ;;
    "")
        usage >&2
        exit 2
        ;;
    *)
        echo "[ERROR] unknown stage: $stage" >&2
        usage >&2
        exit 2
        ;;
esac
