#!/usr/bin/env bash
# Shared defaults for Nova SFT/GRPO scripts.
# Source this file; do not execute it directly.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/nova_sft}"
NOVA_ROOT="${NOVA_ROOT:-$ROOT_DIR/Nova}"
IMAGE_ROOT="${IMAGE_ROOT:-$NOVA_ROOT/images}"

# Base VLM used by the final paper. MODEL can select downloaded local weights.
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
if [[ "$BASE_MODEL" == "Qwen/Qwen3.5-9B" && -f "$MODEL_ROOT/Qwen3.5-9B/config.json" ]]; then
    BASE_MODEL="$MODEL_ROOT/Qwen3.5-9B"
fi

SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
FREEZE_VIT="${FREEZE_VIT:-true}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-false}"

SAVE_STEPS="${SAVE_STEPS:-25}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-25}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

require_file() {
    local path="$1"
    [[ "${DRY_RUN:-0}" == "1" ]] && return 0
    if [[ ! -f "$path" ]]; then
        echo "[ERROR] missing data file: $path" >&2
        echo "        run:" >&2
        echo "        python3 src/build_dataset.py \\" >&2
        echo "          --nova_root Nova --image_root Nova/images \\" >&2
        echo "          --out_dir data/nova_sft --seed 42" >&2
        exit 1
    fi
}

require_swift() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    if ! command -v "$SWIFT_BIN" >/dev/null 2>&1; then
        echo "[ERROR] swift not found in PATH." >&2
        echo '        export PATH="$HOME/venvs/ms-swift/bin:$PATH"' >&2
        echo "        or create it with: bash scripts/setup_swift_env.sh" >&2
        exit 1
    fi
}

show_common() {
    echo "ROOT_DIR      = $ROOT_DIR"
    echo "DATA_DIR      = $DATA_DIR"
    echo "NOVA_ROOT     = $NOVA_ROOT"
    echo "IMAGE_ROOT    = $IMAGE_ROOT"
    echo "MAX_LENGTH    = $MAX_LENGTH"
    echo "MAX_COMPLETION= $MAX_COMPLETION_LENGTH"
}

run_cmd() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}
