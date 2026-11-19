#!/usr/bin/env bash
# Qwen3.5-9B baseline inference (OpenAI-compatible endpoint).
#
#   bash scripts/run_baseline_qwen.sh image
#   bash scripts/run_baseline_qwen.sh text
#
# Prerequisites: the vLLM server must already be running WITHOUT
# --language-model-only for the image mode, e.g.
#   vllm serve $HOME/models/Qwen3.5-9B \
#     --served-model-name Qwen3.5-9B --host 127.0.0.1 --port 8000 \
#     --tensor-parallel-size 2 --dtype bfloat16 \
#     --max-model-len 16384 --max-num-seqs 16 \
#     --limit-mm-per-prompt image=1 --reasoning-parser qwen3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="${1:-image}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
DATA="${DATA:-$ROOT_DIR/data/nova_sft/test.jsonl}"
MAX_TOKENS="${MAX_TOKENS:-512}"

case "$MODE" in
    image)
        OUT="${OUT:-$ROOT_DIR/result/qwen35_baseline/pred.jsonl}"
        EXTRA=()
        ;;
    text)
        OUT="${OUT:-$ROOT_DIR/result/qwen35_baseline_text/pred.jsonl}"
        EXTRA=(--text_only)
        ;;
    *)
        echo "usage: bash scripts/run_baseline_qwen.sh {image|text}" >&2
        exit 2
        ;;
esac

# Qwen3.5 默认 thinking 会先用完 512 token 预算，content 返回空；
# baseline 需要完整三段式输出时设 NO_THINKING=1。
if [[ "${NO_THINKING:-0}" == "1" ]]; then
    EXTRA+=(--no_thinking)
fi

# 零样本 baseline 与 SFT 标签格式对齐：要求三段式 + 英文最终诊断。
if [[ "${FORMAT_HINT:-0}" == "1" ]]; then
    EXTRA+=(--format_hint)
fi

# 断点续跑：跳过 --out 里已有 index（会话中断后续跑用）。
if [[ "${RESUME:-0}" == "1" ]]; then
    EXTRA+=(--resume)
fi

if [[ ! -f "$DATA" ]]; then
    echo "[ERROR] missing data: $DATA" >&2
    exit 1
fi

echo "MODE      = $MODE"
echo "BASE_URL  = $BASE_URL"
echo "MODEL     = $MODEL_NAME"
echo "DATA      = $DATA"
echo "OUT       = $OUT"
echo "MAX_TOKENS= $MAX_TOKENS"
echo "NOTE      = image mode requires vLLM without --language-model-only"

CMD=(python3 "$ROOT_DIR/src/infer_qwen_baseline.py"
    --data "$DATA" --out "$OUT"
    --base_url "$BASE_URL" --model "$MODEL_NAME"
    --max_tokens "$MAX_TOKENS")
CMD+=(${EXTRA[@]+"${EXTRA[@]}"})

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"

echo
echo "evaluate:"
echo "  python3 src/eval_nova.py --pred $OUT --gold $DATA --out ${OUT%.jsonl}_report.json"
