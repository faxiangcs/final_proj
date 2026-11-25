#!/usr/bin/env bash
# Reproduce the ms-swift training environment used for Nova.
#
#   bash scripts/setup_swift_env.sh
#
# Result: $ENV_DIR (default: $HOME/venvs/ms-swift, Python 3.11)
#   torch 2.6.0+cu124, transformers 5.20.2, ms-swift 4.6.1
#   all remaining versions are read from requirements.txt
#
# Notes:
# - transformers must be >= 5.x: Qwen3.5 (qwen3_5) is not in 4.57;
# - ms-swift 4.6.1 accepts transformers <5.23 and registers qwen3_5 templates;
# - causal_conv1d needs nvcc and is optional (Qwen3.5 falls back to a slower
#   reference implementation without it);
# - flash-linear-attention is NOT installed on purpose: with Triton 3.2 it was
#   ~8x slower than the reference path on the project machine (42s vs 5.3s per
#   GRPO step). Revisit if Triton >= 3.3 and causal_conv1d are installed;
# - set PIP_INDEX to a mirror if direct PyPI downloads are slow.
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/venvs/ms-swift}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_VERSION="${PY_VERSION:-3.11}"
PIP_INDEX="${PIP_INDEX:-https://pypi.org/simple}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    conda create -p "$ENV_DIR" "python=$PY_VERSION" -y
fi

PIP=("$ENV_DIR/bin/python" -m pip)
"${PIP[@]}" install -i "$PIP_INDEX" -r "$ROOT_DIR/requirements.txt"
"${PIP[@]}" check

"$ENV_DIR/bin/python" - <<'PY'
import torch, transformers, swift
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('swift', swift.__version__)
PY

cat <<EOF

Environment ready: $ENV_DIR

Use it with:
  export PATH="$ENV_DIR/bin:\$PATH"
  # or
  conda activate "$ENV_DIR"

Smoke tests:
  DRY_RUN=1 bash scripts/train.sh sft
  bash scripts/train.sh sft      # full run
EOF
