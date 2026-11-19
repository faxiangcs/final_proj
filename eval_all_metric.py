#!/usr/bin/env python3
"""兼容入口：旧命令 ``python eval_all_metric.py xxx.jsonl`` 仍可用。

实现已迁移到 ``src/eval_nova.py``（P0-14 ~ P0-16）：

- 多框一对一匹配，不再只看第一个预测框；
- 诊断改为归一化 exact / token-set 匹配，同时报告 strict 与 lenient；
- 预测坐标自动识别 ``[0,1]`` / ``[0,100]`` / 像素 / ``[0,1000]``；
- 指标名与实际定义一致，另给出 case 级诊断准确率。

推荐直接使用：

    python3 src/eval_nova.py --pred pred.jsonl --gold data/nova_sft/test.jsonl \
        --out eval_report.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_nova import main as eval_main  # noqa: E402


def _normalize_argv(argv: list[str]) -> list[str]:
    """允许把预测文件直接作为第一个位置参数。"""
    if not argv:
        return argv
    if argv[0] in ("-h", "--help"):
        return argv
    if not argv[0].startswith("-"):
        return ["--pred", argv[0], *argv[1:]]
    return argv


if __name__ == "__main__":
    raise SystemExit(eval_main(_normalize_argv(sys.argv[1:])))
