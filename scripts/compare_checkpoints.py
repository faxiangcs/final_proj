#!/usr/bin/env python3
"""Compare several eval_nova report.json files as a table.

Usage:
    python3 scripts/compare_checkpoints.py result/qwen_sft_val_checkpoint-*/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    loc = data.get("localization", {})
    diag = data.get("diagnosis", {})
    cases = data.get("cases", {})
    meta = data.get("meta", {})
    return {
        "name": Path(path).parent.name,
        "scale": meta.get("box_scale", 1.0),
        "f1_30": float(loc.get("f1@0.3", 0.0)),
        "f1_50": float(loc.get("f1@0.5", 0.0)),
        "f1_75": float(loc.get("f1@0.75", 0.0)),
        "iou": float(loc.get("mean_iou", 0.0)),
        "map": float(loc.get("map50_95", 0.0)),
        "kw1": float(diag.get("top1_keyword", 0.0)),
        "kw5": float(diag.get("top5_keyword", 0.0)),
        "case_kw": float(cases.get("top1_keyword", 0.0)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare eval reports")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--sort", choices=("f1_50", "iou", "map", "kw1"), default="f1_50")
    args = parser.parse_args(argv)

    rows = [load(path) for path in args.reports]
    rows.sort(key=lambda row: row[args.sort], reverse=True)
    header = (
        f"{'run':<44} {'scale':>5} {'F1@.3':>7} {'F1@.5':>7} "
        f"{'IoU':>7} {'mAP':>7} {'kwT1':>6} {'kwT5':>6} {'case':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name'][:44]:<44} {row['scale']:>5.2f} {row['f1_30']:>7.4f} "
            f"{row['f1_50']:>7.4f} {row['iou']:>7.4f} {row['map']:>7.4f} "
            f"{row['kw1']:>6.4f} {row['kw5']:>6.4f} {row['case_kw']:>6.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
