#!/usr/bin/env python3
"""Quantify systematic box-size bias (regression to the training mean).

For every matched (prediction, gold) box pair, compare areas and regress
``log(pred_area)`` on ``log(gold_area)``.  A slope < 1 means large lesions are
under-predicted and small lesions over-predicted.  This is the evidence used to
justify the post-processing box-scale calibration.

Usage:
    python3 scripts/box_bias_report.py \
        --pred result/qwen_sft_ck50/pred.jsonl \
        --gold data/nova_sft/test.jsonl \
        --out result/qwen_sft_ck50/box_bias.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval_nova import (  # noqa: E402
    COORD_MODES,
    convert_boxes,
    detect_coord_mode,
    extract_label,
    extract_prediction,
    greedy_match,
    read_rows,
)
from src.nova_schema import parse_label, parse_prediction  # noqa: E402


def area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Box-size bias diagnostics")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--coord", choices=COORD_MODES, default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    pred_rows = read_rows(args.pred)
    gold_rows = read_rows(args.gold)
    gold_parsed = [parse_label(extract_label(row), strict=False) for row in gold_rows]
    pred_parsed = [parse_prediction(extract_prediction(row)) for row in pred_rows]
    mode = (
        detect_coord_mode([target.boxes for target in pred_parsed])
        if args.coord == "auto"
        else args.coord
    )

    ratios: list[float] = []
    pred_areas: list[float] = []
    gold_areas: list[float] = []
    log_gold: list[float] = []
    log_pred: list[float] = []
    bins = {"small_lt5k": [], "medium_5k_20k": [], "large_gt20k": []}

    for pred, gold in zip(pred_parsed, gold_parsed):
        pred_boxes, _ = convert_boxes(pred.boxes, mode)
        gold_boxes, _ = convert_boxes(gold.boxes, "pixel")
        for p_index, g_index, _ in greedy_match(pred_boxes, gold_boxes):
            p_area = area(pred_boxes[p_index])
            g_area = area(gold_boxes[g_index])
            if p_area <= 0 or g_area <= 0:
                continue
            ratios.append(p_area / g_area)
            pred_areas.append(p_area)
            gold_areas.append(g_area)
            log_gold.append(math.log(g_area))
            log_pred.append(math.log(p_area))
            if g_area < 5000:
                bins["small_lt5k"].append(p_area / g_area)
            elif g_area <= 20000:
                bins["medium_5k_20k"].append(p_area / g_area)
            else:
                bins["large_gt20k"].append(p_area / g_area)

    result = {
        "pred": str(args.pred),
        "gold": str(args.gold),
        "coord_mode": mode,
        "matched_boxes": len(ratios),
        "pred_area_median": median(pred_areas),
        "gold_area_median": median(gold_areas),
        "area_ratio_median": median(ratios),
        "area_ratio_mean": (sum(ratios) / len(ratios)) if ratios else 0.0,
        "log_area_slope": slope(log_gold, log_pred),
        "bins": {
            key: {"n": len(values), "ratio_median": median(values)}
            for key, values in bins.items()
        },
    }

    print(f"matched boxes : {result['matched_boxes']} (coord={mode})")
    print(
        f"area median   : pred={result['pred_area_median']:.0f} "
        f"gold={result['gold_area_median']:.0f}"
    )
    print(
        f"area ratio    : median={result['area_ratio_median']:.3f} "
        f"mean={result['area_ratio_mean']:.3f}"
    )
    print(f"log-area slope: {result['log_area_slope']:.3f} (1.0 = unbiased)")
    for key, value in result["bins"].items():
        print(f"  {key:<14} n={value['n']:<4} ratio_median={value['ratio_median']:.3f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
