#!/usr/bin/env python3
"""Fit a single box-size calibration factor on a held-out split.

The SFT model tends to regress box size toward the training mean.  This script
scans a grid of scale factors, scales each predicted box about its center, and
picks the factor that maximises F1@IoU0.5 on the calibration split (normally
``val``).  The selected factor can then be applied to ``test`` with:

    python3 src/eval_nova.py --pred test_pred.jsonl --gold test.jsonl \
        --box_scale 1.30 --out test_report_calibrated.json

This is a post-processing calibration, not a model improvement.  Always report
the raw (scale=1.0) and calibrated numbers side by side.

Usage:
    python3 scripts/fit_box_scale.py \
        --pred result/qwen_sft_val_checkpoint-50/pred.jsonl \
        --gold data/nova_sft/val.jsonl \
        --out result/qwen_sft_val_checkpoint-50/box_scale.json
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

from src.eval_nova import (  # noqa: E402
    COORD_MODES,
    DEFAULT_ALIAS_PATH,
    evaluate_rows,
    load_diagnosis_aliases,
    read_rows,
)

LOC_KEYS = ("f1@0.3", "f1@0.5", "f1@0.75", "mean_iou", "map50_95")


def parse_scales(raw: str) -> list[float]:
    scales = sorted({round(float(item), 4) for item in raw.split(",") if item.strip()})
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("--scales must be a comma-separated list of positive numbers")
    return scales


def score(report: dict, metric: str) -> tuple[float, float]:
    loc = report.get("localization", {})
    f1 = float(loc.get("f1@0.5", 0.0))
    iou = float(loc.get("mean_iou", 0.0))
    if metric == "composite":
        value = 0.5 * f1 + 0.25 * float(loc.get("f1@0.3", 0.0)) + 0.25 * iou
    else:
        value = f1
    return value, iou


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit box scale on a calibration split")
    parser.add_argument("--pred", required=True, help="prediction jsonl (calibration split)")
    parser.add_argument("--gold", required=True, help="gold jsonl for the same split")
    parser.add_argument("--coord", choices=COORD_MODES, default="auto")
    parser.add_argument(
        "--scales",
        default="1.0,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.45,1.50",
        help="comma-separated scale grid",
    )
    parser.add_argument(
        "--metric",
        choices=("f1_0.5", "composite"),
        default="f1_0.5",
        help="selection metric (tie-break: mean IoU)",
    )
    parser.add_argument("--aliases", default=None, help=f"default: {DEFAULT_ALIAS_PATH.name}")
    parser.add_argument("--out", default=None, help="write the fit result as JSON")
    args = parser.parse_args(argv)

    pred_rows = read_rows(args.pred)
    gold_rows = read_rows(args.gold)
    aliases = load_diagnosis_aliases(args.aliases) if args.aliases else None
    scales = parse_scales(args.scales)

    grid: list[dict] = []
    best: tuple[float, float, float, dict] | None = None  # (score, iou, scale, report)
    for scale in scales:
        report = evaluate_rows(
            pred_rows,
            gold_rows,
            coord=args.coord,
            aliases=aliases,
            box_scale=scale,
        )
        value, iou = score(report, args.metric)
        loc = report.get("localization", {})
        row = {
            "scale": scale,
            "score": value,
            **{key: float(loc.get(key, 0.0)) for key in LOC_KEYS},
        }
        grid.append(row)
        if best is None or (value, iou) > (best[0], best[1]):
            best = (value, iou, scale, report)

    assert best is not None
    best_score, best_iou, best_scale, best_report = best

    header = f"{'scale':>6}  {'F1@.3':>7}  {'F1@.5':>7}  {'F1@.75':>7}  {'IoU':>7}  {'mAP':>7}"
    print(header)
    print("-" * len(header))
    for row in grid:
        marker = "  <-- best" if abs(row["scale"] - best_scale) < 1e-9 else ""
        print(
            f"{row['scale']:>6.2f}  {row['f1@0.3']:>7.4f}  {row['f1@0.5']:>7.4f}  "
            f"{row['f1@0.75']:>7.4f}  {row['mean_iou']:>7.4f}  {row['map50_95']:>7.4f}{marker}"
        )
    print()
    print(f"best scale = {best_scale:.2f}  ({args.metric} = {best_score:.4f}, mean IoU = {best_iou:.4f})")

    result = {
        "pred": str(args.pred),
        "gold": str(args.gold),
        "coord": args.coord,
        "metric": args.metric,
        "best_scale": best_scale,
        "best_score": best_score,
        "best_mean_iou": best_iou,
        "best_meta": best_report.get("meta", {}),
        "best_localization": best_report.get("localization", {}),
        "grid": grid,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"fit written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
