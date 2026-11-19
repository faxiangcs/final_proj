#!/usr/bin/env python3
"""Paired case-cluster bootstrap for box-size calibration deltas.

Compares raw (scale=1.0) and calibrated predictions on the same rows and
reports the mean per-image delta with a 95% CI obtained by resampling whole
cases (images from one case are correlated, so an image-level CI is too
narrow).

Usage:
    python3 scripts/calibration_ci.py \
        --pred result/qwen_sft_ck50/pred.jsonl \
        --gold data/nova_sft/test.jsonl \
        --scale 1.25
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval_nova import COORD_MODES, evaluate_rows, read_rows  # noqa: E402

METRICS = ("f1@0.5", "mean_iou", "map50_95")


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_case_ci(
    deltas_by_case: dict[str, list[float]],
    n_boot: int = 4000,
    seed: int = 42,
) -> tuple[float, float]:
    cases = sorted(deltas_by_case)
    if not cases:
        return 0.0, 0.0
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        picked = [rng.choice(cases) for _ in cases]
        values = [value for case in picked for value in deltas_by_case[case]]
        stats.append(sum(values) / len(values))
    return percentile(stats, 0.025), percentile(stats, 0.975)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Case-cluster CI for box calibration")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--coord", choices=COORD_MODES, default="auto")
    parser.add_argument("--n-boot", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="write the result as JSON")
    args = parser.parse_args(argv)

    pred_rows = read_rows(args.pred)
    gold_rows = read_rows(args.gold)
    raw = evaluate_rows(pred_rows, gold_rows, coord=args.coord, collect=True)
    calibrated = evaluate_rows(
        pred_rows,
        gold_rows,
        coord=args.coord,
        box_scale=args.scale,
        collect=True,
    )

    raw_samples = {s["index"]: s for s in raw["per_sample"]}
    cal_samples = {s["index"]: s for s in calibrated["per_sample"]}

    result: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        deltas_by_case: dict[str, list[float]] = defaultdict(list)
        flat: list[float] = []
        for index, cal in cal_samples.items():
            base = raw_samples.get(index)
            if not base or not cal.get("loc") or not base.get("loc"):
                continue
            delta = float(cal["loc"][metric]) - float(base["loc"][metric])
            deltas_by_case[str(cal.get("case_id") or index)].append(delta)
            flat.append(delta)
        if not flat:
            continue
        low, high = bootstrap_case_ci(deltas_by_case, args.n_boot, args.seed)
        result[metric] = {
            "raw": float(raw["localization"][metric]),
            "calibrated": float(calibrated["localization"][metric]),
            "delta": sum(flat) / len(flat),
            "ci95_low": low,
            "ci95_high": high,
            "cases": float(len(deltas_by_case)),
            "images": float(len(flat)),
        }

    print(f"scale = {args.scale:.2f}  ({args.pred})")
    header = f"{'metric':<10} {'raw':>8} {'calibrated':>11} {'delta':>8} {'95% CI':>20}"
    print(header)
    print("-" * len(header))
    for metric in METRICS:
        row = result.get(metric)
        if not row:
            continue
        print(
            f"{metric:<10} {row['raw']:>8.4f} {row['calibrated']:>11.4f} "
            f"{row['delta']:>+8.4f} "
            f"[{row['ci95_low']:>+7.4f}, {row['ci95_high']:>+7.4f}]"
        )
    print(f"cases={int(next(iter(result.values()))['cases'])} "
          f"images={int(next(iter(result.values()))['images'])} "
          f"bootstrap={args.n_boot}")

    if args.out:
        payload = {
            "pred": str(args.pred),
            "gold": str(args.gold),
            "scale": args.scale,
            "n_boot": args.n_boot,
            "metrics": result,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
