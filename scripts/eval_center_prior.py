#!/usr/bin/env python3
"""Evaluate the paper's fixed [120,120,360,360] spatial heuristic; no model needed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval_nova import evaluate_rows, read_rows
from src.nova_schema import NovaTarget, render_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, default=ROOT/'data/nova_sft/test.jsonl')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    gold = read_rows(args.gold)
    target = NovaTarget(boxes=[[120,120,360,360]], caption='', final_diagnosis='', differential=[])
    predictions = [{'response': render_label(target)} for _ in gold]
    report = evaluate_rows(predictions, gold)
    print(json.dumps(report['localization'], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
