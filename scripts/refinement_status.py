#!/usr/bin/env python3
"""Read saved controller progress without importing training dependencies."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT/'experiments/grpo_ablation')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'output/grpo_ablation')
    args = parser.parse_args()
    for name in ('controller.json', 'configuration_selection.json', 'complete.json'):
        path = args.work_dir/name
        if path.exists():
            print(name, path.read_text())
    for run in sorted(args.output_dir.glob('*')):
        latest = {}
        for log in sorted(run.glob('v*/logging.jsonl'), key=lambda p: p.stat().st_mtime):
            for line in log.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if 'global_step/max_steps' in row:
                    latest = row
        print(run.name, latest.get('global_step/max_steps', 'initializing'),
              'reward=', latest.get('reward'), 'KL=', latest.get('kl'))
    for path in sorted(args.work_dir.glob('*/validation_done.json')):
        value = json.loads(path.read_text())
        print('validation', value['name'], value['selected'])
    if not args.work_dir.exists():
        print('No experiment has been started in', args.work_dir)


if __name__ == '__main__':
    main()
