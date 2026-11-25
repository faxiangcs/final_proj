#!/usr/bin/env python3
"""Export aggregate paper evidence; never copy case text, predictions, or model paths."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def export(work, training, baseline_report, destination):
    sources = {}

    def load(path):
        raw = path.read_bytes()
        sources[path.name if path.parent == work else str(path.relative_to(ROOT))
                if path.is_relative_to(ROOT) else path.name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    plan = load(work/'plan.json')
    selections = load(work/'selection.json')
    aggregate = load(work/'aggregate_results.json')
    test = load(work/'test_results.json')
    chosen = load(work/'configuration_selection.json')
    baseline = load(baseline_report)
    clean_selection = []
    for row in selections:
        clean_selection.append({
            'name': row['name'],
            'candidates': [{k:c[k] for k in ('step', 'score')} for c in row['candidates']],
            'selected': {k:row['selected'][k] for k in ('step', 'score')},
        })
    signals = {}
    for stage in plan['stages']:
        for seed in plan['seeds']:
            name = f"{stage['name']}_seed{seed}"
            steps = {}
            logs = sorted((training/name).glob('v*/logging.jsonl'), key=lambda p:p.stat().st_mtime)
            for log in logs:
                key = str(log.relative_to(training))
                raw = log.read_bytes()
                sources['training/'+key] = hashlib.sha256(raw).hexdigest()
                for line in raw.decode().splitlines():
                    row = json.loads(line)
                    if 'frac_reward_zero_std' in row:
                        step = int(row['global_step/max_steps'].split('/')[0])
                        steps[step] = row['frac_reward_zero_std']
            if set(steps) != set(range(1, plan['train_steps']+1)):
                raise ValueError(f'{name}: expected every training step exactly once after merging restarts')
            values = [steps[i] for i in range(1, plan['train_steps']+1)]
            if any(not 0 <= value <= 1 for value in values):
                raise ValueError(f'{name}: invalid tied-group fraction')
            signals[name] = {'frac_reward_zero_std': values, 'mean':mean(values)}
    clean_test = [{k:r[k] for k in ('name','kind','raw','caption','diagnosis','scale',
                  'calibrated','delta_vs_sft_raw','nominal_case_ci95')} for r in test]
    clean_baseline = {k:baseline[k] for k in ('localization','caption','diagnosis')}
    payloads = {
        'experiment.json': {k:plan[k] for k in ('seeds','checkpoints','train_steps')},
        'selection.json': clean_selection,
        'configuration_selection.json': chosen,
        'aggregate_results.json': aggregate,
        'test_results.json': clean_test,
        'baseline.json': clean_baseline,
        'training_signal.json': signals,
        'provenance.json': {
            'description': 'Aggregate metrics and scalar training diagnostics from the reported runs. '
                           'No images, case identifiers, report text, or checkpoint paths are exported.',
            'source_sha256': sources,
            'historical_source_sha256': load(work/'source_hashes.json') if (work/'source_hashes.json').exists() else {},
            'portability_note': 'Released launchers/controller support configurable paths and validation-fitted '
                                'baseline calibration. Historical run files are unchanged; original source '
                                'hashes are provenance, not hashes of the revised controller.',
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    for filename, value in payloads.items():
        (destination/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(f'Exported {len(signals)} training runs, {len(clean_test)} test summaries to {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT/'experiments/grpo_ablation')
    parser.add_argument('--training-dir', type=Path, default=ROOT/'output/grpo_ablation')
    parser.add_argument('--baseline-report', type=Path,
                        help='default: WORK_DIR/sft/test/report.json')
    parser.add_argument('--out-dir', type=Path, default=ROOT/'paper/results')
    args = parser.parse_args()
    export(args.work_dir.resolve(), args.training_dir.resolve(),
           (args.baseline_report or args.work_dir/'sft/test/report.json').resolve(),
           args.out_dir.resolve())


if __name__ == '__main__':
    main()
