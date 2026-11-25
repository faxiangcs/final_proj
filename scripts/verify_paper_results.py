#!/usr/bin/env python3
"""Verify public aggregate evidence without private data, training dependencies or GPUs."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def verify(folder):
    def load(name):
        return json.loads((folder/name).read_text())

    plan = load('experiment.json')
    selections = load('selection.json')
    aggregate = load('aggregate_results.json')
    results = load('test_results.json')
    signal = load('training_signal.json')
    choice = load('configuration_selection.json')
    baseline = load('baseline.json')['localization']['f1@0.5']
    expected_names = {f'{stage}_seed{seed}' for stage in ('reference','dense','loc_only')
                      for seed in plan['seeds']}
    assert {r['name'] for r in selections} == expected_names
    assert len(selections) == len(expected_names)
    assert set(signal) == expected_names
    assert len(results) == 2*len(expected_names)
    for row in selections:
        assert [c['step'] for c in row['candidates']] == [0, *plan['checkpoints']]
        best = max(row['candidates'], key=lambda c:(*c['score'], -c['step']))
        assert row['selected'] == best, row['name']+' selection is inconsistent'
    means = {}
    for stage in ('reference','dense','loc_only'):
        names = [f'{stage}_seed{seed}' for seed in plan['seeds']]
        means[stage] = mean(r['selected']['score'][0] for r in selections if r['name'] in names)
        rates = []
        for name in names:
            values = signal[name]['frac_reward_zero_std']
            assert len(values) == plan['train_steps']
            assert all(0 <= value <= 1 for value in values)
            assert abs(mean(values)-signal[name]['mean']) < 1e-12
            rates.append(mean(values))
        for kind in ('selected','final'):
            rows = [next(r for r in results if r['name'] == name and r['kind'] == kind) for name in names]
            scores = [r['raw']['f1@0.5'] for r in rows]
            summary = aggregate[f'{stage}_{kind}']
            assert summary['seeds'] == plan['seeds']
            assert scores == summary['raw_f1_per_seed']
            assert abs(mean(scores)-summary['mean_raw_f1']) < 1e-12
            assert abs(stdev(scores)-summary['seed_std_raw_f1']) < 1e-12
            assert abs(mean(scores)-baseline-summary['mean_delta_raw']) < 1e-12
            assert abs(mean(r['calibrated']['f1@0.5'] for r in rows)-summary['mean_calibrated_f1']) < 1e-12
            low, high = summary['paired_seed_case_ci95_raw']
            assert -1 <= low <= high <= 1
        print(f'{stage}: tied {100*mean(rates):.4f}%; selected test F1 '
              f"{aggregate[stage+'_selected']['mean_raw_f1']:.4f}")
    assert all(abs(means[k]-choice['validation_means'][k]) < 1e-12 for k in means)
    assert max(means, key=means.get) == choice['chosen']
    print('Public evidence is internally consistent. Case/seed CIs require local predictions to recompute.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'paper/results')
    verify(parser.parse_args().results_dir)
