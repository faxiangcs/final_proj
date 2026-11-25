#!/usr/bin/env python3
"""Durable, restartable sequential ablations. Start via start_refinement.sh.

A controller lock prevents duplicate schedulers. Training resumes optimizer,
scheduler and RNG from a complete checkpoint. Inference stores independent
8-row chunks, so an interrupted chunk alone is recomputed. No test predictions
are made until all validation selections are written to selection.json.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval_nova import read_rows, evaluate_rows
from scripts.calibration_ci import bootstrap_case_ci
from collections import defaultdict

WORK = ROOT / 'experiments/grpo_refinement_20261119'
PYTHON = '/data/yanfaxiang/venvs/ms-swift/bin/python'
SWIFT = '/data/yanfaxiang/venvs/ms-swift/bin/swift'
PLAN = json.loads((WORK / 'plan.json').read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def event(kind, **kwargs):
    row = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'event': kind, **kwargs}
    print(json.dumps(row), flush=True)
    with (WORK / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')


def run_command(cmd, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        code = proc.wait()
    if code:
        raise RuntimeError(f'Command failed ({code}); see {log}')


def checkpoints(output):
    found = []
    for path in output.glob('v*/checkpoint-*'):
        # HF writes trainer_state at the end of checkpoint serialization.
        if all((path / f).exists() for f in ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'adapter_model.safetensors']):
            state = json.loads((path / 'trainer_state.json').read_text())
            step = int(state['global_step'])
            if step != int(path.name.split('-')[-1]):
                continue
            found.append((step, path))
    return sorted(found, key=lambda x: (x[0], x[1].stat().st_mtime))


def train(stage, seed, smoke=False, resume_smoke=False):
    name = ('smoke_' if smoke else '') + stage['name'] + f'_seed{seed}'
    output = ROOT / 'output/grpo_refinement_20261119' / name
    marker = WORK / name / 'training_done.json'
    expected = (3 if resume_smoke else 2) if smoke else 266
    if marker.exists():
        saved = Path(json.loads(marker.read_text())['checkpoint'])
        if int(saved.name.split('-')[-1]) == expected:
            return saved
    env = os.environ.copy()
    env.pop('GRPO_RESUME', None)
    env.update({
        'PATH': str(Path(PYTHON).parent) + ':' + env['PATH'],
        'CUDA_VISIBLE_DEVICES': PLAN['train_gpus'], 'NPROC_PER_NODE': '2',
        'MASTER_PORT': '29632', 'MODEL': PLAN['base_model'],
        'ADAPTERS': str(ROOT / PLAN['sft_adapter']),
        'REF_ADAPTERS': str(ROOT / PLAN['sft_adapter']),
        'EXTERNAL_PLUGIN': str(ROOT / 'refinement_plugin.py'),
        'GRPO_CALLBACKS': 'frozen_sft_reference_audit',
        'OUTPUT_DIR': str(output), 'DATA_DIR': str(ROOT / 'data/nova_sft'),
        'LORA_RANK': '16', 'LORA_ALPHA': '32', 'FREEZE_VIT': 'true',
        'FREEZE_ALIGNER': 'false', 'GRPO_LR': '5e-6', 'GRPO_EPOCHS': '1',
        'GRPO_BATCH_SIZE': '1', 'GRPO_GRAD_ACCUM': '4',
        'GRPO_MAX_STEPS': str(expected),
        'NUM_GENERATIONS': '4', 'TEMPERATURE': '0.9', 'TOP_P': '0.95',
        'BETA': '0.001', 'REWARD_FUNCS': stage['rewards'],
        'REWARD_WEIGHTS': stage['weights'], 'MAX_LENGTH': '2048',
        'MAX_COMPLETION_LENGTH': '512', 'ENABLE_THINKING': 'false',
        'GRPO_EVAL_STRATEGY': 'no', 'LOGGING_STEPS': '1',
        'SAVE_STEPS': '1' if smoke else '25', 'SEED': str(seed),
    })
    last = checkpoints(output)
    if last:
        env['GRPO_RESUME'] = str(last[-1][1])
        event('resume_training', name=name, checkpoint=env['GRPO_RESUME'])
    else:
        event('start_training', name=name)
    write(WORK / name / 'environment.json', {k:v for k,v in env.items() if k in ['MODEL','ADAPTERS','REF_ADAPTERS','GRPO_RESUME','SEED','REWARD_FUNCS','REWARD_WEIGHTS','CUDA_VISIBLE_DEVICES','NPROC_PER_NODE','GRPO_MAX_STEPS','GRPO_GRAD_ACCUM']})
    run_command(['bash', 'scripts/run_grpo.sh'], WORK / name / 'train.log', env)
    last = checkpoints(output)
    if not last or last[-1][0] != expected:
        raise RuntimeError(f'Missing final checkpoint: {name}')
    audit = json.loads((last[-1][1].parent / 'reference_audit.json').read_text())
    if not audit['policy_updated'] or not all(c['reference_unchanged'] for c in audit['checks']):
        raise RuntimeError('Reference/policy checks failed')
    write(marker, {'checkpoint': str(last[-1][1]), 'reference_audit': audit})
    event('training_done', name=name)
    return last[-1][1]


def predict(checkpoint, split, name, gpus=None):
    gpus = PLAN['eval_gpus'] if gpus is None else gpus
    out = WORK / name / split
    source = read_rows(ROOT / f'data/nova_sft/{split}.jsonl')
    chunks = []
    for start in range(0, len(source), 8):
        folder = out / f'chunk_{start:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        data = folder / 'input.jsonl'
        data.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in source[start:start+8]))
        chunks.append((start, folder, data, len(source[start:start+8])))

    def worker(gpu, assigned):
        for start, folder, data, count in assigned:
            dest = folder / 'pred.jsonl'
            done = folder / 'done.json'
            if done.exists() and json.loads(done.read_text()).get('checkpoint') == str(checkpoint) and dest.exists() and len(read_rows(dest)) == count:
                continue
            temp = folder / f'partial_{int(time.time())}.jsonl'
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            run_command([SWIFT, 'infer', '--model', PLAN['base_model'], '--adapters', str(checkpoint),
                         '--val_dataset', str(data), '--infer_backend', 'pt', '--temperature', '0',
                         '--max_new_tokens', '512', '--enable_thinking', 'false', '--result_path', str(temp)],
                        folder / 'infer.log', env)
            rows = read_rows(temp)
            if len(rows) != count:
                raise RuntimeError(f'Incomplete inference chunk: {folder}')
            temp.replace(dest)
            write(done, {'rows': count, 'checkpoint': str(checkpoint)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, chunks[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for f in futures:
            f.result()
    pred = [row for _,folder,_,_ in chunks for row in read_rows(folder / 'pred.jsonl')]
    report = evaluate_rows(pred, source, collect=True)
    (out / 'pred.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in pred))
    write(out / 'report.json', report)
    return report


def score(report):
    return (report['localization']['f1@0.5'], report['localization']['mean_iou'])


def validate(stage, seed, final):
    name = stage['name'] + f'_seed{seed}'
    done = WORK / name / 'validation_done.json'
    if done.exists():
        return json.loads(done.read_text())
    baseline = json.loads((ROOT / 'result/qwen_sft_val_checkpoint-201/report.json').read_text())
    candidates = [{'step':0,'checkpoint':str(ROOT / PLAN['sft_adapter']), 'score':list(score(baseline))}]
    for step in PLAN['checkpoints']:
        available = {n:p for n,p in checkpoints(final.parent.parent)}
        ckpt = available.get(step, final.parent / f'checkpoint-{step}')
        if not ckpt.is_dir():
            raise RuntimeError(f'Missing selection checkpoint {ckpt}')
        report = predict(ckpt, 'val', name + f'_ck{step}')
        candidates.append({'step':step,'checkpoint':str(ckpt), 'score':list(score(report))})
    chosen = max(candidates, key=lambda c: (*c['score'], -c['step']))
    value = {'name':name,'candidates':candidates,'selected':chosen,'final':str(final)}
    write(done,value)
    event('validation_done', name=name, selected_step=chosen['step'], val_f1=chosen['score'][0])
    return value


def test_and_summarize(selections):
    from scripts.fit_box_scale import parse_scales
    gold = read_rows(ROOT / 'data/nova_sft/test.jsonl')
    baseline_pred = read_rows(ROOT / 'result/qwen_sft_test_checkpoint-201/pred.jsonl')
    baseline = evaluate_rows(baseline_pred, gold, collect=True)
    results=[]
    for item in selections:
        name = item['name']
        for kind, path in [('selected',item['selected']['checkpoint']),('final',item['final'])]:
            report=predict(Path(path),'test',name+'_'+kind)
            # Each scale is selected using validation predictions only.
            val = predict(Path(path),'val',name+'_'+kind+'_calibration')
            val_pred=read_rows(WORK/(name+'_'+kind+'_calibration')/'val/pred.jsonl')
            val_gold=read_rows(ROOT/'data/nova_sft/val.jsonl')
            scales=parse_scales('1,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.45,1.50')
            scale=max(scales,key=lambda s:score(evaluate_rows(val_pred,val_gold,box_scale=s)))
            test_pred=read_rows(WORK/(name+'_'+kind)/'test/pred.jsonl')
            calibrated=evaluate_rows(test_pred,gold,box_scale=scale)
            deltas=defaultdict(list)
            for a,b in zip(baseline['per_sample'],report['per_sample']):
                deltas[a['case_id']].append(b['loc']['f1@0.5']-a['loc']['f1@0.5'])
            row={'name':name,'kind':kind,'checkpoint':path,'raw':report['localization'],
                 'caption':report['caption'],'diagnosis':report['diagnosis'],
                 'scale':scale,'calibrated':calibrated['localization'],
                 'delta_vs_sft_raw':report['localization']['f1@0.5']-baseline['localization']['f1@0.5'],
                 'nominal_case_ci95':bootstrap_case_ci(deltas)}
            results.append(row)
            write(WORK/'test_results.json',results)
    aggregate_results(results)
    write(WORK/'complete.json',{'completed':time.strftime('%Y-%m-%d %H:%M:%S'),'runs':len(results)})
    lines=['# GRPO refinement results','', 'Exploratory comparison; historical test was already inspected.', '',
           '| Run | Checkpoint rule | Raw F1@0.5 | Calibrated F1@0.5 | Caption F1 | Diagnosis keyword T1 |',
           '|---|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['name']} | {r['kind']} | {r['raw']['f1@0.5']:.4f} | {r['calibrated']['f1@0.5']:.4f} | {r['caption']['f1']:.4f} | {r['diagnosis']['top1_keyword']:.4f} |")
    (WORK/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def aggregate_results(results):
    """Seed dispersion plus a paired seed-and-case bootstrap, all stages shown."""
    from scripts.calibration_ci import percentile
    gold = read_rows(ROOT / 'data/nova_sft/test.jsonl')
    base_pred = read_rows(ROOT / 'result/qwen_sft_test_checkpoint-201/pred.jsonl')
    base = evaluate_rows(base_pred, gold, collect=True)
    base_cal = evaluate_rows(base_pred, gold, box_scale=1.20, collect=True)
    cases = sorted({s['case_id'] for s in base['per_sample']})
    groups = {case:[i for i,s in enumerate(base['per_sample']) if s['case_id']==case] for case in cases}
    summaries={}
    for stage in PLAN['stages']:
        for kind in ['selected','final']:
            rows=[r for r in results if r['name'].startswith(stage['name']+'_') and r['kind']==kind]
            matrix=[]; cal_matrix=[]
            for row in rows:
                folder=WORK/(row['name']+'_'+kind)/'test'
                report=json.loads((folder/'report.json').read_text())
                cal=evaluate_rows(read_rows(folder/'pred.jsonl'),gold,box_scale=row['scale'],collect=True)
                matrix.append([b['loc']['f1@0.5']-a['loc']['f1@0.5'] for a,b in zip(base['per_sample'],report['per_sample'])])
                cal_matrix.append([b['loc']['f1@0.5']-a['loc']['f1@0.5'] for a,b in zip(base_cal['per_sample'],cal['per_sample'])])
            def interval(values):
                rng=random.Random(42); draws=[]
                for _ in range(4000):
                    sampled_seeds=[rng.randrange(len(values)) for _ in values]
                    sampled_cases=[rng.choice(cases) for _ in cases]
                    indices=[i for c in sampled_cases for i in groups[c]]
                    draws.append(sum(values[j][i] for j in sampled_seeds for i in indices)/(len(sampled_seeds)*len(indices)))
                return [percentile(draws,.025),percentile(draws,.975)]
            scores=[r['raw']['f1@0.5'] for r in rows]
            summaries[stage['name']+'_'+kind]={
                'seeds':PLAN['seeds'],'mean_raw_f1':statistics.mean(scores),
                'seed_std_raw_f1':statistics.stdev(scores), 'raw_f1_per_seed':scores,
                'mean_delta_raw':statistics.mean(scores)-base['localization']['f1@0.5'],
                'paired_seed_case_ci95_raw':interval(matrix),
                'mean_calibrated_f1':statistics.mean(r['calibrated']['f1@0.5'] for r in rows),
                'paired_seed_case_ci95_calibrated':interval(cal_matrix),
                'baseline_raw':base['localization']['f1@0.5'],
                'baseline_calibrated':base_cal['localization']['f1@0.5'],
            }
    write(WORK/'aggregate_results.json',summaries)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume-smoke',action='store_true')
    args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(WORK/'controller.json',{'pid':os.getpid(),'started':time.strftime('%Y-%m-%d %H:%M:%S'),'plan_sha256':hashlib.sha256((WORK/'plan.json').read_bytes()).hexdigest()})
    if args.smoke or args.resume_smoke:
        train(PLAN['stages'][0],42,smoke=True,resume_smoke=args.resume_smoke)
        return
    # Source hashes lock the executable experiment independently of shell state.
    sources=['experiments/grpo_refinement_20261119/plan.json','medical_reward.py','refinement_plugin.py','src/refinement_rewards.py','src/grpo_reference_audit.py','scripts/run_grpo.sh', 'scripts/run_refinement.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    manifest=WORK/'source_hashes.json'
    if manifest.exists() and json.loads(manifest.read_text())!=hashes:
        raise RuntimeError('Experiment source changed; review before resuming')
    write(manifest,hashes)
    # Keep one validation job concurrent with the next training run.
    futures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as evaluator:
        for stage in PLAN['stages']:
            for seed in PLAN['seeds']:
                final=train(stage,seed)
                futures.append(evaluator.submit(validate,stage,seed,final))
        selections=[f.result() for f in futures]
    path=WORK/'selection.json'
    if path.exists() and json.loads(path.read_text())!=selections:
        raise RuntimeError('Locked checkpoint selections changed')
    write(path,selections)
    means={stage['name']: sum(item['selected']['score'][0] for item in selections if item['name'].startswith(stage['name']+'_'))/len(PLAN['seeds']) for stage in PLAN['stages']}
    write(WORK/'configuration_selection.json', {'validation_means':means,'chosen':max(means,key=means.get),'test_not_consulted':True})
    event('all_selections_locked',runs=len(selections))
    test_and_summarize(selections)
    event('complete')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        event('failed',error=str(exc))
        raise
