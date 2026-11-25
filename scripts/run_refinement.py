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
import random
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval_nova import read_rows, evaluate_rows
from scripts.calibration_ci import bootstrap_case_ci
from collections import defaultdict

WORK = ROOT / 'experiments/grpo_ablation'
OUTPUT = ROOT / 'output/grpo_ablation'
DATA_DIR = ROOT / 'data/nova_sft'
PYTHON = sys.executable
SWIFT = shutil.which('swift') or 'swift'
# Loaded only by main(), so importing helpers never needs local experiment data.
PLAN = {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def event(kind, **kwargs):
    row = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'event': kind, **kwargs}
    print(json.dumps(row), flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
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


def training_environment(stage, seed, output, expected, smoke=False):
    """One explicit recipe shared by execution and the CPU-only command preview."""
    env = os.environ.copy()
    env.pop('GRPO_RESUME', None)
    env.update({
        'PATH': str(Path(PYTHON).parent) + os.pathsep + env.get('PATH', ''),
        'CUDA_VISIBLE_DEVICES': PLAN['train_gpus'],
        'NPROC_PER_NODE': str(PLAN['nproc_per_node']),
        'MASTER_PORT': str(PLAN['master_port']), 'MODEL': PLAN['base_model'],
        'SWIFT_BIN': SWIFT,
        'ADAPTERS': PLAN['sft_adapter'], 'REF_ADAPTERS': PLAN['sft_adapter'],
        'EXTERNAL_PLUGIN': str(ROOT / 'refinement_plugin.py'),
        'GRPO_CALLBACKS': 'frozen_sft_reference_audit',
        'OUTPUT_DIR': str(output), 'DATA_DIR': str(DATA_DIR),
        'TRAIN_FILE': str(DATA_DIR / 'grpo_train.jsonl'),
        'VAL_FILE': str(DATA_DIR / 'grpo_val.jsonl'),
        'LORA_RANK': str(PLAN['lora_rank']), 'LORA_ALPHA': str(PLAN['lora_alpha']),
        'FREEZE_VIT': 'true', 'FREEZE_ALIGNER': 'false',
        'GRPO_LR': str(PLAN['learning_rate']), 'GRPO_EPOCHS': '1',
        'GRPO_BATCH_SIZE': str(PLAN['batch_size']),
        'GRPO_GRAD_ACCUM': str(PLAN['gradient_accumulation']),
        'GRPO_MAX_STEPS': str(expected), 'NUM_GENERATIONS': str(PLAN['generations']),
        'TEMPERATURE': str(PLAN['temperature']), 'TOP_P': str(PLAN['top_p']),
        'BETA': str(PLAN['beta']), 'REWARD_FUNCS': stage['rewards'],
        'REWARD_WEIGHTS': stage['weights'], 'MAX_LENGTH': str(PLAN['max_length']),
        'MAX_COMPLETION_LENGTH': str(PLAN['max_completion_length']),
        'ENABLE_THINKING': 'false', 'GRPO_EVAL_STRATEGY': 'no',
        'USE_VLLM': 'false', 'TORCH_DTYPE': 'bfloat16',
        'LOGGING_STEPS': '1', 'SAVE_STEPS': '1' if smoke else str(PLAN['save_steps']),
        'SEED': str(seed),
    })
    return env


def train(stage, seed, smoke=False, resume_smoke=False):
    name = ('smoke_' if smoke else '') + stage['name'] + f'_seed{seed}'
    output = OUTPUT / name
    marker = WORK / name / 'training_done.json'
    expected = (3 if resume_smoke else 2) if smoke else PLAN['train_steps']
    if marker.exists():
        saved = Path(json.loads(marker.read_text())['checkpoint'])
        if saved.is_dir() and int(saved.name.split('-')[-1]) == expected:
            return saved
    env = training_environment(stage, seed, output, expected, smoke)
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
    source = read_rows(DATA_DIR / f'{split}.jsonl')
    if not source or not gpus:
        raise ValueError('Inference requires nonempty data and at least one evaluation GPU')
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
            # A launcher may inherit NPROC_PER_NODE for training; inference is single-GPU.
            for key in ('NPROC_PER_NODE', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
                env.pop(key, None)
            run_command([SWIFT, 'infer', '--model', PLAN['base_model'], '--adapters', str(checkpoint),
                         '--val_dataset', str(data), '--infer_backend', 'pt', '--temperature', '0',
                         '--max_new_tokens', str(PLAN['max_completion_length']),
                         '--enable_thinking', 'false', '--result_path', str(temp)],
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
    baseline = json.loads((WORK / 'sft/val/report.json').read_text())
    candidates = [{'step':0,'checkpoint':PLAN['sft_adapter'], 'score':list(score(baseline))}]
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


def fit_scale(pred, gold):
    # Sorted grid + max's first-tie rule prefer the smaller scale on a full tie.
    return max(sorted(PLAN['scale_grid']),
               key=lambda s: score(evaluate_rows(pred, gold, box_scale=s)))


def test_and_summarize(selections):
    gold = read_rows(DATA_DIR / 'test.jsonl')
    predict(Path(PLAN['sft_adapter']), 'test', 'sft')
    baseline_pred = read_rows(WORK / 'sft/test/pred.jsonl')
    baseline = evaluate_rows(baseline_pred, gold, collect=True)
    results=[]
    for item in selections:
        name = item['name']
        for kind, path in [('selected',item['selected']['checkpoint']),('final',item['final'])]:
            report=predict(Path(path),'test',name+'_'+kind)
            # Each scale is selected using validation predictions only.
            predict(Path(path),'val',name+'_'+kind+'_calibration')
            val_pred=read_rows(WORK/(name+'_'+kind+'_calibration')/'val/pred.jsonl')
            val_gold=read_rows(DATA_DIR/'val.jsonl')
            scale=fit_scale(val_pred,val_gold)
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


def paired_seed_case_ci(values, case_ids, n_boot=4000, seed=42):
    """Pair image deltas; resample RL seeds and whole cases, retaining image weights."""
    from scripts.calibration_ci import percentile
    if not values or not case_ids or any(len(row) != len(case_ids) for row in values):
        raise ValueError('Need a nonempty seed-by-image matrix matching case_ids')
    if n_boot <= 0:
        raise ValueError('n_boot must be positive')
    cases = sorted(set(case_ids))
    groups = {case: [i for i, c in enumerate(case_ids) if c == case] for case in cases}
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sampled_seeds = [rng.randrange(len(values)) for _ in values]
        sampled_cases = [rng.choice(cases) for _ in cases]
        indices = [i for c in sampled_cases for i in groups[c]]
        draws.append(sum(values[j][i] for j in sampled_seeds for i in indices)
                     / (len(sampled_seeds) * len(indices)))
    return [percentile(draws, .025), percentile(draws, .975)]


def aggregate_results(results):
    """Seed dispersion plus a paired seed-and-case bootstrap, all stages shown."""
    gold = read_rows(DATA_DIR / 'test.jsonl')
    base_pred = read_rows(WORK / 'sft/test/pred.jsonl')
    base = evaluate_rows(base_pred, gold, collect=True)
    baseline_scale = json.loads((WORK / 'baseline_calibration.json').read_text())['scale']
    base_cal = evaluate_rows(base_pred, gold, box_scale=baseline_scale, collect=True)
    case_ids = [s['case_id'] for s in base['per_sample']]
    summaries={}
    for stage in PLAN['stages']:
        for kind in ['selected','final']:
            rows=[next(r for r in results if r['name'] == f"{stage['name']}_seed{seed}"
                       and r['kind'] == kind) for seed in PLAN['seeds']]
            matrix=[]; cal_matrix=[]
            for row in rows:
                folder=WORK/(row['name']+'_'+kind)/'test'
                report=json.loads((folder/'report.json').read_text())
                cal=evaluate_rows(read_rows(folder/'pred.jsonl'),gold,box_scale=row['scale'],collect=True)
                matrix.append([b['loc']['f1@0.5']-a['loc']['f1@0.5'] for a,b in zip(base['per_sample'],report['per_sample'])])
                cal_matrix.append([b['loc']['f1@0.5']-a['loc']['f1@0.5'] for a,b in zip(base_cal['per_sample'],cal['per_sample'])])
            def interval(values):
                return paired_seed_case_ci(values, case_ids,
                                           PLAN['bootstrap_replicates'], PLAN['bootstrap_seed'])
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


def configure(args):
    global PLAN, WORK, OUTPUT, DATA_DIR, PYTHON, SWIFT
    PLAN = json.loads(args.config.read_text())
    WORK = args.work_dir.resolve()
    OUTPUT = args.output_dir.resolve()
    DATA_DIR = args.data_dir.resolve()
    PYTHON = sys.executable
    SWIFT = args.swift
    if not args.model or not args.sft_adapter:
        raise ValueError('Set --model and --sft-adapter (or MODEL and SFT_ADAPTER)')
    PLAN.update(base_model=args.model, sft_adapter=str(Path(args.sft_adapter).resolve()),
                train_gpus=args.train_gpus, eval_gpus=args.eval_gpus.split(','),
                master_port=args.master_port, output_dir=str(OUTPUT), data_dir=str(DATA_DIR))
    if os.path.exists(args.model):
        PLAN['base_model'] = str(Path(args.model).resolve())
    if len(set(PLAN['seeds'])) != len(PLAN['seeds']) or len(PLAN['seeds']) < 2:
        raise ValueError('At least two distinct seeds are required for seed SD')
    stages = PLAN['stages']
    if not stages or len({s['name'] for s in stages}) != len(stages):
        raise ValueError('Stage names must be nonempty and distinct')
    for stage in stages:
        if not stage['name'].replace('_', '').isalnum():
            raise ValueError('Stage names must be alphanumeric with optional underscores')
        if len(stage['rewards'].split()) != len(stage['weights'].split()):
            raise ValueError('Each reward needs one weight')
    if (sorted(set(PLAN['checkpoints'])) != PLAN['checkpoints']
            or not PLAN['checkpoints'] or PLAN['checkpoints'][0] <= 0
            or PLAN['checkpoints'][-1] != PLAN['train_steps']):
        raise ValueError('Selection checkpoints must increase and end at train_steps')
    if (len(args.train_gpus.split(',')) != PLAN['nproc_per_node']
            or not all(PLAN['eval_gpus'])):
        raise ValueError('Training GPU count must equal nproc_per_node; eval GPUs cannot be empty')
    if set(args.train_gpus.split(',')) & set(PLAN['eval_gpus']):
        raise ValueError('Concurrent training and evaluation require disjoint GPU lists')
    if len(set(args.train_gpus.split(','))) != PLAN['nproc_per_node']:
        raise ValueError('Training GPU IDs must be distinct')
    if len(set(PLAN['eval_gpus'])) != len(PLAN['eval_gpus']):
        raise ValueError('Evaluation GPU IDs must be distinct')
    if not PLAN['scale_grid'] or any(s <= 0 for s in PLAN['scale_grid']):
        raise ValueError('scale_grid must contain positive scales')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/grpo_ablation.json')
    parser.add_argument('--model', default=os.environ.get('MODEL'))
    parser.add_argument('--sft-adapter', default=os.environ.get('SFT_ADAPTER'))
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--work-dir', type=Path, default=WORK)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--train-gpus', default='0,1')
    parser.add_argument('--eval-gpus', default='2,3')
    parser.add_argument('--master-port', type=int, default=29632)
    parser.add_argument('--swift', default=SWIFT, help='swift executable in the active environment')
    parser.add_argument('--dry-run', action='store_true', help='print all nine commands without data or GPUs')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume-smoke',action='store_true')
    parser.add_argument('--summarize-only', action='store_true', help='recompute CIs from saved test predictions; no inference')
    args=parser.parse_args(argv)
    configure(args)
    if args.dry_run:
        for stage in PLAN['stages']:
            for seed in PLAN['seeds']:
                name = f"{stage['name']}_seed{seed}"
                env = training_environment(stage, seed, OUTPUT/name, PLAN['train_steps'])
                env['DRY_RUN'] = '1'
                print(f'\n# {name}', flush=True)
                subprocess.run(['bash', str(ROOT/'scripts/run_grpo.sh')], env=env,
                               cwd=ROOT, check=True)
        return
    if args.summarize_only:
        aggregate_results(json.loads((WORK/'test_results.json').read_text()))
        return
    for split in ('train', 'val', 'test', 'grpo_train', 'grpo_val'):
        if not (DATA_DIR/f'{split}.jsonl').is_file():
            raise FileNotFoundError(f'Missing {DATA_DIR}/{split}.jsonl; run src/build_dataset.py first')
    if not (Path(PLAN['sft_adapter'])/'adapter_config.json').is_file():
        raise FileNotFoundError('Pass the actual SFT checkpoint-201 directory, including its v*/ directory')
    if not shutil.which(SWIFT):
        raise FileNotFoundError(f'swift executable not found: {SWIFT}')
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Source hashes lock the executable experiment independently of shell state.
    sources=['medical_reward.py','refinement_plugin.py','src/refinement_rewards.py',
             'src/grpo_reference_audit.py','src/nova_schema.py','src/eval_nova.py',
             'scripts/common.sh','scripts/run_grpo.sh','scripts/run_refinement.py',
             'scripts/calibration_ci.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    hashes['resolved_plan'] = hashlib.sha256(json.dumps(PLAN, sort_keys=True).encode()).hexdigest()
    for split in ('train', 'val', 'test', 'grpo_train', 'grpo_val'):
        hashes[f'data/{split}'] = hashlib.sha256((DATA_DIR/f'{split}.jsonl').read_bytes()).hexdigest()
    manifest=WORK/'source_hashes.json'
    if manifest.exists() and json.loads(manifest.read_text())!=hashes:
        raise RuntimeError('Experiment source changed; review before resuming')
    write(manifest,hashes)
    write(WORK/'plan.json', PLAN)
    write(WORK/'controller.json',{'pid':os.getpid(),'started':time.strftime('%Y-%m-%d %H:%M:%S'),
                                 'plan_sha256':hashlib.sha256((WORK/'plan.json').read_bytes()).hexdigest()})
    if args.smoke or args.resume_smoke:
        train(PLAN['stages'][0],PLAN['seeds'][0],smoke=True,resume_smoke=args.resume_smoke)
        return
    # Baseline inference and scale selection use validation only until choices lock.
    predict(Path(PLAN['sft_adapter']), 'val', 'sft')
    baseline_scale = fit_scale(read_rows(WORK/'sft/val/pred.jsonl'), read_rows(DATA_DIR/'val.jsonl'))
    write(WORK/'baseline_calibration.json', {'scale':baseline_scale, 'selected_on':'val'})
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
        print(f'Controller failed: {exc}', file=sys.stderr)
        if WORK.exists():
            event('failed',error=str(exc))
        raise
