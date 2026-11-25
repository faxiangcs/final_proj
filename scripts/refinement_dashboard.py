"""Read-only snapshot of the durable GRPO experiment for the existing dashboard."""
from __future__ import annotations

import json
import re
from pathlib import Path
import time

LABELS = {'reference': '① 修正 SFT reference', 'dense': '② 连续定位奖励', 'loc_only': '③ 仅定位奖励'}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def read_rows(path):
    rows=[]
    try:
        for line in path.read_text().splitlines():
            try:
                row=json.loads(line)
                if isinstance(row,dict): rows.append(row)
            except ValueError:
                pass
    except OSError:
        pass
    return rows


def controller_alive(pid):
    try:
        command=(Path('/proc')/str(int(pid))/'cmdline').read_bytes()
        return b'scripts/run_refinement.py' in command
    except (OSError, TypeError, ValueError):
        return False


def collect_refinement(root):
    work=root/'experiments/grpo_refinement_20261119'
    plan=read_json(work/'plan.json')
    if not plan:
        return {'available':False}
    controller=read_json(work/'controller.json')
    alive=controller_alive(controller.get('pid'))
    events=read_rows(work/'events.jsonl')
    # Smoke events remain in history, but formal state follows the controller's
    # most recent start/resume event for that run.
    latest={}
    for event in events:
        if event.get('name'): latest[event['name']]=event
    runs=[]
    for stage in plan['stages']:
        for seed in plan['seeds']:
            name=f"{stage['name']}_seed{seed}"
            output=root/'output/grpo_refinement_20261119'/name
            versions=sorted(output.glob('v*'),key=lambda p:(int(re.match(r'v(\d+)',p.name).group(1)) if re.match(r'v(\d+)',p.name) else -1,p.name)) if output.exists() else []
            log=versions[-1]/'logging.jsonl' if versions else output/'logging.jsonl'
            history={}
            for version in versions:
                for row in read_rows(version/'logging.jsonl'):
                    step=str(row.get('global_step/max_steps','')).split('/')[0]
                    if step.isdigit() and 'reward' in row:
                        history[int(step)]={
                            'step':int(step), 'reward':row.get('reward'), 'kl':row.get('kl'),
                            'localization':next((v for k,v in row.items() if k.startswith('rewards/') and 'Localization' in k and k.endswith('/mean')),None),
                            'diagnosis':next((v for k,v in row.items() if k.startswith('rewards/') and 'Diagnosis' in k and k.endswith('/mean')),None),
                            'zero_variance':row.get('frac_reward_zero_std'),
                            'remaining':row.get('remaining_time'), 'loss':row.get('loss'),
                        }
            curve=[history[k] for k in sorted(history)]
            last=curve[-1] if curve else {}
            training=read_json(work/name/'training_done.json')
            validation=read_json(work/name/'validation_done.json')
            complete_checkpoints=[]
            for version in versions:
                for ckpt in version.glob('checkpoint-*'):
                    if all((ckpt/f).exists() for f in ['trainer_state.json','optimizer.pt','scheduler.pt','adapter_model.safetensors']):
                        state=read_json(ckpt/'trainer_state.json')
                        if isinstance(state.get('global_step'),int):
                            complete_checkpoints.append(state['global_step'])
            audit=read_json(versions[-1]/'reference_audit.json') if versions else {}
            event=latest.get(name,{})
            if training:
                status='done'
            elif event and alive:
                status='running' if last else 'initializing'
            elif versions:
                status='interrupted'
            else:
                status='pending'
            if status=='running' and log.exists() and time.time()-log.stat().st_mtime>600:
                status='stalled'
            chunks=[]
            reports=[]
            for step in plan['checkpoints']:
                folder=work/f'{name}_ck{step}'/'val'
                finished=sum(1 for p in folder.glob('chunk_*/done.json') if read_json(p))
                chunks.append({'step':step,'done':finished})
                report=read_json(folder/'report.json')
                if report:
                    reports.append({'step':step,'f1':report.get('localization',{}).get('f1@0.5'),
                                    'iou':report.get('localization',{}).get('mean_iou')})
            selected=validation.get('selected')
            if selected:
                validation_state='done'
            elif any(c['done'] for c in chunks):
                validation_state='running' if alive else 'interrupted'
            else:
                validation_state='pending'
            runs.append({
                'name':name,'stage':stage['name'],'label':LABELS.get(stage['name'],stage['name']),
                'seed':seed,'status':status,'step':plan['train_steps'] if training else last.get('step',max(complete_checkpoints,default=0)),
                'total_steps':plan['train_steps'],'last':last,'curve':curve,
                'checkpoint_step':max(complete_checkpoints,default=None),
                'reference':{'verified':bool(audit.get('checks')) and all(c.get('reference_unchanged') for c in audit.get('checks',[])),
                             'last_step':audit.get('checks',[{}])[-1].get('step'),
                             'resumed_at_step':audit.get('resumed_at_step',0)},
                'resumed':event.get('event')=='resume_training' or bool(audit.get('resumed_at_step')),
                'validation':{'status':validation_state,'chunks':chunks,'reports':reports,'selected':selected},
            })
    active=next((r for r in runs if r['status'] in ['running','initializing','stalled']
                 or r['validation']['status']=='running'),None)
    last_event=events[-1] if events else {}
    done=read_json(work/'complete.json')
    error=last_event.get('error') if last_event.get('event')=='failed' else None
    # A restarted controller supersedes a prior transient failure; retain the
    # error payload for diagnostics but show the live process as running.
    status='done' if done else ('running' if alive else ('error' if error else 'interrupted'))
    test_results=read_json(work/'test_results.json') or []
    test_jobs=[]
    for run in runs:
        for kind in ['selected','final']:
            test_dir=work/f"{run['name']}_{kind}"/'test'
            cal_dir=work/f"{run['name']}_{kind}_calibration"/'val'
            test_total=len(list(test_dir.glob('chunk_*')))
            test_done=sum(1 for p in test_dir.glob('chunk_*/done.json') if read_json(p))
            cal_total=len(list(cal_dir.glob('chunk_*')))
            cal_done=sum(1 for p in cal_dir.glob('chunk_*/done.json') if read_json(p))
            hit=next((r for r in test_results if r.get('name')==run['name'] and r.get('kind')==kind),None)
            job_status='done' if hit else ('running' if test_done or cal_done else 'pending')
            test_jobs.append({'name':run['name'],'kind':kind,'status':job_status,
                              'test_done':test_done,'test_total':test_total,
                              'calibration_done':cal_done,'calibration_total':cal_total})
    baseline=read_json(root/'result/qwen_sft_val_checkpoint-201/report.json')
    return {
        'available':True,'status':status,'controller_alive':alive,'pid':controller.get('pid'),
        'started':controller.get('started'),'active':active['name'] if active else None,
        'completed_training':sum(r['status']=='done' for r in runs),'total_runs':len(runs),
        'completed_validation':sum(r['validation']['status']=='done' for r in runs),
        'total_steps':len(runs)*plan['train_steps'],
        'completed_steps':sum(min(r['step'],r['total_steps']) for r in runs),
        'selection_locked':(work/'selection.json').exists(),
        'configuration_selection':read_json(work/'configuration_selection.json'),
        'baseline_val_f1':baseline.get('localization',{}).get('f1@0.5'),
        'train_gpus':plan['train_gpus'],'eval_gpus':plan['eval_gpus'],
        'last_event':last_event,'error':error,'events':events[-8:],'runs':runs,
        'test_results':test_results,'test_jobs':test_jobs,
        'aggregate':read_json(work/'aggregate_results.json'),
    }
