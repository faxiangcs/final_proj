#!/usr/bin/env python3
"""Read experiment status without loading models or occupying GPUs."""
import json
import os
from pathlib import Path

root=Path(__file__).resolve().parents[1]
work=root/'experiments/grpo_refinement_20261119'
info=work/'controller.json'
if info.exists():
    state=json.loads(info.read_text())
    try:
        os.kill(state['pid'],0)
        state['alive']=True
    except ProcessLookupError:
        state['alive']=False
    print(json.dumps(state,ensure_ascii=False,indent=2))
for run in sorted((root/'output/grpo_refinement_20261119').glob('*')):
    logs=sorted(run.glob('v*/logging.jsonl'),key=lambda p:p.stat().st_mtime)
    if not logs:
        print(run.name, 'initializing')
        continue
    rows=[]
    for line in logs[-1].read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    last=next((r for r in reversed(rows) if 'global_step/max_steps' in r),{})
    print(run.name, last.get('global_step/max_steps','no step yet'),
          'reward=',last.get('reward'), 'KL=',last.get('kl'),
          'remaining=',last.get('remaining_time'))
for p in sorted(work.glob('*/validation_done.json')):
    d=json.loads(p.read_text());print('validation',d['name'],d['selected'])
if (work/'complete.json').exists():
    print('COMPLETE', (work/'complete.json').read_text())
else:
    events=work/'events.jsonl'
    if events.exists():print('Last event:',events.read_text().splitlines()[-1])
