"""Check that an explicit SFT reference adapter remains byte-for-byte frozen."""
import hashlib
import json
from pathlib import Path

import torch
from transformers import TrainerCallback
from swift.callbacks import callbacks_map


def adapter_hash(model, adapter):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        marker = '.' + adapter + '.'
        if marker not in name:
            continue
        digest.update(name.replace(marker, '.ADAPTER.').encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        count += parameter.numel()
    if not count:
        raise RuntimeError(f'Missing adapter parameters: {adapter}')
    return digest.hexdigest(), count


class FrozenSFTReferenceAudit(TrainerCallback):
    def __init__(self, args=None, trainer=None):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # The framework loads ref_adapters with is_trainable=True for PEFT
        # compatibility. Explicitly select the policy before the first update.
        model.set_adapter('default')
        for name, parameter in model.named_parameters():
            if '.ref_adapter.' in name:
                parameter.requires_grad_(False)
        self.initial_ref_hash, self.parameter_count = adapter_hash(model, 'ref_adapter')
        self.initial_policy_hash, policy_count = adapter_hash(model, 'default')
        self.starts_equal = self.initial_ref_hash == self.initial_policy_hash
        self.resume_step = state.global_step
        if policy_count != self.parameter_count:
            raise RuntimeError('Policy/reference adapter sizes differ')
        if state.global_step == 0 and not self.starts_equal:
            raise RuntimeError('Policy/reference adapters differ before training')
        self.records = []
        self.check(args, state, model)

    def check(self, args, state, model):
        current, _ = adapter_hash(model, 'ref_adapter')
        if current != self.initial_ref_hash:
            raise RuntimeError('Frozen SFT reference changed during training')
        if any(p.grad is not None for n, p in model.named_parameters() if '.ref_adapter.' in n):
            raise RuntimeError('Reference adapter received gradients')
        if state.is_world_process_zero:
            self.records.append({'step': state.global_step, 'reference_unchanged': True})
            policy_hash, _ = adapter_hash(model, 'default')
            out = Path(args.output_dir) / 'reference_audit.json'
            out.write_text(json.dumps({
                'reference_parameter_count': self.parameter_count,
                'initial_reference_sha256': self.initial_ref_hash,
                'initial_policy_equals_reference': self.starts_equal,
                'resumed_at_step': getattr(self, 'resume_step', 0),
                'latest_policy_sha256': policy_hash,
                'policy_updated': policy_hash != self.initial_ref_hash,
                'policy_changed_this_session': policy_hash != self.initial_policy_hash,
                'checks': self.records,
            }, indent=2))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 1 or state.global_step % 25 == 0:
            self.check(args, state, model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self.check(args, state, model)


callbacks_map['frozen_sft_reference_audit'] = FrozenSFTReferenceAudit
