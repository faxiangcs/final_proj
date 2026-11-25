"""Exercise checkpoint discovery, selection and chunk recovery without private data or GPUs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from scripts import run_refinement as runner
from tests.test_rewards import GOLD_TEXT


class ResumeTest(unittest.TestCase):
    def test_partial_checkpoint_is_not_resumed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            good, partial = root/'v0/checkpoint-25', root/'v0/checkpoint-50'
            for folder in (good, partial):
                folder.mkdir(parents=True)
                (folder/'trainer_state.json').write_text(json.dumps({'global_step':int(folder.name.split('-')[-1])}))
                (folder/'adapter_model.safetensors').write_bytes(b'weights')
            for name in ('optimizer.pt','scheduler.pt'):
                (good/name).write_bytes(b'state')
            self.assertEqual(runner.checkpoints(root), [(25,good)])

    def test_inference_reuses_only_finished_chunks(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            data.mkdir()
            rows = [{'messages':[{'role':'user','content':'Synthetic prompt'},
                                 {'role':'assistant','content':GOLD_TEXT}],
                     'images':[f'/synthetic/case_{i:03d}_image.png']} for i in range(18)]
            (data/'val.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            calls = []

            def fake(cmd, log, env):
                source = Path(cmd[cmd.index('--val_dataset')+1])
                destination = Path(cmd[cmd.index('--result_path')+1])
                predictions = [{'response':r['messages'][-1]['content']} for r in runner.read_rows(source)]
                destination.write_text(''.join(json.dumps(row)+'\n' for row in predictions))
                calls.append(str(source))
                self.assertNotIn('NPROC_PER_NODE', env)

            with patch.object(runner, 'WORK', root/'work'), patch.object(runner, 'DATA_DIR', data), \
                    patch.object(runner, 'PLAN', {'base_model':'synthetic','max_completion_length':512}), \
                    patch.object(runner, 'run_command', fake):
                runner.predict(Path('/mock/adapter'), 'val', 'test', gpus=[0])
                self.assertEqual(len(calls), 3)
                calls.clear()
                runner.predict(Path('/mock/adapter'), 'val', 'test', gpus=[0])
                self.assertEqual(calls, [])
                (root/'work/test/val/chunk_0000/done.json').unlink()
                runner.predict(Path('/mock/adapter'), 'val', 'test', gpus=[0])
                self.assertEqual(len(calls), 1)
                calls.clear()
                runner.predict(Path('/mock/other'), 'val', 'test', gpus=[0])
                self.assertEqual(len(calls), 3)

    def test_validation_uses_iou_then_earlier_step_and_allows_sft(self):
        for scores, expected in [({50:(.4,.3),100:(.4,.3)},0),
                                 ({50:(.4,.35),100:(.4,.35)},50),
                                 ({50:(.4,.35),100:(.5,.2)},100)]:
            with self.subTest(expected=expected), TemporaryDirectory() as tmp:
                work = Path(tmp)
                base = work/'sft/val'
                base.mkdir(parents=True)
                (base/'report.json').write_text(json.dumps({'localization':{'f1@0.5':.4,'mean_iou':.3}}))
                final = work/'output/v0/checkpoint-100'
                final.mkdir(parents=True)
                (final.parent/'checkpoint-50').mkdir()

                def fake_predict(checkpoint, split, name):
                    self.assertEqual(split, 'val')
                    f1, iou = scores[int(checkpoint.name.split('-')[-1])]
                    return {'localization':{'f1@0.5':f1,'mean_iou':iou}}

                with patch.object(runner, 'WORK', work), \
                        patch.object(runner, 'PLAN', {'sft_adapter':'/synthetic/sft','checkpoints':[50,100]}), \
                        patch.object(runner, 'predict', fake_predict):
                    value = runner.validate({'name':'reference'}, 42, final)
                self.assertEqual(value['selected']['step'], expected)


if __name__ == '__main__':
    unittest.main()
