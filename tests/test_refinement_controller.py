"""Exercise a fresh controller start and prove test is gated by validation locks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_refinement as runner
from tests.test_rewards import GOLD_TEXT


class ControllerTest(unittest.TestCase):
    def test_fresh_run_locks_all_choices_before_test(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            data.mkdir()
            gold = [{'messages':[{'role':'user','content':'Synthetic input'},
                                 {'role':'assistant','content':GOLD_TEXT}],
                     'images':['/synthetic/case_001_img.png']}]
            for split in ('train','val','test','grpo_train','grpo_val'):
                (data/f'{split}.jsonl').write_text(json.dumps(gold[0])+'\n')
            adapter = root/'sft/checkpoint-201'
            adapter.mkdir(parents=True)
            (adapter/'adapter_config.json').write_text('{}')
            config = json.loads((runner.ROOT/'configs/grpo_ablation.json').read_text())
            config.update(seeds=[42,43],checkpoints=[1,2],train_steps=2)
            config_path = root/'config.json'
            config_path.write_text(json.dumps(config))
            work, output = root/'work', root/'output'
            calls = []

            def train(stage, seed):
                final = output/f"{stage['name']}_seed{seed}"/'v0/checkpoint-2'
                final.mkdir(parents=True)
                (final.parent/'checkpoint-1').mkdir()
                return final

            def predict(checkpoint, split, name):
                self.assertEqual(split, 'val')
                calls.append((name, split))
                folder = work/name/split
                folder.mkdir(parents=True, exist_ok=True)
                pred = [{'response':GOLD_TEXT}]
                (folder/'pred.jsonl').write_text(json.dumps(pred[0])+'\n')
                report = runner.evaluate_rows(pred, gold, collect=True)
                runner.write(folder/'report.json',report)
                return report

            def test(selections):
                self.assertEqual(len(selections), 6)
                self.assertTrue((work/'selection.json').exists())
                self.assertTrue((work/'configuration_selection.json').exists())
                self.assertTrue((work/'source_hashes.json').exists())
                self.assertEqual(json.loads((work/'baseline_calibration.json').read_text())['scale'],1.0)
                self.assertEqual({s['selected']['step'] for s in selections}, {0})
                calls.append(('test','test'))

            argv = ['--model','synthetic-model','--sft-adapter',str(adapter),
                    '--config',str(config_path),'--data-dir',str(data),
                    '--work-dir',str(work),'--output-dir',str(output)]
            with patch.object(runner,'PLAN',{}), patch.object(runner,'WORK',work), \
                    patch.object(runner,'OUTPUT',output), patch.object(runner,'DATA_DIR',data), \
                    patch.object(runner,'train',train), patch.object(runner,'predict',predict), \
                    patch.object(runner,'test_and_summarize',test), \
                    patch.object(runner.shutil,'which',return_value='swift'):
                runner.main(argv)
                self.assertEqual(calls[-1], ('test','test'))
                self.assertTrue((work/'controller.json').exists())
                config['beta'] = .002
                config_path.write_text(json.dumps(config))
                with self.assertRaisesRegex(RuntimeError,'source changed'):
                    runner.main(argv)


if __name__ == '__main__':
    unittest.main()
