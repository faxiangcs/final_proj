"""Exercise checkpoint discovery and chunk recovery without GPU inference."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from scripts import run_refinement as runner


class ResumeTest(unittest.TestCase):
    def test_partial_checkpoint_is_not_resumed(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            good=root/'v0/checkpoint-25'
            partial=root/'v0/checkpoint-50'
            for folder in [good,partial]:
                folder.mkdir(parents=True)
                (folder/'trainer_state.json').write_text(json.dumps({'global_step':int(folder.name.split('-')[-1])}))
                (folder/'adapter_model.safetensors').write_bytes(b'weights')
            (good/'optimizer.pt').write_bytes(b'optimizer')
            (good/'scheduler.pt').write_bytes(b'scheduler')
            self.assertEqual(runner.checkpoints(root),[(25,good)])

    def test_inference_reuses_only_finished_chunks(self):
        with TemporaryDirectory() as tmp:
            work=Path(tmp)
            calls=[]
            def fake(cmd,log,env):
                source=Path(cmd[cmd.index('--val_dataset')+1])
                destination=Path(cmd[cmd.index('--result_path')+1])
                rows=runner.read_rows(source)
                # Report perfect outputs while retaining realistic file shape.
                predictions=[{'response':row['messages'][-1]['content']} for row in rows]
                destination.write_text(''.join(json.dumps(row)+'\n' for row in predictions))
                calls.append(str(source))
            with patch.object(runner,'WORK',work),patch.object(runner,'run_command',fake):
                runner.predict(Path('/mock/adapter'),'val','test',gpus=[0])
                self.assertGreater(len(calls),1)
                calls.clear()
                runner.predict(Path('/mock/adapter'),'val','test',gpus=[0])
                self.assertEqual(calls,[])
                # Only the failed chunk should be repeated.
                (work/'test/val/chunk_0000/done.json').unlink()
                runner.predict(Path('/mock/adapter'),'val','test',gpus=[0])
                self.assertEqual(len(calls),1)
                # A different model may not reuse the previous model's chunks.
                calls.clear()
                runner.predict(Path('/mock/other'),'val','test',gpus=[0])
                self.assertGreater(len(calls),1)


if __name__=='__main__':
    unittest.main()
