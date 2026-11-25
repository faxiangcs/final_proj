"""Monitor persisted scheduler state, including partial writes and restarts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.refinement_dashboard import collect_refinement
from scripts.monitor_dashboard import HTML


class RefinementDashboardTest(unittest.TestCase):
    def test_absent_experiment(self):
        with TemporaryDirectory() as tmp:
            self.assertFalse(collect_refinement(Path(tmp))['available'])

    def test_resume_and_interrupted_status(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            work=root/'experiments/grpo_refinement_20261119'
            work.mkdir(parents=True)
            plan={'stages':[{'name':'reference'}],'seeds':[42],'checkpoints':[50],
                  'train_steps':266,'train_gpus':'2,3','eval_gpus':[0,1]}
            (work/'plan.json').write_text(json.dumps(plan))
            (work/'controller.json').write_text('{"pid":123}')
            (work/'events.jsonl').write_text(json.dumps({'event':'resume_training','name':'reference_seed42'})+'\n')
            out=root/'output/grpo_refinement_20261119/reference_seed42'
            old=out/'v0'; old.mkdir(parents=True)
            (old/'logging.jsonl').write_text(json.dumps({'global_step/max_steps':'25/266','reward':.4,'kl':.01})+'\n')
            new=out/'v1';new.mkdir()
            (new/'logging.jsonl').write_text(json.dumps({'global_step/max_steps':'26/266','reward':.5,'kl':.02})+'\n{partial')
            (new/'reference_audit.json').write_text(json.dumps({'resumed_at_step':25,'checks':[{'step':26,'reference_unchanged':True}]}))
            ck=old/'checkpoint-25';ck.mkdir()
            for f in ['optimizer.pt','scheduler.pt','adapter_model.safetensors']:(ck/f).touch()
            (ck/'trainer_state.json').write_text('{"global_step":25}')
            with patch('scripts.refinement_dashboard.controller_alive',return_value=True):
                snapshot=collect_refinement(root)
            run=snapshot['runs'][0]
            self.assertEqual(run['step'],26)
            self.assertEqual(len(run['curve']),2)
            self.assertEqual(run['checkpoint_step'],25)
            self.assertTrue(run['reference']['verified'])
            self.assertTrue(run['resumed'])
            self.assertEqual(run['status'],'running')
            with patch('scripts.refinement_dashboard.controller_alive',return_value=False):
                self.assertEqual(collect_refinement(root)['runs'][0]['status'],'interrupted')
            done=work/'reference_seed42';done.mkdir()
            (done/'training_done.json').write_text('{"checkpoint":"done"}')
            (done/'validation_done.json').write_text(json.dumps({'selected':{'step':50,'score':[.45,.4]}}))
            with patch('scripts.refinement_dashboard.controller_alive',return_value=True):
                snapshot=collect_refinement(root)
            self.assertEqual(snapshot['completed_training'],1)
            self.assertEqual(snapshot['completed_validation'],1)

    def test_page_has_refinement_widgets(self):
        self.assertIn('id="refinement-queue"',HTML)
        self.assertIn('renderRefinement(data.refinement)',HTML)


if __name__=='__main__':unittest.main()
