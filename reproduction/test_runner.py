import argparse
import json
from pathlib import Path
import tempfile
import unittest

import run
from benchmark_report import read_baseline


class ProtocolTests(unittest.TestCase):
    def test_benchmark_report_requires_full_protocol_and_case_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / 'status.json'
            path = root / 'metrics.json'
            status.write_text('{"exit_code": 0, "smoke": true}')
            with self.assertRaisesRegex(ValueError, 'full benchmark'):
                read_baseline(path)
            status.write_text('{"exit_code": 0, "smoke": false}')
            result = {
                "config": {
                    "world": {"env_name": "swm/PushT-v1"}, "policy": "released/pusht/weights.pt", "seed": 42,
                    "eval": {"dataset_name": "pusht_expert_train", "num_eval": 50,
                             "goal_offset_steps": 25, "eval_budget": 50, "img_size": 224},
                    "plan_config": {"horizon": 5, "receding_horizon": 5, "action_block": 5},
                    "solver": {"num_samples": 300, "topk": 30, "n_steps": 30, "var_scale": 1},
                },
                "metrics": {"episode_successes": [True] * 47 + [False] * 3, "success_rate": 94},
                "episode_ids": list(range(50)), "start_steps": [0] * 50,
            }
            path.write_text(json.dumps(result))
            task, _, row = read_baseline(path)
            self.assertEqual(task, 'pusht')
            self.assertEqual(row['delta_from_published_percentage_points'], -2)
            result['config']['solver']['n_steps'] = 10
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'solver.n_steps'):
                read_baseline(path)

    def args(self, task, **kwargs):
        values = dict(task=task, action='eval', seed=42, training_seed=3072,
                      retrained=False, smoke=False)
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_released_eval_requires_real_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run.command(self.args('pusht'), Path(directory), Path(directory))

    def test_paper_planner_iterations_for_each_task(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for task, steps in [('pusht', 30), ('reacher', 10), ('cube', 10)]:
                weights = base / 'checkpoints' / 'released' / task / 'weights.pt'
                weights.parent.mkdir(parents=True)
                weights.touch()
                cmd = run.command(self.args(task), base, base)
                self.assertIn(f'solver.n_steps={steps}', cmd)
                self.assertIn('eval.num_eval=50', cmd)

    def test_smoke_does_not_change_full_training_checkpoint(self):
        args = self.args('cube', action='train', smoke=True, seed=3072)
        cmd = run.command(args, Path('.'), Path('.'))
        names = [x for x in cmd if x.startswith('output_model_name=')]
        self.assertEqual(names[-1], 'output_model_name=smoke/cube/seed-3072')
        self.assertIn('data=ogb', cmd)
        self.assertIn('data.dataset.name=ogbench/cube_single_expert.h5', cmd)

    def test_training_protocol_and_reacher_path(self):
        cmd = run.command(self.args('reacher', action='train'), Path('.'), Path('.'))
        self.assertIn('trainer.max_epochs=10', cmd)
        self.assertIn('data=dmc', cmd)
        self.assertIn('data.dataset.name=dmc/reacher_random.h5', cmd)


if __name__ == '__main__':
    unittest.main()
