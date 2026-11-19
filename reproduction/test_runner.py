import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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

    def test_sigreg_sweep_uses_upstream_override_and_distinct_checkpoint_paths(self):
        commands = [run.command(self.args('pusht', action='train', seed=3072, split_seed=13,
                                          action_holdout=0.8, sigreg_weight=weight), Path('.'), Path('.'))
                    for weight in (0.009, 0.09, 0.45)]
        for weight, command in zip((0.009, 0.09, 0.45), commands):
            self.assertIn(f'loss.sigreg.weight={weight}', command)
            self.assertIn(f'output_model_name=restricted/0.8/split-13/sigreg-{weight}/pusht/seed-3072', command)

    def test_uncertainty_artifact_hashes_reject_changed_bank_and_member_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {
                'base_checkpoint': 'base/weights.pt',
                'training_bank': 'bank.pt', 'action_bank': 'action-bank.pt',
                'diagnostic_prior_stats': 'prior-stats.pt',
                'predictor_checkpoints': ['member/weights.pt'],
            }
            paths = run.uncertainty_artifact_paths(root, settings, root)
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name)
            with self.assertRaisesRegex(ValueError, 'content hashes'):
                run.verify_uncertainty_artifacts(root, settings, root, required=True)
            settings['artifact_sha256'] = {name: run.sha256(path) for name, path in paths.items()}
            run.verify_uncertainty_artifacts(root, settings, root, required=True)
            paths['diagnostic_prior_stats'].write_text('replacement')
            with self.assertRaisesRegex(ValueError, 'diagnostic_prior_stats'):
                run.verify_uncertainty_artifacts(root, settings, root, required=True)
            paths['diagnostic_prior_stats'].write_text(paths['diagnostic_prior_stats'].name)
            paths['training_bank'].write_text('replacement')
            with self.assertRaisesRegex(ValueError, 'training_bank'):
                run.verify_uncertainty_artifacts(root, settings, root, required=True)
            paths['training_bank'].write_text(paths['training_bank'].name)
            paths['predictor_0_weights'].write_text('replacement')
            with self.assertRaisesRegex(ValueError, 'predictor_0_weights'):
                run.verify_uncertainty_artifacts(root, settings, root, required=True)

    def test_download_rejects_an_extracted_dataset_from_a_different_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'extracted' / 'pusht' / '.complete'
            marker.parent.mkdir(parents=True)
            marker.write_text('older-revision')
            with patch('run.data_root', return_value=root), patch('huggingface_hub.hf_hub_download') as download:
                with self.assertRaisesRegex(ValueError, 'differs from the pinned revision'):
                    run.download('pusht')
                download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
