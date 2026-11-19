import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from action_holdout import clip_magnitudes, id_action_rows, partition_validation, reference_action_bands
from compare_planning import paired_results, paired_statistics
from select_penalty import select
from summarize_uncertainty import summarize
from uncertainty import (
    UncertainLeWM, check_prior, ensemble_disagreement, fit_threshold, knn_distance,
    mahalanobis_distance, prior_deviation,
)
from uncertainty_metrics import episode_metrics, prediction_metrics, warning_metrics


class UncertaintySmokeTests(unittest.TestCase):
    def test_offline_summary_keeps_undefined_scores_and_requires_distinct_checkpoints_for_ci(self):
        reports = []
        for seed, error in enumerate([1., 2., 3.]):
            condition = {"count": 10, "latent_error": error, "physical_error": 0.5, "probe_error": 0.2}
            for estimator in ("knn", "prior", "ensemble"):
                condition[estimator] = {"shift_auroc": None, "physical_error_spearman": 0.5}
            reports.append({
                "task": "pusht", "dataset": "pusht_expert_train", "prior_mode": "chi_square",
                "physical_state_column": "state", "probe_rmse_units": "standardized",
                "fit_protocol": {"train_clip_limit": None}, "probe_clip_limit": 12, "test_clip_limit": 10,
                "checkpoint": f"run-{seed}/weights.pt", "fit_seed": seed, "evaluation_seed": 42,
                "clean": condition, "visual_tint": {},
            })
        clean = summarize(reports)["conditions"]["clean"]["metrics"]
        self.assertEqual(clean["latent_error"]["mean"], 2.0)
        self.assertIsNotNone(clean["latent_error"]["checkpoint_run_95_ci"])
        self.assertEqual(clean["knn.shift_auroc"]["defined_runs"], 0)
        self.assertIsNone(clean["knn.shift_auroc"]["mean"])
        for report in reports:
            report["checkpoint"] = "shared/weights.pt"
        clean = summarize(reports)["conditions"]["clean"]["metrics"]
        self.assertIsNone(clean["latent_error"]["checkpoint_run_95_ci"])

    def test_validation_roles_never_share_episodes(self):
        parts = partition_validation([9, 2, 4, 7, 3])
        self.assertEqual(sorted(sum(parts, [])), [2, 3, 4, 7, 9])
        self.assertTrue(all(set(left).isdisjoint(right) for i, left in enumerate(parts) for right in parts[i + 1:]))
        with self.assertRaisesRegex(ValueError, "three validation"):
            partition_validation([1, 2])

    def test_penalty_selection_uses_clean_matched_validation_and_id_tolerance(self):
        from pathlib import Path
        report = {
            "config": {"policy": "released/pusht/weights.pt"}, "dataset_rows": [4, 5],
            "episode_ids": [1, 1], "start_steps": [2, 3],
            "metrics": {"evaluation_split": "validation", "visual_tint": 0, "episode_successes": [True, False]},
        }
        settings = {"method": "prior", "evaluation_split": "validation", "visual_tint": 0}
        paths = [Path("zero/metrics.json"), Path("positive/metrics.json")]
        with patch("select_penalty.read_run", side_effect=[
            (report, dict(settings, beta=0), [], 50), (report, dict(settings, beta=0.1), [], 50),
        ]):
            chosen, _ = select(paths)
        self.assertEqual(chosen, {"method": "prior", "beta": 0.1})
        worse = dict(report, metrics=dict(report["metrics"], episode_successes=[False, False]))
        with patch("select_penalty.read_run", side_effect=[
            (report, dict(settings, beta=0), [], 50), (worse, dict(settings, beta=0.1), [], 50),
        ]):
            chosen, candidates = select(paths)
        self.assertEqual(chosen["beta"], 0)
        self.assertFalse(candidates[0.1]["admissible"])
        mismatched = dict(report, start_steps=[2, 4])
        with patch("select_penalty.read_run", side_effect=[
            (report, dict(settings, beta=0), [], 50), (mismatched, dict(settings, beta=0.1), [], 50),
        ]):
            with self.assertRaisesRegex(ValueError, "identical settings and cases"):
                select(paths)

    def test_paired_planning_rejects_different_start_cases(self):
        report = {"config": {}, "dataset_rows": [1], "episode_ids": [0], "start_steps": [1]}
        other = dict(report, start_steps=[2])
        with patch("compare_planning.read_run", side_effect=[
            (report, {"beta": 0}, [], 50), (other, {"beta": 0.1}, [], 50),
        ]):
            with self.assertRaisesRegex(ValueError, "different start_steps"):
                paired_results(None, None)

    def test_paired_statistics_respects_reused_episode_clusters(self):
        result = paired_statistics([0, 0, 1, 1], [1, 1, 0, 0], [3, 3, 7, 7])
        self.assertEqual(result["independent_episode_clusters"], 2)
        self.assertIsNone(result["exact_mcnemar_pvalue"])
        self.assertEqual(result["episode_cluster_bootstrap_95_ci_percentage_points"], [-100, 100])
        independent = paired_statistics([0] * 6, [1] * 6, list(range(6)))
        self.assertEqual(independent["exact_mcnemar_pvalue"], 0.03125)

    def test_holdout_uses_only_actions_consumed_by_the_predictor(self):
        dataset = SimpleNamespace(clip_indices=[(0, 0), (0, 1), (1, 0)],
                                  offsets=np.array([0, 5]), span=4, frameskip=1)
        raw = np.array([[0], [1], [2], [99], [3], [4], [5], [6], [100]])
        np.testing.assert_array_equal(clip_magnitudes(dataset, raw), [2, 99, 6])
        np.testing.assert_array_equal(id_action_rows(dataset, raw, [0, 2]), [[0], [1], [2], [4], [5], [6]])

    def test_goal_path_action_bands_respect_eligible_rows_and_cutoff(self):
        actions = np.array([[1], [2], [99], [4], [5], [0]])
        valid = np.array([True, False, False, True, True, False])
        np.testing.assert_array_equal(reference_action_bands(actions, valid, 2, [2, 6, 8]),
                                      [0, -1, -1, 1, 1, -1])

    def test_ensemble_detects_disagreement(self):
        predictions = torch.tensor([[[0., 0.], [0., 0.]], [[0., 0.], [2., 2.]]])
        torch.testing.assert_close(ensemble_disagreement(predictions), torch.tensor([0., 1.]))

    def test_knn_scores_distant_point_higher(self):
        bank = torch.tensor([[0.], [2.], [4.]])
        scores = knn_distance(torch.tensor([[1.], [10.]]), bank, k=2)
        torch.testing.assert_close(scores, torch.tensor([1., 7.]))

    def test_new_ensemble_excludes_unequally_trained_base(self):
        class Rollout(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def rollout(self, info, actions):
                info["predicted_emb"] = torch.tensor([[[[0.], [self.value]]]])
                return info

            def get_cost(self, info, actions):
                self.rollout(info, actions)
                return torch.zeros(1, 1)

        scorer = UncertainLeWM(Rollout(100.), "ensemble", predictors=[Rollout(1.), Rollout(3.)],
                               include_base_predictor=False)
        _, scores = scorer.score_candidates({"pixels": torch.zeros(1, 1, 1, 3, 1, 1)}, None)
        torch.testing.assert_close(scores, torch.ones(1, 1, 1))

    def test_prior_scores_both_tails(self):
        scores = prior_deviation(torch.tensor([[0., 0.], [1.18, 0.], [10., 0.]]))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(scores[0] > scores[1] and scores[2] > scores[1])

    def test_saturated_validation_prior_falls_back_to_training_covariance(self):
        training = torch.tensor([[0., 0.], [1., 0.], [0., 2.]])
        validation = torch.full((4, 2), 100.)
        diagnostics, stats = check_prior(training, validation, validation)
        self.assertEqual(diagnostics["mode"], "mahalanobis")
        self.assertEqual(diagnostics["predicted"]["saturation_fraction"], 1.0)
        torch.testing.assert_close(stats["center"], training.mean(dim=0))
        center = stats["center"]
        distances = mahalanobis_distance(torch.stack([center, center + 1]), **stats)
        self.assertEqual(float(distances[0]), 0.0)
        self.assertGreater(float(distances[1]), 0.0)

    def test_threshold_uses_validation_scores(self):
        torch.testing.assert_close(fit_threshold(torch.arange(6), 0.8), torch.tensor(4.))

    def test_planning_penalty_can_change_candidate_ranking(self):
        class FixedRollout(torch.nn.Module):
            def get_cost(self, info, actions):
                info["predicted_emb"] = torch.tensor([[[[0.], [10.]], [[0.], [1.]]]])
                return torch.tensor([[0., 1.]])

        info = {"pixels": torch.zeros(1, 2, 1, 3, 1, 1)}
        scorer = UncertainLeWM(FixedRollout(), "knn", training_bank=torch.tensor([[1.]]), k=1)
        torch.testing.assert_close(scorer.get_cost(info, None), torch.tensor([[0., 1.]]))
        scorer.beta = 0.5
        torch.testing.assert_close(scorer.get_cost(info, None), torch.tensor([[4.5, 1.]]))

    def test_metrics_on_ranked_failures(self):
        scores = [0, 1, 2, 3, 4]
        results = episode_metrics(scores, [1, 1, 1, 1, 0], threshold=3)
        self.assertEqual(results["failure_auroc"], 1)
        self.assertEqual(results["success_at_80_coverage"], 1)
        self.assertAlmostEqual(results["aurc"], 0.04)
        self.assertEqual(prediction_metrics(scores, scores, [0, 0, 0, 1, 1])["shift_auroc"], 1)

    def test_warning_metrics_include_missed_failures(self):
        result = warning_metrics(
            [[0, 2, 3, 4], [0, 0, 2], [2, 0], [0, 0]],
            [3, 2, None, None], 1, [False, False, True, True],
        )
        self.assertEqual(result["lead_times"], [2])
        self.assertEqual(result["failure_detection_rate"], 0.5)
        self.assertEqual(result["id_false_alarm_rate"], 0.5)
