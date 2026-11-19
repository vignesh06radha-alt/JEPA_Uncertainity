import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from stable_worldmodel import World

from action_holdout import (clip_magnitudes, episode_membership, id_action_rows, partition_validation,
                            reference_action_bands, restrict_training, sample_cases)
from compare_planning import hierarchical_interval, paired_results, paired_statistics, summarize as summarize_planning
from encoder_control import candidate_goal_cost, scalar_cost_disagreement
from fit_uncertainty import bootstrap_clips
from plan_trace import TracedExecution
from report_episode_metrics import complete_outcomes, outcome_warnings, policy_identity, verify_frozen_threshold
from select_penalty import select
from summarize_uncertainty import summarize
from uncertainty import (
    UncertainLeWM, check_prior, ensemble_disagreement, fit_threshold, knn_distance,
    mahalanobis_distance, prior_deviation,
)
from uncertainty_metrics import episode_metrics, prediction_metrics, warning_metrics
from visual_shift import shift_info


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
                "training_provenance": {
                    "training_seed": seed, "split_seed": 42, "sigreg_weight": 0.09, "smoke": False,
                    "checkpoint_sha256": f"checkpoint-{seed}", "holdout_manifest_sha256": f"split-{seed}",
                },
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

    def test_holdout_roles_and_scaler_are_independent_of_training_seed(self):
        actions = np.tile(np.arange(8, dtype=float), 30)[:, None]
        row_ids = np.repeat(np.arange(30) + 500, 8)
        dataset = SimpleNamespace(
            clip_indices=[(ep, start) for ep in range(30) for start in range(5)],
            span=4, frameskip=1, offsets=np.arange(0, 240, 8),
            get_col_data={"action": actions, "episode_idx": row_ids}.__getitem__,
        )
        with TemporaryDirectory() as directory:
            import json
            path = Path(directory) / "manifest.json"
            first_train, validation, scaler = restrict_training(dataset, 1, 42, 0.8, path, "fixture", "seed1", .09, True)
            first = json.loads(path.read_text())
            second_train, _, _ = restrict_training(dataset, 2, 42, 0.8, path, "fixture", "seed2", .09, True)
            second = json.loads(path.read_text())
        self.assertNotEqual(first["training_seed"], second["training_seed"])
        self.assertEqual(first["roles"], second["roles"])
        self.assertEqual(first_train.indices, second_train.indices)
        self.assertTrue(set(validation.indices) <= set(first_train.indices))
        np.testing.assert_array_equal(scaler.mean_, id_action_rows(dataset, actions, first_train.indices).mean(axis=0))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            episode_membership(dataset, train=[1, 2], test=[2, 3])

    def test_bootstrap_uses_equal_clip_counts_with_unequal_episode_lengths(self):
        groups = {1: [1], 2: list(range(2, 20)), 3: [20, 21]}
        for seed in range(20):
            episodes, clips = bootstrap_clips(groups, list(groups), 21, np.random.default_rng(seed))
            self.assertEqual(len(clips), 21)
            self.assertEqual(len(episodes), 3)
            self.assertTrue(set(clips) <= {clip for ep in episodes for clip in groups[ep]})

    def test_case_sampler_prefers_distinct_episodes(self):
        ids = np.repeat([5, 6, 7], 3)
        rows = np.arange(len(ids))
        chosen = sample_cases(rows, ids, 3, np.random.default_rng(42))
        self.assertEqual(len(set(ids[chosen])), 3)

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

    def test_hierarchical_resampling_shares_episode_draw_between_seeds(self):
        runs = [{"case_pairs": [
            {"episode_id": 9, "dataset_row": 5, "start_step": 2, "success_delta": delta},
            {"episode_id": 8, "dataset_row": 6, "start_step": 1, "success_delta": -delta},
        ]} for delta in (1, 1, 1)]
        self.assertEqual(hierarchical_interval(runs), [-100, 100])
        runs[-1]["case_pairs"][-1]["start_step"] = 5
        self.assertIsNone(hierarchical_interval(runs))

    def test_planning_seed_intervals_require_matching_uncapped_fit_protocols(self):
        fits = [{"members": 5, "latent_k_requested": 10, "train_clip_limit": 8} for _ in range(3)]
        pairs = [((seed, 0.0), (seed, 0.1)) for seed in range(3)]

        def read(path):
            seed, beta = path
            report = {
                "dataset_rows": [2, 4], "episode_ids": [1, 3], "start_steps": [0, 0],
                "config": {"world": {}, "eval": {}, "plan_config": {}, "solver": {},
                           "policy": f"checkpoint-{seed}", "seed": 42},
                "metrics": {"evaluation_split": "test", "visual_tint": 0,
                            "episode_successes": [True, bool(beta)]},
            }
            settings = {
                "beta": beta, "method": "ensemble", "fit_protocol": dict(fits[seed]),
                "training_provenance": {
                    "training_seed": seed, "split_seed": 42, "sigreg_weight": 0.09, "smoke": False,
                    "checkpoint_sha256": f"checkpoint-{seed}", "holdout_manifest_sha256": f"manifest-{seed}",
                },
            }
            return report, settings, [[], []], 50

        with patch("compare_planning.read_run", side_effect=read):
            capped = summarize_planning(pairs)
            self.assertTrue(capped["verified_independent_training_runs"])
            self.assertIsNone(capped["checkpoint_run_95_ci_percentage_points"])
            self.assertIsNone(capped["training_seed_and_shared_episode_bootstrap_95_ci_percentage_points"])
            for fit in fits:
                fit["train_clip_limit"] = None
            full = summarize_planning(pairs)
            self.assertEqual(full["checkpoint_run_95_ci_percentage_points"], [50, 50])
            self.assertIsNotNone(full["training_seed_and_shared_episode_bootstrap_95_ci_percentage_points"])
            fits[2]["latent_k_requested"] = 5
            with self.assertRaisesRegex(ValueError, "same task, shift, and planning protocol"):
                summarize_planning(pairs)

    def test_initial_gate_steps_only_accepted_slots_through_upstream_wait_loop(self):
        class Pool:
            num_envs = 3

            def __init__(self):
                self.calls = []
                self.counts = np.zeros(3, dtype=int)

            def step(self, actions, mask):
                self.calls.append(mask.tolist())
                self.counts += mask
                terminated = mask & (self.counts >= np.array([1, 2, 3]))
                truncated = np.zeros(3, dtype=bool)
                return None, np.zeros(3), terminated, truncated, {"state": self.counts[:, None, None]}

        for warning_scores, expected_actions in (([0.1, 0.9, 0.2], [1, 0, 3]),
                                                 ([0.9, 0.6, 0.7], [0, 0, 0]),
                                                 ([0.1, 0.2, 0.3], [1, 2, 3])):
            pool = Pool()
            traces = {index: [{"step": 0, "scores": [0.1], "warning_score": score}]
                      for index, score in enumerate(warning_scores)}
            execution = TracedExecution(pool, traces, threshold=0.5)
            world = World.__new__(World)
            world.envs = execution
            world.policy = SimpleNamespace(get_action=lambda _: np.ones((3, 1)))
            world.infos = {}
            list(world._run_iter(max_steps=5, mode="wait"))
            self.assertEqual(execution.counts.tolist(), expected_actions)
            self.assertEqual(pool.counts.tolist(), expected_actions)
            self.assertEqual([event and event["step"] for event in execution.events],
                             [count or None for count in expected_actions])
            self.assertEqual(execution.action_exposure(0.5)["commands_above_cutoff"], expected_actions)
            self.assertEqual(len(pool.calls), max(expected_actions) or 1)

    def test_frozen_threshold_matches_cost_policy_even_when_visual_mode_changes(self):
        settings = {"method": "ensemble", "beta": 0.05, "observation_scale": 0.7, "threshold_episode_ids": [3],
                    "penalty_selection_complete": True, "action_band": 0, "action_severity_edges": [0.5, 0.7, 0.9],
                    "artifact_sha256": {"training_bank": "example-checksum"}}
        config = {"policy": "base.pt", "plan_config": {"horizon": 5}, "solver": {"seed": 42},
                  "world": {"num_envs": 5}, "eval": {"num_eval": 5, "eval_budget": 50}}
        gate = {"version": 1, "threshold": 0.7, "target_coverage": 0.8, "calibration_episode_ids": [3],
                "score": "hybrid", "policy": policy_identity(settings, config)}
        verify_frozen_threshold(dict(settings, gate=gate, visual_shift={"mode": "goal"}), config)
        with self.assertRaisesRegex(ValueError, "frozen threshold"):
            verify_frozen_threshold(dict(settings, gate=gate, beta=0), config)
        with self.assertRaisesRegex(ValueError, "frozen threshold"):
            verify_frozen_threshold(dict(settings, gate=gate), dict(config, solver={"seed": 43}))
        with self.assertRaisesRegex(ValueError, "frozen threshold"):
            verify_frozen_threshold(dict(settings, gate=gate, action_band=1), config)
        with self.assertRaisesRegex(ValueError, "frozen threshold"):
            verify_frozen_threshold(dict(settings, gate=gate, artifact_sha256={"training_bank": "other"}), config)
        with self.assertRaisesRegex(ValueError, "ungated"):
            complete_outcomes({"episode_successes": [None]})
        with self.assertRaisesRegex(ValueError, "every episode"):
            episode_metrics([1], [None], threshold=0.5)

    def test_timeout_lead_time_uses_event_boundary_and_excludes_rejections_and_truncation(self):
        trace = [{"step": 0, "warning_score": 0.1}, {"step": 4, "warning_score": 0.9},
                 {"step": 7, "warning_score": 0.95}]
        report = outcome_warnings([trace] * 3, [
            {"step": 7, "reason": "budget_exhausted"}, None,
            {"step": 6, "reason": "environment_truncation"},
        ], [False, None, False], 0.5, False, "budget_exhausted")
        self.assertEqual(report["lead_times"], [3])
        self.assertEqual(report["event_count"], 1)

    def test_visual_modes_transform_only_the_selected_input_deterministically(self):
        pixels = torch.ones(1, 3, 12, 12)
        frames = {"pixels": pixels, "goal": pixels}
        observed = shift_info(frames, "occlusion", 0.25, "observation")
        self.assertIs(observed["goal"], pixels)
        self.assertEqual(float(observed["pixels"].sum()), 3 * (144 - 36))
        goal = shift_info(frames, "distractor", 0.25, "goal")
        self.assertIs(goal["pixels"], pixels)
        torch.testing.assert_close(goal["goal"], shift_info(frames, "distractor", 0.25, "both")["pixels"])
        self.assertFalse(torch.equal(goal["goal"], shift_info(frames, "distractor", 0.25, "goal", seed=99)["goal"]))

    def test_independent_encoder_control_uses_normalized_scalar_costs(self):
        costs = torch.tensor([[[2., 4.]], [[20., 40.]], [[12., 36.]]])
        disagreement = scalar_cost_disagreement(costs, torch.tensor([2., 20., 12.]))
        self.assertEqual(float(disagreement[0, 0]), 0)
        self.assertGreater(float(disagreement[0, 1]), 0)

    def test_control_goal_embedding_broadcasts_over_cases_and_candidates(self):
        def criterion(info):
            forecast = info["predicted_emb"]
            goal = info["goal_emb"].expand_as(forecast)
            return (forecast - goal).square().sum(dim=(2, 3))

        model = SimpleNamespace(
            rollout=lambda info, actions: {"predicted_emb": actions[:, :, -1:]},
            encode=lambda info: {"emb": info["pixels"].reshape(2, 1, -1)},
            criterion=criterion,
        )
        goals = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1, 1, 1)
        actions = torch.zeros(2, 3, 3, 1)
        costs = candidate_goal_cost(model, goals, goals, actions)
        torch.testing.assert_close(costs, torch.tensor([[9., 9., 9.], [49., 49., 49.]]))

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
