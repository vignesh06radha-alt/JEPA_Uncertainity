"""Summarize offline uncertainty metrics across matching experiment runs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from compare_planning import independent_training_runs


def conditions(report):
    values = {"clean": report["clean"]}
    for family, levels in report.get("visual", {"tint": report["visual_tint"]}).items():
        for level, result in levels.items():
            values[f"visual_{family}:{level}"] = result
    for name in ("action_holdout", "action_magnitude_diagnostic"):
        if name in report:
            for band, result in report[name]["higher_magnitude_bins"].items():
                values[f"{name}:{band}"] = result
    return values


def extract_metrics(condition):
    if condition is None:
        return {}
    output = {name: condition[name] for name in ("latent_error", "clean_target_error", "physical_error", "probe_error")
              if name in condition}
    for estimator in ("knn", "prior", "ensemble", "observed_knn", "observed_prior",
                      "predicted_norm", "observed_norm", "predicted_mahalanobis", "observed_mahalanobis",
                      "action_history_knn", "predicted_squared_norm", "observed_squared_norm"):
        for metric, value in condition.get(estimator, {}).items():
            output[f"{estimator}.{metric}"] = value
    return output


def summarize(reports):
    if not reports:
        raise ValueError("Provide at least one offline report.")
    keys = ("task", "dataset", "prior_mode", "physical_state_column", "probe_rmse_units",
            "fit_protocol", "probe_clip_limit", "test_clip_limit", "visual_seed", "test_clip_hash", "test_episode_ids")
    protocol = {key: reports[0].get(key) for key in keys}
    expected = set(conditions(reports[0]))
    for report in reports:
        if {key: report.get(key) for key in keys} != protocol or set(conditions(report)) != expected:
            raise ValueError("Offline reports must use the same task, estimators, shift levels, and protocol.")
    runs = [{key: report[key] for key in ("checkpoint", "fit_seed", "evaluation_seed")} for report in reports]
    if len({tuple(run.values()) for run in runs}) != len(runs):
        raise ValueError("Duplicate checkpoint, fit seed and evaluation seed.")
    eligible = (independent_training_runs([report.get("training_provenance") for report in reports])
                and all(report["fit_protocol"] and report["fit_protocol"]["train_clip_limit"] is None
                        for report in reports))
    summary = {}
    for condition in sorted(expected):
        samples = [conditions(report)[condition] for report in reports]
        scores = [extract_metrics(sample) for sample in samples]
        metrics = {}
        for key in sorted(set().union(*scores)):
            observed = [(score[key], run["checkpoint"]) for score, run in zip(scores, runs)
                        if score.get(key) is not None]
            values = np.asarray([value for value, _ in observed])
            mean = float(values.mean()) if len(values) else None
            ci = None
            if eligible and len(values) >= 3 and len({checkpoint for _, checkpoint in observed}) == len(values):
                margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
                ci = [mean - margin, mean + margin]
            metrics[key] = {"mean": mean, "defined_runs": len(values), "checkpoint_run_95_ci": ci}
        summary[condition] = {
            "sample_counts": [0 if sample is None else sample["count"] for sample in samples],
            "metrics": metrics,
        }
    return {"protocol": protocol, "runs": runs, "conditions": summary}


def episode_cluster_analysis(path, replicates):
    """Use each episode once per draw, retaining paired clean and shifted clips."""
    groups = {}
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            groups.setdefault(row["condition"], []).append(row)
    id_rows = groups.get("clean", [])
    output = {}
    for condition, rows in groups.items():
        control = [] if condition == "clean" else id_rows
        episodes = np.unique([row["episode_id"] for row in rows + control])
        episode_ids = np.asarray([row["episode_id"] for row in rows])
        control_ids = np.asarray([row["episode_id"] for row in control])
        case_indices = [np.flatnonzero(episode_ids == episode) for episode in episodes]
        id_indices = [np.flatnonzero(control_ids == episode) for episode in episodes]
        requested = ("ensemble", "observed_knn", "observed_norm", "observed_mahalanobis",
                     "knn", "action_history_knn")
        methods = [method for method in requested if method in rows[0]["scores"]]
        errors = ("latent_error", "clean_target_error", "physical_error")
        score_values = {method: np.asarray([row["scores"][method] for row in rows]) for method in methods}
        id_scores = {method: np.asarray([row["scores"][method] for row in control]) for method in methods}
        error_values = {key: np.asarray([row["errors"][key] for row in rows]) for key in errors}
        samples = {method: {key: [] for key in (*errors, "shift_auroc")} for method in methods}
        if len(episodes) >= 2 and len(np.unique(episode_ids)) >= 2:
            rng = np.random.default_rng(42)
            for _ in range(replicates):
                draw = rng.integers(0, len(episodes), len(episodes))
                indices = np.concatenate([case_indices[index] for index in draw])
                if not len(indices):
                    continue
                reference = np.concatenate([id_indices[index] for index in draw])
                labels = np.r_[np.zeros(len(reference)), np.ones(len(indices))]
                for method in methods:
                    scores = score_values[method][indices]
                    for key in errors:
                        values = error_values[key][indices]
                        if np.ptp(scores) > 0 and np.ptp(values) > 0:
                            samples[method][key].append(float(spearmanr(scores, values).statistic))
                    if len(reference) and len(np.unique(control_ids)) >= 2:
                        combined = np.r_[id_scores[method][reference], scores]
                        samples[method]["shift_auroc"].append(float(roc_auc_score(labels, combined)))
        output[condition] = {
            "episode_clusters": int(len(np.unique(episode_ids))),
            "id_reference_episode_clusters": int(len(np.unique(control_ids))),
            "metrics": {method: {
                key if key == "shift_auroc" else f"{key}_spearman": {
                    "ci_95": np.quantile(values, [0.025, 0.975]).tolist() if len(values) >= 2 else None,
                    "defined_bootstrap_draws": len(values),
                } for key, values in measures.items()
            } for method, measures in samples.items()},
        }
    return {"replicates": replicates, "seed": 42, "conditions": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-bootstrap", type=int, default=0,
                        help="Episode-cluster bootstrap replicates for each linked clip export")
    args = parser.parse_args()
    if args.episode_bootstrap < 0:
        parser.error("--episode-bootstrap must be nonnegative")
    reports = [json.loads(path.read_text()) for path in args.reports]
    result = summarize(reports)
    if args.episode_bootstrap:
        result["per_run_episode_cluster_bootstrap"] = [
            episode_cluster_analysis(path.parent / report["clip_records"], args.episode_bootstrap)
            for path, report in zip(args.reports, reports)
        ]
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(f"Wrote offline uncertainty summary to {args.output}.")
