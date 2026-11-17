"""Summarize offline uncertainty metrics across matching experiment runs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def conditions(report):
    values = {"clean": report["clean"]}
    for level, result in report["visual_tint"].items():
        values[f"visual_tint:{level}"] = result
    for name in ("action_holdout", "action_magnitude_diagnostic"):
        if name in report:
            for band, result in report[name]["higher_magnitude_bins"].items():
                values[f"{name}:{band}"] = result
    return values


def extract_metrics(condition):
    if condition is None:
        return {}
    output = {name: condition[name] for name in ("latent_error", "physical_error", "probe_error")}
    for estimator in ("knn", "prior", "ensemble"):
        for metric, value in condition[estimator].items():
            output[f"{estimator}.{metric}"] = value
    return output


def summarize(reports):
    if not reports:
        raise ValueError("Provide at least one offline report.")
    keys = ("task", "dataset", "prior_mode", "physical_state_column", "probe_rmse_units",
            "fit_protocol", "probe_clip_limit", "test_clip_limit")
    protocol = {key: reports[0][key] for key in keys}
    expected = set(conditions(reports[0]))
    for report in reports:
        if {key: report[key] for key in keys} != protocol or set(conditions(report)) != expected:
            raise ValueError("Offline reports must use the same task, estimators, shift levels, and protocol.")
    runs = [{key: report[key] for key in ("checkpoint", "fit_seed", "evaluation_seed")} for report in reports]
    if len({tuple(run.values()) for run in runs}) != len(runs):
        raise ValueError("Duplicate checkpoint, fit seed and evaluation seed.")
    eligible = all(report["fit_protocol"] and report["fit_protocol"]["train_clip_limit"] is None
                   for report in reports)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(path.read_text()) for path in args.reports])
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(f"Wrote offline uncertainty summary to {args.output}.")
