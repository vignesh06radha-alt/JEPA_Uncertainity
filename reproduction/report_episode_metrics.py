"""Calibrate on clean validation traces and score held-out planning episodes."""

import argparse
import json
from pathlib import Path

import numpy as np

from uncertainty_metrics import episode_metrics, warning_metrics


def read_run(path):
    report = json.loads(path.read_text())
    settings = json.loads(path.with_name("uncertainty.json").read_text())
    metrics = report["metrics"]
    traces = metrics["selected_plan_traces"]
    budget = report["config"]["eval"]["eval_budget"]
    if len(traces) != len(metrics["episode_successes"]) or any(not trace or trace[0]["step"] != 0 for trace in traces):
        raise ValueError("Expected a first selected plan and one outcome per episode.")
    return report, settings, traces, budget


def evaluate(calibration_path, test_path, target_coverage=0.8):
    validation, validation_settings, val_traces, _ = read_run(calibration_path)
    test, test_settings, traces, budget = read_run(test_path)
    if validation["metrics"]["evaluation_split"] != "threshold" or validation["metrics"]["visual_tint"] != 0:
        raise ValueError("Threshold calibration needs the separate clean threshold partition; refit old artifacts.")
    if validation["metrics"].get("action_band", 0) != 0:
        raise ValueError("Threshold calibration needs in-range action cases.")
    exposure = validation["metrics"].get("action_command_exposure")
    if exposure and any(exposure["commands_above_cutoff"]):
        raise ValueError("Calibration issued commands outside the training action range.")
    if test["metrics"]["evaluation_split"] != "test":
        raise ValueError("The evaluation file must contain test episodes.")
    if validation["config"]["eval"]["dataset_name"] != test["config"]["eval"]["dataset_name"]:
        raise ValueError("Calibration and test must be drawn from the same dataset.")
    if set(validation["episode_ids"]) & set(test["episode_ids"]):
        raise ValueError("Calibration and test episode IDs overlap.")
    provenance = {"evaluation_split", "visual_tint", "action_band"}
    if {k: v for k, v in validation_settings.items() if k not in provenance} != {
        k: v for k, v in test_settings.items() if k not in provenance
    } or validation["config"]["policy"] != test["config"]["policy"]:
        raise ValueError("Calibration and test must use the same model and uncertainty settings.")
    threshold = float(np.quantile([max(trace[0]["scores"]) for trace in val_traces], target_coverage))
    first_plan_scores = [max(trace[0]["scores"]) for trace in traces]
    outcomes = test["metrics"]["episode_successes"]
    warnings = []
    for trace in traces:
        # Only count warnings at recorded replans, measured in simulator steps.
        dense = np.full(budget + 1, threshold)
        for record in trace:
            dense[record["step"]] = max(record["scores"])
        warnings.append(dense)
    is_id = test["metrics"]["visual_tint"] == 0 and test["metrics"].get("action_band", 0) == 0
    return {
        "threshold": threshold,
        "target_calibration_coverage": target_coverage,
        "validation_episodes": len(val_traces),
        "test_episodes": len(traces),
        "visual_tint": test["metrics"]["visual_tint"],
        "action_band": test["metrics"].get("action_band"),
        "action_command_exposure": test["metrics"].get("action_command_exposure"),
        "abstention": episode_metrics(first_plan_scores, outcomes, threshold),
        "warnings": warning_metrics(warnings, [None if ok else budget for ok in outcomes], threshold,
                                    [is_id] * len(traces)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be in (0, 1]")
    summary = evaluate(args.calibration, args.test, args.coverage)
    with args.output.open("x") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
    print(f"Wrote episode metrics to {args.output}.")
