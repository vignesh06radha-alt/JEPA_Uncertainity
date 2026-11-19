"""Calibrate on clean validation traces and score held-out planning episodes."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from uncertainty_metrics import episode_metrics, trace_score, warning_metrics


def shifted(metrics):
    return bool(metrics.get("visual_tint") or metrics.get("visual_shift"))


def policy_identity(settings, config):
    excluded = {"evaluation_split", "visual_tint", "visual_shift", "action_band", "gate"}
    return {
        "settings": {key: value for key, value in settings.items() if key not in excluded},
        "raw_action_cutoff": (settings["action_severity_edges"][0]
                              if settings.get("action_band") == 0 else None),
        "policy": config["policy"], "plan_config": config["plan_config"], "solver": config["solver"],
        "world": {key: value for key, value in config["world"].items() if key != "num_envs"},
        "eval": {key: value for key, value in config["eval"].items() if key != "num_eval"},
    }


def verify_frozen_threshold(settings, config):
    gate = settings["gate"]
    ids = set(gate["calibration_episode_ids"])
    if (gate.get("version") != 1 or not math.isfinite(gate["threshold"])
            or not 0 < gate["target_coverage"] <= 1
            or gate.get("score") != "hybrid" or not settings.get("observation_scale") or not settings.get("artifact_sha256")
            or not settings.get("penalty_selection_complete")
            or not ids or not ids <= set(settings.get("threshold_episode_ids", []))
            or ids & set(settings.get("test_episode_ids", []))
            or gate["policy"] != policy_identity(settings, config)):
        raise ValueError("The frozen threshold must match the selected uncertainty policy and planning configuration.")


def complete_outcomes(metrics):
    outcomes = metrics["episode_successes"]
    if metrics.get("gate") is not None or any(ok is None for ok in outcomes):
        raise ValueError("This analysis requires complete ungated outcomes.")
    return outcomes


def read_run(path):
    report = json.loads(path.read_text())
    settings = json.loads(path.with_name("uncertainty.json").read_text())
    metrics = report["metrics"]
    traces = metrics["selected_plan_traces"]
    budget = report["config"]["eval"]["eval_budget"]
    if len(traces) != len(metrics["episode_successes"]) or any(not trace or trace[0]["step"] != 0 for trace in traces):
        raise ValueError("Expected a first selected plan and one outcome per episode.")
    return report, settings, traces, budget


def calibrate(path, target_coverage):
    validation, settings, traces, _ = read_run(path)
    metrics = validation["metrics"]
    complete_outcomes(metrics)
    if not 0 < target_coverage <= 1:
        raise ValueError("Calibration coverage must be in (0, 1].")
    if metrics["evaluation_split"] != "threshold" or shifted(metrics):
        raise ValueError("Threshold calibration needs the separate clean threshold partition; refit old artifacts.")
    if metrics.get("action_band", 0) != 0:
        raise ValueError("Threshold calibration needs in-range action cases.")
    exposure = metrics.get("action_command_exposure")
    if exposure and any(exposure["commands_above_cutoff"]):
        raise ValueError("Calibration issued commands outside the training action range.")
    scores = [trace_score(trace[0]) for trace in traces]
    if not scores or not np.isfinite(scores).all():
        raise ValueError("Threshold calibration needs finite initial-plan scores.")
    ids = sorted(set(validation["episode_ids"]))
    if set(ids) & set(settings.get("test_episode_ids", [])) or not set(ids) <= set(settings["threshold_episode_ids"]):
        raise ValueError("Threshold cases must belong to the held-out calibration partition.")
    return {
        "version": 1, "threshold": float(np.quantile(scores, target_coverage)),
        "target_coverage": target_coverage, "calibration_episode_ids": ids, "calibration_cases": len(traces),
        "policy": policy_identity(settings, validation["config"]),
        "score": "hybrid" if settings.get("observation_scale") else settings["method"],
    }


def outcome_warnings(traces, events, outcomes, threshold, is_id, reason):
    scores, steps, successes = [], [], []
    for trace, event, ok in zip(traces, events, outcomes):
        if ok is None or event is None or (not ok and event["reason"] != reason):
            continue
        dense = np.full(event["step"] + 1, threshold)
        for record in trace:
            if record["step"] < event["step"]:
                dense[record["step"]] = trace_score(record)
        scores.append(dense)
        steps.append(None if ok else event["step"])
        successes.append(is_id)
    summary = warning_metrics(scores, steps, threshold, successes)
    count = sum(step is not None for step in steps)
    summary.update(event_count=count, misses=count - len(summary["lead_times"]))
    return summary


def evaluate(calibration_path, test_path, target_coverage=0.8):
    calibration = calibrate(calibration_path, target_coverage)
    test, test_settings, traces, _ = read_run(test_path)
    if test["metrics"]["evaluation_split"] != "test":
        raise ValueError("The evaluation file must contain test episodes.")
    if set(calibration["calibration_episode_ids"]) & set(test["episode_ids"]):
        raise ValueError("Calibration and test episode IDs overlap.")
    if calibration["policy"] != policy_identity(test_settings, test["config"]):
        raise ValueError("Calibration and test must use the same model and uncertainty settings.")
    threshold = calibration["threshold"]
    metrics = test["metrics"]
    outcomes = metrics["episode_successes"]
    first_plan_scores = [trace_score(trace[0]) for trace in traces]
    gate = metrics.get("gate")
    if gate:
        if test_settings.get("gate") != calibration or gate["accepted"] != [ok is not None for ok in outcomes]:
            raise ValueError("Recorded execution gate does not match the frozen threshold and outcomes.")
        if any(count for count, accepted in zip(metrics["executed_actions"], gate["accepted"]) if not accepted):
            raise ValueError("A rejected case executed a task action.")
    is_id = not shifted(metrics) and metrics.get("action_band", 0) == 0
    events = metrics.get("episode_events")
    report = {
        "threshold": threshold,
        "target_calibration_coverage": target_coverage,
        "validation_cases": calibration["calibration_cases"],
        "test_episodes": len(traces),
        "visual_shift": metrics.get("visual_shift"),
        "visual_tint": metrics["visual_tint"],
        "action_band": metrics.get("action_band"),
        "action_command_exposure": metrics.get("action_command_exposure"),
        "frozen_threshold_score": calibration["score"],
        "actual_gate": gate,
        "retrospective_abstention": None if gate else episode_metrics(first_plan_scores, outcomes, threshold),
    }
    report["warnings"] = None if events is None else {
        "budget_exhaustion": outcome_warnings(traces, events, outcomes, threshold, is_id, "budget_exhausted"),
        "environment_truncations": sum(event is not None and event["reason"] == "environment_truncation"
                                       for event in events),
        "validated_physical_failure": None,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate-output", type=Path, help="Write the frozen threshold for a selected hybrid policy")
    parser.add_argument("--coverage", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be in (0, 1]")
    if bool(args.test) != bool(args.output) or not (args.output or args.gate_output):
        parser.error("Provide --test and --output for an episode report or --gate-output to freeze the threshold.")
    if args.gate_output:
        gate = calibrate(args.calibration, args.coverage)
        settings = gate["policy"]["settings"]
        if gate["score"] != "hybrid" or not settings.get("penalty_selection_complete") or not settings.get("artifact_sha256"):
            parser.error("Execution gating requires a hashed hybrid fit and separate penalty selection.")
        with args.gate_output.open("x") as output:
            json.dump(gate, output, indent=2, allow_nan=False)
        print(f"Wrote frozen execution threshold to {args.gate_output}.")
    if args.output:
        summary = evaluate(args.calibration, args.test, args.coverage)
        with args.output.open("x") as output:
            json.dump(summary, output, indent=2, allow_nan=False)
        print(f"Wrote episode metrics to {args.output}.")
