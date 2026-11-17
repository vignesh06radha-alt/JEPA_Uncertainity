"""Choose a planning penalty from matched, clean validation episodes."""

import argparse
import json
import math
from pathlib import Path

from report_episode_metrics import read_run


def select(paths):
    matches = None
    candidates = {}
    for path in paths:
        report, settings, _, _ = read_run(path)
        metrics = report["metrics"]
        if (metrics["evaluation_split"] != "validation" or metrics["visual_tint"] != 0
                or metrics.get("action_band", 0) != 0):
            raise ValueError("Penalty selection requires clean ID validation episodes.")
        exposure = metrics.get("action_command_exposure")
        if exposure and any(exposure["commands_above_cutoff"]):
            raise ValueError("Validation issued out-of-range commands.")
        beta = settings.get("beta", 0.0)
        if not math.isfinite(beta) or beta < 0 or beta in candidates:
            raise ValueError("Expected distinct, finite, nonnegative beta values.")
        experiment = (
            {key: value for key, value in settings.items() if key != "beta"},
            report["config"], report["dataset_rows"], report["episode_ids"], report["start_steps"],
        )
        if matches is not None and experiment != matches:
            raise ValueError("Validation runs must have identical settings and cases except for beta.")
        matches = experiment
        outcomes = metrics["episode_successes"]
        if not outcomes or len(outcomes) != len(report["dataset_rows"]):
            raise ValueError("Expected one outcome per validation case.")
        candidates[beta] = {
            "successful_cases": int(sum(outcomes)), "cases": len(outcomes), "file": str(path.resolve()),
        }
    if 0 not in candidates or len(candidates) < 2:
        raise ValueError("Include both beta zero and at least one positive beta.")
    chosen = min(candidates, key=lambda beta: (-candidates[beta]["successful_cases"], beta))
    chosen_settings = dict(matches[0], beta=chosen)
    for key in ("evaluation_split", "visual_tint", "action_band"):
        chosen_settings.pop(key, None)
    return chosen_settings, candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    settings, candidates = select(args.metrics)
    with args.output.open("x") as output:
        json.dump(settings, output, indent=2, allow_nan=False)
    print(json.dumps({"selected_beta": settings["beta"], "validation_runs": candidates}, indent=2))
