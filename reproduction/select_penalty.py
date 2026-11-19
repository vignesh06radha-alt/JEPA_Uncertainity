"""Choose a planning penalty from matched, clean validation episodes."""

import argparse
import json
import math
from pathlib import Path

from report_episode_metrics import read_run


def select(paths, max_id_drop=0.02):
    if not math.isfinite(max_id_drop) or not 0 <= max_id_drop <= 1:
        raise ValueError("The tolerated ID success drop must lie in [0, 1].")
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
    baseline = candidates[0]["successful_cases"] / candidates[0]["cases"]
    for candidate in candidates.values():
        candidate["success_rate"] = candidate["successful_cases"] / candidate["cases"]
        candidate["admissible"] = candidate["success_rate"] >= baseline - max_id_drop - 1e-12
    chosen = max(beta for beta, candidate in candidates.items() if candidate["admissible"])
    chosen_settings = dict(matches[0], beta=chosen)
    for key in ("evaluation_split", "visual_tint", "action_band"):
        chosen_settings.pop(key, None)
    return chosen_settings, candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-id-drop", type=float, default=0.02,
                        help="Maximum empirical ID success drop as a fraction (0.02 = two percentage points)")
    args = parser.parse_args()
    settings, candidates = select(args.metrics, args.max_id_drop)
    with args.output.open("x") as output:
        json.dump(settings, output, indent=2, allow_nan=False)
    record = {"selected_beta": settings["beta"], "max_id_drop": args.max_id_drop,
              "validation_runs": candidates}
    with args.output.with_suffix(".selection.json").open("x") as output:
        json.dump(record, output, indent=2, allow_nan=False)
    print(json.dumps(record, indent=2))
