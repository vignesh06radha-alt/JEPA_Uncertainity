"""Compare completed unmodified LeWM runs with the published Figure 6."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from run import TASKS


PUBLISHED = {"pusht": 96.0, "reacher": 86.0, "cube": 74.0}
ENVIRONMENTS = {
    "pusht": "swm/PushT-v1", "reacher": "swm/ReacherDMControl-v0", "cube": "swm/OGBCube-v0",
}


def read_baseline(path):
    status = json.loads(path.with_name("status.json").read_text())
    if status.get("exit_code") != 0 or status.get("smoke") is not False:
        raise ValueError(f"Expected a successful full benchmark run: {path}")
    if path.with_name("uncertainty.json").exists():
        raise ValueError(f"Expected an unmodified baseline: {path}")
    result = json.loads(path.read_text())
    cfg = result["config"]
    task = next((task for task, env in ENVIRONMENTS.items() if cfg["world"]["env_name"] == env), None)
    if task is None:
        raise ValueError(f"Unknown benchmark environment: {path}")
    _, dataset, iterations = TASKS[task]
    expected = {
        "eval": {"dataset_name": dataset, "num_eval": 50, "goal_offset_steps": 25,
                 "eval_budget": 50, "img_size": 224},
        "plan_config": {"horizon": 5, "receding_horizon": 5, "action_block": 5},
        "solver": {"num_samples": 300, "topk": 30, "n_steps": iterations, "var_scale": 1},
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            if cfg[section].get(key) != value:
                raise ValueError(f"{path}: {section}.{key} differs from the benchmark protocol.")
    successes = result["metrics"]["episode_successes"]
    if len(successes) != 50 or any(value not in (True, False) for value in successes):
        raise ValueError(f"{path}: expected 50 binary outcomes.")
    rate = 2 * sum(successes)
    if not np.isclose(rate, result["metrics"]["success_rate"]):
        raise ValueError(f"{path}: the reported rate disagrees with the case outcomes.")
    cases = list(zip(result["episode_ids"], result["start_steps"]))
    if len(cases) != 50 or len(set(cases)) != 50:
        raise ValueError(f"{path}: expected 50 distinct episode/start pairs.")
    if cfg["policy"] == "random":
        raise ValueError(f"{path}: the evaluation requires a LeWM checkpoint.")
    return task, cases, {
        "checkpoint": cfg["policy"], "evaluation_seed": cfg["seed"],
        "successes": int(sum(successes)), "cases": 50, "success_rate_percent": rate,
        "delta_from_published_percentage_points": rate - PUBLISHED[task],
    }


def report(paths):
    groups = {task: [] for task in PUBLISHED}
    for path in paths:
        task, cases, result = read_baseline(path)
        groups[task].append((cases, result))
    summary = {}
    for task, entries in groups.items():
        rows = [row for _, row in entries]
        if len({(row["checkpoint"], row["evaluation_seed"]) for row in rows}) != len(rows):
            raise ValueError(f"{task}: repeated checkpoint and evaluation seed.")
        matched = len(entries) > 0 and all(cases == entries[0][0] for cases, _ in entries)
        rates = np.asarray([row["success_rate_percent"] for row in rows])
        ci = None
        if matched and len(rates) >= 3 and len({row["checkpoint"] for row in rows}) == len(rates):
            mean = float(rates.mean())
            margin = float(t.ppf(0.975, len(rates) - 1) * rates.std(ddof=1) / np.sqrt(len(rates)))
            ci = [max(0, mean - margin), min(100, mean + margin)]
        summary[task] = {
            "published_success_rate_percent": PUBLISHED[task], "runs": rows,
            "mean_success_rate_percent": float(rates.mean()) if len(rates) else None,
            "matched_case_set": matched, "checkpoint_run_95_ci_percent": ci,
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = report(args.metrics)
    with args.output.open("x") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
    print(f"Wrote baseline reproduction comparison to {args.output}.")
