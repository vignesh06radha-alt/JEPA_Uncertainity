"""Compare planning success for paired zero-penalty and penalized runs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t

from report_episode_metrics import read_run


def paired_statistics(original, penalized, episode_ids):
    """Resample episode clusters; exact McNemar requires independent case pairs."""
    delta = np.asarray(penalized, dtype=int) - np.asarray(original, dtype=int)
    ids = np.asarray(episode_ids)
    unique = np.unique(ids)
    totals = np.array([delta[ids == episode].sum() for episode in unique])
    sizes = np.array([(ids == episode).sum() for episode in unique])
    interval = None
    if len(unique) >= 2:
        rng = np.random.default_rng(42)
        # Keep memory bounded when evaluating many episodes.
        samples = []
        for _ in range(2000):
            selected = rng.integers(0, len(unique), size=len(unique))
            samples.append(100 * totals[selected].sum() / sizes[selected].sum())
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
    recovered, lost = int((delta == 1).sum()), int((delta == -1).sum())
    pvalue = None
    if len(unique) == len(delta):
        pvalue = float(binomtest(recovered, recovered + lost, 0.5).pvalue) if recovered + lost else 1.0
    return {"episode_cluster_bootstrap_95_ci_percentage_points": interval,
            "exact_mcnemar_pvalue": pvalue, "independent_episode_clusters": len(unique),
            "bootstrap_replicates": 2000, "bootstrap_seed": 42}


def paired_results(baseline_path, penalty_path):
    baseline, baseline_settings, _, _ = read_run(baseline_path)
    penalty, penalty_settings, _, _ = read_run(penalty_path)
    if baseline_settings.get("beta", 0) != 0 or not penalty_settings.get("beta", 0) > 0:
        raise ValueError("The first run must use beta=0 and the second a positive penalty.")
    if {k: v for k, v in baseline_settings.items() if k != "beta"} != {
        k: v for k, v in penalty_settings.items() if k != "beta"
    }:
        raise ValueError("Paired uncertainty settings may differ only in beta.")
    for key in ("config", "dataset_rows", "episode_ids", "start_steps"):
        if baseline[key] != penalty[key]:
            raise ValueError(f"Paired planning runs have different {key}.")
    if baseline["metrics"]["evaluation_split"] != "test" or penalty["metrics"]["evaluation_split"] != "test":
        raise ValueError("Pair only held-out test runs.")
    original = np.asarray(baseline["metrics"]["episode_successes"], dtype=bool)
    penalized = np.asarray(penalty["metrics"]["episode_successes"], dtype=bool)
    if len(original) != len(baseline["dataset_rows"]) or len(original) != len(penalized) or not len(original):
        raise ValueError("Expected one success outcome per paired case.")
    cfg = baseline["config"]
    summary = {
        "checkpoint": cfg["policy"],
        "evaluation_seed": cfg["seed"],
        "cases": len(original),
        "baseline_success_rate_percent": 100 * float(original.mean()),
        "penalty_success_rate_percent": 100 * float(penalized.mean()),
        "delta_percentage_points": 100 * float((penalized.astype(int) - original).mean()),
        "failures_recovered": int((~original & penalized).sum()),
        "successes_lost": int((original & ~penalized).sum()),
    }
    summary.update(paired_statistics(original, penalized, baseline["episode_ids"]))
    protocol = {key: cfg[key] for key in ("world", "eval", "plan_config")}
    protocol.update(method=penalty_settings["method"], beta=penalty_settings["beta"],
                    visual_tint=penalty["metrics"]["visual_tint"],
                    action_band=penalty["metrics"].get("action_band"),
                    prior_mode=penalty_settings.get("prior_mode"))
    protocol["solver"] = {key: value for key, value in cfg["solver"].items() if key != "seed"}
    return summary, protocol


def summarize(pairs):
    summaries = []
    protocol = None
    for baseline, penalty in pairs:
        summary, run_protocol = paired_results(baseline, penalty)
        if protocol is not None and protocol != run_protocol:
            raise ValueError("Aggregate only the same task, shift, and planning protocol.")
        protocol = run_protocol
        summaries.append(summary)
    identities = {(item["checkpoint"], item["evaluation_seed"]) for item in summaries}
    if len(identities) != len(summaries):
        raise ValueError("Duplicate checkpoint and evaluation seed.")
    deltas = np.asarray([item["delta_percentage_points"] for item in summaries])
    mean = float(deltas.mean())
    unique_checkpoints = len({item["checkpoint"] for item in summaries})
    interval = None
    if len(deltas) >= 3 and unique_checkpoints == len(deltas):
        half_width = float(t.ppf(0.975, len(deltas) - 1)) * float(deltas.std(ddof=1)) / np.sqrt(len(deltas))
        interval = [max(-100.0, mean - half_width), min(100.0, mean + half_width)]
    return {
        "protocol": protocol, "runs": summaries,
        "distinct_checkpoints": unique_checkpoints,
        "mean_delta_percentage_points": mean,
        "checkpoint_run_95_ci_percentage_points": interval,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, type=Path, action="append", required=True,
                        metavar=("BASELINE_METRICS", "PENALTY_METRICS"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.pair)
    with args.output.open("x") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
    print(f"Wrote paired planning comparison to {args.output}.")
