"""Compare planning success for paired zero-penalty and penalized runs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t

from report_episode_metrics import complete_outcomes, read_run


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
    baseline, baseline_settings, baseline_traces, _ = read_run(baseline_path)
    penalty, penalty_settings, penalty_traces, _ = read_run(penalty_path)
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
    original = np.asarray(complete_outcomes(baseline["metrics"]), dtype=bool)
    penalized = np.asarray(complete_outcomes(penalty["metrics"]), dtype=bool)
    if len(original) != len(baseline["dataset_rows"]) or len(original) != len(penalized) or not len(original):
        raise ValueError("Expected one success outcome per paired case.")
    cfg = baseline["config"]
    deltas = penalized.astype(int) - original
    summary = {
        "checkpoint": cfg["policy"],
        "evaluation_seed": cfg["seed"],
        "treatment_beta": penalty_settings["beta"],
        "training_provenance": baseline_settings.get("training_provenance"),
        "case_pairs": [
            {"dataset_row": row, "episode_id": episode, "start_step": start, "success_delta": int(delta)}
            for row, episode, start, delta in zip(baseline["dataset_rows"], baseline["episode_ids"],
                                                 baseline["start_steps"], deltas)
        ],
        "cases": len(original),
        "baseline_success_rate_percent": 100 * float(original.mean()),
        "penalty_success_rate_percent": 100 * float(penalized.mean()),
        "delta_percentage_points": 100 * float(deltas.mean()),
        "failures_recovered": int((~original & penalized).sum()),
        "successes_lost": int((original & ~penalized).sum()),
    }
    for arm, run, traces in (("baseline", baseline, baseline_traces), ("penalty", penalty, penalty_traces)):
        times = [item["planning_seconds"] for trace in traces for item in trace if "planning_seconds" in item]
        summary[f"{arm}_timing"] = {
            "evaluation_seconds": run.get("seconds"),
            "mean_planning_seconds_per_case": float(np.mean(times)) if times else None,
            "profiling": run["metrics"].get("profiling"),
        }
    summary.update(paired_statistics(original, penalized, baseline["episode_ids"]))
    protocol = {key: cfg[key] for key in ("world", "eval", "plan_config")}
    protocol.update(method=penalty_settings["method"],
                    fit_protocol=penalty_settings.get("fit_protocol"),
                    latent_k=penalty_settings.get("k"),
                    visual_tint=penalty["metrics"]["visual_tint"],
                    visual_shift=penalty["metrics"].get("visual_shift"),
                    action_band=penalty["metrics"].get("action_band"),
                    prior_mode=penalty_settings.get("prior_mode"),
                    fitted_test_episode_ids=penalty_settings.get("test_episode_ids"),
                    action_severity_edges=penalty_settings.get("action_severity_edges"))
    if penalty_settings["method"] == "ensemble":
        protocol["predictors"] = len(penalty_settings.get("predictor_checkpoints", [])) + int(
            penalty_settings.get("include_base_predictor", True)
        )
    if "penalty_selection_rule" in penalty_settings:
        protocol["penalty_selection_rule"] = penalty_settings["penalty_selection_rule"]
    else:
        protocol["beta"] = penalty_settings["beta"]
    protocol["solver"] = {key: value for key, value in cfg["solver"].items() if key != "seed"}
    return summary, protocol


def independent_training_runs(provenance):
    keys = ("training_seed", "checkpoint_sha256", "split_seed", "sigreg_weight", "holdout_manifest_sha256")
    if len(provenance) < 3 or any(
        not item or item.get("smoke") is not False or any(item.get(key) is None for key in keys)
        for item in provenance
    ):
        return False
    count = len(provenance)
    return (len({item["training_seed"] for item in provenance}) == count
            and len({item["checkpoint_sha256"] for item in provenance}) == count
            and len({item["holdout_manifest_sha256"] for item in provenance}) == count
            and len({(item["split_seed"], item["sigreg_weight"]) for item in provenance}) == 1)


def hierarchical_interval(summaries):
    case_keys = ("dataset_row", "episode_id", "start_step")
    cases = [[tuple(case[key] for key in case_keys) for case in run["case_pairs"]] for run in summaries]
    if not all(selected == cases[0] for selected in cases):
        return None
    episode_ids = np.asarray([case[1] for case in cases[0]])
    episodes = np.unique(episode_ids)
    if len(episodes) < 2:
        return None
    deltas = np.asarray([[pair["success_delta"] for pair in run["case_pairs"]] for run in summaries])
    masks = [episode_ids == episode for episode in episodes]
    totals = np.stack([deltas[:, mask].sum(axis=1) for mask in masks], axis=1)
    sizes = np.asarray([mask.sum() for mask in masks])
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(2000):
        train_draw = rng.integers(0, len(summaries), size=len(summaries))
        episode_draw = rng.integers(0, len(episodes), size=len(episodes))
        estimates.append(100 * totals[np.ix_(train_draw, episode_draw)].mean(axis=0).sum() / sizes[episode_draw].sum())
    return np.quantile(estimates, [0.025, 0.975]).tolist()


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
    interval = hierarchical = None
    verified = independent_training_runs([item["training_provenance"] for item in summaries])
    fit_protocol = protocol.get("fit_protocol") or {}
    if verified and "train_clip_limit" in fit_protocol and fit_protocol["train_clip_limit"] is None:
        half_width = float(t.ppf(0.975, len(deltas) - 1)) * float(deltas.std(ddof=1)) / np.sqrt(len(deltas))
        interval = [max(-100.0, mean - half_width), min(100.0, mean + half_width)]
        if len({item["evaluation_seed"] for item in summaries}) == 1:
            hierarchical = hierarchical_interval(summaries)
    return {
        "protocol": protocol, "runs": summaries,
        "distinct_checkpoints": unique_checkpoints,
        "verified_independent_training_runs": verified,
        "mean_delta_percentage_points": mean,
        "checkpoint_run_95_ci_percentage_points": interval,
        "training_seed_and_shared_episode_bootstrap_95_ci_percentage_points": hierarchical,
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
