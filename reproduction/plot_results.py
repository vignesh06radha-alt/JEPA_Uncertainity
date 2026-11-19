"""Generate uncertainty diagnostics from offline clip and ungated planning records."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report_episode_metrics import complete_outcomes, read_run
from uncertainty_metrics import trace_score


def plot(offline, output, planning=None, episode_report=None):
    report = json.loads(offline.read_text())
    with (offline.parent / report["clip_records"]).open() as stream:
        records = [json.loads(line) for line in stream]
    clean = [row for row in records if row["condition"] == "clean"]
    shifted = [row for row in records if row["condition"] != "clean"]
    if not clean:
        raise ValueError("Plotting needs held-out ID clips.")
    output.mkdir(exist_ok=False)
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for group, label, color in ((clean, "Clean ID", "#1d5b94"), (shifted, "Shifted", "#ad5427")):
        if not group:
            continue
        for axis, method, title in (
            (axes[0, 0], "ensemble", "Forecast disagreement"),
            (axes[0, 1], "observed_knn", "Current observation kNN"),
        ):
            axis.scatter([row["scores"][method] for row in group],
                         [row["errors"]["clean_target_error"] for row in group],
                         s=7, alpha=0.3, color=color, label=label)
            axis.set(xlabel=title, ylabel="Clean-target latent error")
            axis.legend()
        for axis, method, title in (
            (axes[1, 0], "observed_squared_norm", "Current observation squared norm"),
            (axes[1, 1], "predicted_squared_norm", "Forecast squared norm"),
        ):
            axis.hist([row["scores"][method] for row in group], bins=25, density=True,
                      histtype="step", linewidth=1.5, color=color, label=label)
            axis.set(xlabel=title, ylabel="Density")
            axis.legend()
    figure.savefig(output / "offline-diagnostics.png", dpi=160)
    plt.close(figure)

    examples = {}
    for condition in sorted({row["condition"] for row in records}):
        rows = [row for row in records if row["condition"] == condition]
        cutoff = np.median([row["scores"]["ensemble"] for row in rows])
        low_score = [row for row in rows if row["scores"]["ensemble"] <= cutoff]
        examples[condition] = max(low_score, key=lambda row: row["errors"]["clean_target_error"])
    result = {"high_error_with_below_median_disagreement": examples}
    if planning:
        run, _, traces, _ = read_run(planning)
        successes = complete_outcomes(run["metrics"])
        candidates = [(trace_score(trace[0]), index) for index, trace in enumerate(traces) if successes[index]]
        if candidates:
            score, index = max(candidates)
            result["highest_warning_success"] = {
                "score": score, "episode_id": run["episode_ids"][index], "start_step": run["start_steps"][index],
                "case_index": index, "trace": traces[index],
            }
    if episode_report:
        metrics = json.loads(episode_report.read_text())["retrospective_abstention"]
        if metrics is None:
            raise ValueError("Risk–coverage plotting requires an ungated episode report.")
        curve = metrics["risk_coverage"]
        figure, axis = plt.subplots(figsize=(5, 4), layout="constrained")
        axis.step([0, *curve["coverage"]], [curve["risk"][0], *curve["risk"]], where="pre")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Coverage", ylabel="Failure rate among accepted cases")
        figure.savefig(output / "risk-coverage.png", dpi=160)
        plt.close(figure)
    (output / "case-examples.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Wrote diagnostic figures and case examples to {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--planning", type=Path)
    parser.add_argument("--episode-report", type=Path)
    args = parser.parse_args()
    plot(args.offline, args.output, args.planning, args.episode_report)
