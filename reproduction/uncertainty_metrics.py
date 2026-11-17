"""Offline metrics for uncertainty, shift detection, and selective execution."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _auroc(labels, scores):
    if np.unique(labels).size != 2:
        return None
    return float(roc_auc_score(labels, scores))


def prediction_metrics(scores, errors, shifted):
    """Scores and errors are aligned per prediction, across ID and shift."""
    scores, errors, shifted = map(np.asarray, (scores, errors, shifted))
    if scores.ndim != 1 or scores.shape != errors.shape or scores.shape != shifted.shape:
        raise ValueError("Provide one score, error and shift label per prediction.")
    if not np.isfinite(scores).all() or not np.isfinite(errors).all():
        raise ValueError("Scores and errors must be finite.")
    correlation = None
    if np.unique(scores).size > 1 and np.unique(errors).size > 1:
        correlation = float(spearmanr(scores, errors).statistic)
    return {
        "spearman": correlation,
        "shift_auroc": _auroc(shifted, scores),
    }


def episode_metrics(scores, success, threshold):
    """Compare episode risk at the fixed ID cutoff and across the score ranking."""
    scores, success = np.asarray(scores), np.asarray(success, dtype=bool)
    if scores.ndim != 1 or scores.shape != success.shape or not scores.size:
        raise ValueError("Provide one score and outcome per episode.")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite.")
    accepted = scores <= threshold
    order = np.argsort(scores, kind="stable")
    # Accept all episodes with the same score together.
    ends = np.r_[np.flatnonzero(np.diff(scores[order])) + 1, len(scores)]
    coverage = ends / len(scores)
    risk = np.cumsum(~success[order])[ends - 1] / ends
    point = int(np.searchsorted(coverage, 0.8))
    return {
        "planning_success_rate": float(success.mean()),
        "coverage": float(accepted.mean()),
        "accepted_success": float(success[accepted].mean()) if accepted.any() else None,
        "overall_completion": float((accepted & success).mean()),
        "failure_auroc": _auroc(~success, scores),
        "risk_coverage": {"coverage": coverage.tolist(), "risk": risk.tolist()},
        "aurc": float(np.dot(np.diff(np.r_[0, coverage]), risk)),
        "coverage_at_80": float(coverage[point]),
        "success_at_80_coverage": float(1 - risk[point]),
    }


def warning_metrics(score_sequences, failure_steps, threshold, in_distribution):
    """Measure advance warnings and false alarms on successful ID episodes."""
    if len(score_sequences) != len(failure_steps) or len(failure_steps) != len(in_distribution):
        raise ValueError("Provide one score sequence, failure step, and ID flag per episode.")
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    lead_times = []
    failures = false_alarms = id_successes = 0
    for scores, failure_step, is_id in zip(score_sequences, failure_steps, in_distribution):
        scores = np.asarray(scores)
        if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
            raise ValueError("Each episode needs a nonempty, finite sequence of step scores.")
        warnings = np.flatnonzero(scores > threshold)
        if failure_step is None:
            if is_id:
                id_successes += 1
                false_alarms += bool(warnings.size)
        else:
            if not 0 <= failure_step < len(scores):
                raise ValueError("Failure step is outside the recorded episode.")
            failures += 1
            if warnings.size and warnings[0] < failure_step:
                lead_times.append(int(failure_step - warnings[0]))
    return {
        "lead_times": lead_times,
        "mean_lead_time": float(np.mean(lead_times)) if lead_times else None,
        "failure_detection_rate": len(lead_times) / failures if failures else None,
        "id_false_alarm_rate": false_alarms / id_successes if id_successes else None,
    }
