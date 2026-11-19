"""Episode-disjoint action support restriction for end-to-end LeWM training."""

import json
import hashlib
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset


def clip_magnitudes(dataset, raw_actions):
    """Maximum raw action component in each model input window."""
    coordinates = np.asarray(dataset.clip_indices, dtype=np.int64).reshape(-1, 2)
    if not len(coordinates):
        raise ValueError("Dataset has no clips with a complete action history.")
    rows = dataset.offsets[coordinates[:, 0]] + coordinates[:, 1]
    # The final block belongs to the prediction target and is unused by LeWM.
    return window_magnitudes(raw_actions, rows, dataset.span - dataset.frameskip)


def window_magnitudes(raw_actions, rows, span):
    if not len(rows):
        return np.empty(0)
    action_magnitude = np.nan_to_num(np.abs(raw_actions), nan=0.0).max(axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(action_magnitude, span)
    return np.concatenate(
        [windows[rows[start:start + 65536]].max(axis=1) for start in range(0, len(rows), 65536)]
    )


def reference_action_bands(actions, valid, goal_offset, edges):
    """Classify candidate goal paths by the recorded actions between start and goal."""
    rows = np.flatnonzero(valid)
    magnitudes = window_magnitudes(actions, rows, goal_offset)
    bands = np.full(len(valid), -1)
    bands[rows] = np.searchsorted(edges, magnitudes, side="left")
    return bands


def sample_cases(eligible_rows, row_episode_ids, count, rng):
    """Prefer one start-goal case per episode, then fill from remaining starts."""
    if len(eligible_rows) < count:
        raise ValueError(f"Only {len(eligible_rows)} eligible cases for {count} requested.")
    shuffled = rng.permutation(eligible_rows)
    first_positions = np.sort(np.unique(row_episode_ids[shuffled], return_index=True)[1])
    selected = shuffled[first_positions[:count]]
    if len(selected) < count:
        remaining = shuffled[~np.isin(shuffled, selected)]
        selected = np.r_[selected, remaining[:count - len(selected)]]
    return np.sort(selected)


def split_episodes(dataset, rng):
    episodes = rng.permutation(np.unique([ep for ep, _ in dataset.clip_indices]))
    if len(episodes) < 3:
        raise ValueError("At least three usable episodes are needed for disjoint splits.")
    held_out = max(1, len(episodes) // 10)
    return episodes[2 * held_out:], episodes[:held_out], episodes[held_out:2 * held_out]


def partition_validation(episodes):
    """Reserve disjoint episodes for score fitting, beta selection, and thresholds."""
    if len(episodes) < 3:
        raise ValueError("At least three validation episodes are required; enlarge the smoke fixture.")
    return [part.tolist() for part in np.array_split(np.asarray(episodes), 3)]


def id_action_rows(dataset, raw_actions, training_indices, *, return_indices=False):
    """Return only actions that appear in a permitted training input."""
    coordinates = np.asarray(dataset.clip_indices)[training_indices]
    starts = np.asarray(dataset.offsets[coordinates[:, 0]], dtype=np.int64) + coordinates[:, 1]
    span = dataset.span - dataset.frameskip
    edges = np.bincount(starts, minlength=len(raw_actions) + 1)
    edges -= np.bincount(starts + span, minlength=len(raw_actions) + 1)
    eligible = np.flatnonzero(np.cumsum(edges[:-1]) > 0)
    eligible = eligible[np.isfinite(raw_actions[eligible]).all(axis=1)]
    return eligible if return_indices else raw_actions[eligible]


def dataset_episode_ids(dataset):
    try:
        row_ids = dataset.get_col_data("episode_idx")
    except KeyError:
        row_ids = dataset.get_col_data("ep_idx")
    return row_ids[np.asarray(dataset.offsets)]


def episode_membership(dataset, **roles):
    episode_ids = dataset_episode_ids(dataset)
    membership = {}
    seen = set()
    for role, episodes in roles.items():
        indices = np.asarray(episodes, dtype=np.int64)
        ids = episode_ids[indices].tolist()
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Dataset episode IDs must be unique and disjoint across roles.")
        seen.update(ids)
        membership[role] = {"indices": indices.tolist(), "episode_ids": ids}
    return membership


def clip_index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def restrict_training(dataset, training_seed, split_seed, quantile, manifest_path, dataset_name,
                      model_name, sigreg_weight, smoke, dataset_provenance=None):
    raw_actions = dataset.get_col_data("action")
    training, validation, test = split_episodes(dataset, np.random.default_rng(split_seed))
    episodes = np.asarray(dataset.clip_indices)[:, 0]
    magnitude = clip_magnitudes(dataset, raw_actions)
    training_mask = np.isin(episodes, training)
    cutoff = float(np.quantile(magnitude[training_mask], quantile))
    outside = magnitude[training_mask & (magnitude > cutoff)]
    if len(outside):
        severity_edges = np.quantile(outside, [1 / 3, 2 / 3]).tolist()
    else:
        severity_edges = [cutoff * 1.10, cutoff * 1.25]
    train_indices = np.flatnonzero(training_mask & (magnitude <= cutoff))
    validation_indices = np.flatnonzero(np.isin(episodes, validation) & (magnitude <= cutoff))
    if not len(train_indices):
        raise ValueError("The action cutoff leaves no ID training clips.")
    eligible = np.unique(episodes[validation_indices])
    calibration, selection, threshold = partition_validation(validation[np.isin(validation, eligible)])
    rows = id_action_rows(dataset, raw_actions, train_indices)
    scaler = StandardScaler().fit(rows)
    manifest = {
        "split_seed": split_seed, "training_seed": training_seed, "sigreg_weight": sigreg_weight,
        "smoke": smoke, "quantile": quantile, "cutoff": cutoff,
        "severity_edges": [cutoff, *severity_edges],
        "dataset": dataset_name, "model_name": model_name,
        "dataset_provenance": dataset_provenance,
        "train": training.tolist(), "validation": validation.tolist(), "test": test.tolist(),
        "roles": episode_membership(dataset, train=training, score_calibration=calibration,
                                    penalty_selection=selection, threshold_calibration=threshold, test=test),
        "action_mean": scaler.mean_.tolist(), "action_scale": scaler.scale_.tolist(),
        "training_clips": len(train_indices), "validation_clips": len(validation_indices),
        "training_clips_before_filter": int(training_mask.sum()),
        "validation_clips_before_filter": int(np.isin(episodes, validation).sum()),
        "retained_training_episodes": int(len(np.unique(episodes[train_indices]))),
        "retained_validation_episodes": int(len(eligible)),
        "scaler_rows": len(rows), "training_clip_hash": clip_index_hash(train_indices),
        "validation_clip_hash": clip_index_hash(validation_indices),
        "action_magnitude_quantiles": {
            "probabilities": [0, 0.25, 0.5, 0.75, 1],
            "before_filter": np.quantile(magnitude[training_mask], [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "retained": np.quantile(magnitude[train_indices], [0, 0.25, 0.5, 0.75, 1]).tolist(),
        },
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2))
    # Upstream sanity/validation runs use training data. The separate roles are
    # reserved for the evaluator and never affect fitting or checkpoint selection.
    train_set = Subset(dataset, train_indices.tolist())
    val_set = Subset(dataset, train_indices[:128].tolist())
    return train_set, val_set, scaler
