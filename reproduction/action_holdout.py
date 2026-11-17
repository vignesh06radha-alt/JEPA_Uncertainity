"""Episode-disjoint action support restriction for end-to-end LeWM training."""

import json
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


def split_episodes(dataset, rng):
    episodes = rng.permutation(np.unique([ep for ep, _ in dataset.clip_indices]))
    if len(episodes) < 3:
        raise ValueError("At least three usable episodes are needed for disjoint splits.")
    held_out = max(1, len(episodes) // 10)
    return episodes[2 * held_out:], episodes[:held_out], episodes[held_out:2 * held_out]


def id_action_rows(dataset, raw_actions, training_indices):
    """Return only actions that appear in a permitted training input."""
    coordinates = np.asarray(dataset.clip_indices)[training_indices]
    starts = np.asarray(dataset.offsets[coordinates[:, 0]], dtype=np.int64) + coordinates[:, 1]
    span = dataset.span - dataset.frameskip
    edges = np.bincount(starts, minlength=len(raw_actions) + 1)
    edges -= np.bincount(starts + span, minlength=len(raw_actions) + 1)
    rows = raw_actions[np.cumsum(edges[:-1]) > 0]
    return rows[np.isfinite(rows).all(axis=1)]


def restrict_training(dataset, seed, quantile, manifest_path, dataset_name, model_name):
    raw_actions = dataset.get_col_data("action")
    training, validation, test = split_episodes(dataset, np.random.default_rng(seed))
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
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("The action cutoff leaves no ID training or validation clips.")
    scaler = StandardScaler().fit(id_action_rows(dataset, raw_actions, train_indices))
    manifest = {
        "seed": seed, "quantile": quantile, "cutoff": cutoff,
        "severity_edges": [cutoff, *severity_edges],
        "dataset": dataset_name, "model_name": model_name,
        "train": training.tolist(), "validation": validation.tolist(), "test": test.tolist(),
        "action_mean": scaler.mean_.tolist(), "action_scale": scaler.scale_.tolist(),
        "training_clips": len(train_indices), "validation_clips": len(validation_indices),
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2))
    return Subset(dataset, train_indices.tolist()), Subset(dataset, validation_indices.tolist()), scaler
