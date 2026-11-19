"""Evaluate uncertainty and prediction errors on held-out LeWM episodes."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.wm.utils import load_pretrained
import torch
from torch.utils.data import DataLoader, Subset

from action_holdout import clip_index_hash, clip_magnitudes, dataset_episode_ids
from fit_uncertainty import batch_latents, predict
from run import SOURCE, TASKS, data_root, sha256, verify_uncertainty_artifacts
from uncertainty import (ensemble_disagreement, knn_distance, load_uncertain_model,
                         mahalanobis_distance, prior_deviation)
from uncertainty_metrics import prediction_metrics
from visual_shift import LEVELS, shift_pixels, patch_shape


STATE_COLUMNS = {"pusht": "state", "reacher": "qpos", "cube": "privileged_block_0_pos"}
ERRORS = ("latent_error", "clean_target_error", "physical_error", "probe_error")
SCORES = ("prior", "knn", "ensemble", "observed_knn", "observed_prior",
          "predicted_norm", "observed_norm", "predicted_mahalanobis", "observed_mahalanobis",
          "action_history_knn", "predicted_squared_norm", "observed_squared_norm")


def compare(reference, shifted, mask=None):
    if mask is None:
        mask = np.ones(len(shifted["latent_error"]), dtype=bool)
    if not mask.any() or not len(reference["latent_error"]):
        return None
    result = {"count": int(mask.sum())}
    for key in ERRORS:
        result[key] = float(shifted[key][mask].mean())
    for method in SCORES:
        if method not in shifted:
            continue
        id_scores = reference[method]
        shift_scores = shifted[method][mask]
        pooled = np.r_[id_scores, shift_scores]
        labels = np.r_[np.zeros(len(id_scores)), np.ones(len(shift_scores))]
        comparison = {
            "shift_auroc": None if reference is shifted and mask.all()
            else prediction_metrics(pooled, pooled, labels)["shift_auroc"]
        }
        for error in ("latent_error", "clean_target_error", "physical_error"):
            metrics = prediction_metrics(shift_scores, shifted[error][mask], np.zeros(len(shift_scores)))
            comparison[f"{error}_spearman"] = metrics["spearman"]
        result[method] = comparison
    return result


def image_pair(batch, shifted, index, path):
    frames = torch.stack([batch["pixels"][index, -2], shifted[index, -2],
                          batch["pixels"][index, -1], shifted[index, -1]])
    mean = frames.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = frames.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    rgb = ((frames * std + mean).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    panel = np.concatenate([np.concatenate(rgb[:2], axis=1), np.concatenate(rgb[2:], axis=1)], axis=0)
    Image.fromarray(panel).save(path)


def coverage(states, rows):
    data = states[rows].reshape(len(rows), -1)
    data = data[np.isfinite(data).all(axis=1)]
    if not len(data):
        return None
    return {"count": len(data), "mean": data.mean(axis=0).tolist(), "std": data.std(axis=0).tolist(),
            "quantiles_05_50_95": np.quantile(data, [0.05, 0.5, 0.95], axis=0).tolist()}


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows_path = args.output.with_suffix(".clips.jsonl")
    examples_dir = args.output.with_suffix(".examples")
    examples_dir.mkdir(exist_ok=False)
    with rows_path.open("x") as records:
        summary = evaluate_clips(args, records, examples_dir)
    summary["clip_records"] = rows_path.name
    with args.output.open("x") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
    print(f"Wrote held-out uncertainty report to {args.output}.")


def evaluate_clips(args, records, examples_dir):
    sys.path.insert(0, str(SOURCE))
    from utils import get_img_preprocessor

    root = data_root()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    directory = args.artifacts.resolve()
    split = json.loads((directory / "splits.json").read_text())
    ensemble_config = json.loads((directory / "ensemble.json").read_text())
    knn_config = json.loads((directory / "knn.json").read_text())
    prior_config = json.loads((directory / "prior.json").read_text())
    for config in (ensemble_config, knn_config, prior_config):
        verify_uncertainty_artifacts(root, config, directory)
    for config in (ensemble_config, prior_config):
        for name in ("training_bank", "prior_stats", "action_bank"):
            if name in config:
                config[name] = str(directory / config[name])
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ensemble_config["action_mean"])
    scaler.scale_ = np.asarray(ensemble_config["action_scale"])
    checkpoint = split["checkpoint"]
    provenance = split.get("training_provenance")
    if provenance and sha256(root / "checkpoints" / checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("The fitted base checkpoint content has changed.")
    model = load_pretrained(checkpoint).to(device).eval().requires_grad_(False)
    ensemble = load_uncertain_model(model, json.dumps(ensemble_config))
    prior = load_uncertain_model(model, json.dumps(prior_config))
    bank = torch.load(directory / knn_config["training_bank"], map_location=device, weights_only=True)
    stats = torch.load(directory / prior_config.get("diagnostic_prior_stats", "prior-stats.pt"),
                       map_location=device, weights_only=True)
    model_config = OmegaConf.load((root / "checkpoints" / checkpoint).parent / "config.json")
    state_key = args.state_column or STATE_COLUMNS[args.task]
    dataset = HDF5Dataset(split["dataset"], frameskip=5, num_steps=model_config.predictor.num_frames + 1,
                          keys_to_load=["pixels", "action", state_key], keys_to_cache=["action", state_key],
                          transform=get_img_preprocessor("pixels", "pixels"))
    cutoff = split.get("action_cutoff")
    magnitude_values = clip_magnitudes(dataset, dataset.get_col_data("action"))
    allowed = None if cutoff is None else magnitude_values <= cutoff
    coordinates = np.asarray(dataset.clip_indices, dtype=np.int64)
    offsets = np.asarray(dataset.offsets)
    episode_ids = dataset_episode_ids(dataset)
    target_rows = offsets[coordinates[:, 0]] + coordinates[:, 1] + (dataset.span - dataset.frameskip)

    def clips(episodes, cap, restrict=False):
        mask = np.isin(coordinates[:, 0], episodes)
        if restrict and allowed is not None:
            mask &= allowed
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError("An evaluation split has no usable clips.")
        if len(indices) > cap:
            indices = sorted(rng.choice(indices, size=cap, replace=False))
        return np.asarray(indices), DataLoader(Subset(dataset, list(indices)), batch_size=args.batch_size)

    train_z, train_state = [], []
    physical_states = dataset.get_col_data(state_key)
    training_mask = np.isin(coordinates[:, 0], split["train"])
    train_indices, train_loader = clips(split["train"], args.probe_clips, restrict=True)
    train_all = np.isin(coordinates[:, 0], split.get("episode_roles", {}).get("train", {}).get("indices", split["train"]))
    observed_rows = target_rows - dataset.frameskip
    training_coverage = {
        "before_filter": coverage(physical_states, np.unique(observed_rows[train_all])),
        "retained": coverage(physical_states, np.unique(observed_rows[training_mask if allowed is None else training_mask & allowed])),
    }

    conditions = [(None, 0.0)] + [(family, level) for family in args.visual_families for level in LEVELS[family]]
    buckets = {condition: {} for condition in conditions}
    examples = []

    with torch.inference_mode():
        for batch in train_loader:
            latents, actions = batch_latents(model, batch, scaler, device)
            train_z.append(latents[:, -1].cpu().numpy())
            train_state.append(batch[state_key][:, -1].reshape(len(actions), -1).numpy())
        states = np.concatenate(train_state)
        state_scale = StandardScaler().fit(states)
        probe = Ridge(alpha=1.0).fit(np.concatenate(train_z), state_scale.transform(states))
        if cutoff is None:
            action_cuts = np.quantile(magnitude_values[training_mask], [0.50, 0.75, 0.90])
        else:
            action_cuts = np.asarray(split["action_severity_edges"])
        test_indices, test_loader = clips(split["test"], args.test_clips)
        action_bins = np.searchsorted(action_cuts, magnitude_values[test_indices], side="left")
        position = 0
        for batch in test_loader:
            batch_size = len(batch["action"])
            indices = test_indices[position:position + batch_size]
            bands = action_bins[position:position + batch_size]
            position += batch_size
            true_state = state_scale.transform(batch[state_key][:, -1].reshape(len(batch["action"]), -1))
            clean_latents, actions = batch_latents(model, batch, scaler, device)
            clean_target = clean_latents[:, -1]
            for (family, level), bucket in buckets.items():
                if family:
                    pixels = shift_pixels(batch["pixels"], family, level, args.visual_seed)
                    latents = batch_latents(model, dict(batch, pixels=pixels), scaler, device)[0]
                else:
                    pixels, latents = batch["pixels"], clean_latents
                forecast, observed = predict(model, latents, actions)[:, -1], latents[:, -1]
                predictions = [forecast] if ensemble.include_base_predictor else []
                predictions.extend(predict(member, latents, actions)[:, -1] for member in ensemble.predictors)
                current = latents[:, -2]  # Last input observation; the target is unavailable online.
                scores = {
                    "prior": prior.score_prior(forecast).cpu().numpy(),
                    "knn": knn_distance(forecast, bank, knn_config["k"]).cpu().numpy(),
                    "ensemble": ensemble_disagreement(torch.stack(predictions)).cpu().numpy(),
                    "observed_knn": knn_distance(current, bank, knn_config["k"]).cpu().numpy(),
                    "observed_prior": prior.score_prior(current).cpu().numpy(),
                    "predicted_norm": prior_deviation(forecast).cpu().numpy(),
                    "observed_norm": prior_deviation(current).cpu().numpy(),
                    "predicted_mahalanobis": mahalanobis_distance(forecast, **stats).cpu().numpy(),
                    "observed_mahalanobis": mahalanobis_distance(current, **stats).cpu().numpy(),
                    "predicted_squared_norm": forecast.float().square().sum(-1).cpu().numpy(),
                    "observed_squared_norm": current.float().square().sum(-1).cpu().numpy(),
                    "latent_error": (forecast - observed).norm(dim=-1).cpu().numpy(),
                    "clean_target_error": (forecast - clean_target).norm(dim=-1).cpu().numpy(),
                    "physical_error": np.sqrt(((probe.predict(forecast.cpu().numpy()) - true_state) ** 2).mean(axis=1)),
                    "probe_error": np.sqrt(((probe.predict(observed.cpu().numpy()) - true_state) ** 2).mean(axis=1)),
                }
                action_score = ensemble.score_actions(actions[:, :-1])
                if action_score is not None:
                    scores["action_history_knn"] = action_score.amax(dim=(1, 2)).cpu().numpy()
                for name, data in scores.items():
                    bucket.setdefault(name, []).append(data)
                matches = np.flatnonzero(bands == 0) if cutoff is not None else np.arange(len(bands))
                if family and len(matches) and not any(item["family"] == family and item["severity"] == level for item in examples):
                    row = int(matches[0])
                    name = f"{family}-{level:g}.png"
                    image_pair(batch, pixels, row, examples_dir / name)
                    height, width = pixels.shape[-2:]
                    patch_height, patch_width = patch_shape(height, width, level)
                    examples.append({
                        "family": family, "severity": level, "path": f"{examples_dir.name}/{name}",
                        "clip_index": int(indices[row]), "episode_id": int(episode_ids[coordinates[indices[row], 0]]),
                        "panel": "top row: input history; bottom row: target; clean left and transformed right",
                        "patch_xyxy": [0, 0, patch_width, patch_height] if family != "tint" else None,
                        "occluded_fraction": patch_width * patch_height / (width * height) if family == "occlusion" else None,
                        "task_critical_overlap": None,
                    })
                for row, index in enumerate(indices):
                    band = int(bands[row])
                    if family and cutoff is not None and band != 0:
                        continue
                    episode, start = coordinates[index]
                    label = f"visual:{family}:{level:g}" if family else ("clean" if band == 0 or cutoff is None else f"action:{band}")
                    record = {
                        "condition": label, "clip_index": int(index), "episode_index": int(episode),
                        "episode_id": int(episode_ids[episode]), "start": int(start),
                        "raw_action_magnitude": float(magnitude_values[index]), "action_band": band,
                        "visual_family": family, "visual_severity": level, "visual_seed": args.visual_seed,
                        "visual_mode": "history_and_target" if family else None,
                        "checkpoint": checkpoint, "evaluation_seed": args.seed,
                        "fit_seed": split["seed"],
                        "scores": {key: float(data[row]) for key, data in scores.items() if key in SCORES},
                        "errors": {key: float(scores[key][row]) for key in ERRORS},
                    }
                    records.write(json.dumps(record, allow_nan=False) + "\n")
    buckets = {condition: {key: np.concatenate(parts) for key, parts in bucket.items()}
               for condition, bucket in buckets.items()}
    clean = buckets[None, 0.0]
    action_reference = {key: data[action_bins == 0] for key, data in clean.items()}
    visual_mask = action_bins == 0 if cutoff is not None else np.ones(len(action_bins), dtype=bool)
    visual_clean = action_reference if cutoff is not None else clean
    visual = {family: {} for family in args.visual_families}
    for (family, level), scores in buckets.items():
        if family:
            visual[family][str(level)] = compare(visual_clean, scores, visual_mask)
    probe_id = clean["probe_error"][action_bins == 0] if cutoff is not None else clean["probe_error"]
    return {
        "task": args.task, "dataset": split["dataset"], "checkpoint": checkpoint,
        "training_provenance": provenance,
        "fit_seed": split["seed"], "evaluation_seed": args.seed,
        "fit_protocol": split.get("fit_protocol"),
        "test_clip_hash": clip_index_hash(test_indices),
        "test_episode_ids": split.get("episode_roles", {}).get("test", {}).get("episode_ids"),
        "probe_clip_limit": args.probe_clips, "test_clip_limit": args.test_clips,
        "prior_mode": prior_config.get("prior_mode", "chi_square"),
        "physical_state_column": state_key, "probe_rmse_units": "training-state standard deviations",
        "physical_state_training_coverage": training_coverage,
        "probe_quality": {
            "training_clip_count": len(train_indices), "training_state_mean": state_scale.mean_.tolist(),
            "training_state_scale": state_scale.scale_.tolist(), "held_out_id_clips": len(probe_id),
            "held_out_id_representation_rmse": float(probe_id.mean()) if len(probe_id) else None,
        },
        "observed_score_timing": "last input observation, before prediction target",
        "physical_error_interpretation": "ID-trained probe diagnostic in standardized state coordinates",
        "visual_timing": "transformed history and next target with clean paired next target; no goal cost",
        "visual_seed": args.visual_seed, "visual_examples": examples,
        "clean": compare(visual_clean, visual_clean),
        "visual": visual, "visual_tint": visual.get("tint", {}),
        ("action_holdout" if cutoff is not None else "action_magnitude_diagnostic"): {
            "bin_edges": action_cuts.tolist(),
            "higher_magnitude_bins": {
                str(severity): compare(action_reference, clean, action_bins == severity)
                for severity in (1, 2, 3)
            },
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state-column")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-clips", type=int, default=4096)
    parser.add_argument("--test-clips", type=int, default=2048)
    parser.add_argument("--visual-families", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--visual-seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.probe_clips, args.test_clips) < 1:
        parser.error("Batch and clip limits must be positive.")
    evaluate(args)
