"""Evaluate uncertainty and prediction errors on held-out LeWM episodes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.wm.utils import load_pretrained
import torch
from torch.utils.data import DataLoader, Subset

from action_holdout import clip_magnitudes
from fit_uncertainty import batch_latents, predict
from run import SOURCE, TASKS, data_root
from uncertainty import ensemble_disagreement, knn_distance, load_uncertain_model, tint_pixels
from uncertainty_metrics import prediction_metrics


STATE_COLUMNS = {"pusht": "state", "reacher": "qpos", "cube": "privileged_block_0_pos"}
VISUAL_LEVELS = (0.15, 0.35, 0.60)


def compare(reference, shifted, mask=None):
    if mask is None:
        mask = np.ones(len(shifted["latent_error"]), dtype=bool)
    if not mask.any() or not len(reference["latent_error"]):
        return None
    result = {"count": int(mask.sum())}
    for key in ("latent_error", "physical_error", "probe_error"):
        result[key] = float(shifted[key][mask].mean())
    for method in ("prior", "knn", "ensemble"):
        id_scores = reference[method]
        shift_scores = shifted[method][mask]
        pooled = np.r_[id_scores, shift_scores]
        labels = np.r_[np.zeros(len(id_scores)), np.ones(len(shift_scores))]
        comparison = {
            "shift_auroc": None if reference is shifted and mask.all()
            else prediction_metrics(pooled, pooled, labels)["shift_auroc"]
        }
        for error in ("latent_error", "physical_error"):
            metrics = prediction_metrics(shift_scores, shifted[error][mask], np.zeros(len(shift_scores)))
            comparison[f"{error}_spearman"] = metrics["spearman"]
        result[method] = comparison
    return result


def evaluate(args):
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
    if "prior_stats" in prior_config:
        prior_config["prior_stats"] = str(directory / prior_config["prior_stats"])
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ensemble_config["action_mean"])
    scaler.scale_ = np.asarray(ensemble_config["action_scale"])
    checkpoint = split["checkpoint"]
    model = load_pretrained(checkpoint).to(device).eval().requires_grad_(False)
    ensemble = load_uncertain_model(model, json.dumps(ensemble_config))
    prior = load_uncertain_model(model, json.dumps(prior_config))
    bank = torch.load(directory / knn_config["training_bank"], map_location=device, weights_only=True)
    model_config = OmegaConf.load((root / "checkpoints" / checkpoint).parent / "config.json")
    state_key = args.state_column or STATE_COLUMNS[args.task]
    dataset = HDF5Dataset(split["dataset"], frameskip=5, num_steps=model_config.predictor.num_frames + 1,
                          keys_to_load=["pixels", "action", state_key], keys_to_cache=["action", state_key],
                          transform=get_img_preprocessor("pixels", "pixels"))
    cutoff = split.get("action_cutoff")
    allowed = None if cutoff is None else clip_magnitudes(dataset, dataset.get_col_data("action")) <= cutoff

    def clips(episodes, cap, restrict=False):
        episodes = set(episodes)
        indices = [i for i, (ep, _) in enumerate(dataset.clip_indices)
                   if ep in episodes and (not restrict or allowed is None or allowed[i])]
        if not indices:
            raise ValueError("An evaluation split has no usable clips.")
        if len(indices) > cap:
            indices = sorted(rng.choice(indices, size=cap, replace=False))
        return DataLoader(Subset(dataset, indices), batch_size=args.batch_size)

    train_z, train_state, action_sizes = [], [], []

    def magnitude(actions):
        return actions[:, :-1].abs().amax(dim=(1, 2)).cpu().numpy()

    with torch.inference_mode():
        for batch in clips(split["train"], args.probe_clips, restrict=True):
            latents, actions = batch_latents(model, batch, scaler, device)
            train_z.append(latents[:, -1].cpu().numpy())
            train_state.append(batch[state_key][:, -1].reshape(len(actions), -1).numpy())
            action_sizes.append(magnitude(actions))
        states = np.concatenate(train_state)
        state_scale = StandardScaler().fit(states)
        probe = Ridge(alpha=1.0).fit(np.concatenate(train_z), state_scale.transform(states))
        if cutoff is None:
            action_cuts = np.quantile(np.concatenate(action_sizes), [0.50, 0.75, 0.90])
        else:
            action_cuts = np.asarray(split["action_severity_edges"])
        buckets = {level: {} for level in (0.0, *VISUAL_LEVELS)}
        magnitudes = []
        for batch in clips(split["test"], args.test_clips):
            true_state = state_scale.transform(batch[state_key][:, -1].reshape(len(batch["action"]), -1))
            for level, records in buckets.items():
                inputs = dict(batch, pixels=tint_pixels(batch["pixels"], level)) if level else batch
                latents, actions = batch_latents(model, inputs, scaler, device)
                members = [model, *ensemble.predictors]
                predictions = torch.stack([predict(member, latents, actions)[:, -1] for member in members])
                forecast, observed = predictions[0], latents[:, -1]
                if not level:
                    if cutoff is None:
                        magnitudes.append(magnitude(actions))
                    else:
                        raw = torch.nan_to_num(batch["action"][:, :-1].abs(), nan=0.0)
                        magnitudes.append(raw.amax(dim=(1, 2)).numpy())
                scores = {
                    "prior": prior.score_prior(forecast).cpu().numpy(),
                    "knn": knn_distance(forecast, bank, knn_config["k"]).cpu().numpy(),
                    "ensemble": ensemble_disagreement(predictions).cpu().numpy(),
                    "latent_error": (forecast - observed).norm(dim=-1).cpu().numpy(),
                    "physical_error": np.sqrt(((probe.predict(forecast.cpu().numpy()) - true_state) ** 2).mean(axis=1)),
                    "probe_error": np.sqrt(((probe.predict(observed.cpu().numpy()) - true_state) ** 2).mean(axis=1)),
                }
                for name, data in scores.items():
                    records.setdefault(name, []).append(data)
    buckets = {level: {key: np.concatenate(parts) for key, parts in records.items()}
               for level, records in buckets.items()}
    clean = buckets[0.0]
    magnitude_values = np.concatenate(magnitudes)
    action_bins = np.searchsorted(action_cuts, magnitude_values, side="left")
    action_reference = {key: data[action_bins == 0] for key, data in clean.items()}
    visual_mask = action_bins == 0 if cutoff is not None else np.ones(len(action_bins), dtype=bool)
    visual_clean = action_reference if cutoff is not None else clean
    summary = {
        "task": args.task, "dataset": split["dataset"], "checkpoint": checkpoint,
        "fit_seed": split["seed"], "evaluation_seed": args.seed,
        "fit_protocol": split.get("fit_protocol"),
        "probe_clip_limit": args.probe_clips, "test_clip_limit": args.test_clips,
        "prior_mode": prior_config.get("prior_mode", "chi_square"),
        "physical_state_column": state_key, "probe_rmse_units": "training-state standard deviations",
        "clean": compare(visual_clean, visual_clean),
        "visual_tint": {str(level): compare(visual_clean, buckets[level], visual_mask) for level in VISUAL_LEVELS},
        ("action_holdout" if cutoff is not None else "action_magnitude_diagnostic"): {
            "bin_edges": action_cuts.tolist(),
            "higher_magnitude_bins": {
                str(severity): compare(action_reference, clean, action_bins == severity)
                for severity in (1, 2, 3)
            },
        },
    }
    with args.output.open("x") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
    print(f"Evaluated {len(clean['latent_error'])} held-out clips at four visual conditions. Wrote {args.output}.")


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
    args = parser.parse_args()
    if min(args.batch_size, args.probe_clips, args.test_clips) < 1:
        parser.error("Batch and clip limits must be positive.")
    evaluate(args)
