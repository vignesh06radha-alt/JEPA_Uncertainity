"""Fit shared-encoder predictors and ID calibration artifacts for LeWM."""

import argparse
import json
from pathlib import Path
import sys

import h5py
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.wm.utils import load_pretrained, save_pretrained
import torch
from torch.utils.data import DataLoader, Subset

from action_holdout import clip_magnitudes, partition_validation, split_episodes
from run import TASKS, SOURCE, data_root
from uncertainty import check_prior, ensemble_disagreement, knn_distance, mahalanobis_distance, prior_deviation


def batch_latents(model, batch, scaler, device):
    action = batch["action"].to(device).float()
    raw_dim = len(scaler.mean_)
    raw = action.reshape(*action.shape[:-1], -1, raw_dim)
    mean = raw.new_tensor(scaler.mean_)
    scale = raw.new_tensor(scaler.scale_)
    action = torch.nan_to_num((raw - mean) / scale).reshape_as(action)
    with torch.no_grad():
        encoded = model.encode({"pixels": batch["pixels"].to(device)})
    return encoded["emb"], action


def predict(model, latents, actions):
    return model.predict(latents[:, :-1], model.action_encoder(actions[:, :-1]))


def fit(args):
    sys.path.insert(0, str(SOURCE))
    from utils import get_img_preprocessor

    root = data_root()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = args.checkpoint or f"released/{args.task}/weights.pt"
    model = load_pretrained(checkpoint).to(device).eval().requires_grad_(False)
    weights = root / "checkpoints" / checkpoint
    config = OmegaConf.load(weights.parent / "config.json")
    history = int(config.predictor.num_frames)
    dataset_name = args.dataset or TASKS[args.task][1]
    dataset = HDF5Dataset(dataset_name, frameskip=5, num_steps=history + 1,
                          keys_to_load=["pixels", "action"], keys_to_cache=["action"],
                          transform=get_img_preprocessor("pixels", "pixels"))

    holdout = json.loads(args.holdout_manifest.read_text()) if args.holdout_manifest else None
    if holdout:
        if holdout["dataset"] != f"{dataset_name}.h5" or Path(checkpoint).parent.as_posix() != holdout["model_name"]:
            raise ValueError("The holdout manifest must match the dataset and base checkpoint.")
        training, validation, test = [np.asarray(holdout[key]) for key in ("train", "validation", "test")]
        allowed = clip_magnitudes(dataset, dataset.get_col_data("action")) <= holdout["cutoff"]
        held_in = set(np.r_[training, validation])
    else:
        training, validation, test = split_episodes(dataset, rng)
    with h5py.File(dataset.h5_path, "r") as source:
        column = "episode_idx" if "episode_idx" in source else "ep_idx"
        row_ids = source[column][:]
    test_episode_ids = row_ids[np.asarray(dataset.offsets)[test]].tolist()
    groups = {}
    for index, (episode, _) in enumerate(dataset.clip_indices):
        if holdout and episode in held_in and not allowed[index]:
            continue
        groups.setdefault(episode, []).append(index)
    training = [i for i in training if i in groups]
    validation = [i for i in validation if i in groups]
    if not training or not validation:
        raise ValueError("No clips remain in the training or validation partition.")
    calibration, selection, threshold = partition_validation(validation)
    if holdout:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(holdout["action_mean"])
        scaler.scale_ = np.asarray(holdout["action_scale"])
    else:
        training_rows = np.concatenate([
            np.arange(dataset.offsets[i], dataset.offsets[i] + dataset.lengths[i]) for i in training
        ])
        raw_actions = dataset.get_col_data("action")[training_rows]
        scaler = StandardScaler().fit(raw_actions[np.isfinite(raw_actions).all(axis=1)])
    out = root / "uncertainty" / args.task / args.name
    out.mkdir(parents=True, exist_ok=False)
    (out / "splits.json").write_text(json.dumps({
        "dataset": dataset_name, "checkpoint": checkpoint, "seed": args.seed,
        "fit_protocol": {
            "epochs": args.epochs, "members": args.members, "batch_size": args.batch_size,
            "bank_size_limit": args.bank_size, "validation_clip_limit": args.validation_clips,
            "train_clip_limit": args.max_clips,
            "action_holdout_quantile": holdout["quantile"] if holdout else None,
        },
        "train": np.asarray(training).tolist(), "validation": np.asarray(validation).tolist(),
        "score_calibration": calibration, "penalty_selection": selection, "threshold_calibration": threshold,
        "test": test.tolist(), "action_cutoff": holdout["cutoff"] if holdout else None,
        "action_severity_edges": holdout["severity_edges"] if holdout else None,
    }, indent=2))

    def loader(indices, cap=None):
        if cap is not None and len(indices) > cap:
            indices = rng.choice(indices, size=cap, replace=False)
        return DataLoader(Subset(dataset, list(indices)), batch_size=args.batch_size)

    bank_indices = [i for ep in training for i in groups[ep]]
    bank = torch.cat([batch_latents(model, batch, scaler, device)[0][:, 0].cpu()
                     for batch in loader(bank_indices, args.bank_size)])
    bank_path = out / "bank.pt"
    torch.save(bank, bank_path)
    bank = bank.to(device)
    k = min(10, len(bank))

    predictors = []
    names = []
    for index in range(args.members):
        torch.manual_seed(args.seed + index)
        member = load_pretrained(checkpoint).to(device).eval().requires_grad_(False)
        member.encoder = model.encoder
        member.projector = model.projector
        member.predictor = instantiate(config.predictor).to(device)
        member.pred_proj = instantiate(config.pred_proj).to(device)
        parameters = list(member.predictor.parameters()) + list(member.pred_proj.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=5e-5)
        for _ in range(args.epochs):
            sampled = rng.choice(training, size=len(training), replace=True)
            indices = [i for ep in sampled for i in groups[ep]]
            rng.shuffle(indices)
            for batch in loader(indices, args.max_clips):
                latents, actions = batch_latents(model, batch, scaler, device)
                loss = (predict(member, latents, actions) - latents[:, 1:]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite ensemble training loss.")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        member.eval().requires_grad_(False)
        name = f"uncertainty/{args.task}/{args.name}/member-{index}"
        save_pretrained(member, run_name=name, config=config)
        names.append(f"{name}/weights.pt")
        predictors.append(member)

    validation_indices = [i for ep in calibration for i in groups[ep]]
    values = {method: [] for method in ("prior", "knn", "ensemble")}
    observed, forecasts = [], []
    with torch.inference_mode():
        for batch in loader(validation_indices, args.validation_clips):
            latents, actions = batch_latents(model, batch, scaler, device)
            predicted = torch.stack([predict(member, latents, actions)[:, -1] for member in predictors])
            observed.append(latents[:, -1].cpu())
            base_forecast = predict(model, latents, actions)[:, -1]
            forecasts.append(base_forecast.cpu())
            values["knn"].append(knn_distance(base_forecast, bank, k).cpu())
            values["ensemble"].append(ensemble_disagreement(predicted).cpu())
    validation_predictions = torch.cat(forecasts)
    diagnostics, prior_stats = check_prior(bank, torch.cat(observed), validation_predictions)
    (out / "prior-diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    torch.save(prior_stats, out / "prior-stats.pt")
    if diagnostics["mode"] == "chi_square":
        values["prior"].append(prior_deviation(validation_predictions))
    else:
        values["prior"].append(mahalanobis_distance(validation_predictions, **prior_stats))
    for method, parts in values.items():
        scores = torch.cat(parts)
        settings = {
            "method": method, "beta": args.beta, "scale": max(float(scores.mean()), 1e-8),
            "action_mean": scaler.mean_.tolist(), "action_scale": scaler.scale_.tolist(),
            "dataset_name": dataset_name, "test_episode_ids": test_episode_ids,
            "validation_episode_ids": row_ids[np.asarray(dataset.offsets)[selection]].tolist(),
            "score_calibration_episode_ids": row_ids[np.asarray(dataset.offsets)[calibration]].tolist(),
            "threshold_episode_ids": row_ids[np.asarray(dataset.offsets)[threshold]].tolist(),
            "base_checkpoint": checkpoint,
        }
        if holdout:
            settings["action_severity_edges"] = holdout["severity_edges"]
        if method == "knn":
            settings.update(training_bank=bank_path.name, k=k)
        elif method == "ensemble":
            settings["predictor_checkpoints"] = names
            settings["include_base_predictor"] = False
        else:
            settings["prior_mode"] = diagnostics["mode"]
            if diagnostics["mode"] == "mahalanobis":
                settings["prior_stats"] = "prior-stats.pt"
        (out / f"{method}.json").write_text(json.dumps(settings, indent=2))
    print(f"Saved {len(bank)} training latents and {len(names)} equally trained predictors to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", help="Override the dataset name in the HDF5 cache")
    parser.add_argument("--checkpoint", help="Path to weights relative to the checkpoint cache")
    parser.add_argument("--holdout-manifest", type=Path, help="Action holdout manifest from restricted training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--validation-clips", type=int, default=512)
    parser.add_argument("--max-clips", type=int, help="Cap training clips per epoch for smoke checks")
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.bank_size, args.validation_clips) < 1 or args.members < 2:
        parser.error("Training counts must be positive and the ensemble needs at least two members.")
    if args.beta < 0 or (args.max_clips is not None and args.max_clips < 1):
        parser.error("beta must be nonnegative and max-clips must be positive.")
    if args.holdout_manifest and not args.checkpoint:
        parser.error("--holdout-manifest requires the restricted base --checkpoint")
    fit(args)
