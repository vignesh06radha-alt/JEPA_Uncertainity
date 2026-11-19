"""Fit shared-encoder predictors and ID calibration artifacts for LeWM."""

import argparse
import json
from pathlib import Path
import sys

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.wm.utils import load_pretrained, save_pretrained
import torch
from torch.utils.data import DataLoader, Subset

from action_holdout import (clip_index_hash, clip_magnitudes, episode_membership, id_action_rows,
                            partition_validation, split_episodes)
from run import TASKS, SOURCE, data_root, dataset_provenance, sha256, uncertainty_artifact_paths
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


def bootstrap_clips(groups, training, budget, rng):
    episodes = rng.choice(training, size=len(training), replace=True)
    pool = [clip for episode in episodes for clip in groups[episode]]
    if len(pool) >= budget:
        indices = rng.choice(pool, size=budget, replace=False)
    else:
        indices = np.r_[pool, rng.choice(pool, size=budget - len(pool), replace=True)]
        rng.shuffle(indices)
    return episodes, indices


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
    checkpoint_hash = sha256(weights)
    config = OmegaConf.load(weights.parent / "config.json")
    history = int(config.predictor.num_frames)
    dataset_name = args.dataset or TASKS[args.task][1]
    dataset = HDF5Dataset(dataset_name, frameskip=5, num_steps=history + 1,
                          keys_to_load=["pixels", "action"], keys_to_cache=["action"],
                          transform=get_img_preprocessor("pixels", "pixels"))

    holdout = json.loads(args.holdout_manifest.read_text()) if args.holdout_manifest else None
    provenance = dataset_provenance(dataset.h5_path)
    raw_actions = dataset.get_col_data("action")
    if holdout:
        if holdout["dataset"] != f"{dataset_name}.h5" or Path(checkpoint).parent.as_posix() != holdout["model_name"]:
            raise ValueError("The holdout manifest must match the dataset and base checkpoint.")
        training, validation, test = [np.asarray(holdout[key]) for key in ("train", "validation", "test")]
        allowed = clip_magnitudes(dataset, raw_actions) <= holdout["cutoff"]
        held_in = set(np.r_[training, validation])
        split_seed = holdout.get("split_seed", holdout.get("seed"))
        if "roles" in holdout:
            cfg = OmegaConf.load(weights.parent / "config.yaml")
            if (cfg.seed != holdout["training_seed"] or cfg.loss.sigreg.weight != holdout["sigreg_weight"]
                    or cfg.output_model_name != holdout["model_name"]):
                raise ValueError("Resolved training configuration and action holdout manifest disagree.")
        if holdout.get("dataset_provenance") and holdout["dataset_provenance"] != provenance:
            raise ValueError("Dataset provenance differs from restricted training.")
    else:
        split_seed = args.split_seed
        training, validation, test = split_episodes(dataset, np.random.default_rng(split_seed))
    split_training = training
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
    roles = episode_membership(dataset, train=split_training, score_calibration=calibration,
                               penalty_selection=selection, threshold_calibration=threshold, test=test)
    train_indices = sorted(i for ep in training for i in groups[ep])
    action_row_ids = id_action_rows(dataset, raw_actions, train_indices, return_indices=True)
    action_rows = raw_actions[action_row_ids]
    scaler = StandardScaler().fit(action_rows)
    if holdout:
        if not (np.allclose(scaler.mean_, holdout["action_mean"]) and
                np.allclose(scaler.scale_, holdout["action_scale"])):
            raise ValueError("Training-only action scaler does not match the holdout manifest.")
        scaler.mean_ = np.asarray(holdout["action_mean"])
        scaler.scale_ = np.asarray(holdout["action_scale"])
        if "roles" in holdout and (
            holdout["roles"] != roles or holdout["training_clip_hash"] != clip_index_hash(train_indices)
            or holdout["validation_clip_hash"] != clip_index_hash(sorted(i for ep in validation for i in groups[ep]))
        ):
            raise ValueError("Episode roles or retained clip membership differ from restricted training.")
    clip_budget = min(len(train_indices), args.max_clips or len(train_indices))
    latent_bank_count = min(args.bank_size, len(train_indices))
    fit_protocol = {
        "epochs": args.epochs, "members": args.members, "batch_size": args.batch_size,
        "bank_size_limit": args.bank_size, "validation_clip_limit": args.validation_clips,
        "latent_bank_count": latent_bank_count, "latent_k_requested": args.k,
        "latent_k_effective": min(args.k, latent_bank_count),
        "train_clip_limit": args.max_clips, "clips_per_member_epoch": clip_budget,
        "action_holdout_quantile": holdout["quantile"] if holdout else None,
    }
    training_provenance = {
        "checkpoint_sha256": checkpoint_hash, "training_seed": holdout.get("training_seed") if holdout else None,
        "sigreg_weight": holdout.get("sigreg_weight") if holdout else None,
        "split_seed": split_seed, "smoke": holdout.get("smoke") if holdout else None,
        "holdout_manifest_sha256": sha256(args.holdout_manifest) if holdout else None,
    }
    out = root / "uncertainty" / args.task / args.name
    out.mkdir(parents=True, exist_ok=False)
    (out / "splits.json").write_text(json.dumps({
        "dataset": dataset_name, "checkpoint": checkpoint, "seed": args.seed,
        "training_provenance": training_provenance,
        "dataset_provenance": provenance,
        "episode_roles": roles,
        "action_scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "rows": len(action_rows)},
        "fit_protocol": fit_protocol,
        "train": np.asarray(training).tolist(), "validation": np.asarray(validation).tolist(),
        "score_calibration": calibration, "penalty_selection": selection, "threshold_calibration": threshold,
        "test": test.tolist(), "action_cutoff": holdout["cutoff"] if holdout else None,
        "action_severity_edges": holdout["severity_edges"] if holdout else None,
    }, indent=2))

    def sample(indices, cap):
        return rng.choice(indices, size=cap, replace=False) if len(indices) > cap else indices

    def loader(indices):
        return DataLoader(Subset(dataset, list(indices)), batch_size=args.batch_size)

    bank_indices = sample(train_indices, args.bank_size)
    bank = torch.cat([batch_latents(model, batch, scaler, device)[0][:, 0].cpu()
                     for batch in loader(bank_indices)])
    bank_path = out / "bank.pt"
    torch.save(bank, bank_path)
    bank = bank.to(device)
    k = fit_protocol["latent_k_effective"]
    action_indices = sample(np.arange(len(action_rows)), args.bank_size)
    action_bank = torch.from_numpy(scaler.transform(action_rows[action_indices])).float()
    torch.save(action_bank, out / "action-bank.pt")
    action_bank_info = {"action_bank": "action-bank.pt", "action_k": min(10, len(action_bank)),
                        "action_dim": len(scaler.mean_)}

    predictors = []
    names = []
    member_audit = []
    for index in range(args.members):
        member_seed = args.seed + index
        torch.manual_seed(member_seed)
        member_rng = np.random.default_rng(member_seed)
        member = load_pretrained(checkpoint).to(device).eval().requires_grad_(False)
        member.encoder = model.encoder
        member.projector = model.projector
        member.action_encoder = model.action_encoder
        member.predictor = instantiate(config.predictor).to(device)
        member.pred_proj = instantiate(config.pred_proj).to(device)
        parameters = list(member.predictor.parameters()) + list(member.pred_proj.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=5e-5)
        audit = {"seed": member_seed, "epochs": []}
        for _ in range(args.epochs):
            sampled, indices = bootstrap_clips(groups, training, clip_budget, member_rng)
            updates = 0
            total_loss = torch.zeros((), device=device)
            for batch in loader(indices):
                latents, actions = batch_latents(model, batch, scaler, device)
                loss = (predict(member, latents, actions) - latents[:, 1:]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite ensemble training loss.")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates += 1
                total_loss += loss.detach() * len(latents)
            audit["epochs"].append({"episode_bootstrap": sampled.tolist(), "clip_hash": clip_index_hash(indices),
                                    "updates": updates, "clips": len(indices), "mean_loss": float(total_loss / len(indices))})
        member.eval().requires_grad_(False)
        name = f"uncertainty/{args.task}/{args.name}/member-{index}"
        save_pretrained(member, run_name=name, config=config)
        names.append(f"{name}/weights.pt")
        predictors.append(member)
        member_audit.append(audit)

    validation_indices = sample([i for ep in calibration for i in groups[ep]], args.validation_clips)
    values = {method: [] for method in ("prior", "knn", "ensemble")}
    observed, forecasts, observed_knn, member_mse = [], [], [], []
    with torch.inference_mode():
        for batch in loader(validation_indices):
            latents, actions = batch_latents(model, batch, scaler, device)
            predicted = torch.stack([predict(member, latents, actions)[:, -1] for member in predictors])
            observed.append(latents[:, -2].cpu())
            observed_knn.append(knn_distance(latents[:, -2], bank, k).cpu())
            member_mse.append((predicted - latents[:, -1]).square().mean(dim=-1).cpu())
            base_forecast = predict(model, latents, actions)[:, -1]
            forecasts.append(base_forecast.cpu())
            values["knn"].append(knn_distance(base_forecast, bank, k).cpu())
            values["ensemble"].append(ensemble_disagreement(predicted).cpu())
    validation_predictions = torch.cat(forecasts)
    diagnostics, prior_stats = check_prior(bank, torch.cat(observed), validation_predictions)
    (out / "prior-diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    torch.save(prior_stats, out / "prior-stats.pt")
    for audit, error in zip(member_audit, torch.cat(member_mse, dim=1).mean(dim=1).tolist()):
        audit["score_calibration_prediction_mse"] = error
    (out / "fit-audit.json").write_text(json.dumps({
        "members": member_audit, "calibration_clips": np.asarray(validation_indices).tolist(),
        "bank_clips": np.asarray(bank_indices).tolist(), "action_rows": action_row_ids[action_indices].tolist(),
        "bank_sampling_seed": args.seed, "bank_residency": str(device),
        "bank_query_chunk": 512, "bank_reference_chunk": 4096, "action_bank": action_bank_info,
    }, indent=2, allow_nan=False))
    if diagnostics["mode"] == "chi_square":
        values["prior"].append(prior_deviation(validation_predictions))
    else:
        values["prior"].append(mahalanobis_distance(validation_predictions, **prior_stats))
    artifact_hashes = {}
    for method, parts in values.items():
        scores = torch.cat(parts)
        settings = {
            "method": method, "beta": args.beta, "scale": max(float(scores.mean()), 1e-8),
            "action_mean": scaler.mean_.tolist(), "action_scale": scaler.scale_.tolist(),
            "dataset_name": dataset_name, "test_episode_ids": roles["test"]["episode_ids"],
            "validation_episode_ids": roles["penalty_selection"]["episode_ids"],
            "score_calibration_episode_ids": roles["score_calibration"]["episode_ids"],
            "threshold_episode_ids": roles["threshold_calibration"]["episode_ids"],
            "base_checkpoint": checkpoint, "base_checkpoint_sha256": checkpoint_hash,
            "training_provenance": training_provenance,
            "fit_protocol": fit_protocol,
            **action_bank_info,
        }
        if holdout:
            settings["action_severity_edges"] = holdout["severity_edges"]
        if method == "knn":
            settings.update(training_bank=bank_path.name, k=k)
        elif method == "ensemble":
            settings["predictor_checkpoints"] = names
            settings["include_base_predictor"] = False
            settings.update(training_bank=bank_path.name, k=k,
                            observation_scale=max(float(torch.cat(observed_knn).mean()), 1e-8))
        else:
            settings["prior_mode"] = diagnostics["mode"]
            settings["diagnostic_prior_stats"] = "prior-stats.pt"
            if diagnostics["mode"] == "mahalanobis":
                settings["prior_stats"] = "prior-stats.pt"
        paths = uncertainty_artifact_paths(root, settings, out)
        for path in paths.values():
            if path not in artifact_hashes:
                artifact_hashes[path] = sha256(path)
        settings["artifact_sha256"] = {name: artifact_hashes[path] for name, path in paths.items()}
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
    parser.add_argument("--split-seed", type=int, default=42, help="Episode split seed for fits without a holdout manifest")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--k", type=int, default=10, help="Neighbors used for the latent bank")
    parser.add_argument("--validation-clips", type=int, default=512)
    parser.add_argument("--max-clips", type=int, help="Cap training clips per epoch for smoke checks")
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.bank_size, args.k, args.validation_clips) < 1 or args.members < 2:
        parser.error("Training counts must be positive and the ensemble needs at least two members.")
    if args.beta < 0 or (args.max_clips is not None and args.max_clips < 1):
        parser.error("beta must be nonnegative and max-clips must be positive.")
    if args.holdout_manifest and not args.checkpoint:
        parser.error("--holdout-manifest requires the restricted base --checkpoint")
    fit(args)
