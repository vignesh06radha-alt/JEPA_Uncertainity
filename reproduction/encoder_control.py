"""Score identical offline action candidates with independently trained LeWM encoders."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.wm.utils import load_pretrained
import torch
from torch.utils.data import DataLoader, Subset

from action_holdout import clip_magnitudes
from run import SOURCE, data_root, sha256
from visual_shift import LEVELS, shift_pixels


def scalar_cost_disagreement(costs, scales):
    """Reduce across models only after each model's scalar cost has been scaled."""
    return (costs / scales[:, None, None]).std(dim=0, unbiased=False)


def candidate_goal_cost(model, pixels, clean_goal, actions):
    observations = pixels[:, None, :-1].expand(-1, actions.shape[1], -1, -1, -1, -1)
    rollout = model.rollout({"pixels": observations}, actions)
    rollout["goal_emb"] = model.encode({"pixels": clean_goal[:, -1:]})["emb"].unsqueeze(1)
    return model.criterion(rollout)


def load_protocol(artifacts):
    splits = [json.loads((path / "splits.json").read_text()) for path in artifacts]
    first = splits[0]
    keys = ("dataset", "episode_roles", "action_cutoff", "action_scaler")
    if any(key not in first for key in keys) or first["action_cutoff"] is None:
        raise ValueError("Use artifacts from the audited restricted training protocol.")
    protocol = {key: first[key] for key in keys}
    if any({key: item.get(key) for key in keys} != protocol for item in splits):
        raise ValueError("Encoder controls require matching data, episode roles, and action support.")
    provenance = [item["training_provenance"] for item in splits]
    identities = [(item["training_seed"], item["checkpoint_sha256"]) for item in provenance]
    if any(seed is None for seed, _ in identities) or any(len(set(values)) != len(splits) for values in zip(*identities)):
        raise ValueError("Encoder controls require independent training seeds and checkpoint contents.")
    if any(len({item[key] for item in provenance}) != 1 for key in ("sigreg_weight", "split_seed", "smoke")):
        raise ValueError("Encoder controls require a matched SIGReg coefficient, split seed, and training budget.")
    return splits, protocol


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(SOURCE))
    from utils import get_img_preprocessor

    splits, protocol = load_protocol(args.artifacts)
    root = data_root()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    models, configurations = [], []
    for split in splits:
        weights = root / "checkpoints" / split["checkpoint"]
        if sha256(weights) != split["training_provenance"]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint content changed: {split['checkpoint']}")
        configurations.append(OmegaConf.to_container(OmegaConf.load(weights.parent / "config.json"), resolve=True))
        models.append(load_pretrained(split["checkpoint"]).to(device).eval().requires_grad_(False))
    if any(config != configurations[0] for config in configurations):
        raise ValueError("Independent encoders must use the same architecture.")
    history = configurations[0]["predictor"]["num_frames"]
    dataset = HDF5Dataset(protocol["dataset"], frameskip=5, num_steps=history + 1,
                          keys_to_load=["pixels", "action"], keys_to_cache=["action"],
                          transform=get_img_preprocessor("pixels", "pixels"))
    magnitude = clip_magnitudes(dataset, dataset.get_col_data("action"))
    coordinates = np.asarray(dataset.clip_indices)
    cutoff = protocol["action_cutoff"]
    mean = torch.tensor(protocol["action_scaler"]["mean"], device=device, dtype=torch.float32)
    scale = torch.tensor(protocol["action_scaler"]["scale"], device=device, dtype=torch.float32)

    def sample(role, in_support):
        partition = protocol["episode_roles"][role]
        mask = np.isin(coordinates[:, 0], partition["indices"])
        if in_support:
            mask &= magnitude <= cutoff
        indices = np.flatnonzero(mask)
        if len(indices) > args.clips:
            indices = np.sort(rng.choice(indices, size=args.clips, replace=False))
        if not len(indices):
            raise ValueError(f"No usable clips in {role}.")
        ids = dict(zip(partition["indices"], partition["episode_ids"]))
        loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size)
        return indices, ids, loader

    def candidates(batch, in_support):
        raw = batch["action"][:, :-1].to(mean).unflatten(-1, (-1, len(mean)))
        standardized = torch.nan_to_num((raw - mean) / scale)
        population = standardized.unsqueeze(1).repeat(1, args.candidates, 1, 1, 1)
        final = population[:, 1:, -1]
        final.add_(torch.randn(final.shape, generator=generator, device=device) * 0.5)
        lower = (-cutoff - mean) / scale
        upper = (cutoff - mean) / scale
        # All ID candidates use the training support. Leave shifted candidates unconstrained.
        final.copy_(torch.where(in_support[:, None, None, None], final.clamp(lower, upper), final))
        return population.flatten(-2)

    def candidate_costs(pixels, clean_goal, actions):
        return torch.stack([candidate_goal_cost(model, pixels, clean_goal, actions) for model in models])

    rows_path = args.output.with_suffix(".candidates.jsonl")
    encoder_calls = [0] * len(models)
    records = {}
    began = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), ExitStack() as stack:
        for index, model in enumerate(models):
            def count_encoder(_module, _inputs, slot=index):
                encoder_calls[slot] += 1
            hook = model.encoder.register_forward_pre_hook(count_encoder)
            stack.callback(hook.remove)
        _, _, calibration = sample("score_calibration", True)
        calibration_costs = []
        for batch in calibration:
            pixels = batch["pixels"].to(device)
            actions = candidates(batch, torch.ones(len(pixels), dtype=torch.bool, device=device))
            calibration_costs.append(candidate_costs(pixels, pixels, actions))
        scales = torch.cat(calibration_costs, dim=1).mean(dim=(1, 2)).clamp_min(1e-8)
        test_indices, ids, test = sample("test", False)
        stream = stack.enter_context(rows_path.open("x"))
        position = 0
        for batch in test:
            pixels = batch["pixels"].to(device)
            indices = test_indices[position:position + len(pixels)]
            position += len(pixels)
            within_support = magnitude[indices] <= cutoff
            actions = candidates(batch, torch.from_numpy(within_support).to(device))
            shifted = shift_pixels(pixels, args.visual_family, args.visual_severity, args.seed)
            raw_costs = candidate_costs(pixels, pixels, actions)
            actual = actions.cpu().numpy()
            for condition, costs, include in (
                ("clean", raw_costs, within_support), ("action", raw_costs, ~within_support),
                ("visual", candidate_costs(shifted, pixels, actions), within_support),
            ):
                disagreement = scalar_cost_disagreement(costs, scales).cpu().numpy()
                normalized = (costs / scales[:, None, None]).cpu().numpy()
                for row in np.flatnonzero(include):
                    index = indices[row]
                    episode, start = coordinates[index]
                    ranks = [spearmanr(normalized[0, row], other).statistic for other in normalized[1:, row]
                             if np.ptp(normalized[0, row]) > 0 and np.ptp(other) > 0]
                    rank = float(np.mean(ranks)) if ranks else None
                    record = {"condition": condition, "clip_index": int(index), "episode_id": ids[episode],
                              "start": int(start), "raw_action_magnitude": float(magnitude[index]),
                              "standardized_candidate_actions": actual[row].tolist(),
                              "normalized_goal_costs": normalized[:, row].tolist(),
                              "candidate_cost_disagreement": disagreement[row].tolist(), "rank_agreement": rank}
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    records.setdefault(condition, []).append((float(disagreement[row].mean()), rank))
    summary = {
        "method": "ID-normalized scalar goal-cost disagreement",
        "goal": "Paired clean next frame. Each encoder scores its own forecast against its own goal representation.",
        "checkpoints": [split["checkpoint"] for split in splits],
        "training_provenance": [split["training_provenance"] for split in splits],
        "visual_family": args.visual_family, "visual_severity": args.visual_severity,
        "evaluation_seed": args.seed, "candidate_count": args.candidates, "id_mean_cost_scales": scales.tolist(),
        "candidate_records": rows_path.name, "encoder_calls": encoder_calls, "device": str(device),
        "elapsed_seconds": perf_counter() - began,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "conditions": {},
    }
    for name, observations in records.items():
        ranks = [rank for _, rank in observations if rank is not None]
        summary["conditions"][name] = {"clips": len(observations),
                                       "mean_candidate_disagreement": float(np.mean([score for score, _ in observations])),
                                       "mean_rank_agreement_with_first_encoder": float(np.mean(ranks)) if ranks else None}
    with args.output.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(f"Wrote independent-encoder control to {args.output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clips", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--visual-family", choices=LEVELS, default="tint")
    parser.add_argument("--visual-severity", type=float, default=0.35)
    args = parser.parse_args()
    if not 3 <= len(args.artifacts) <= 5 or min(args.batch_size, args.clips) < 1 or args.candidates < 2:
        parser.error("Provide three to five checkpoints, positive clip and batch counts, and two or more candidates.")
    if args.visual_severity not in LEVELS[args.visual_family]:
        parser.error("Choose a preregistered severity for the specified visual family.")
    evaluate(args)
