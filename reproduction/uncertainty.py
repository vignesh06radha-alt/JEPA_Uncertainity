"""Scores used by the SIGReg JEPA uncertainty experiments."""

import json
import math

import torch


def tint_pixels(pixels, alpha):
    """Apply a fixed tint in ImageNet-normalized RGB space."""
    from stable_pretraining.data.dataset_stats import ImageNet

    shape = (1,) * (pixels.ndim - 3) + (3, 1, 1)
    color = pixels.new_tensor([0.15, 0.65, 0.9]).reshape(shape)
    mean = pixels.new_tensor(ImageNet["mean"]).reshape(shape)
    std = pixels.new_tensor(ImageNet["std"]).reshape(shape)
    return pixels.lerp((color - mean) / std, alpha)


def ensemble_disagreement(predictions):
    """Mean latent variance, with the predictor index on the first axis."""
    if predictions.shape[0] < 2:
        raise ValueError("Disagreement requires at least two predictors.")
    return predictions.float().var(dim=0, unbiased=False).mean(dim=-1)


def knn_distance(latents, training_bank, k=10):
    """Mean Euclidean distance to the k closest training latents."""
    if not 1 <= k <= len(training_bank):
        raise ValueError("k must be between one and the training bank size.")
    queries = latents.reshape(-1, latents.shape[-1]).float()
    output = []
    for batch in queries.split(512):
        nearest = batch.new_full((len(batch), k), float("inf"))
        for reference in training_bank.split(4096):
            distances = torch.cdist(batch, reference.to(batch))
            local = distances.topk(min(k, len(reference)), largest=False).values
            nearest = torch.cat((nearest, local), dim=-1).topk(k, largest=False).values
        output.append(nearest.mean(dim=-1))
    return torch.cat(output).reshape(latents.shape[:-1])


def prior_deviation(latents):
    """Negative log two-sided chi-square tail under an isotropic Gaussian."""
    squared_norm = latents.float().square().sum(dim=-1)
    shape = squared_norm.new_tensor(latents.shape[-1] / 2)
    lower = torch.special.gammainc(shape, squared_norm / 2)
    upper = torch.special.gammaincc(shape, squared_norm / 2)
    return -(2 * torch.minimum(lower, upper)).clamp_min(1e-12).log()


def mahalanobis_distance(latents, center, whitening):
    return torch.linalg.vector_norm((latents.float() - center) @ whitening.T, dim=-1)


def check_prior(training_latents, validation_observed, validation_predicted):
    """Check the chi-square radial distribution on clean validation latents."""
    from scipy.stats import kstest
    from sklearn.covariance import LedoitWolf

    diagnostics = {}
    for name, latents in (("observed", validation_observed), ("predicted", validation_predicted)):
        scores = prior_deviation(latents).cpu().numpy()
        diagnostics[name] = {
            "count": len(scores),
            "squared_norm_mean": float(latents.float().square().sum(-1).mean()),
            "squared_norm_variance": float(latents.float().square().sum(-1).var(unbiased=False)),
            "ks_distance": float(kstest(scores, "expon").statistic),
            "saturation_fraction": float((scores >= -math.log(1e-12) - 1e-5).mean()),
        }
    fallback = any(item["ks_distance"] > 0.20 or item["saturation_fraction"] >= 0.05
                   for item in diagnostics.values())
    diagnostics["mode"] = "mahalanobis" if fallback else "chi_square"
    fit = LedoitWolf().fit(training_latents.detach().cpu().double().numpy())
    covariance = torch.from_numpy(fit.covariance_)
    identity = torch.eye(len(covariance), dtype=covariance.dtype)
    ridge = max(float(covariance.diag().mean()) * 1e-6, 1e-8)
    lower = torch.linalg.cholesky(covariance + ridge * identity)
    whitening = torch.linalg.solve_triangular(lower, identity, upper=False)
    return diagnostics, {"center": torch.from_numpy(fit.location_).float(), "whitening": whitening.float()}


def fit_threshold(id_validation_scores, coverage=0.8):
    """Fit the upper uncertainty cutoff from in-distribution validation."""
    if not 0 < coverage <= 1 or id_validation_scores.numel() == 0:
        raise ValueError("A nonempty validation set and coverage in (0, 1] are required.")
    return torch.quantile(id_validation_scores.float().flatten(), coverage)


class UncertainLeWM(torch.nn.Module):
    """Keep LeWM's goal cost and score only newly predicted rollout steps."""

    def __init__(self, model, method, *, beta=0.0, scale=1.0, training_bank=None,
                 predictors=(), k=10, prior_stats=None, include_base_predictor=True):
        super().__init__()
        if method not in {"ensemble", "knn", "prior"} or beta < 0 or scale <= 0:
            raise ValueError("Choose a supported score, a nonnegative beta, and a positive scale.")
        if method == "ensemble" and not predictors:
            raise ValueError("Provide trained ensemble members.")
        if method == "ensemble" and len(predictors) + int(include_base_predictor) < 2:
            raise ValueError("Disagreement requires at least two predictors.")
        if method == "knn" and training_bank is None:
            raise ValueError("Provide a bank of training latents.")
        self.model = model
        self.predictors = torch.nn.ModuleList(predictors)
        self.include_base_predictor = include_base_predictor
        self.register_buffer("training_bank", training_bank)
        self.register_buffer("prior_center", prior_stats["center"] if prior_stats else None)
        self.register_buffer("prior_whitening", prior_stats["whitening"] if prior_stats else None)
        self.method = method
        self.beta = beta
        self.scale = scale
        self.k = k
        self.register_buffer("action_lower", None)
        self.register_buffer("action_upper", None)

    def bound_raw_actions(self, cutoff, mean, scale, action_block):
        """Restrict CEM candidates to the trained componentwise raw action range."""
        device = next(self.parameters()).device
        mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
        self.action_lower = ((-cutoff - mean) / scale).repeat(action_block)
        self.action_upper = ((cutoff - mean) / scale).repeat(action_block)

    def score_candidates(self, info, actions):
        goal_cost = self.model.get_cost(info, actions)
        history = info["pixels"].size(2)
        predicted = info["predicted_emb"][..., history:, :]
        if self.method == "knn":
            scores = knn_distance(predicted, self.training_bank, self.k)
        elif self.method == "prior":
            scores = self.score_prior(predicted)
        else:
            # Use the base model's cached observation latents for every member.
            members = [predicted] if self.include_base_predictor else []
            for member in self.predictors:
                result = member.rollout(dict(info), actions)
                members.append(result["predicted_emb"][..., history:, :])
            scores = ensemble_disagreement(torch.stack(members))
        return goal_cost, scores / self.scale

    def score_prior(self, latents):
        if self.prior_center is not None:
            return mahalanobis_distance(latents, self.prior_center, self.prior_whitening)
        return prior_deviation(latents)

    def get_cost(self, info, actions):
        if self.action_lower is not None:
            # CEM gathers elites from this same tensor, so its returned mean stays in range.
            actions.clamp_(min=self.action_lower.to(actions), max=self.action_upper.to(actions))
        if self.beta == 0:
            return self.model.get_cost(info, actions)
        goal_cost, scores = self.score_candidates(info, actions)
        return goal_cost + self.beta * scores.sum(dim=-1)


def load_uncertain_model(model, config_json):
    """Load artifacts using the same checkpoint cache as the baseline."""
    from stable_worldmodel.wm.utils import load_pretrained

    config = json.loads(config_json)
    for field in ("action_mean", "action_scale", "dataset_name", "test_episode_ids",
                  "validation_episode_ids", "base_checkpoint", "evaluation_split", "visual_tint",
                  "action_band", "action_severity_edges", "prior_mode"):
        config.pop(field, None)
    for field in ("score_calibration_episode_ids", "threshold_episode_ids"):
        config.pop(field, None)
    device = next(model.parameters()).device
    bank_path = config.pop("training_bank", None)
    bank = torch.load(bank_path, map_location=device, weights_only=True) if bank_path else None
    stats_path = config.pop("prior_stats", None)
    prior_stats = torch.load(stats_path, map_location=device, weights_only=True) if stats_path else None
    predictors = []
    for checkpoint in config.pop("predictor_checkpoints", []):
        member = load_pretrained(checkpoint)
        member.encoder = model.encoder
        member.projector = model.projector
        predictors.append(member)
    return UncertainLeWM(model, training_bank=bank, predictors=predictors,
                         prior_stats=prior_stats, **config).to(device).eval().requires_grad_(False)
