"""Trace the final CEM plan on each replanning step."""

import numpy as np
from stable_worldmodel.policy import WorldModelPolicy
from stable_worldmodel.solver.cem import CEMSolver
import torch

from uncertainty import tint_pixels


class TracedCEM(CEMSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces = {}

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        episodes = info_dict.pop("_trace_episode").tolist()
        steps = info_dict.pop("_trace_step").tolist()
        result = super().solve(info_dict, init_action)
        for first in range(0, len(episodes), self.batch_size):
            last = first + self.batch_size
            actions = result["actions"][first:last].to(self.device, dtype=self.dtype).unsqueeze(1)
            info = {
                key: info_dict[key][first:last].to(self.device, dtype=self.dtype).unsqueeze(1)
                for key in ("pixels", "goal")
            }
            info["action"] = actions[:, :, :info["pixels"].shape[2]]
            _, uncertainty = self.model.score_candidates(info, actions)
            for episode, step, scores in zip(episodes[first:last], steps[first:last], uncertainty[:, 0].tolist()):
                self.traces.setdefault(episode, []).append({"step": step, "scores": scores})
        return result


class TracedPolicy(WorldModelPolicy):
    def __init__(self, *args, visual_tint=0.0, action_cutoff=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.visual_tint = visual_tint
        self.action_cutoff = action_cutoff
        self.action_commands = []
        self.step = 0

    def _prepare_info(self, info_dict):
        prepared = super()._prepare_info(info_dict)
        if self.visual_tint:
            for key in ("pixels", "goal"):
                prepared[key] = tint_pixels(prepared[key], self.visual_tint)
        return prepared

    def get_action(self, info_dict, **kwargs):
        annotated = dict(info_dict)
        annotated["_trace_episode"] = np.arange(self.env.num_envs)
        annotated["_trace_step"] = np.full(self.env.num_envs, self.step)
        result = super().get_action(annotated, **kwargs)
        if self.action_cutoff is not None:
            self.action_commands.append(result.copy())
        self.step += 1
        return result

    def action_exposure(self):
        commands = np.asarray(self.action_commands)
        valid = np.isfinite(commands).all(axis=-1)
        magnitude = np.where(valid, np.nan_to_num(np.abs(commands)).max(axis=-1), 0)
        return {
            "cutoff": self.action_cutoff,
            "max_raw_command": magnitude.max(axis=0).tolist(),
            "commands_above_cutoff": ((magnitude > self.action_cutoff + 1e-6) & valid).sum(axis=0).tolist(),
            "finite_commands": valid.sum(axis=0).tolist(),
        }
