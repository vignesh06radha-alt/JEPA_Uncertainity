"""Trace selected plans and actual simulator steps during dataset evaluation."""

from time import perf_counter

import numpy as np
from stable_worldmodel.policy import WorldModelPolicy
from stable_worldmodel.solver.cem import CEMSolver
import torch

from uncertainty_metrics import trace_score
from visual_shift import shift_info


def state_snapshot(infos, slot):
    fields = ("state", "qpos", "qvel", "proprio", "privileged", "distance", "success",
              "goal_state", "goal_qpos", "goal_privileged_block_0_pos")
    snapshot = {}
    for key, values in infos.items():
        if key not in fields and not key.startswith("privileged_block_"):
            continue
        if isinstance(values, dict):
            snapshot[key] = state_snapshot(values, slot) or {
                name: np.asarray(value[slot]).tolist() for name, value in values.items()
                if isinstance(value, (np.ndarray, torch.Tensor))
            }
        elif isinstance(values, (np.ndarray, torch.Tensor)):
            value = np.asarray(values[slot])
            if value.dtype.kind in "biuf" and np.isfinite(value).all():
                snapshot[key] = value[-1].tolist() if value.ndim else value.tolist()
    return snapshot


class TracedCEM(CEMSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces = {}

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        episodes = info_dict.pop("_trace_episode").tolist()
        steps = info_dict.pop("_trace_step").tolist()
        start = perf_counter()
        result = super().solve(info_dict, init_action)
        solve_seconds = perf_counter() - start
        for first in range(0, len(episodes), self.batch_size):
            last = first + self.batch_size
            actions = result["actions"][first:last].to(self.device, dtype=self.dtype).unsqueeze(1)
            info = {
                key: info_dict[key][first:last].to(self.device, dtype=self.dtype).unsqueeze(1)
                for key in ("pixels", "goal")
            }
            info["action"] = actions[:, :, :info["pixels"].shape[2]]
            start = perf_counter()
            costs, uncertainty = self.model.score_candidates(info, actions)
            observation = (self.model.score_observation(info["emb"][:, 0, -1])
                           if self.model.observation_scale is not None else None)
            action_scores = self.model.score_actions(actions[:, 0])
            scores = uncertainty[:, 0].tolist()
            observation = observation.tolist() if observation is not None else [None] * len(scores)
            action_scores = action_scores.tolist() if action_scores is not None else [None] * len(scores)
            costs = costs[:, 0].tolist()
            score_seconds = perf_counter() - start
            for row, episode in enumerate(episodes[first:last]):
                forecast = max(scores[row])
                current = observation[row]
                self.traces.setdefault(episode, []).append({
                    "step": steps[first + row], "scores": scores[row],
                    "forecast_score": forecast, "observation_knn": current,
                    "warning_score": max(forecast, current) if current is not None else forecast,
                    "goal_cost": costs[row], "standardized_plan": result["actions"][first + row].tolist(),
                    "candidate_action_knn": action_scores[row],
                    "planning_seconds": solve_seconds / len(episodes),
                    "selected_plan_score_seconds": score_seconds / len(scores),
                })
        return result


class TracedPolicy(WorldModelPolicy):
    def __init__(self, *args, visual_shift=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.visual_shift = visual_shift
        self.action_cutoff = None
        self.execution = None

    def _prepare_info(self, info_dict):
        prepared = super()._prepare_info(info_dict)
        return shift_info(prepared, **self.visual_shift) if self.visual_shift else prepared

    def get_action(self, info_dict, **kwargs):
        annotated = dict(info_dict)
        execution = self.execution
        initial_records = {slot: len(trace) for slot, trace in self.solver.traces.items()}
        annotated["_trace_episode"] = np.arange(self.env.num_envs)
        annotated["_trace_step"] = execution.counts.copy()
        annotated["terminated"] = execution.retired.copy()
        result = super().get_action(annotated, **kwargs)
        for slot, traces in self.solver.traces.items():
            if len(traces) == initial_records.get(slot, 0):
                continue
            current = traces[-1]
            current["physical_state"] = state_snapshot(info_dict, slot)
            current["history_action_knn"] = None
            commands = execution.commands[slot][-self.flatten_receding_horizon:]
            if commands and self.solver.model.action_bank is not None:
                started = perf_counter()
                standardized = self.process["action"].transform(np.asarray(commands))
                actions = torch.as_tensor(standardized, device=self.solver.device)
                current["history_action_knn"] = self.solver.model.score_actions(actions).flatten().tolist()
                current["history_action_score_seconds"] = perf_counter() - started
        return result


class TracedExecution:
    """Apply initial acceptance at the environment's selective step boundary."""

    def __init__(self, pool, traces, threshold=None):
        self.pool = pool
        self.traces = traces
        self.threshold = threshold
        self.accepted = np.ones(pool.num_envs, dtype=bool)
        self.retired = np.zeros(pool.num_envs, dtype=bool)
        self.counts = np.zeros(pool.num_envs, dtype=int)
        self.commands = [[] for _ in range(pool.num_envs)]
        self.events = [None] * pool.num_envs
        self.last_state = [None] * pool.num_envs
        self.last_reward = [None] * pool.num_envs
        self.started = False

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def step(self, actions, mask=None):
        active = np.ones_like(self.accepted) if mask is None else np.asarray(mask, dtype=bool)
        rejected = np.zeros_like(active)
        if not self.started:
            self.started = True
            if self.threshold is not None:
                for slot in np.flatnonzero(active):
                    records = self.traces.get(slot, [])
                    if not records or records[0]["step"] != 0:
                        raise RuntimeError("A first selected plan is required before any gated action.")
                    score = trace_score(records[0])
                    if not np.isfinite(score):
                        raise RuntimeError("The gate requires a finite initial-plan score before execution.")
                    self.accepted[slot] = score <= self.threshold
                rejected = active & ~self.accepted
        execute = active & self.accepted & ~self.retired
        if not np.isfinite(actions[execute]).all():
            raise RuntimeError("Only finite actions can be executed.")
        observations, rewards, terminated, truncated, infos = self.pool.step(actions, mask=execute)
        for slot in np.flatnonzero(execute):
            self.counts[slot] += 1
            self.commands[slot].append(actions[slot].copy())
            self.last_state[slot] = state_snapshot(infos, slot)
            self.last_reward[slot] = float(rewards[slot]) if np.isfinite(rewards[slot]) else None
            if terminated[slot] or truncated[slot]:
                self.events[slot] = {
                    "step": int(self.counts[slot]), "reason": "terminated" if terminated[slot] else "environment_truncation",
                    "terminated": bool(terminated[slot]), "truncated": bool(truncated[slot]),
                    "physical_state": self.last_state[slot],
                    "reward": self.last_reward[slot],
                }
        self.retired |= rejected | terminated | truncated
        # The upstream wait loop retires these slots. They did not physically truncate.
        return observations, rewards, terminated, truncated | rejected, infos

    def action_exposure(self, cutoff):
        magnitudes = [np.abs(actions).max(axis=-1) if len(actions) else np.empty(0)
                      for actions in self.commands]
        return {
            "cutoff": cutoff,
            "max_raw_command": [float(values.max()) if len(values) else None for values in magnitudes],
            "commands_above_cutoff": [int((values > cutoff + 1e-6).sum()) for values in magnitudes],
            "finite_commands": self.counts.tolist(),
        }


def evaluate_with_trace(world, policy, settings, **options):
    gate = settings.get("gate")
    pool = world.envs
    execution = TracedExecution(pool, policy.solver.traces, gate["threshold"] if gate else None)
    policy.execution = execution
    world.envs = execution
    encoder_calls = 0

    def count_encode(_module, inputs):
        nonlocal encoder_calls
        encoder_calls += 1

    device = torch.device(policy.solver.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    handle = policy.solver.model.model.encoder.register_forward_pre_hook(count_encode)
    try:
        metrics = world.evaluate(**options)
    finally:
        world.envs = pool
        handle.remove()
    accepted = execution.accepted
    successes = [bool(ok) if accept else None for ok, accept in zip(metrics["episode_successes"], accepted)]
    metrics["episode_successes"] = successes
    metrics["executed_actions"] = execution.counts.tolist()
    metrics["episode_events"] = [
        None if not accept else (event or {
            "step": int(count), "reason": "budget_exhausted", "terminated": False, "truncated": False,
            "physical_state": state, "reward": reward,
        })
        for accept, event, count, state, reward in zip(
            accepted, execution.events, execution.counts, execution.last_state, execution.last_reward
        )
    ]
    for event, success in zip(metrics["episode_events"], successes):
        if event is not None:
            event["task_success"] = success
    metrics["selected_plan_traces"] = [policy.solver.traces.get(i, []) for i in range(pool.num_envs)]
    metrics["visual_shift"] = policy.visual_shift
    metrics["profiling"] = {"vision_encoder_calls": encoder_calls, "device": str(device),
                            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    if policy.action_cutoff is not None:
        metrics["action_command_exposure"] = execution.action_exposure(policy.action_cutoff)
    if gate:
        metrics["gate"] = {"threshold": gate["threshold"], "accepted": accepted.tolist(), "coverage": float(accepted.mean()),
                           "accepted_success": float(np.mean([ok for ok in successes if ok is not None]))
                           if accepted.any() else None}
        metrics["success_rate"] = (100 * metrics["gate"]["accepted_success"]
                                   if metrics["gate"]["accepted_success"] is not None else None)
    return metrics
