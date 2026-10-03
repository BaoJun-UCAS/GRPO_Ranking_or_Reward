"""Method 1: historical quantiles of dimensionless within-prompt differences.

This adapts reward units and affected pair proportions, not reward reliability.
State belongs to one configured estimator/trainer, never a module-global cache.
"""

from collections import deque
import json
import math
from numbers import Real
from pathlib import Path

import torch


def _real(name, value, *, minimum=0, strict=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or (value <= minimum if strict else value < minimum)):
        raise ValueError(f"{name} must be a finite real number {'>' if strict else '>='} {minimum}")


def _thresholds(ell, tau):
    _real("ell", ell)
    if isinstance(tau, bool) or not isinstance(tau, Real) or math.isnan(tau) or tau < 0:
        raise ValueError("tau must be nonnegative (positive infinity disables clipping)")


def standardized_differences(batch, epsilon):
    _real("epsilon", epsilon, strict=True)
    rewards = batch.rewards.detach().double()
    std = batch.group_std.detach().double()
    if rewards.ndim != 2 or rewards.shape[1] < 2 or std.shape != (rewards.shape[0], 1):
        raise ValueError("Expected complete reward groups and unbiased per-group standard deviations")
    if not torch.isfinite(rewards).all() or not torch.isfinite(std).all() or (std < 0).any():
        raise ValueError("Rewards and group standard deviations must be finite")
    return (rewards.unsqueeze(2) - rewards.unsqueeze(1)) / (std + epsilon).unsqueeze(2)


@torch.no_grad()
def normalized_pairwise_advantages(batch, *, ell, tau, epsilon=1e-4):
    r"""A_i = sum_j sign(z_ij) min((|z_ij|-ell)_+, tau) / G.

    Unit slope, cap=tau, divisor=G. No division by c, multiplication by 2.46,
    or post-hoc group standardization. ell=0,tau=inf restores standard GRPO.
    """
    _thresholds(ell, tau)
    z = standardized_differences(batch, epsilon)
    if ell == 0 and math.isinf(tau):
        # Match the existing GRPO path exactly for its usual fp32/fp64 inputs.
        if batch.rewards.dtype in (torch.float32, torch.float64):
            return (batch.rewards - batch.group_mean) / (batch.group_std + epsilon)
        return ((batch.rewards.double() - batch.group_mean.double())
                / (batch.group_std.double() + epsilon)).to(batch.rewards.dtype)
    strength = (z.abs() - ell).clamp(min=0, max=tau)
    return (z.sign() * strength).sum(2).div(batch.rewards.shape[1]).to(batch.rewards.dtype)


class RollingQuantilePairwise:
    """One rolling window of GLOBAL rollouts, identical on every policy rank.

    p,q are explicit hyperparameters: lower p quantile, upper (1-q) quantile.
    Thresholds use only previously observed rollouts; all fresh rollouts within
    an optimizer step share the same thresholds. With no history, use GRPO.
    Evaluation uses frozen thresholds and does not change any training state.
    """

    state_filename = "rolling_advantage_state.json"

    def __init__(self, *, p, q, window_size=32, epsilon=1e-4):
        _real("p", p)
        _real("q", q)
        if p + q >= 1:
            raise ValueError("p + q must be less than 1")
        if type(window_size) is not int or window_size < 1:
            raise ValueError("window_size must be a positive integer (global rollouts)")
        _real("epsilon", epsilon, strict=True)
        self.p, self.q, self.window_size, self.epsilon = p, q, window_size, epsilon
        self._history = deque(maxlen=window_size)
        self._step = None
        self._active = None
        self.last_metrics = {}
        self.resume_loaded = False
        self.checkpoint_global_step = None

    def _fit(self):
        if not self._history:
            return {"ell": 0., "tau": math.inf, "history_rollouts": 0, "history_pairs": 0}
        gaps = torch.cat(list(self._history))
        ell, upper = torch.quantile(gaps, torch.tensor([self.p, 1 - self.q], dtype=torch.float64)).tolist()
        return {"ell": ell, "tau": max(0., upper - ell),
                "history_rollouts": len(self._history), "history_pairs": gaps.numel()}

    @torch.no_grad()
    def __call__(self, batch, *, step=None, update_history=True):
        if not isinstance(update_history, bool):
            raise TypeError("update_history must be boolean")
        z = standardized_differences(batch, self.epsilon)
        i, j = torch.triu_indices(z.shape[1], z.shape[1], offset=1, device=z.device)
        gaps = z[:, i, j].abs().flatten().cpu()
        if update_history:
            step = (0 if self._step is None else self._step + 1) if step is None else step
            if type(step) is not int or step < 0 or (self._step is not None and step < self._step):
                raise ValueError("Training step must be a nonnegative, nondecreasing integer")
            if self._step != step:
                self._active, self._step = self._fit(), step
            active = self._active
        else:
            active = self._active if self._active is not None else self._fit()
        ell, tau = active["ell"], active["tau"]
        advantage = normalized_pairwise_advantages(batch, ell=ell, tau=tau, epsilon=self.epsilon)
        # Ratios measure this rollout using thresholds from older rollouts.
        # Degenerate tau=0 is valid and yields zero signal; masks stay disjoint.
        self.last_metrics = {
            "ell": ell, "history_rollouts": active["history_rollouts"],
            "history_pairs": active["history_pairs"], "warmup_grpo": float(math.isinf(tau)),
            "degenerate_thresholds": float(tau == 0),
            "dead_pair_fraction": (gaps <= ell).double().mean().item(),
            "saturated_pair_fraction": ((gaps > ell) & (gaps >= ell + tau)).double().mean().item(),
            "final_rms": advantage.double().square().mean().sqrt().item(),
            "zero_group_fraction": (advantage == 0).all(1).double().mean().item(),
        }
        # Trainer JSON logs must not contain infinite numeric values in warmup.
        if math.isfinite(tau):
            self.last_metrics.update(tau=tau, upper_threshold=ell + tau)
        if update_history:
            self._history.append(gaps.detach().clone())
        return advantage

    def state_dict(self):
        active = dict(self._active) if self._active is not None else None
        if active is not None and math.isinf(active["tau"]):
            active["tau"] = None  # JSON representation of the GRPO warmup cap.
        return {"version": 1, "config": {"p": self.p, "q": self.q,
                "window_size": self.window_size, "epsilon": self.epsilon},
                "step": self._step, "active": active,
                "history": [gaps.tolist() for gaps in self._history]}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("config") != self.state_dict()["config"]:
            raise ValueError("Rolling advantage state version/config does not match this estimator")
        history = state.get("history")
        if not isinstance(history, list) or len(history) > self.window_size:
            raise ValueError("Invalid rolling advantage history")
        tensors = [torch.tensor(row, dtype=torch.float64) for row in history]
        if any(t.ndim != 1 or not t.numel() or not torch.isfinite(t).all() or (t < 0).any() for t in tensors):
            raise ValueError("History must contain finite nonnegative absolute pair gaps")
        step, active = state.get("step"), state.get("active")
        if step is not None and (type(step) is not int or step < 0):
            raise ValueError("Invalid saved optimizer step")
        if (step is None) != (active is None) or (step is None) != (len(tensors) == 0):
            raise ValueError("Inconsistent saved rolling advantage state")
        if active is not None:
            active = dict(active)
            active["tau"] = math.inf if active["tau"] is None else active["tau"]
            _thresholds(active["ell"], active["tau"])
            for key in ("history_rollouts", "history_pairs"):
                if type(active.get(key)) is not int or active[key] < 0:
                    raise ValueError("Invalid saved history counts")
        self._history = deque(tensors, maxlen=self.window_size)
        self._step, self._active, self.last_metrics = step, active, {}
        self.resume_loaded = True
        self.checkpoint_global_step = None

    def save(self, directory, *, global_step=None):
        if global_step is not None and (type(global_step) is not int or global_step < 0):
            raise ValueError("Checkpoint global_step must be a nonnegative integer")
        path = Path(directory) / self.state_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        state = self.state_dict()
        if global_step is not None:
            state["checkpoint_global_step"] = global_step
        temporary.write_text(json.dumps(state, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def load(self, directory):
        path = Path(directory) / self.state_filename
        if not path.is_file():
            raise ValueError(f"Missing rolling advantage history: {path}; cannot resume with an empty window")
        state = json.loads(path.read_text(encoding="utf-8"))
        global_step = state.get("checkpoint_global_step")
        if global_step is not None and (type(global_step) is not int or global_step < 0):
            raise ValueError("Invalid saved checkpoint global_step")
        self.load_state_dict(state)
        self.checkpoint_global_step = global_step
