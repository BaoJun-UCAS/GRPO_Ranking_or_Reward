"""Pluggable sequence advantages, computed on complete, globally gathered groups.

An estimator receives an :class:`AdvantageBatch` and returns a floating tensor
with shape ``(num_prompts, num_generations)`` on the same device. Register one
with ``@register_advantage("my_method")`` or configure a Python import path such
as ``advantage: my_package.estimators:my_method``. ``advantage_kwargs`` are passed
as keyword arguments, so new experiments do not need trainer changes.
"""

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from numbers import Real
from typing import Any

import torch

from .rolling_advantages import RollingQuantilePairwise


@dataclass(frozen=True)
class AdvantageBatch:
    """Complete groups in sampler order; no distributed operations are needed.

    ``rewards`` is the weighted aggregate, shaped ``(P, G)``. Optional reward
    components have shape ``(P, G, R)`` and may contain NaNs for inapplicable
    reward functions; weights have shape ``(R,)``. Means and unbiased standard
    deviations have shape ``(P, 1)``. Estimators must not mutate these tensors.
    """

    rewards: torch.Tensor
    group_mean: torch.Tensor
    group_std: torch.Tensor
    rewards_per_func: torch.Tensor | None = None
    reward_weights: torch.Tensor | None = None


@dataclass(frozen=True)
class AdvantageResult:
    """Flat advantages in original global order and per-group reward metrics."""

    advantages: torch.Tensor
    group_mean: torch.Tensor
    group_std: torch.Tensor


AdvantageEstimator = Callable[..., torch.Tensor]
_ESTIMATORS: dict[str, AdvantageEstimator] = {}


def register_advantage(name: str) -> Callable[[AdvantageEstimator], AdvantageEstimator]:
    """Register an estimator; fail on duplicate names to avoid silent changes."""
    if not isinstance(name, str) or not name or ":" in name:
        raise ValueError("An advantage registry name must be non-empty and cannot contain ':'")

    def register(estimator: AdvantageEstimator) -> AdvantageEstimator:
        if not callable(estimator):
            raise TypeError("An advantage estimator must be callable")
        if name in _ESTIMATORS:
            raise ValueError(f"Advantage estimator {name!r} is already registered")
        _ESTIMATORS[name] = estimator
        return estimator

    return register


def resolve_advantage(method: str | AdvantageEstimator) -> AdvantageEstimator:
    """Resolve a built-in/registered name, ``module:callable`` path, or callable."""
    if callable(method):
        return method
    if not isinstance(method, str):
        raise TypeError("Advantage method must be a name, module:callable path, or callable")
    if method in _ESTIMATORS:
        return _ESTIMATORS[method]
    if ":" in method:
        module_name, attribute = method.split(":", 1)
        if not module_name or not attribute:
            raise ValueError("Custom advantage paths must use module:callable syntax")
        estimator = getattr(import_module(module_name), attribute)
        if not callable(estimator):
            raise TypeError(f"Custom advantage estimator {method!r} is not callable")
        return estimator
    raise ValueError(
        f"Unknown advantage method {method!r}. Available: {', '.join(sorted(_ESTIMATORS))}; "
        "or use a module:callable import path."
    )


def parse_advantage_kwargs(value: Mapping[str, Any] | str | None) -> dict[str, Any]:
    """Accept YAML mappings and JSON strings supplied by the CLI parser."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("advantage_kwargs must be a valid JSON object") from error
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError("advantage_kwargs must be a mapping with string keys")
    return dict(value)


@register_advantage("grpo")
@register_advantage("studentization")
def studentization(
    batch: AdvantageBatch, *, epsilon: float = 1e-4, scale_rewards: bool = True
) -> torch.Tensor:
    """Center rewards, optionally divide by their original unbiased group std."""
    if not isinstance(scale_rewards, bool):
        raise TypeError("scale_rewards must be a boolean")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("studentization epsilon must be finite and greater than zero")
    centered = batch.rewards - batch.group_mean
    return centered / (batch.group_std + epsilon) if scale_rewards else centered


@register_advantage("ranking")
def ranking(batch: AdvantageBatch) -> torch.Tensor:
    """Map ascending average ranks to [-2, 2], assigning equal rewards equal ranks.

    Without ties this equals the original double-argsort formula. Average ranks
    keep every group's mean at zero, including an all-equal group's zero signal.
    Sorting and cumulative scans use O(G log G) time and O(G) temporary memory
    per group, avoiding a pairwise G-by-G comparison tensor.
    """
    sorted_rewards, order = batch.rewards.sort(dim=1)
    group_size = sorted_rewards.shape[1]
    positions = torch.arange(group_size, device=order.device).expand_as(order)
    changed = sorted_rewards[:, 1:] != sorted_rewards[:, :-1]
    first = torch.ones_like(sorted_rewards[:, :1], dtype=torch.bool)
    starts = torch.cat((first, changed), dim=1)
    ends = torch.cat((changed, first), dim=1)
    start_positions = torch.where(starts, positions, 0).cummax(dim=1).values
    end_positions = torch.where(ends, positions, group_size - 1).flip(1).cummin(dim=1).values.flip(1)
    average_ranks = (start_positions + end_positions).to(batch.rewards.dtype) * 0.5
    ranks = torch.empty_like(batch.rewards).scatter_(1, order, average_ranks)
    return 4.0 * ranks / (group_size - 1) - 2.0


@register_advantage("rank_reward")
def rank_reward(
    batch: AdvantageBatch, *, rank_weight: float = 0.5, epsilon: float = 1e-4, scale_rewards: bool = True
) -> torch.Tensor:
    """Example experiment: convex mix of raw ranking and normalized reward.

    ``rank_weight=0`` reproduces studentization; ``rank_weight=1`` reproduces
    ranking. This intentionally retains each method's original scale.
    ``scale_rewards=False`` disables std scaling in the reward branch only.
    """
    if not math.isfinite(rank_weight) or not 0 <= rank_weight <= 1:
        raise ValueError("rank_weight must be finite and between 0 and 1")
    reward_advantage = studentization(batch, epsilon=epsilon, scale_rewards=scale_rewards)
    return (1 - rank_weight) * reward_advantage + rank_weight * ranking(batch)


def _validate_pairwise_thresholds(delta: float, c: float) -> None:
    for name, value in (("delta", delta), ("c", c)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"robust_pairwise {name} must be a finite real number")
    if delta < 0:
        raise ValueError("robust_pairwise delta must be non-negative")
    if c <= 0:
        raise ValueError("robust_pairwise c must be greater than zero")


@register_advantage("robust_pairwise")
def robust_pairwise(batch: AdvantageBatch, *, delta: float = 0.02, c: float = 0.2) -> torch.Tensor:
    r"""Uncertainty-constrained pairwise advantages on the original reward scale.

    A_i = sum_j sign(r_i-r_j) * clamp((abs(r_i-r_j)-delta)/c, 0, 1) / (G-1).

    The diagonal contributes zero. Each group is centered by antisymmetry and
    bounded by [-1, 1], including ties. No group standardization follows: doing
    so would amplify precisely the weak evidence this estimator suppresses.
    The fixed defaults are example hyperparameters, not a noise calibration.
    ``delta`` is a reward-difference threshold, unrelated to PPO ratio clipping.

    Arithmetic and temporary storage are O(P*G**2), with no extra model calls.
    Use double precision for this small comparison tensor to avoid threshold
    underflow in half precision and overflow of float32 reward differences.
    The result is cast back to the input dtype, as required by the interface.
    """
    _validate_pairwise_thresholds(delta, c)
    rewards = batch.rewards.to(dtype=torch.float64)
    differences = rewards.unsqueeze(2) - rewards.unsqueeze(1)
    strengths = ((differences.abs() - delta).clamp(min=0) / c).clamp(max=1)
    pairwise = differences.sign() * strengths
    return (pairwise.sum(dim=2) / (rewards.shape[1] - 1)).to(dtype=batch.rewards.dtype)


def configure_advantage(
    method: str | AdvantageEstimator,
    method_kwargs: Mapping[str, Any] | str | None = None,
    *,
    scale_rewards: bool = True,
) -> tuple[AdvantageEstimator, dict[str, Any]]:
    """Bind trainer reward scaling to built-ins without altering custom APIs.

    The top-level ``scale_rewards`` setting and an explicit estimator keyword
    must agree. Raising on conflict prevents a CLI/YAML override from silently
    changing the intended experiment. Ranking has no reward scaling branch.
    """
    estimator = resolve_advantage(method)
    options = parse_advantage_kwargs(method_kwargs)
    if estimator is RollingQuantilePairwise:
        if scale_rewards is not True:
            raise ValueError("rolling_quantile_pairwise requires scale_rewards=True for its standardized differences")
        # Construct once per trainer; never reset historical state per batch.
        return RollingQuantilePairwise(**options), {}
    if estimator in (studentization, rank_reward):
        if not isinstance(scale_rewards, bool):
            raise TypeError("scale_rewards must be a boolean")
        if "scale_rewards" in options:
            if not isinstance(options["scale_rewards"], bool):
                raise TypeError("advantage_kwargs.scale_rewards must be a boolean")
            if options["scale_rewards"] != scale_rewards:
                raise ValueError(
                    "advantage_kwargs.scale_rewards conflicts with the top-level scale_rewards setting; "
                    "set scale_rewards once at the top level or make both values agree"
                )
        options["scale_rewards"] = scale_rewards
    if estimator is robust_pairwise:
        unknown = set(options) - {"delta", "c"}
        if unknown:
            raise ValueError(f"Unknown robust_pairwise options: {', '.join(sorted(unknown))}; use delta and c")
        _validate_pairwise_thresholds(options.get("delta", 0.02), options.get("c", 0.2))
    return estimator, options


def compute_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    *,
    method: str | AdvantageEstimator = "studentization",
    method_kwargs: Mapping[str, Any] | str | None = None,
    rewards_per_func: torch.Tensor | None = None,
    reward_weights: torch.Tensor | None = None,
) -> AdvantageResult:
    """Compute and validate advantages *before* slicing by distributed rank.

    The caller must gather rewards from all ranks in sampler order first; one
    prompt's group may cross rank boundaries. Inputs are detached, and returned
    advantages are detached and retain the aggregate rewards' dtype/device.
    Aggregate rewards must be finite; handle missing components during reward
    aggregation, not by silently changing group membership here.
    """
    if not isinstance(num_generations, int) or isinstance(num_generations, bool) or num_generations < 2:
        raise ValueError("num_generations must be an integer of at least 2")
    if rewards.ndim != 1 or rewards.numel() == 0 or rewards.numel() % num_generations:
        raise ValueError("rewards must be a non-empty flat tensor containing complete generation groups")
    if not rewards.is_floating_point():
        raise TypeError("rewards must be a floating-point tensor")
    if not torch.isfinite(rewards).all():
        raise ValueError("Aggregated rewards must be finite")
    method_kwargs = parse_advantage_kwargs(method_kwargs)

    grouped = rewards.detach().reshape(-1, num_generations)
    components = None
    if rewards_per_func is not None:
        if rewards_per_func.ndim != 2 or rewards_per_func.shape[0] != rewards.numel():
            raise ValueError("rewards_per_func must have shape (num_completions, num_reward_functions)")
        if rewards_per_func.device != rewards.device:
            raise ValueError("rewards_per_func and rewards must use the same device")
        components = rewards_per_func.detach().reshape(*grouped.shape, rewards_per_func.shape[1])
    if reward_weights is not None:
        if components is None or reward_weights.shape != (components.shape[2],):
            raise ValueError("reward_weights must match the reward component dimension")
        if reward_weights.device != rewards.device:
            raise ValueError("reward_weights and rewards must use the same device")
        reward_weights = reward_weights.detach()

    batch = AdvantageBatch(
        rewards=grouped,
        group_mean=grouped.mean(dim=1, keepdim=True),
        group_std=grouped.std(dim=1, keepdim=True, unbiased=True),
        rewards_per_func=components,
        reward_weights=reward_weights,
    )
    estimator = resolve_advantage(method)
    if estimator is RollingQuantilePairwise:
        raise ValueError("rolling_quantile_pairwise is stateful; use configure_advantage once and reuse its estimator")
    advantages = estimator(batch, **method_kwargs)
    if not isinstance(advantages, torch.Tensor) or advantages.shape != grouped.shape:
        raise ValueError(f"Advantage estimator must return a tensor of shape {tuple(grouped.shape)}")
    if not advantages.is_floating_point() or advantages.device != rewards.device:
        raise ValueError("Advantage estimator must return floating-point values on the rewards' device")
    advantages = advantages.detach().to(dtype=rewards.dtype)
    if not torch.isfinite(advantages).all():
        raise ValueError("Advantage estimator returned non-finite values")
    return AdvantageResult(
        advantages=advantages.reshape(-1),
        group_mean=batch.group_mean.squeeze(1),
        group_std=batch.group_std.squeeze(1),
    )


register_advantage("rolling_quantile_pairwise")(RollingQuantilePairwise)
