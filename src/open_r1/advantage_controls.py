"""Isolated scale/weight controls; the original advantage estimators are unchanged."""

import math
from numbers import Real

import torch

from .advantages import AdvantageBatch, robust_pairwise, studentization


def validate_scale(k):
    if isinstance(k, bool) or not isinstance(k, Real) or not math.isfinite(k) or k <= 0:
        raise ValueError("k must be a finite positive real number")


@torch.no_grad()
def control_components(batch: AdvantageBatch, *, delta=.002, c=.08, k=2.46, epsilon=1e-4):
    """Compute q from THIS rollout's complete groups, never from historic CSVs.

    The standard GRPO branch uses the same unbiased std and epsilon as before.
    RMS/ratios use float64; zero-GRPO groups receive q=0 without epsilon bias.
    Both outputs retain the original reward dtype/device and stop gradients.
    """
    validate_scale(k)
    grpo = studentization(batch, epsilon=epsilon, scale_rewards=True).detach()
    robust = robust_pairwise(batch, delta=delta, c=c).detach()
    grpo_rms = grpo.double().square().mean(1, keepdim=True).sqrt()
    robust_rms = robust.double().square().mean(1, keepdim=True).sqrt()
    nonzero = grpo_rms > 0
    denominator = torch.where(nonzero, grpo_rms, torch.ones_like(grpo_rms))
    q = torch.where(nonzero, robust_rms / denominator, torch.zeros_like(grpo_rms)).detach()
    return {
        "grpo": grpo, "robust": robust, "q": q,
        "robust_scaled": (k * robust.double()).to(batch.rewards.dtype),
        "weight_only_scaled": (k * q * grpo.double()).to(batch.rewards.dtype),
    }


def robust_scaled(batch: AdvantageBatch, *, delta=.002, c=.08, k=2.46, epsilon=1e-4):
    """Experiment A: k * A_robust, without subsequent standardization."""
    return control_components(batch, delta=delta, c=c, k=k, epsilon=epsilon)["robust_scaled"]


def weight_only_scaled(batch: AdvantageBatch, *, delta=.002, c=.08, k=2.46, epsilon=1e-4):
    """Experiment B: k * q_g * A_GRPO; same group RMS as A on fixed rewards."""
    return control_components(batch, delta=delta, c=c, k=k, epsilon=epsilon)["weight_only_scaled"]
