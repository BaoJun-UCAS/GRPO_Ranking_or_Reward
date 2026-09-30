"""Loss reductions shared by GRPO experiments, independent of rollout backends."""

import torch


def reduce_policy_loss(
    per_token_loss: torch.Tensor,
    completion_mask: torch.Tensor,
    loss_type: str,
    max_completion_length: int,
) -> torch.Tensor:
    """Reduce masked token losses using the selected TRL 0.18 objective.

    Empty (fully truncated) rows contribute zero. An entirely masked batch
    produces a differentiable zero rather than NaN, so distributed ranks can
    still participate in backward together.
    """
    masked_loss = per_token_loss * completion_mask
    if loss_type == "grpo":
        return (masked_loss.sum(-1) / completion_mask.sum(-1).clamp(min=1)).mean()
    if loss_type == "bnpo":
        return masked_loss.sum() / completion_mask.sum().clamp(min=1)
    if loss_type == "dr_grpo":
        if max_completion_length <= 0:
            raise ValueError("max_completion_length must be positive")
        return masked_loss.sum() / (per_token_loss.shape[0] * max_completion_length)
    raise ValueError(f"Unknown loss_type: {loss_type!r}; choose grpo, bnpo or dr_grpo")
