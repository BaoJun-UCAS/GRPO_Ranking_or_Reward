"""Numerical objective contracts; no weights, CUDA, or network required."""

import pytest

torch = pytest.importorskip("torch")

from open_r1.objectives import reduce_policy_loss


@pytest.mark.parametrize("kind, expected", [("grpo", 3.0), ("bnpo", 3.5), ("dr_grpo", 1.75)])
def test_distinct_normalizations_with_unequal_lengths(kind, expected):
    loss = torch.tensor([[2.0, 99.0, 99.0], [4.0, 4.0, 4.0]], requires_grad=True)
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
    result = reduce_policy_loss(loss, mask, kind, 4)
    assert result.item() == pytest.approx(expected)
    result.backward()
    assert loss.grad[0, 1:].eq(0).all()
    assert torch.isfinite(loss.grad).all()


@pytest.mark.parametrize("kind", ["grpo", "bnpo", "dr_grpo"])
def test_fully_masked_batch_has_finite_zero_gradient(kind):
    loss = torch.randn(2, 3, requires_grad=True)
    result = reduce_policy_loss(loss, torch.zeros_like(loss), kind, 3)
    assert result.item() == 0
    result.backward()
    assert loss.grad.eq(0).all()


def test_unknown_objective_does_not_silently_use_bnpo():
    with pytest.raises(ValueError, match="Unknown loss_type"):
        reduce_policy_loss(torch.ones(1, 1), torch.ones(1, 1), "typo", 1)
