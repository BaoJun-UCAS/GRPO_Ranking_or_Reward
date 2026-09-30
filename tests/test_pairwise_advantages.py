"""Numerical contracts for the attachment's uncertainty-aware pairwise advantage.

These use Torch on CPU, independent scalar examples and mathematical properties;
no models, GPU kernels or external reward services are needed.
"""

import pytest


torch = pytest.importorskip("torch")

from open_r1.advantages import compute_advantages, configure_advantage


def pairwise(rewards, group_size, **options):
    return compute_advantages(
        rewards, group_size, method="robust_pairwise", method_kwargs=options,
    ).advantages


@pytest.mark.parametrize("rewards, expected", [
    ([0.100, 0.101, 0.102, 0.103], [0.0, 0.0, 0.0, 0.0]),
    ([0.0, 0.1, 0.2, 0.3], [-23 / 30, -0.3, 0.3, 23 / 30]),
    ([0.0, 0.1, 0.2, 10.0], [-23 / 30, -1 / 3, 0.1, 1.0]),
])
def test_three_numerical_examples_from_the_attachment(rewards, expected):
    actual = pairwise(torch.tensor(rewards, dtype=torch.float64), 4, delta=0.02, c=0.2)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_documented_example_thresholds_are_the_defaults():
    rewards = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    torch.testing.assert_close(pairwise(rewards, 4), pairwise(rewards, 4, delta=0.02, c=0.2))


@pytest.mark.parametrize("difference, signal", [
    (0.0, 0.0), (0.0625, 0.0), (0.125, 0.0),
    (0.25, 0.5), (0.375, 1.0), (0.75, 1.0),
])
def test_exact_dead_zone_and_saturation_boundaries(difference, signal):
    # Binary fractions make the <= delta and >= delta+c boundaries exact.
    rewards = torch.tensor([0.0, difference], dtype=torch.float64)
    actual = pairwise(rewards, 2, delta=0.125, c=0.25)
    torch.testing.assert_close(actual, torch.tensor([-signal, signal], dtype=torch.float64), atol=0, rtol=0)


@pytest.mark.parametrize("group_size", [2, 3, 4, 7, 16])
def test_centering_bound_order_and_permutation_equivariance(group_size):
    generator = torch.Generator().manual_seed(789 + group_size)
    rewards = torch.randn(9, group_size, generator=generator, dtype=torch.float64)
    rewards[0].fill_(3.0)
    if group_size > 2:
        rewards[1, 1] = rewards[1, 0]
    actual = pairwise(rewards.flatten(), group_size, delta=0.11, c=0.9).reshape_as(rewards)
    torch.testing.assert_close(actual.sum(dim=1), torch.zeros(9, dtype=torch.float64), atol=1e-12, rtol=0)
    assert (actual.abs() <= 1).all()
    order = rewards.argsort(dim=1)
    sorted_advantages = actual.gather(1, order)
    assert (sorted_advantages[:, 1:] >= sorted_advantages[:, :-1] - 1e-12).all()
    permutation = torch.randperm(group_size, generator=generator)
    permuted = pairwise(rewards[:, permutation].flatten(), group_size, delta=0.11, c=0.9).reshape_as(rewards)
    torch.testing.assert_close(permuted, actual[:, permutation], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("scale", [0.01, 3.5, 10000.0])
def test_invariant_to_translation_and_joint_scaling_of_rewards_and_thresholds(scale):
    rewards = torch.tensor([-0.1, 0.0, 0.025, 0.4, 1.1, 1.2, 1.23, 1.6], dtype=torch.float64)
    expected = pairwise(rewards, 4, delta=0.02, c=0.2)
    translated = pairwise(rewards + 17.25, 4, delta=0.02, c=0.2)
    scaled = pairwise(rewards * scale, 4, delta=0.02 * scale, c=0.2 * scale)
    torch.testing.assert_close(translated, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(scaled, expected, atol=1e-12, rtol=1e-12)


def test_scaling_only_rewards_does_change_the_signal():
    rewards = torch.tensor([0.0, 0.01, 0.02, 0.03], dtype=torch.float64)
    small = pairwise(rewards, 4, delta=0.02, c=0.2)
    large = pairwise(rewards * 10, 4, delta=0.02, c=0.2)
    assert not torch.allclose(small, large)


def test_without_dead_zone_or_saturation_is_scaled_centered_reward():
    rewards = torch.tensor([[0.0, 0.2, 0.5, 0.9], [-2.0, -1.9, -1.8, -1.7]], dtype=torch.float64)
    c = 4.0  # Larger than every within-group difference.
    actual = pairwise(rewards.flatten(), 4, delta=0.0, c=c).reshape_as(rewards)
    expected = 4 / (3 * c) * (rewards - rewards.mean(dim=1, keepdim=True))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("rewards", [
    [0.0, 0.1, 0.2, 10.0],
    [3.0, 1.0, 3.0, 2.0],
    [8.0, 8.0, 8.0, 8.0],
])
def test_small_c_zero_delta_limit_is_half_the_existing_ranking(rewards):
    rewards = torch.tensor(rewards, dtype=torch.float64)
    expected = 0.5 * compute_advantages(rewards, 4, method="ranking").advantages
    actual = pairwise(rewards, 4, delta=0.0, c=1e-8)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_positive_delta_small_c_limit_treats_close_pairs_as_ties():
    rewards = torch.tensor([0.0, 0.125, 0.25], dtype=torch.float64)
    actual = pairwise(rewards, 3, delta=0.125, c=1e-8)
    # Only the first/last pair exceeds delta; equal-to-delta pairs abstain.
    torch.testing.assert_close(actual, torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64), atol=0, rtol=0)


def test_equal_scores_get_equal_advantages_and_all_ties_get_zero():
    rewards = torch.tensor([0.0, 0.1, 0.1, 0.8, 4.0, 4.0, 4.0, 4.0], dtype=torch.float64)
    actual = pairwise(rewards, 4, delta=0.02, c=0.2)
    assert actual[1] == actual[2]
    torch.testing.assert_close(actual[4:], torch.zeros(4, dtype=torch.float64), atol=0, rtol=0)


@pytest.mark.parametrize("scale_rewards", [False, True])
def test_weak_evidence_is_not_renormalized_by_trainer_scale_rewards(scale_rewards):
    rewards = torch.tensor([0.0, 0.001, 0.002, 0.003], dtype=torch.float64)
    estimator, options = configure_advantage(
        "robust_pairwise", {"delta": 0.0, "c": 0.2}, scale_rewards=scale_rewards,
    )
    actual = compute_advantages(rewards, 4, method=estimator, method_kwargs=options).advantages
    expected = torch.tensor([-0.01, -1 / 300, 1 / 300, 0.01], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert actual.abs().max() < 0.011


def test_comparisons_do_not_cross_global_prompt_groups_or_rank_boundaries():
    # Each worker owns six completions, splitting the second prompt's group.
    rank_zero = torch.tensor([0.100, 0.101, 0.102, 0.103, 0.0, 0.1], dtype=torch.float64)
    rank_one = torch.tensor([0.2, 0.3, 0.0, 0.1, 0.2, 10.0], dtype=torch.float64)
    all_rewards = torch.cat((rank_zero, rank_one))
    actual = pairwise(all_rewards, 4, delta=0.02, c=0.2)
    expected = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, -23 / 30, -0.3, 0.3, 23 / 30, -23 / 30, -1 / 3, 0.1, 1.0],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual[:6], expected[:6], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual[6:], expected[6:], atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="complete generation groups"):
        pairwise(rank_zero, 4, delta=0.02, c=0.2)


def test_reward_autograd_graph_is_detached_and_input_is_not_mutated():
    rewards = torch.tensor([0.0, 0.1, 0.2, 0.3], requires_grad=True)
    original = rewards.detach().clone()
    result = compute_advantages(rewards, 4, method="robust_pairwise", method_kwargs={"delta": 0.02, "c": 0.2})
    assert not result.advantages.requires_grad
    assert result.advantages.grad_fn is None
    assert not result.group_mean.requires_grad
    assert not result.group_std.requires_grad
    torch.testing.assert_close(rewards.detach(), original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_rewards_are_finite_and_keep_their_dtype(dtype):
    # c=1e-8 rounds to zero in float16; diagonal/true-tie comparisons must
    # remain zero instead of becoming 0/0 when the arithmetic is promoted.
    rewards = torch.tensor([0.0, 1.0, 1.0, 3.0], dtype=dtype)
    actual = pairwise(rewards, 4, delta=0.0, c=1e-8)
    expected = torch.tensor([-1.0, 0.0, 0.0, 1.0], dtype=dtype)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("delta", [-0.1, float("nan"), float("inf"), -float("inf"), True, "0.02", None])
def test_invalid_delta_is_rejected(delta):
    with pytest.raises((TypeError, ValueError)):
        pairwise(torch.tensor([0.0, 0.1]), 2, delta=delta, c=0.2)


@pytest.mark.parametrize("c", [0.0, -0.1, float("nan"), float("inf"), -float("inf"), True, "0.2", None])
def test_invalid_c_is_rejected(c):
    with pytest.raises((TypeError, ValueError)):
        pairwise(torch.tensor([0.0, 0.1]), 2, delta=0.02, c=c)


def test_saturated_outlier_cannot_rescale_other_comparisons():
    rewards = torch.tensor([0.0, 0.1, 0.2, 10.0], dtype=torch.float64)
    larger_outlier = rewards.clone()
    larger_outlier[-1] = 1000000
    torch.testing.assert_close(
        pairwise(rewards, 4, delta=0.02, c=0.2), pairwise(larger_outlier, 4, delta=0.02, c=0.2),
        atol=0, rtol=0,
    )


def test_single_outlier_has_bounded_influence_on_unchanged_responses():
    rewards = torch.tensor([-0.5, -0.1, 0.0, 0.2, 0.6, 0.9, 1.1], dtype=torch.float64)
    perturbed = rewards.clone()
    perturbed[2] = 10000
    before = pairwise(rewards, 7, delta=0.02, c=0.2)
    after = pairwise(perturbed, 7, delta=0.02, c=0.2)
    unchanged = torch.arange(7) != 2
    assert ((after - before)[unchanged].abs() <= 2 / 6 + 1e-12).all()


def test_standard_grpo_alias_preserves_unbiased_std_and_original_epsilon():
    rewards = torch.tensor([0.0, 1.0, 2.0, 4.0, 10.0, 11.0, 12.0, 14.0], dtype=torch.float64)
    actual = compute_advantages(rewards, 4, method="grpo").advantages
    legacy = compute_advantages(rewards, 4, method="studentization").advantages
    grouped = rewards.reshape(2, 4)
    expected = (grouped - grouped.mean(dim=1, keepdim=True)) / (grouped.std(dim=1, keepdim=True, unbiased=True) + 1e-4)
    torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected.flatten(), atol=0, rtol=0)


def test_grpo_alias_keeps_top_level_scale_rewards_false_compatibility():
    rewards = torch.tensor([0.0, 2.0, 4.0, 18.0], dtype=torch.float64)
    estimator, options = configure_advantage("grpo", scale_rewards=False)
    actual = compute_advantages(rewards, 4, method=estimator, method_kwargs=options).advantages
    torch.testing.assert_close(actual, rewards - rewards.mean(), atol=0, rtol=0)


@pytest.mark.parametrize("options", [
    {"delta": -0.1}, {"c": 0.0}, {"delta": float("nan")}, {"c": float("inf")},
    {"epsilon": 0.001}, {"scale_rewards": True},
])
def test_invalid_or_normalizing_options_fail_before_loading_models(options):
    with pytest.raises(ValueError, match="robust_pairwise"):
        configure_advantage("robust_pairwise", options)
