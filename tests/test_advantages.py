"""Numerical CPU tests for global-group and custom advantage estimators."""

import pytest


torch = pytest.importorskip("torch")

from open_r1 import advantages as module
from open_r1.advantages import compute_advantages, parse_advantage_kwargs, register_advantage, resolve_advantage


def test_studentization_matches_legacy_formula_and_unbiased_std():
    rewards = torch.tensor([0.0, 1.0, 5.0, 3.0, 40.0, 42.0, 45.0, 55.0], dtype=torch.float64)
    grouped = rewards.reshape(2, 4)
    expected_std = grouped.std(dim=1, unbiased=True)
    expected = (grouped - grouped.mean(dim=1, keepdim=True)) / (expected_std[:, None] + 1e-4)
    result = compute_advantages(rewards, 4)
    torch.testing.assert_close(result.advantages, expected.reshape(-1))
    torch.testing.assert_close(result.group_std, expected_std)
    torch.testing.assert_close(result.group_mean, torch.tensor([2.25, 45.5], dtype=torch.float64))
    assert result.advantages.dtype == rewards.dtype


def test_global_groups_cross_rank_boundaries_without_changing_advantages():
    # Two workers each own six samples; group 1 crosses their boundary.
    rank_zero = torch.tensor([0.0, 2.0, 4.0, 6.0, 10.0, 12.0])
    rank_one = torch.tensor([14.0, 16.0, 20.0, 22.0, 24.0, 26.0])
    result = compute_advantages(torch.cat((rank_zero, rank_one)), 4)
    expected_group = torch.tensor([-3.0, -1.0, 1.0, 3.0]) / (torch.tensor(20.0 / 3.0).sqrt() + 1e-4)
    torch.testing.assert_close(result.advantages[:6], expected_group.repeat(2)[:6])
    torch.testing.assert_close(result.advantages[6:], expected_group.repeat(2)[2:])
    with pytest.raises(ValueError, match="complete generation groups"):
        compute_advantages(rank_zero, 4)


def test_ranking_matches_legacy_formula_when_rewards_are_distinct():
    rewards = torch.tensor([4.0, 9.0, 2.0, 0.0, -1.0, -2.0, 9.0, 0.0])
    legacy_ranks = rewards.reshape(2, 4).argsort(dim=1, descending=True).argsort(dim=1) + 1
    legacy = 2.0 - 4.0 * (legacy_ranks.float() - 1.0) / 3
    result = compute_advantages(rewards, 4, method="ranking")
    torch.testing.assert_close(result.advantages, legacy.reshape(-1))


def test_ranking_uses_average_ranks_and_is_permutation_equivariant():
    rewards = torch.tensor([3.0, 1.0, 3.0, 2.0])
    result = compute_advantages(rewards, 4, method="ranking").advantages
    expected = torch.tensor([4.0 / 3, -2.0, 4.0 / 3, -2.0 / 3])
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result.mean(), torch.tensor(0.0), atol=1e-7, rtol=0)
    permutation = torch.tensor([2, 3, 0, 1])
    shuffled = compute_advantages(rewards[permutation], 4, method="ranking").advantages
    torch.testing.assert_close(shuffled, result[permutation])


@pytest.mark.parametrize("method", ["studentization", "ranking", "rank_reward"])
def test_constant_reward_group_has_no_policy_signal(method):
    result = compute_advantages(torch.tensor([5.0] * 4 + [-2.0] * 4), 4, method=method)
    torch.testing.assert_close(result.advantages, torch.zeros(8), atol=0, rtol=0)
    torch.testing.assert_close(result.group_std, torch.zeros(2))


@pytest.mark.parametrize("rank_weight", [0.0, 0.3, 1.0])
def test_rank_reward_interpolates_the_documented_methods(rank_weight):
    rewards = torch.tensor([1.0, 2.0, 2.0, 20.0])
    student = compute_advantages(rewards, 4).advantages
    ranking = compute_advantages(rewards, 4, method="ranking").advantages
    result = compute_advantages(rewards, 4, method="rank_reward", method_kwargs={"rank_weight": rank_weight})
    torch.testing.assert_close(result.advantages, (1 - rank_weight) * student + rank_weight * ranking)


def test_custom_estimator_receives_grouped_components_and_detached_rewards(monkeypatch):
    monkeypatch.setattr(module, "_ESTIMATORS", dict(module._ESTIMATORS))
    components = torch.tensor([[1.0, 4.0], [2.0, float("nan")], [3.0, 9.0], [5.0, 1.0]], requires_grad=True)
    weights = torch.tensor([0.4, 0.6], requires_grad=True)
    rewards = (components * weights).nansum(dim=1)

    @register_advantage("unit_component_estimator")
    def custom(batch, *, scale):
        assert batch.rewards.shape == (1, 4)
        assert batch.rewards_per_func.shape == (1, 4, 2)
        assert batch.group_mean.shape == (1, 1)
        assert batch.group_std.shape == (1, 1)
        assert not batch.rewards.requires_grad
        assert not batch.rewards_per_func.requires_grad
        assert not batch.reward_weights.requires_grad
        torch.testing.assert_close(batch.reward_weights, weights.detach())
        # Deliberately create a tensor with grad to verify the output boundary.
        return (batch.rewards_per_func[:, :, 0] * scale).requires_grad_(True)

    result = compute_advantages(
        rewards, 4, method="unit_component_estimator", method_kwargs={"scale": 2},
        rewards_per_func=components, reward_weights=weights,
    )
    torch.testing.assert_close(result.advantages, torch.tensor([2.0, 4.0, 6.0, 10.0]))
    assert not result.advantages.requires_grad
    assert not result.group_mean.requires_grad
    assert not result.group_std.requires_grad
    with pytest.raises(ValueError, match="already registered"):
        register_advantage("unit_component_estimator")(custom)


def test_import_path_loads_an_external_estimator_with_json_options(tmp_path, monkeypatch):
    (tmp_path / "experiment_advantage.py").write_text(
        "def estimate(batch, *, scale=1.0):\n"
        "    return scale * (batch.rewards - batch.group_mean)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    result = compute_advantages(
        torch.tensor([1.0, 3.0]), 2, method="experiment_advantage:estimate", method_kwargs='{"scale": 2.0}',
    )
    torch.testing.assert_close(result.advantages, torch.tensor([-2.0, 2.0]))


@pytest.mark.parametrize("group_size", [0, 1, -1, True, 2.5])
def test_invalid_group_size_is_rejected(group_size):
    with pytest.raises(ValueError, match="at least 2"):
        compute_advantages(torch.ones(4), group_size)


@pytest.mark.parametrize("rewards", [torch.ones(3), torch.ones(2, 2), torch.tensor([])])
def test_invalid_group_shape_is_rejected(rewards):
    with pytest.raises(ValueError, match="complete generation groups"):
        compute_advantages(rewards, 2)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_aggregate_rewards_are_rejected(bad_value):
    with pytest.raises(ValueError, match="Aggregated rewards must be finite"):
        compute_advantages(torch.tensor([0.0, bad_value]), 2)


@pytest.mark.parametrize("bad_estimator", [
    lambda batch: batch.rewards.reshape(-1),
    lambda batch: batch.rewards.long(),
    lambda batch: torch.full_like(batch.rewards, float("nan")),
    lambda batch: [1.0, 2.0],
])
def test_bad_custom_output_fails_at_the_estimator_boundary(bad_estimator):
    with pytest.raises(ValueError, match="Advantage estimator"):
        compute_advantages(torch.tensor([1.0, 3.0]), 2, method=bad_estimator)


@pytest.mark.parametrize("options", [[1, 2], "[]", '{"broken"', {1: "value"}])
def test_invalid_options_fail_with_a_clear_error(options):
    with pytest.raises((TypeError, ValueError), match="advantage_kwargs"):
        parse_advantage_kwargs(options)


@pytest.mark.parametrize("epsilon", [0, -1, float("nan"), float("inf")])
def test_invalid_epsilon_is_rejected(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        compute_advantages(torch.ones(4), 4, method_kwargs={"epsilon": epsilon})


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_mix_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="rank_weight"):
        compute_advantages(torch.ones(4), 4, method="rank_reward", method_kwargs={"rank_weight": weight})


def test_unknown_estimator_error_lists_builtins_and_extension_syntax():
    with pytest.raises(ValueError, match="rank_reward.*ranking.*studentization.*module:callable"):
        resolve_advantage("not_registered")


def test_component_and_weight_shapes_are_validated():
    rewards = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="rewards_per_func"):
        compute_advantages(rewards, 2, rewards_per_func=torch.ones(3, 1))
    with pytest.raises(ValueError, match="reward_weights"):
        compute_advantages(rewards, 2, rewards_per_func=torch.ones(2, 3), reward_weights=torch.ones(2))


@pytest.mark.parametrize("method", ["studentization", "open_r1.advantages:studentization"])
def test_trainer_scale_rewards_false_centers_without_std_normalization(method):
    estimator, options = module.configure_advantage(method, scale_rewards=False)
    result = compute_advantages(torch.tensor([0.0, 2.0, 4.0, 18.0]), 4, method=estimator, method_kwargs=options)
    torch.testing.assert_close(result.advantages, torch.tensor([-6.0, -4.0, -2.0, 12.0]))


def test_unscaled_rank_reward_preserves_rank_branch():
    rewards = torch.tensor([0.0, 2.0, 4.0, 18.0])
    estimator, options = module.configure_advantage("rank_reward", {"rank_weight": 0.25}, scale_rewards=False)
    result = compute_advantages(rewards, 4, method=estimator, method_kwargs=options)
    ranks = compute_advantages(rewards, 4, method="ranking").advantages
    expected = 0.75 * (rewards - rewards.mean()) + 0.25 * ranks
    torch.testing.assert_close(result.advantages, expected)


@pytest.mark.parametrize("method", ["studentization", "rank_reward"])
@pytest.mark.parametrize("scale_rewards", [False, True])
def test_conflicting_scaling_configuration_is_rejected(method, scale_rewards):
    with pytest.raises(ValueError, match="conflicts with the top-level scale_rewards"):
        module.configure_advantage(method, {"scale_rewards": not scale_rewards}, scale_rewards=scale_rewards)
    _, options = module.configure_advantage(method, {"scale_rewards": scale_rewards}, scale_rewards=scale_rewards)
    assert options["scale_rewards"] is scale_rewards


@pytest.mark.parametrize("method", ["ranking", lambda batch: batch.rewards])
def test_top_level_scaling_is_not_injected_into_other_estimator_apis(method):
    _, options = module.configure_advantage(method, scale_rewards=False)
    assert options == {}


@pytest.mark.parametrize("value", ["false", 0, None])
def test_nonboolean_scaling_is_rejected(value):
    with pytest.raises(TypeError, match="must be a boolean"):
        module.configure_advantage("studentization", {"scale_rewards": value})
    with pytest.raises(TypeError, match="must be a boolean"):
        compute_advantages(torch.tensor([1.0, 2.0]), 2, method_kwargs={"scale_rewards": value})
