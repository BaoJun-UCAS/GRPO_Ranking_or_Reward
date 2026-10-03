"""Method-1 algebra, causal rolling history, checkpoint and trainer contracts."""

import ast
from collections import defaultdict
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from open_r1.advantages import AdvantageBatch, compute_advantages, configure_advantage
from open_r1.rolling_advantages import RollingQuantilePairwise, normalized_pairwise_advantages


def batch(rows, dtype=torch.float64):
    rewards = torch.tensor(rows, dtype=dtype)
    return AdvantageBatch(rewards, rewards.mean(1, keepdim=True), rewards.std(1, keepdim=True, unbiased=True))


def gaps_of(b, epsilon=1e-4):
    # Independent scalar extraction: only i<j, with no zero diagonal entries.
    return torch.tensor([abs(float(row[i] - row[j])) / (float(std) + epsilon)
                         for row, std in zip(b.rewards, b.group_std.flatten())
                         for i in range(len(row)) for j in range(i + 1, len(row))], dtype=torch.float64)


def scalar_advantage(b, ell, tau, epsilon=1e-4):
    rows = []
    for rewards, std in zip(b.rewards.tolist(), b.group_std.flatten().tolist()):
        rows.append([sum(math.copysign(min(max(abs((r - other) / (std + epsilon)) - ell, 0), tau), r - other)
                         for other in rewards) / len(rewards) for r in rewards])
    return torch.tensor(rows, dtype=b.rewards.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_filter_no_cap_exactly_restores_grpo(dtype):
    b = batch([[0, 1, 2, 8], [2, 2, 2, 2]], dtype)
    actual = normalized_pairwise_advantages(b, ell=0, tau=math.inf)
    expected = compute_advantages(b.rewards.flatten(), 4, method="grpo").advantages.reshape_as(actual)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_unit_slope_cap_tau_and_divisor_g_match_scalar_formula():
    b = batch([[0, 1, 2], [0, 1, 10]])
    actual = normalized_pairwise_advantages(b, ell=.5, tau=1.25)
    torch.testing.assert_close(actual, scalar_advantage(b, .5, 1.25), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual.sum(1), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
    assert actual.abs().max() <= 2 / 3 * 1.25


def test_first_step_uses_grpo_next_step_uses_previous_pairs_only():
    estimator = RollingQuantilePairwise(p=.2, q=.2)
    previous = batch([[0, 1, 2, 3], [0, 0, 0, 1]])
    current = batch([[0, 1, 100, 1000]])
    actual = estimator(previous, step=0)
    torch.testing.assert_close(actual, normalized_pairwise_advantages(previous, ell=0, tau=math.inf))
    assert estimator.last_metrics["history_pairs"] == 0
    estimator(current, step=1)
    ell, upper = torch.quantile(gaps_of(previous), torch.tensor([.2, .8], dtype=torch.float64)).tolist()
    assert estimator.last_metrics["ell"] == pytest.approx(ell)
    assert estimator.last_metrics["tau"] == pytest.approx(upper - ell)
    assert estimator.last_metrics["history_pairs"] == 12
    assert len(estimator.state_dict()["history"][0]) == 12
    torch.testing.assert_close(estimator(current, step=1), scalar_advantage(current, ell, upper - ell))


def test_thresholds_frozen_across_multiple_rollouts_in_same_optimizer_step():
    estimator = RollingQuantilePairwise(p=.2, q=.1)
    estimator(batch([[0, 1, 2, 3]]), step=0)
    current = batch([[0, 0, 0, 1]])
    first = estimator(current, step=1)
    frozen = estimator.state_dict()["active"]
    estimator(batch([[0, 1, 10, 100]]), step=1)
    second = estimator(current, step=1)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert estimator.state_dict()["active"] == frozen
    with pytest.raises(ValueError, match="nondecreasing"):
        estimator(current, step=0)


def test_window_evicts_old_rollouts_instead_of_accumulating_forever():
    estimator = RollingQuantilePairwise(p=.1, q=.2, window_size=1)
    estimator(batch([[0, 1, 2, 3]]), step=0)
    previous = batch([[0, 0, 0, 1]])
    estimator(previous, step=1)
    estimator(batch([[0, 10, 20, 30]]), step=2)
    ell, upper = torch.quantile(gaps_of(previous), torch.tensor([.1, .8], dtype=torch.float64)).tolist()
    assert estimator.last_metrics["ell"] == pytest.approx(ell)
    assert estimator.last_metrics["upper_threshold"] == pytest.approx(upper)
    assert estimator.last_metrics["history_rollouts"] == 1


def test_evaluation_does_not_change_history_step_or_frozen_thresholds():
    estimator = RollingQuantilePairwise(p=.1, q=.1)
    estimator(batch([[0, 1, 2, 3]]), step=0)
    estimator(batch([[0, 0, 1, 2]]), step=1)
    before = estimator.state_dict()
    estimator(batch([[0, 10, 100, 1000]]), step=100, update_history=False)
    assert estimator.state_dict() == before


def test_constant_history_degenerate_quantiles_produce_finite_zero_signal():
    estimator = RollingQuantilePairwise(p=.1, q=.1)
    assert torch.count_nonzero(estimator(batch([[2, 2, 2, 2]]), step=0)) == 0
    actual = estimator(batch([[0, 1, 2, 3]]), step=1)
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual) == 0
    assert estimator.last_metrics["degenerate_thresholds"] == 1
    assert estimator.last_metrics["dead_pair_fraction"] + estimator.last_metrics["saturated_pair_fraction"] == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_keeps_dtype_and_does_not_mutate_rewards(dtype):
    b = batch([[0, 1, 2, 3], [1, 1, 1, 1]], dtype)
    original = b.rewards.clone()
    estimator = RollingQuantilePairwise(p=.1, q=.1)
    estimator(b)
    actual = estimator(b)
    assert actual.dtype == dtype and actual.device == b.rewards.device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(b.rewards, original, atol=0, rtol=0)


def test_reward_translation_and_joint_scaling_of_epsilon():
    a = RollingQuantilePairwise(p=.1, q=.1, epsilon=1e-4)
    b = RollingQuantilePairwise(p=.1, q=.1, epsilon=1e-3)
    for rows in ([[0, 1, 2, 3]], [[0, 0, 1, 10]], [[0, 1, 10, 100]]):
        original = batch(rows)
        changed = batch((original.rewards * 10 + 100).tolist())
        torch.testing.assert_close(a(original), b(changed), atol=1e-12, rtol=1e-12)


def test_checkpoint_roundtrip_matches_uninterrupted_training_and_is_json_finite(tmp_path):
    a = RollingQuantilePairwise(p=.1, q=.2, window_size=2)
    a(batch([[0, 1, 2, 3]]), step=0)
    # Cold-start state contains infinite tau encoded as null, not JSON Infinity.
    a.save(tmp_path)
    text = (tmp_path / a.state_filename).read_text()
    assert "Infinity" not in text
    b = RollingQuantilePairwise(p=.1, q=.2, window_size=2)
    b.load(tmp_path)
    for step, rows in enumerate(([[0, 0, 1, 10]], [[0, 1, 5, 6]], [[0, 1, 2, 10]]), start=1):
        torch.testing.assert_close(a(batch(rows), step=step), b(batch(rows), step=step), atol=0, rtol=0)
        assert a.state_dict() == b.state_dict()


def test_two_configured_trainers_and_simulated_ranks_have_independent_state():
    a, opts = configure_advantage("rolling_quantile_pairwise", {"p": .1, "q": .2})
    b, _ = configure_advantage("rolling_quantile_pairwise", {"p": .1, "q": .2})
    rewards = torch.tensor([0., 1., 2., 3., 10., 11., 12., 100.], requires_grad=True)
    for step in range(2):
        left = compute_advantages(rewards, 4, method=a, method_kwargs={"step": step, **opts}).advantages
        right = compute_advantages(rewards, 4, method=b, method_kwargs={"step": step}).advantages
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert not left.requires_grad
    a(batch([[0, 1, 2, 3]]), step=2)
    assert a.state_dict() != b.state_dict()
    with pytest.raises(ValueError, match="stateful"):
        compute_advantages(rewards, 4, method="rolling_quantile_pairwise")


@pytest.mark.parametrize("options", [
    {"p": -.1, "q": .1}, {"p": .5, "q": .5}, {"p": True, "q": .1},
    {"p": .1, "q": float("nan")}, {"p": .1, "q": .1, "window_size": 0},
    {"p": .1, "q": .1, "epsilon": 0},
])
def test_invalid_configuration_rejected(options):
    with pytest.raises(ValueError):
        configure_advantage("rolling_quantile_pairwise", options)


def test_required_proportions_and_scaling_flag_are_explicit():
    with pytest.raises(TypeError):
        configure_advantage("rolling_quantile_pairwise", {})
    with pytest.raises(ValueError, match="scale_rewards=True"):
        configure_advantage("rolling_quantile_pairwise", {"p": .1, "q": .1}, scale_rewards=False)


def test_corrupt_or_mismatched_checkpoint_rejected(tmp_path):
    estimator = RollingQuantilePairwise(p=.1, q=.1)
    with pytest.raises(ValueError, match="Missing"):
        estimator.load(tmp_path)
    state = estimator.state_dict()
    state["config"]["p"] = .2
    with pytest.raises(ValueError, match="config"):
        estimator.load_state_dict(state)
    state = estimator.state_dict()
    state.update(step=0, active={"ell": 0, "tau": 1, "history_pairs": 1, "history_rollouts": 1}, history=[[float("nan")]])
    with pytest.raises(ValueError, match="finite"):
        estimator.load_state_dict(state)


def test_checkpoint_callback_writes_history_and_checks_optimizer_position(tmp_path):
    from open_r1.rolling_advantage_state import RollingAdvantageStateCallback
    estimator = RollingQuantilePairwise(p=.1, q=.1)
    estimator(batch([[0, 1, 2, 3]]), step=0)
    callback = RollingAdvantageStateCallback(estimator)
    args = SimpleNamespace(output_dir=str(tmp_path), should_save=True)
    state = SimpleNamespace(global_step=1, is_world_process_zero=True)
    callback.on_save(args, state, None)
    assert (tmp_path / "checkpoint-1" / estimator.state_filename).is_file()
    with pytest.raises(ValueError, match="Resumed"):
        callback.on_train_begin(args, state, None)
    estimator.load(tmp_path / "checkpoint-1")
    callback.on_train_begin(args, state, None)
    with pytest.raises(ValueError, match="Resumed"):
        callback.on_train_begin(args, SimpleNamespace(global_step=2), None)
    callback.on_train_end(args, state, None)
    assert (tmp_path / estimator.state_filename).is_file()


def test_trainer_resume_restores_state_before_parent_training(tmp_path, monkeypatch):
    from open_r1.grpo_trainer import GRPOTrainer
    estimator = RollingQuantilePairwise(p=.1, q=.1)
    estimator(batch([[0, 1, 2, 3]]), step=0)
    estimator.save(tmp_path / "checkpoint-1", global_step=1)
    trainer = object.__new__(GRPOTrainer)
    trainer.advantage_estimator = RollingQuantilePairwise(p=.1, q=.1)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))
    parent = GRPOTrainer.__mro__[1]

    def train(self, **kwargs):
        assert self.advantage_estimator.resume_loaded
        return "restored"

    monkeypatch.setattr(parent, "train", train)
    assert trainer.train(resume_from_checkpoint=True) == "restored"
    assert trainer.advantage_estimator.state_dict() == estimator.state_dict()


def test_trainer_advantage_block_passes_optimizer_step_and_freezes_eval_state():
    # Execute the actual trainer integration block without constructing a model,
    # generating text, contacting services, or running an optimizer.
    source = Path(__file__).resolve().parents[1] / "src/open_r1/grpo_trainer.py"
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    generate = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_generate_and_score_completions")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, node in enumerate(generate.body) if assigns(node, "advantage_options"))
    end = next(i for i, node in enumerate(generate.body) if assigns(node, "process_slice"))
    code = compile(ast.Module(body=generate.body[start:end], type_ignores=[]), str(source), "exec")
    estimator, options = configure_advantage("rolling_quantile_pairwise", {"p": .1, "q": .1})
    fake = SimpleNamespace(advantage_estimator=estimator, advantage_kwargs=options,
                           state=SimpleNamespace(global_step=0), num_generations=4,
                           reward_weights=torch.tensor([1.]), _metrics={mode: defaultdict(list)
                                                                       for mode in ("train", "eval")})
    rewards = torch.tensor([0., 1., 2., 3., 0., 0., 0., 1.])
    namespace = {"self": fake, "rewards": rewards, "rewards_per_func": rewards[:, None],
                 "device": torch.device("cpu"), "mode": "train", "torch": torch,
                 "compute_advantages": compute_advantages, "RollingQuantilePairwise": RollingQuantilePairwise}
    exec(code, namespace)
    assert fake._metrics["train"]["advantage/warmup_grpo"] == [1.]
    assert len(estimator.state_dict()["history"]) == 1
    fake.state.global_step = 1
    exec(code, namespace)
    before_eval = estimator.state_dict()
    namespace.update(mode="eval", rewards=rewards * 100, rewards_per_func=rewards[:, None] * 100)
    exec(code, namespace)
    assert estimator.state_dict() == before_eval
    assert fake._metrics["eval"]["advantage/history_pairs"] == [12]
