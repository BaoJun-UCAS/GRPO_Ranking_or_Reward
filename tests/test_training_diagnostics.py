"""Offline diagnostic statistics must not silently accept broken rollout groups."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


spec = importlib.util.spec_from_file_location(
    "training_diagnostics", Path(__file__).resolve().parents[1] / "scripts/diagnose_training_comparison.py"
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_exact_thresholds_and_exclusion_of_self_comparisons():
    # Three unordered gaps: 0.125 (dead), 0.25 (linear), 0.375 (saturated).
    d = diagnostics.advantage_diagnostics([[0, .125, .375]], .125, .25, 1e-4)
    for key in ("dead_pair_fraction", "linear_pair_fraction", "saturated_pair_fraction"):
        assert d[key][0] == pytest.approx(1 / 3)
    assert d["gaps"].shape == (1, 3)
    # Pair contributions yield [-.5, -.25, .75].
    assert d["robust_rms"][0] == pytest.approx(np.sqrt(.875 / 3))


def test_zero_vectors_are_excluded_from_cosines_and_constant_grpo_ratios():
    d = diagnostics.advantage_diagnostics([[1, 1], [0, .0625], [0, .25]], .125, .25, 1e-4)
    summary = diagnostics.summarize_diagnostics(d, np.ones(3, dtype=bool))
    assert summary["cosine_undefined_groups"] == 2
    assert summary["rms_ratio_undefined_groups"] == 1
    assert summary["all_zero_robust_group_fraction"] == pytest.approx(2 / 3)
    assert summary["distributions"]["cosine"]["mean"] == pytest.approx(1)
    assert d["rms_ratio"][1] == 0
    json.dumps(summary, allow_nan=False)


def test_reconstruction_matches_real_cpu_estimators():
    torch = pytest.importorskip("torch")
    from open_r1.advantages import compute_advantages, configure_advantage

    rewards = np.random.default_rng(123).normal(.8, .03, (25, 8))
    d = diagnostics.advantage_diagnostics(rewards, .002, .08, 1e-4)
    for method, key, kwargs in (("grpo", "grpo_rms", {}),
                                ("robust_pairwise", "robust_rms", {"delta": .002, "c": .08})):
        estimator, options = configure_advantage(method, kwargs, scale_rewards=True)
        actual = compute_advantages(torch.tensor(rewards.ravel(), dtype=torch.float64), 8,
                                    method=estimator, method_kwargs=options).advantages.numpy().reshape(-1, 8)
        np.testing.assert_allclose(d[key], np.sqrt(np.mean(actual ** 2, axis=1)), atol=1e-12)
    assert d["linear_identity_max_abs_error"].max() < 1e-12


def make_run(tmp_path):
    run = tmp_path / "run"
    (run / "config").mkdir(parents=True)
    (run / "reward_data").mkdir()
    cfg = {"max_steps": 1, "num_generations": 4, "generation_batch_size": 4,
           "num_iterations": 1, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 2}
    (run / "config/resolved_training_config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "config/accelerate_config.yaml").write_text("num_processes: 2\n")
    (run / "RUN_STATUS").write_text("status=success\n")
    (run / "trainer_state.json").write_text(json.dumps({
        "global_step": 1, "max_steps": 1,
        "log_history": [{"step": 1, "reward": .375, "reward_std": float(np.std([0, .25, .5, .75], ddof=1))}]}))
    for rank in range(2):
        rows = [{"prompt": "same prompt", "completion": "answer", "reward": (rank * 2 + i) / 4,
                 "training_step": 0, "process_index": rank, "rollout_index": i} for i in range(2)]
        (run / f"reward_data/reward_data_train_step_000000_micro_00000000_proc_{rank}_abc.json").write_text(json.dumps(rows))
    return run


def test_group_spanning_ranks_is_gathered_in_rank_order(tmp_path):
    run = diagnostics.load_run(make_run(tmp_path), 1)
    np.testing.assert_array_equal(run["rewards"], [[0, .25, .5, .75]])
    assert run["audit"]["groups"] == 1


@pytest.mark.parametrize("damage", ["missing_rank", "duplicate_rollout", "wrong_prompt", "wrong_reward"])
def test_corrupt_rollouts_fail_closed(tmp_path, damage):
    run = make_run(tmp_path)
    path = next((run / "reward_data").glob("*proc_1*"))
    if damage == "missing_rank":
        path.unlink()
    elif damage == "duplicate_rollout":
        path.with_name(path.name.replace("abc", "def")).write_bytes(path.read_bytes())
    else:
        rows = json.loads(path.read_text())
        rows[0]["prompt" if damage == "wrong_prompt" else "reward"] = "other prompt" if damage == "wrong_prompt" else 1.5
        path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        diagnostics.load_run(run, 1)
