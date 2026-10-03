"""CPU algebra and loss-input transport checks; no model loading or training."""

from collections import defaultdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from open_r1.advantage_controls import control_components
from open_r1.advantages import AdvantageBatch, compute_advantages, configure_advantage

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("controls_runner", ROOT / "scripts/run_advantage_controls.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def components(rewards, **options):
    return control_components(AdvantageBatch(
        rewards, rewards.mean(1, keepdim=True), rewards.std(1, keepdim=True, unbiased=True)), **options)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_controls_have_same_rms_and_weight_control_preserves_grpo_direction(dtype):
    rewards = torch.tensor([[.1, .1001, .1002, .1003], [0, .01, .02, 1], [2, 2, 2, 2]], dtype=dtype)
    parts = components(rewards)
    a, b = parts["robust_scaled"], parts["weight_only_scaled"]
    torch.testing.assert_close(a, 2.46 * parts["robust"])
    torch.testing.assert_close(a.double().square().mean(1), b.double().square().mean(1), atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(a[[0, 2]]) == torch.count_nonzero(b[[0, 2]]) == 0
    assert parts["q"][2].item() == 0
    assert torch.nn.functional.cosine_similarity(b[1:2], parts["grpo"][1:2]).item() == pytest.approx(1)
    assert not torch.allclose(a[1], b[1])  # Deliberately different within-group shapes.
    assert a.dtype == b.dtype == dtype


def test_weights_recomputed_and_detached_from_each_current_batch():
    rewards = torch.tensor([[0, .01, .02, .03]], requires_grad=True)
    first, second = components(rewards), components(rewards * 2)
    assert not torch.allclose(first["q"], second["q"])
    assert all(not tensor.requires_grad for tensor in first.values())


@pytest.mark.parametrize("k", [0, -1, float("inf"), float("nan"), True])
def test_invalid_scale_rejected(k):
    with pytest.raises(ValueError, match="k must"):
        components(torch.tensor([[0., 1.]]), k=k)


@pytest.mark.parametrize("arm", runner.ARMS)
def test_plugin_path_is_not_renormalized_by_scale_rewards(arm):
    rewards = torch.tensor([0., .01, .02, .03, 1., 1., 1., 1.])
    expected = components(rewards.reshape(-1, 4))[arm].flatten()
    for scale_rewards in (True, False):
        method, options = configure_advantage(f"open_r1.advantage_controls:{arm}", {"k": 2.46}, scale_rewards=scale_rewards)
        actual = compute_advantages(rewards, 4, method=method, method_kwargs=options).advantages
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_final_loss_audit_follows_real_permutation_split_and_trim(tmp_path, monkeypatch):
    from open_r1.control_trainer import ControlGRPOTrainer
    from open_r1.grpo_trainer import GRPOTrainer, split_tensor_dict, trim_padded_microbatch
    from open_r1.training_schedule import permute_rollout_payload

    trainer = object.__new__(ControlGRPOTrainer)
    trainer.accelerator = SimpleNamespace(process_index=1)
    trainer.state = SimpleNamespace(global_step=0)
    trainer._step = 1
    trainer.control_audit_dir = tmp_path
    trainer._control_snapshot = torch.tensor([-.7, -.1, .2, .6])
    payload = {"advantages": trainer._control_snapshot[2:].clone(),
               "completion_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
               "_completion_lengths": torch.tensor([2, 1])}
    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", lambda self, inputs: payload)
    received = []

    def original_loss(self, model, inputs, **kwargs):
        assert not any(key.startswith("_control_") for key in inputs)
        received.append(inputs["advantages"].clone())
        return inputs["advantages"].sum()

    monkeypatch.setattr(GRPOTrainer, "compute_loss", original_loss)
    audited = trainer._generate_and_score_completions([])
    ordered = permute_rollout_payload(audited, [1, 0])
    for chunk in split_tensor_dict(ordered, 2):
        trainer.compute_loss(None, trim_padded_microbatch(chunk))
    rows = [json.loads(line) for line in (tmp_path / "loss_inputs_rank_1.jsonl").read_text().splitlines()]
    assert [row["global_rollout_rows"] for row in rows] == [[3], [2]]
    assert all(row["exact_match"] for row in rows)
    assert len(received) == 2
    bad = dict(ordered, advantages=ordered["advantages"] * 2)
    with pytest.raises(ValueError, match="changed after"):
        trainer.compute_loss(None, bad)


def test_capture_logs_scaled_rms_without_changing_estimator_output(tmp_path, monkeypatch):
    from open_r1.advantage_controls import robust_scaled
    from open_r1.control_trainer import ControlGRPOTrainer
    from open_r1.grpo_trainer import GRPOTrainer

    def fake_init(self):
        self.advantage_estimator = robust_scaled
        self.args = SimpleNamespace(token_broadcast="uniform", output_dir=str(tmp_path))
        self.use_liger_loss = False
        self.model = SimpleNamespace(training=True)
        self.state = SimpleNamespace(global_step=0)
        self.accelerator = SimpleNamespace(is_main_process=True)
        self._metrics = {"train": defaultdict(list)}

    monkeypatch.setattr(GRPOTrainer, "__init__", fake_init)
    trainer = ControlGRPOTrainer()
    r = torch.tensor([[0., .01, .02, .03]])
    batch = AdvantageBatch(r, r.mean(1, keepdim=True), r.std(1, keepdim=True))
    actual = trainer.advantage_estimator(batch, delta=.002, c=.08, k=2.46, epsilon=1e-4)
    torch.testing.assert_close(actual, robust_scaled(batch), atol=0, rtol=0)
    logs = trainer._metrics["train"]
    assert logs["advantage/final_rms"][0] / logs["advantage/robust_rms"][0] == pytest.approx(2.46)
    assert (tmp_path / "advantage_audit/rollout_step_001.json").is_file()


def test_recipe_retains_optimizer_budget_and_only_changes_control_and_paths(tmp_path):
    source = {"learning_rate": 1e-6, "beta": .04, "loss_type": "bnpo", "seed": 42, "lora_r": 32,
              "num_generations": 8, "gradient_accumulation_steps": 128, "resume_from_checkpoint": "/bad"}
    config = {"model_name": "/local/base", "model_revision": "pinned"}
    options = {"delta": .002, "c": .08, "k": 2.46, "epsilon": 1e-4}
    a, b = [runner.recipe_for_arm(source, tmp_path, arm, config, options) for arm in runner.ARMS]
    assert {key for key in a if a[key] != b[key]} == {"advantage", "output_dir", "reward_data_save_path", "run_name"}
    for key in source.keys() - {"resume_from_checkpoint"}:
        assert a[key] == b[key] == source[key]
    assert a["max_steps"] == b["max_steps"] == 200
    assert a["resume_from_checkpoint"] is b["resume_from_checkpoint"] is None
    assert source["resume_from_checkpoint"] == "/bad"  # No source mutation.


def test_offline_environment_blocks_stale_training_overrides(tmp_path, monkeypatch):
    for key, value in {"ADVANTAGE": "ranking", "MAX_STEPS": "1", "MAX_TRAIN_SAMPLES": "1",
                       "RESUME_FROM_CHECKPOINT": "/bad", "TRAINING_ENTRYPOINT": "wrong.py",
                       "HF_HUB_OFFLINE": "0"}.items():
        monkeypatch.setenv(key, value)
    config = yaml.safe_load((ROOT / "recipes/Qwen3-1.7B/advantage_comparison_200.yaml").read_text())
    config["generation_batch_size"] = 256
    manifest = {"config": config, "hub_cache": str(tmp_path / "hub"), "numerical_environment": {}}
    env = runner.environment(tmp_path, manifest, "robust_scaled")
    assert env["MAX_STEPS"] == "200" and env["RESUME_FROM_CHECKPOINT"] == ""
    assert "MAX_TRAIN_SAMPLES" not in env and "ADVANTAGE" not in env
    assert env["HF_HUB_OFFLINE"] == env["HF_DATASETS_OFFLINE"] == "1"
    assert env["TRAINING_ENTRYPOINT"].endswith("scripts/grpo_control_train.py")


def test_missing_cache_fails_locally(tmp_path):
    with pytest.raises(ValueError, match="no download attempted"):
        runner.cached_snapshot(tmp_path, "org/model", "a" * 40)


def test_existing_second_arm_rejects_before_first_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "verify", lambda directory: {})
    (tmp_path / "weight_only_scaled").mkdir()
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="already exists"):
        runner.train(tmp_path, runner.ARMS, False)
    assert not calls
