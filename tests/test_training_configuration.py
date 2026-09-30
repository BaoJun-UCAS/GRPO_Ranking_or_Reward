"""Exercise the trainer's checkpoint setup without importing CUDA libraries."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/open_r1/grpo_trainer.py"


def source_function(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def checkpoint_method(peft, zero3):
    # Compile the real method with lightweight model/distribution stand-ins.
    tree = ast.parse(SOURCE.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_enable_gradient_checkpointing")
    namespace = {
        "PreTrainedModel": object, "GRPOConfig": object,
        "is_peft_model": lambda model: peft, "is_deepspeed_zero3_enabled": lambda: zero3,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("peft,zero3,reentrant", [(True, True, True), (True, False, False), (False, True, False)])
def test_checkpoint_options_reach_model_without_mutating_arguments(peft, zero3, reentrant):
    model = Mock()
    options = {"use_reentrant": reentrant, "preserve_rng_state": False}
    args = SimpleNamespace(gradient_checkpointing_kwargs=options.copy())
    assert checkpoint_method(peft, zero3)(None, model, args) is model
    target = model.base_model if peft else model
    target.gradient_checkpointing_enable.assert_called_once_with(gradient_checkpointing_kwargs=options)
    assert args.gradient_checkpointing_kwargs == options
    assert model.config.use_cache is False
    assert model.enable_input_require_grads.called is reentrant


def test_unsafe_zero3_peft_configuration_is_rejected_before_model_mutation():
    model = Mock()
    with pytest.raises(ValueError, match="PEFT.*ZeRO-3.*use_reentrant=true"):
        checkpoint_method(True, True)(None, model, SimpleNamespace(gradient_checkpointing_kwargs={"use_reentrant": False}))
    model.base_model.gradient_checkpointing_enable.assert_not_called()


def test_default_reentrant_configuration_enables_input_gradients():
    model = Mock()
    checkpoint_method(True, True)(None, model, SimpleNamespace(gradient_checkpointing_kwargs=None))
    model.base_model.gradient_checkpointing_enable.assert_called_once_with(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads.assert_called_once()


def test_trim_padded_microbatch_removes_only_shared_padding_columns():
    class FakeScalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        @property
        def shape(self):
            if self.values and isinstance(self.values[0], list):
                return len(self.values), len(self.values[0])
            return (len(self.values),)

        def max(self):
            values = self.values
            if values and isinstance(values[0], list):
                values = [item for row in values for item in row]
            return FakeScalar(max(values))

        def __getitem__(self, index):
            rows, columns = index
            return FakeTensor([row[columns] for row in self.values[rows]])

    fake_torch = SimpleNamespace(Tensor=FakeTensor)
    trim = source_function("trim_padded_microbatch", {"torch": fake_torch, "Optional": Optional})
    inputs = {
        "prompt_ids": FakeTensor([[0, 0, 1, 2, 3], [0, 0, 4, 5, 6]]),
        "prompt_mask": FakeTensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1]]),
        "completion_ids": FakeTensor([[7, 8, 0, 0, 0, 0], [9, 10, 11, 12, 0, 0]]),
        "completion_mask": FakeTensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]]),
        "old_per_token_logps": FakeTensor([[0] * 6, [0] * 6]),
        "_prompt_lengths": FakeTensor([3, 3]),
        "_completion_lengths": FakeTensor([2, 4]),
    }
    trimmed = trim(inputs)
    assert trimmed["prompt_ids"].shape == (2, 3)
    assert trimmed["completion_ids"].shape == (2, 4)
    assert trimmed["completion_mask"].shape == (2, 4)
    assert trimmed["old_per_token_logps"].shape == (2, 4)
    assert "_prompt_lengths" not in trimmed
    assert "_completion_lengths" not in trimmed
    assert inputs["completion_ids"].shape == (2, 6)


def test_policy_timing_summary_reports_optimizer_step_total_and_micro_step_mean():
    summarize = source_function("summarize_timing_samples", {})
    summary = summarize({"training_step": [5.0, 2.0], "policy_train": [3.0, 1.0]})
    assert summary == {
        "training_step_total": 7.0,
        "training_step_mean": 3.5,
        "policy_train_total": 4.0,
        "policy_train_mean": 2.0,
    }
    with pytest.raises(ValueError, match="equally sized"):
        summarize({"training_step": [1.0], "policy_train": []})


@pytest.mark.parametrize("terminated,total,expected", [(0, 32, 1.0), (11, 32, 0.65625), (32, 32, 0.0)])
def test_completion_clipped_ratio_uses_global_batch(terminated, total, expected):
    tree = ast.parse(SOURCE.read_text())
    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "clipped_completions_ratio" for target in node.targets)
    )
    ratio = eval(compile(ast.Expression(assignment.value), str(SOURCE), "eval"), {
        "term_completion_lengths": [0] * terminated,
        "agg_completion_lengths": [0] * total,
        "completion_lengths": [0] * (total // 4),
    })
    assert ratio == expected
    assert 0 <= ratio <= 1


@pytest.mark.parametrize("enabled", [False, True])
def test_completion_logging_controls_expensive_text_collectives(enabled):
    # Execute the production logging block with observable collectives. This
    # catches accidental all-gathers even when completion tables are disabled.
    tree = ast.parse(SOURCE.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    generate = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_generate_and_score_completions")
    block = next(node for node in generate.body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Attribute) and node.test.attr == "log_completions")
    logs = {"prompt": [], "completion": [], "rewards": {}, "advantages": []}
    gather = Mock(side_effect=lambda items: items)
    mock_trainer = SimpleNamespace(log_completions=enabled, _textual_logs=logs,
                                   _gather_python_objects=gather, reward_func_names=[])
    namespace = {
        "self": mock_trainer, "prompts_text": ["question"], "completions_text": ["answer"],
        "all_process_advantages": SimpleNamespace(tolist=lambda: [1.0]),
    }
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(SOURCE), "exec"), namespace)
    assert gather.call_count == (2 if enabled else 0)
    assert logs["prompt"] == (["question"] if enabled else [])
    assert logs["advantages"] == ([1.0] if enabled else [])
