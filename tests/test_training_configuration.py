"""Exercise the trainer's checkpoint setup without importing CUDA libraries."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/open_r1/grpo_trainer.py"


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
