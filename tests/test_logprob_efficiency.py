"""Numerical and allocation checks for the actual log-probability scoring helper."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from trl.trainer.utils import selective_log_softmax


def load_logprob_method():
    source = Path(__file__).resolve().parents[1] / "src/open_r1/grpo_trainer.py"
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_get_per_token_logps")
    method.decorator_list = []  # Profiling context is unrelated to this numerical contract.
    namespace = {"torch": torch, "selective_log_softmax": selective_log_softmax}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_get_per_token_logps"]


class TokenTableModel(torch.nn.Module):
    """Small differentiable token-to-logit table with the same scoring interface."""

    def __init__(self, weights):
        super().__init__()
        self.weights = torch.nn.Parameter(weights.clone())
        self.calls = []

    def forward(self, *, input_ids, attention_mask, logits_to_keep, use_cache):
        self.calls.append((input_ids.clone(), attention_mask.clone(), logits_to_keep, use_cache))
        return SimpleNamespace(logits=self.weights[input_ids][:, -logits_to_keep:, :])


def previous_logprob_path(model, input_ids, mask, keep, batch_size, temperature):
    """The previous scoring path, including unconditional division and cat."""
    parts = []
    size = batch_size or len(input_ids)
    for start in range(0, len(input_ids), size):
        ids = input_ids[start : start + size]
        logits = model(
            input_ids=ids, attention_mask=mask[start : start + size],
            logits_to_keep=keep + 1, use_cache=False,
        ).logits[:, :-1, :][:, -keep:, :]
        parts.append(selective_log_softmax(logits / temperature, ids[:, -keep:]))
    return torch.cat(parts, dim=0)


@pytest.mark.parametrize("temperature", [1.0, 0.7, 1.3])
@pytest.mark.parametrize("batch_size", [None, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_outputs_and_parameter_gradients_are_identical_to_previous_path(temperature, batch_size, dtype):
    generator = torch.Generator().manual_seed(147)
    weights = torch.randn(16, 19, generator=generator, dtype=torch.float64).to(dtype)
    current_model = TokenTableModel(weights)
    previous_model = TokenTableModel(weights)
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8]])
    mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
    current = load_logprob_method()(
        SimpleNamespace(temperature=temperature), current_model, input_ids, mask, 3, batch_size,
    )
    previous = previous_logprob_path(previous_model, input_ids, mask, 3, batch_size, temperature)
    assert current.shape == (3, 3)
    assert current.dtype == dtype
    torch.testing.assert_close(current, previous, atol=0, rtol=0)
    objective_weights = torch.linspace(0.2, 1.4, 9).reshape(3, 3).to(dtype)
    (current * objective_weights).sum().backward()
    (previous * objective_weights).sum().backward()
    torch.testing.assert_close(current_model.weights.grad, previous_model.weights.grad, atol=0, rtol=0)
    assert len(current_model.calls) == len(previous_model.calls)
    for new_call, old_call in zip(current_model.calls, previous_model.calls):
        torch.testing.assert_close(new_call[0], old_call[0], atol=0, rtol=0)
        torch.testing.assert_close(new_call[1], old_call[1], atol=0, rtol=0)
        assert new_call[2:] == old_call[2:] == (4, False)


@pytest.mark.parametrize("temperature, expected_divisions", [(1.0, 0), (0.5, 1)])
@pytest.mark.parametrize("batch_size, chunks", [(None, 1), (1, 3), (2, 2), (8, 1)])
def test_redundant_division_and_single_chunk_cat_are_eliminated(
    temperature, expected_divisions, batch_size, chunks,
):
    from torch.profiler import ProfilerActivity, profile

    model = TokenTableModel(torch.arange(16 * 19, dtype=torch.float32).reshape(16, 19) / 100)
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8]])
    method = load_logprob_method()
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        actual = method(
            SimpleNamespace(temperature=temperature), model, input_ids, torch.ones_like(input_ids), 3, batch_size,
        )
    operators = {event.key: event.count for event in profiler.key_averages()}
    assert operators.get("aten::div", 0) == expected_divisions * chunks
    # selective_log_softmax uses stack, whose implementation may itself call
    # cat. Each chunk contributes one such cat; only multi-chunk scoring adds
    # the final concatenation of chunk results.
    assert operators.get("aten::cat", 0) == chunks + int(chunks > 1)
    assert len(model.calls) == chunks
    assert actual.shape == (3, 3)
