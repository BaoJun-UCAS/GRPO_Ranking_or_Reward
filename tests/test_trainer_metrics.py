"""Exercise the real policy-loss method with a two-rank tensor collective stub."""

import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from open_r1.objectives import reduce_policy_loss


def load_policy_loss_method():
    # Run the actual method without importing optional vLLM or initializing CUDA.
    source = Path(__file__).resolve().parents[1] / "src/open_r1/grpo_trainer.py"
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss")
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"nanmin", "nanmax"}]
    namespace = {"torch": torch, "reduce_policy_loss": reduce_policy_loss}
    exec(compile(ast.Module(body=helpers + [method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_compute_loss"]


@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("fully_masked", [False, True])
def test_policy_metrics_use_one_collective_and_preserve_rank_reductions(beta, training, fully_masked, monkeypatch):
    policy_loss = load_policy_loss_method()
    model = SimpleNamespace(training=training)
    reference = object()
    logps = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], requires_grad=True)
    ratios = torch.tensor([[0.5, 1.5], [1.5, 0.5]])
    old_logps = logps.detach() - ratios.log()
    reference_logps = logps.detach() + 0.1
    mask = torch.zeros(2, 2, dtype=torch.long) if fully_masked else torch.tensor([[1, 0], [1, 1]])
    collected = []
    host_transfers = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        host_transfers.append((tuple(tensor.shape), tensor.requires_grad))
        return original_cpu(tensor, *args, **kwargs)

    # Verify a single bulk host transfer, not one transfer per statistic.
    # The gather stub and reductions still operate on real Torch tensors.
    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)

    def gather(local):
        collected.append(local)
        assert local.shape == (1, 4 if beta else 3)
        assert not local.requires_grad
        # Different remote values and NaNs make flattening/misordered columns
        # fail, and verify the original nanmean/nanmin/nanmax behavior.
        remote = [float("nan"), 0.75, 0.25]
        if beta:
            remote.append(float("nan"))
        return torch.cat((local, local.new_tensor([remote])), dim=0)

    def get_logps(current_model, *_args):
        return reference_logps if current_model is reference else logps

    trainer = SimpleNamespace(
        model=model, ref_model=reference, beta=beta, epsilon_low=0.2, epsilon_high=0.2,
        args=SimpleNamespace(delta=None, token_broadcast="uniform"),
        loss_type="bnpo", max_completion_length=2,
        _get_per_token_logps=get_logps, accelerator=SimpleNamespace(gather=gather),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )
    inputs = {
        "prompt_ids": torch.tensor([[1], [1]]), "prompt_mask": torch.ones(2, 1, dtype=torch.long),
        "completion_ids": torch.tensor([[2, 3], [4, 5]]), "completion_mask": mask,
        "advantages": torch.tensor([-1.0, 1.0]), "old_per_token_logps": old_logps,
    }
    result = policy_loss(trainer, model, inputs)
    assert len(collected) == 1
    assert host_transfers == [((2, 4 if beta else 3), False)]
    metrics = trainer._metrics["train" if training else "eval"]
    local_clip = 0.0 if fully_masked else 1 / 3
    local_region = 0.0 if fully_masked else 2 / 3
    assert metrics["clip_ratio/low_mean"] == pytest.approx([local_clip])
    assert metrics["clip_ratio/low_min"] == pytest.approx([local_clip])
    assert metrics["clip_ratio/high_mean"] == pytest.approx([(local_clip + 0.75) / 2])
    assert metrics["clip_ratio/high_max"] == pytest.approx([0.75])
    assert metrics["clip_ratio/region_mean"] == pytest.approx([(local_region + 0.25) / 2])
    if beta:
        differences = reference_logps - logps.detach()
        kl = ((differences.exp() - differences - 1) * mask).sum() / mask.sum().clamp(min=1)
        assert metrics["kl"] == pytest.approx([kl.item()])
    else:
        assert "kl" not in metrics
    result.backward()
    assert torch.isfinite(result)
    assert torch.isfinite(logps.grad).all()
    assert logps.grad[mask == 0].eq(0).all()
    if fully_masked:
        assert result.item() == 0
        assert logps.grad.eq(0).all()


@pytest.mark.parametrize("method_name", ["_get_per_token_logps", "_get_last_hidden_state"])
def test_scoring_explicitly_disables_kv_cache_for_right_padded_completions(method_name):
    source = Path(__file__).resolve().parents[1] / "src/open_r1/grpo_trainer.py"
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    scope = {
        "torch": torch, "is_peft_model": lambda model: False,
        "selective_log_softmax": lambda logits, ids: logits.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    calls = []

    class Model:
        config = SimpleNamespace(use_cache=True)

        def __call__(self, **kwargs):
            assert kwargs.get("use_cache") is False
            calls.append(kwargs)
            batch, length = kwargs["input_ids"].shape
            return SimpleNamespace(logits=torch.zeros(batch, length, 8),
                                   last_hidden_state=torch.zeros(batch, length, 3))

    model = Model()
    model.model = model
    ids = torch.tensor([[1, 2, 3, 0], [1, 2, 4, 5]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    result = scope[method_name](SimpleNamespace(temperature=1.0), model, ids, mask, logits_to_keep=2)
    assert len(calls) == 1
    assert result.shape[:2] == (2, 2)
    assert model.config.use_cache is True  # The explicit call does not mutate inference settings.
