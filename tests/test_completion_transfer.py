"""Bulk token transfer must preserve reward, decoding and loss-mask semantics."""

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def load_helper():
    source = Path(__file__).resolve().parents[1] / "src/open_r1/grpo_trainer.py"
    tree = ast.parse(source.read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "completion_lists_from_tensors")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[helper.name]


@pytest.mark.parametrize("eos_token_id", [0, 2])
def test_bulk_transfer_matches_legacy_first_eos_and_full_row_decoding(eos_token_id):
    # First-token EOS, interior EOS, repeated EOS/padding, no EOS (truncated).
    ids = torch.tensor([
        [eos_token_id, 17, 19, 23, 29],
        [11, eos_token_id, 17, eos_token_id, eos_token_id],
        [11, 13, 17, 19, eos_token_id],
        [11, 13, 17, 19, 23],
    ])
    original = ids.clone()
    eos = ids.eq(eos_token_id)
    ends = torch.full((len(ids),), ids.shape[1], dtype=torch.long)
    ends[eos.any(1)] = eos.int().argmax(1)[eos.any(1)]
    mask = (torch.arange(ids.shape[1]).unsqueeze(0) <= ends.unsqueeze(1)).int()
    legacy = [[token.item() for token, keep in zip(row, valid) if keep]
              for row, valid in zip(ids, mask)]
    padded_rows, reward_rows, lengths = load_helper()(ids, mask.sum(1))
    assert reward_rows == legacy
    assert padded_rows == ids.tolist()  # Including tokens after EOS, as the existing decoder does.
    assert lengths.tolist() == [len(row) for row in legacy]
    assert lengths.device.type == "cpu"
    torch.testing.assert_close(ids, original)
    # Masking a truncated sample for the loss must not erase the reward input.
    loss_mask = mask * eos.any(1).unsqueeze(1)
    assert loss_mask[-1].sum() == 0
    assert reward_rows[-1] == [11, 13, 17, 19, 23]


def test_bulk_transfer_handles_empty_valid_prefix_and_does_not_read_scalar_tensors(monkeypatch):
    ids = torch.tensor([[1, 2], [3, 4]])
    lengths = torch.tensor([0, 2])

    def scalar_read_forbidden(*args, **kwargs):
        raise AssertionError("Per-token scalar reads synchronize CUDA")

    monkeypatch.setattr(torch.Tensor, "item", scalar_read_forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", scalar_read_forbidden)
    padded, valid, cpu_lengths = load_helper()(ids, lengths)
    assert padded == [[1, 2], [3, 4]]
    assert valid == [[], [3, 4]]
    assert cpu_lengths.tolist() == [0, 2]
