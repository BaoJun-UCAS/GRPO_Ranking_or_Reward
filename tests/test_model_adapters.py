"""Tokenizer/model configuration contracts, without model downloads or CUDA."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src/open_r1/utils/model_utils.py"


def tokenizer_loader(tokenizer):
    function = next(node for node in ast.parse(SOURCE.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "get_tokenizer")
    auto = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    namespace = {"AutoTokenizer": auto, "ModelConfig": object, "SFTConfig": object, "GRPOConfig": object,
                 "PreTrainedTokenizer": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name], auto


def model_args():
    return SimpleNamespace(model_name_or_path="policy-model", model_revision="policy-commit", trust_remote_code=False)


def test_default_tokenizer_uses_model_revision_and_reuses_eos_for_padding():
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=2, eos_token="</s>", chat_template="original")
    load, auto = tokenizer_loader(tokenizer)
    assert load(model_args(), SimpleNamespace(chat_template=None)) is tokenizer
    auto.from_pretrained.assert_called_once_with("policy-model", revision="policy-commit", trust_remote_code=False)
    assert tokenizer.pad_token == "</s>"
    assert tokenizer.chat_template == "original"


def test_explicit_tokenizer_and_template_override_are_honored():
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, chat_template="original")
    load, auto = tokenizer_loader(tokenizer)
    load(model_args(), SimpleNamespace(tokenizer_name_or_path="compatible-tokenizer", tokenizer_revision="tokenizer-commit",
                                       chat_template="custom"))
    auto.from_pretrained.assert_called_once_with("compatible-tokenizer", revision="tokenizer-commit", trust_remote_code=False)
    assert tokenizer.chat_template == "custom"


def test_separate_tokenizer_does_not_inherit_unrelated_model_commit():
    tokenizer = SimpleNamespace(pad_token_id=0)
    load, auto = tokenizer_loader(tokenizer)
    load(model_args(), SimpleNamespace(tokenizer_name_or_path="compatible-tokenizer", chat_template=None))
    assert auto.from_pretrained.call_args.kwargs["revision"] is None


def test_missing_padding_and_eos_are_rejected_before_training():
    load, _ = tokenizer_loader(SimpleNamespace(pad_token_id=None, eos_token_id=None))
    with pytest.raises(ValueError, match="neither pad_token nor eos_token"):
        load(model_args(), SimpleNamespace(chat_template=None))
