"""CPU-only provenance tests for adapter merging; no model downloads."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "generate/merge_lora_adapter.py"
SPEC = importlib.util.spec_from_file_location("merge_lora_adapter", SCRIPT)
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


def write_manifest(path, model="owner/base", revision="training-commit"):
    (path / "run_manifest.json").write_text(json.dumps({
        "model_arguments": {"model_name_or_path": model, "model_revision": revision},
    }))


def test_base_revision_precedence_and_checkpoint_provenance(tmp_path):
    peft_config = SimpleNamespace(base_model_name_or_path="owner/base", revision="adapter-commit")
    assert MERGE.resolve_base_revision(tmp_path, peft_config) == ("adapter-commit", "adapter_config")
    write_manifest(tmp_path)
    assert MERGE.resolve_base_revision(tmp_path, peft_config) == ("training-commit", "run_manifest")
    assert MERGE.resolve_base_revision(tmp_path, peft_config, "explicit") == ("explicit", "cli")
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    assert MERGE.resolve_base_revision(checkpoint, peft_config) == ("training-commit", "run_manifest")


def test_base_model_mismatch_fails_even_with_an_explicit_revision(tmp_path):
    write_manifest(tmp_path, model="owner/different-model")
    peft_config = SimpleNamespace(base_model_name_or_path="owner/base", revision=None)
    with pytest.raises(ValueError, match="does not match adapter base"):
        MERGE.resolve_base_revision(tmp_path, peft_config, "explicit")


def test_missing_provenance_records_default_and_empty_revision_is_invalid(tmp_path):
    peft_config = SimpleNamespace(base_model_name_or_path="owner/base", revision=None)
    assert MERGE.resolve_base_revision(tmp_path, peft_config) == ("main", "default")
    with pytest.raises(ValueError, match="Invalid base model revision"):
        MERGE.resolve_base_revision(tmp_path, peft_config, "")


def test_merge_passes_revision_to_weights_and_fallback_tokenizer_and_records_it(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    write_manifest(adapter)
    output = tmp_path / "merged"
    config = SimpleNamespace(base_model_name_or_path="owner/base", revision=None)
    base_model = SimpleNamespace(config=SimpleNamespace(_commit_hash="resolved-commit"))
    weights_loader = Mock(return_value=base_model)
    tokenizer = SimpleNamespace(save_pretrained=Mock())
    tokenizer_loader = Mock(side_effect=[OSError("No local tokenizer"), tokenizer])
    merged = SimpleNamespace(save_pretrained=Mock())
    wrapped = SimpleNamespace(merge_and_unload=Mock(return_value=merged))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), bfloat16="bf16",
    ))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
        PeftModel=SimpleNamespace(from_pretrained=Mock(return_value=wrapped)),
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=weights_loader),
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
    ))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--adapter", str(adapter), "--output", str(output)])
    MERGE.main()
    assert weights_loader.call_args.kwargs["revision"] == "training-commit"
    assert tokenizer_loader.call_args.kwargs["revision"] == "training-commit"
    manifest = json.loads((output / "merge_manifest.json").read_text())
    assert manifest["base_model_revision"] == "training-commit"
    assert manifest["revision_source"] == "run_manifest"
    assert manifest["base_model_commit"] == "resolved-commit"
    wrapped.merge_and_unload.assert_called_once_with(safe_merge=True)
