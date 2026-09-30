"""Offline contracts for interchangeable training datasets and prompt adapters."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

datasets = pytest.importorskip("datasets")
pytest.importorskip("trl")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("data_under_test", ROOT / "src/open_r1/utils/data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class Tokenizer:
    chat_template = "fixture"

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{message['role']}>{message['content']}" for message in messages) + "<assistant>"


def arguments(**overrides):
    values = dict(dataset_train_split="train", dataset_test_split="test", dataset_prompt_column="prompt",
                  dataset_adapter="auto", max_train_samples=None, max_eval_samples=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def training(**overrides):
    values = dict(do_eval=False, eval_strategy="no", system_prompt=None, max_prompt_length=128)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_saved_datasetdict_and_single_dataset(tmp_path):
    rows = datasets.Dataset.from_dict({"prompt": ["question"], "solution": ["answer"]})
    dataset_dict_path = tmp_path / "splits"
    datasets.DatasetDict(train=rows, validation=rows).save_to_disk(dataset_dict_path)
    loaded = data.get_dataset(SimpleNamespace(dataset_name=str(dataset_dict_path), dataset_config=None,
                                            dataset_mixture=None, dataset_train_split="train"))
    assert set(loaded) == {"train", "validation"}
    assert loaded["train"][0]["solution"] == "answer"
    dataset_path = tmp_path / "single"
    rows.save_to_disk(dataset_path)
    loaded = data._load_dataset(str(dataset_path), default_split="custom")
    assert set(loaded) == {"custom"}


@pytest.mark.parametrize("extension", ["jsonl", "csv", "parquet"])
def test_load_local_files(tmp_path, monkeypatch, extension):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    path = tmp_path / f"examples.{extension}"
    rows = datasets.Dataset.from_dict({"prompt": ["question"], "solution": ["answer"]})
    if extension == "jsonl":
        rows.to_json(path)
    elif extension == "csv":
        rows.to_csv(path)
    else:
        rows.to_parquet(path)
    assert data._load_dataset(str(path))["train"][0] == {"prompt": "question", "solution": "answer"}


def test_hub_loader_preserves_dataset_configuration(monkeypatch):
    expected = datasets.DatasetDict(train=datasets.Dataset.from_dict({"prompt": ["question"]}))
    calls = []
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: calls.append((args, kwargs)) or expected)
    assert data._load_dataset("organization/dataset", "subset") is expected
    assert calls == [(("organization/dataset", "subset"), {})]


def test_chat_adapter_removes_only_target_and_does_not_mutate_history():
    original = [{"role": "system", "content": "old"}, {"role": "user", "content": "first"},
                {"role": "assistant", "content": "previous answer"}, {"role": "user", "content": "second"},
                {"role": "assistant", "content": "target answer"}]
    output = data.auto_adapter({"messages": original}, prompt_column="prompt", system_prompt="new")["prompt"]
    assert output == [{"role": "system", "content": "new"}, *original[1:-1]]
    assert original[0]["content"] == "old"
    assert original[-1]["content"] == "target answer"


@pytest.mark.parametrize("value", [[], 1, [{"role": "user", "content": ["image"]}],
                                   [{"role": "system", "content": "instructions"}]])
def test_bad_chat_examples_fail_with_useful_errors(value):
    with pytest.raises(ValueError):
        data.auto_adapter({"prompt": value}, prompt_column="prompt", system_prompt=None)


def test_prepare_caps_before_mapping_preserves_reward_columns_and_skips_unused_splits():
    calls = []

    @contextmanager
    def main_first(**kwargs):
        calls.append(kwargs)
        yield

    original = datasets.DatasetDict(
        train=datasets.Dataset.from_dict({"prompt": ["first", "second"], "solution": ["a", "b"]}),
        test=datasets.Dataset.from_dict({"unusable": ["never mapped"]}),
    )
    result = data.prepare_grpo_dataset(original, arguments(max_train_samples=1),
                                      training(main_process_first=main_first), Tokenizer())
    assert list(result) == ["train"]
    assert result["train"][:] == {"prompt": [[{"role": "user", "content": "first"}]], "solution": ["a"]}
    assert calls == [{"desc": "Prepare GRPO dataset"}]
    assert original["train"][0]["prompt"] == "first"


def test_prepare_end_only_eval_maps_messages_without_answer_leakage():
    rows = datasets.Dataset.from_dict({"messages": [[{"role": "user", "content": "p"},
                                                    {"role": "assistant", "content": "secret target"}]] * 2,
                                      "solution": ["a", "b"]})
    result = data.prepare_grpo_dataset(datasets.DatasetDict(train=rows, test=rows),
                                      arguments(max_eval_samples=1), training(do_eval=True), Tokenizer())
    assert len(result["test"]) == 1
    assert "messages" not in result["test"].column_names
    assert result["test"][0]["prompt"] == [{"role": "user", "content": "p"}]
    assert result["test"][0]["solution"] == "a"


def test_prepare_rejects_missing_split_before_mapping():
    rows = datasets.Dataset.from_dict({"prompt": ["p"]})
    with pytest.raises(ValueError, match="Required dataset split 'test'"):
        data.prepare_grpo_dataset(datasets.DatasetDict(train=rows), arguments(), training(do_eval=True), Tokenizer())


def test_prepare_validates_chat_template_and_sample_count():
    rows = datasets.DatasetDict(train=datasets.Dataset.from_dict({"prompt": ["p"]}))
    with pytest.raises(ValueError, match="positive integers"):
        data.prepare_grpo_dataset(rows, arguments(max_train_samples=0), training(), Tokenizer())
    with pytest.raises(ValueError, match="chat template"):
        data.prepare_grpo_dataset(rows, arguments(), training(), SimpleNamespace(chat_template=None))


def test_external_adapter_can_add_reward_inputs(tmp_path, monkeypatch):
    module = ModuleType("fixture_dataset_adapter")

    def adapt(example, *, prompt_column, system_prompt):
        return {"prompt": example["question"], "solution": str(example["answer"])}

    module.adapt = adapt
    monkeypatch.setitem(sys.modules, module.__name__, module)
    rows = datasets.DatasetDict(train=datasets.Dataset.from_dict({"question": ["abcdefghij"], "answer": [42]}))
    result = data.prepare_grpo_dataset(rows, arguments(dataset_adapter="fixture_dataset_adapter:adapt"),
                                      training(max_prompt_length=4), Tokenizer())
    assert result["train"][0] == {"question": "abcdefghij", "answer": 42, "prompt": "ghij", "solution": "42"}


def test_truncation_keeps_history_and_exact_token_budget():
    tokenizer = Tokenizer()
    prompt = [{"role": "system", "content": "rules"}, {"role": "user", "content": "abcdefghij"}]
    overhead = len(tokenizer.apply_chat_template([prompt[0], {"role": "user", "content": ""}]))
    truncated = data.truncate_conversation_prompt(prompt, tokenizer, overhead + 4)
    assert truncated == [prompt[0], {"role": "user", "content": "ghij"}]
    assert prompt[-1]["content"] == "abcdefghij"
    with pytest.raises(ValueError, match="overhead exceeds"):
        data.truncate_conversation_prompt(prompt, tokenizer, overhead - 1)



def test_raw_prompts_support_models_without_a_chat_template():
    rows = datasets.DatasetDict(train=datasets.Dataset.from_dict({"prompt": ["question"]}))
    tokenizer = Tokenizer()
    tokenizer.chat_template = None
    result = data.prepare_grpo_dataset(rows, arguments(dataset_adapter="raw"), training(), tokenizer)
    assert result["train"][0]["prompt"] == "question"
    assert data.raw_adapter({"p": "question"}, prompt_column="p", system_prompt="rules") == {
        "prompt": "rules\n\nquestion"
    }


def test_preprocessing_cache_is_shared_between_distributed_ranks(tmp_path):
    rows = datasets.DatasetDict(train=datasets.Dataset.from_dict({"prompt": ["question"]}))
    rows.save_to_disk(tmp_path / "data")
    loaded = datasets.load_from_disk(tmp_path / "data")
    rank0 = data.prepare_grpo_dataset(loaded, arguments(), training(local_rank=0), Tokenizer())
    rank1 = data.prepare_grpo_dataset(loaded, arguments(), training(local_rank=1), Tokenizer())
    assert rank0["train"]._fingerprint == rank1["train"]._fingerprint
    assert rank0["train"].cache_files == rank1["train"].cache_files



def test_training_sample_cap_cannot_silently_empty_the_grpo_sampler():
    rows = datasets.DatasetDict(train=datasets.Dataset.from_dict({"prompt": ["a", "b", "c", "d"]}))
    args = training(generation_batch_size=8, num_generations=2)
    with pytest.raises(ValueError, match="GRPO requires at least.*4 unique prompt"):
        data.prepare_grpo_dataset(rows, arguments(max_train_samples=2), args, Tokenizer())
    assert len(data.prepare_grpo_dataset(rows, arguments(), args, Tokenizer())["train"]) == 4
