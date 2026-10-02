"""Offline freezing, provenance and integrity contracts for paired experiments."""

import json
from pathlib import Path
import random
import shutil

import pytest

datasets = pytest.importorskip("datasets")
pytest.importorskip("trl")

from open_r1.evaluation import digest, load_frozen_prompts
from open_r1.paired_data import freeze_data, verify_frozen_data


def chat(question, history="previous response", final="follow up"):
    return [{"role": "system", "content": "original system"},
            {"role": "user", "content": question}, {"role": "assistant", "content": history},
            {"role": "user", "content": final}, {"role": "assistant", "content": "target answer"}]


def save_source(path, train=None, test=None):
    train = [chat(f"train {i}") for i in range(12)] if train is None else train
    test = [chat(f"held out {i}") for i in range(6)] if test is None else test
    datasets.DatasetDict(
        train=datasets.Dataset.from_dict({"prompt": train}),
        test=datasets.Dataset.from_dict({"prompt": test}),
    ).save_to_disk(str(path))
    return path


def freeze(source, target, **kwargs):
    options = dict(steps=2, world_size=2, per_device_train_batch_size=1,
                   gradient_accumulation_steps=2, num_generations=2, num_eval_prompts=3)
    options.update(kwargs)
    return freeze_data(source, target, **options)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_freeze_is_reproducible_preserves_history_and_supports_eval_contract(tmp_path):
    source = save_source(tmp_path / "source")
    first = freeze(source, tmp_path / "first")
    state = random.getstate()
    second = freeze(source, tmp_path / "second")
    assert random.getstate() == state
    assert first["train_count"] == 4 and first["eval_count"] == 3
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["training_schedule_sha256"] == second["training_schedule_sha256"]
    assert first["eval_prompts_sha256"] == second["eval_prompts_sha256"]
    assert first == verify_frozen_data(tmp_path / "first")
    expected = list(range(12))
    random.Random(42).shuffle(expected)
    assert [r["source_row_index"] for r in first["train"]["records"]] == expected[:4]
    assert first["train"]["records"][0]["source_row_sha256"] == digest({"prompt": chat(f"train {expected[0]}")})
    loaded = datasets.load_from_disk(first["dataset_path"])
    assert set(loaded) == {"train"}
    assert set(loaded["train"].column_names) == {"prompt", "comparison_sample_id"}
    rows = list(loaded["train"])
    assert rows[0]["prompt"] == [dict(role="system", content=""), *chat(f"train {expected[0]}")[1:-1]]
    assert digest(rows) == first["dataset_sha256"]
    schedule = read(Path(first["training_schedule_path"]))
    assert schedule["prompt_ids"] == [row["comparison_sample_id"] for row in rows]
    assert schedule["generation_batch_size"] == 4
    prompts, identity = load_frozen_prompts(first["eval_prompts_path"], 3)
    assert identity["prompts_sha256"] == first["eval_prompts_sha256"]
    assert all(prompt[-1]["role"] == "user" for prompt in prompts)


def test_question_dedup_and_eval_leakage_ignore_case_whitespace_unicode_and_history(tmp_path):
    source = save_source(tmp_path / "source", train=[
        chat("Ａ  Question", "history one"), chat("a\tquestion", "history two"),
        chat("A QUESTION", "history three"), chat("second question"),
    ], test=[chat("a question", "unseen history"), chat(" SECOND QUESTION "),
             chat("new evaluation one"), chat("new evaluation two")])
    manifest = freeze(source, tmp_path / "frozen", steps=1, seed=0, eval_seed=0, num_eval_prompts=2)
    assert manifest["train"]["selection"]["duplicate_question"] == 2
    assert manifest["eval"]["selection"]["training_question_overlap"] >= 1
    assert sum(manifest["eval"]["selection"][key] for key in
               ("training_question_overlap", "training_prompt_overlap")) == 2
    train_keys = {r["first_user_sha256"] for r in manifest["train"]["records"]}
    eval_keys = {r["first_user_sha256"] for r in manifest["eval"]["records"]}
    assert not train_keys & eval_keys
    assert verify_frozen_data(tmp_path / "frozen") == manifest


def test_exact_duplicate_prompts_are_counted_separately(tmp_path):
    source = save_source(tmp_path / "source", train=[chat("same")] * 3 + [chat("different")])
    manifest = freeze(source, tmp_path / "frozen", steps=1, seed=0)
    assert manifest["train"]["selection"]["duplicate_prompt"] == 2
    assert manifest["train"]["selection"]["duplicate_question"] == 0


@pytest.mark.parametrize("phase", ["train", "eval"])
def test_insufficient_prompts_fail_without_output_or_partial_directory(tmp_path, phase):
    source = save_source(tmp_path / "source", **{phase if phase == "train" else "test": [chat("only")] * 10})
    with pytest.raises(ValueError, match=f"Insufficient unique, disjoint {phase} prompts"):
        freeze(source, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()
    assert not list(tmp_path.glob(".frozen.*"))


def test_missing_eval_split_is_not_replaced_by_training_data(tmp_path):
    path = tmp_path / "source"
    datasets.DatasetDict(train=datasets.Dataset.from_dict({"prompt": ["one"]})).save_to_disk(str(path))
    with pytest.raises(ValueError, match="Required eval split 'test' is missing"):
        freeze(path, tmp_path / "frozen")


def test_same_source_and_split_are_rejected(tmp_path):
    source = save_source(tmp_path / "source")
    with pytest.raises(ValueError, match="independent source splits"):
        freeze(source, tmp_path / "frozen", eval_source=source, eval_split="train")


def test_external_eval_source_and_column_and_raw_text(tmp_path):
    source = save_source(tmp_path / "source", train=[f"train {i}" for i in range(12)], test=["unused"])
    external = tmp_path / "external"
    datasets.DatasetDict(validation=datasets.Dataset.from_dict({"question": [f"eval {i}" for i in range(4)]})).save_to_disk(str(external))
    manifest = freeze(source, tmp_path / "frozen", eval_source=external, eval_split="validation",
                      eval_prompt_column="question", adapter="raw", system_prompt=None)
    assert all(isinstance(row["prompt"], str) for row in datasets.load_from_disk(manifest["dataset_path"])["train"])
    assert all(isinstance(prompt, str) for prompt in read(Path(manifest["eval_prompts_path"]))["prompts"])
    assert manifest["eval"]["split"] == "validation"
    assert verify_frozen_data(tmp_path / "frozen") == manifest


def test_existing_destination_is_never_overwritten(tmp_path):
    source = save_source(tmp_path / "source")
    target = tmp_path / "frozen"
    manifest = freeze(source, target)
    with pytest.raises(FileExistsError):
        freeze(source, target, seed=99)
    assert verify_frozen_data(target) == manifest


@pytest.mark.parametrize("kwargs", [
    {"steps": 0}, {"world_size": True}, {"num_generations": 1}, {"num_generations": 3},
    {"num_eval_prompts": 0}, {"seed": False}, {"eval_seed": 1.5},
])
def test_invalid_counts_and_seeds_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        freeze("must-not-be-loaded", tmp_path / "frozen", **kwargs)


@pytest.mark.parametrize("artifact,key", [
    ("training_schedule.json", "prompt_ids"), ("training_schedule.json", "steps"),
    ("eval_prompts.json", "prompts"), ("eval_prompts.json", "prompt_ids"),
    ("data_manifest.json", "train_count"),
])
def test_metadata_tampering_is_detected(tmp_path, artifact, key):
    source = save_source(tmp_path / "source")
    target = tmp_path / "frozen"
    freeze(source, target)
    payload = read(target / artifact)
    payload[key] = "tampered"
    write(target / artifact, payload)
    with pytest.raises(ValueError, match="integrity"):
        verify_frozen_data(target)


def test_saved_arrow_rows_are_rehashed_on_every_verification(tmp_path):
    source = save_source(tmp_path / "source")
    target = tmp_path / "frozen"
    freeze(source, target)
    loaded = datasets.load_from_disk(str(target / "dataset"))
    rows = list(loaded["train"])
    rows[0]["prompt"][1]["content"] = "changed after freezing"
    replacement = tmp_path / "replacement"
    datasets.DatasetDict(train=datasets.Dataset.from_list(rows)).save_to_disk(str(replacement))
    shutil.rmtree(target / "dataset")
    replacement.rename(target / "dataset")
    with pytest.raises(ValueError, match="dataset SHA256"):
        verify_frozen_data(target)


def test_atomic_preparation_cleans_temporary_artifacts_on_save_failure(tmp_path, monkeypatch):
    source = save_source(tmp_path / "source")
    def fail(*args, **kwargs):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(datasets.DatasetDict, "save_to_disk", fail)
    with pytest.raises(OSError, match="synthetic disk failure"):
        freeze(source, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()
    assert not list(tmp_path.glob(".frozen.*"))


def test_eval_cannot_have_unknown_schema_fields(tmp_path):
    source = save_source(tmp_path / "source")
    target = tmp_path / "frozen"
    freeze(source, target)
    payload = read(target / "eval_prompts.json")
    payload["extra"] = "unsupported"
    write(target / "eval_prompts.json", payload)
    with pytest.raises(ValueError, match="unexpected"):
        verify_frozen_data(target)
