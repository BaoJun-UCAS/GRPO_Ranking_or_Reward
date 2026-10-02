"""Freeze disjoint, ordered prompts for paired training and held-out evaluation.

Freezing never loads a tokenizer/model or truncates conversation history. The
training recipe must use ``shuffle_dataset=False`` and the saved schedule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import unicodedata

from .evaluation import digest, frozen_prompt_messages


def _positive(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _prompt_keys(prompt):
    messages = frozen_prompt_messages(prompt)
    first_user = next(message["content"] for message in messages if message["role"] == "user")
    question = _normalize(first_user)
    if not question:
        raise ValueError("The first user question must not be empty")
    normalized = [{"role": m["role"], "content": _normalize(m["content"])} for m in messages]
    return digest(normalized), digest(question)


def _source_name(source):
    source = str(source)
    path = Path(source).expanduser()
    return str(path.resolve()) if path.exists() else source


def _sample_id(phase, index, prompt):
    return f"{phase}_{index:09d}_{digest(prompt)}"


def _select(dataset, count, *, phase, adapter, prompt_column, system_prompt, seed, blocked=None):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected, records = [], []
    seen_prompts, seen_questions = set(), set()
    blocked_prompts, blocked_questions = blocked or (set(), set())
    stats = dict(examined_rows=0, duplicate_prompt=0, duplicate_question=0,
                 training_prompt_overlap=0, training_question_overlap=0)
    for index in indices:
        stats["examined_rows"] += 1
        try:
            source_row = dataset[index]
            output = adapter(source_row, prompt_column=prompt_column, system_prompt=system_prompt)
            if not isinstance(output, dict) or "prompt" not in output:
                raise ValueError("Dataset adapter must return a mapping containing prompt")
            prompt = output["prompt"]
            prompt_key, question_key = _prompt_keys(prompt)
        except (ValueError, TypeError, KeyError, StopIteration) as error:
            raise ValueError(f"Invalid {phase} source row {index}: {error}") from error
        reason = (
            "training_prompt_overlap" if prompt_key in blocked_prompts else
            "training_question_overlap" if question_key in blocked_questions else
            "duplicate_prompt" if prompt_key in seen_prompts else
            "duplicate_question" if question_key in seen_questions else None
        )
        if reason is not None:
            stats[reason] += 1
            continue
        sample_id = _sample_id(phase, index, prompt)
        selected.append({"prompt": prompt, "comparison_sample_id": sample_id})
        records.append({"source_row_index": index, "source_row_sha256": digest(source_row), "prompt_id": sample_id,
                        "prompt_sha256": digest(prompt), "normalized_prompt_sha256": prompt_key,
                        "first_user_sha256": question_key})
        seen_prompts.add(prompt_key)
        seen_questions.add(question_key)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Insufficient unique, disjoint {phase} prompts: requested {count}, "
                         f"available {len(selected)} after examining {stats['examined_rows']} rows")
    return selected, records, stats, (seen_prompts, seen_questions)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read frozen artifact {path}: {error}") from error


def freeze_data(source, directory, *, train_split="train", eval_source=None, eval_split="test",
                prompt_column="prompt", eval_prompt_column=None, adapter="auto", system_prompt="",
                steps=200, world_size=2, per_device_train_batch_size=1,
                gradient_accumulation_steps=128, num_generations=8, seed=42, eval_seed=31415,
                num_eval_prompts=200):
    """Create a new frozen directory atomically; return its data manifest.

    Sampling uses independent stdlib RNGs and rejects duplicates by full prompt
    and normalized first user question. Counts in ``selection`` describe the
    candidates examined until the requested sample size was reached.
    """
    directory = Path(directory).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"Frozen data directory already exists: {directory}")
    parameters = dict(steps=steps, world_size=world_size,
                      per_device_train_batch_size=per_device_train_batch_size,
                      gradient_accumulation_steps=gradient_accumulation_steps,
                      num_generations=num_generations)
    for name, value in parameters.items():
        _positive(name, value)
    _positive("num_eval_prompts", num_eval_prompts)
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2")
    if type(seed) is not int or type(eval_seed) is not int:
        raise ValueError("seed and eval_seed must be integers")
    generation_batch_size = world_size * per_device_train_batch_size * gradient_accumulation_steps
    if generation_batch_size % num_generations:
        raise ValueError("generation_batch_size must be divisible by num_generations")
    source = _source_name(source)
    eval_source = source if eval_source is None else _source_name(eval_source)
    if source == eval_source and train_split == eval_split:
        raise ValueError("Training and evaluation must use independent source splits")
    eval_prompt_column = eval_prompt_column or prompt_column

    from datasets import Dataset, DatasetDict
    from .utils.data import _load_dataset, get_dataset_adapter

    loaded = _load_dataset(source, default_split=train_split)
    eval_loaded = loaded if source == eval_source else _load_dataset(eval_source, default_split=eval_split)
    for phase, splits, name in (("train", loaded, train_split), ("eval", eval_loaded, eval_split)):
        if name not in splits:
            raise ValueError(f"Required {phase} split {name!r} is missing; available splits: {list(splits)}")
    train_dataset, eval_dataset = loaded[train_split], eval_loaded[eval_split]
    adapt = get_dataset_adapter(adapter)
    train_count = steps * generation_batch_size // num_generations
    train_rows, train_records, train_stats, training_keys = _select(
        train_dataset, train_count, phase="train", adapter=adapt, prompt_column=prompt_column,
        system_prompt=system_prompt, seed=seed)
    eval_rows, eval_records, eval_stats, _ = _select(
        eval_dataset, num_eval_prompts, phase="eval", adapter=adapt, prompt_column=eval_prompt_column,
        system_prompt=system_prompt, seed=eval_seed, blocked=training_keys)
    schedule = dict(version=1, **parameters, generation_batch_size=generation_batch_size, seed=seed,
                    prompt_ids=[row["comparison_sample_id"] for row in train_rows],
                    dataset_sha256=digest(train_rows))
    evaluation = dict(version=1, prompts=[row["prompt"] for row in eval_rows],
                      prompt_ids=[row["comparison_sample_id"] for row in eval_rows])
    evaluation["prompts_sha256"] = digest(evaluation["prompts"])
    manifest = dict(
        version=1, directory=str(directory), dataset_path=str(directory / "dataset"),
        training_schedule_path=str(directory / "training_schedule.json"),
        eval_prompts_path=str(directory / "eval_prompts.json"),
        train_count=train_count, eval_count=num_eval_prompts,
        dataset_sha256=schedule["dataset_sha256"], eval_prompts_sha256=evaluation["prompts_sha256"],
        training_schedule_sha256=digest(schedule), eval_payload_sha256=digest(evaluation),
        parameters=dict(**parameters, generation_batch_size=generation_batch_size, seed=seed,
                        eval_seed=eval_seed, adapter=adapter, prompt_column=prompt_column,
                        eval_prompt_column=eval_prompt_column, system_prompt=system_prompt),
        normalization="NFKC + casefold + whitespace collapse; full prompt and first user question",
    )
    for phase, ds, location, split, records, stats in (
        ("train", train_dataset, source, train_split, train_records, train_stats),
        ("eval", eval_dataset, eval_source, eval_split, eval_records, eval_stats),
    ):
        manifest[phase] = dict(source=location, split=split, fingerprint=ds._fingerprint,
                               source_rows=len(ds), selected_count=len(records), records=records, selection=stats)
    manifest["manifest_sha256"] = digest(manifest)

    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        DatasetDict(train=Dataset.from_list(train_rows)).save_to_disk(str(temporary / "dataset"))
        for name, value in (("training_schedule.json", schedule), ("eval_prompts.json", evaluation),
                            ("data_manifest.json", manifest)):
            (temporary / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify_frozen_data(temporary)
        if directory.exists():
            raise FileExistsError(f"Frozen data directory already exists: {directory}")
        os.rename(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def verify_frozen_data(directory):
    """Re-read frozen rows and all contracts; raise ValueError on any drift."""
    from datasets import DatasetDict, load_from_disk

    directory = Path(directory).expanduser().resolve()
    manifest = _read_json(directory / "data_manifest.json")
    schedule = _read_json(directory / "training_schedule.json")
    evaluation = _read_json(directory / "eval_prompts.json")
    if not all(isinstance(item, dict) and type(item.get("version")) is int and item["version"] == 1
               for item in (manifest, schedule, evaluation)):
        raise ValueError("Frozen data artifacts require version=1")
    expected_manifest = dict(manifest)
    expected_manifest.pop("manifest_sha256", None)
    if manifest.get("manifest_sha256") != digest(expected_manifest):
        raise ValueError("Frozen data manifest integrity check failed")
    try:
        loaded = load_from_disk(str(directory / "dataset"))
        if not isinstance(loaded, DatasetDict) or set(loaded) != {"train"}:
            raise ValueError("Frozen dataset must contain only the train split")
        if set(loaded["train"].column_names) != {"prompt", "comparison_sample_id"}:
            raise ValueError("Frozen train rows require prompt and comparison_sample_id columns")
        rows = list(loaded["train"])
        if digest(rows) != schedule["dataset_sha256"] or digest(rows) != manifest["dataset_sha256"]:
            raise ValueError("Frozen dataset SHA256 integrity check failed")
        if digest(schedule) != manifest["training_schedule_sha256"]:
            raise ValueError("Frozen training schedule integrity check failed")
        if set(evaluation) != {"version", "prompts", "prompt_ids", "prompts_sha256"}:
            raise ValueError("Frozen eval prompts contain unexpected or missing fields")
        if digest(evaluation) != manifest["eval_payload_sha256"]:
            raise ValueError("Frozen eval payload integrity check failed")
        if digest(evaluation["prompts"]) != evaluation["prompts_sha256"] or (
            evaluation["prompts_sha256"] != manifest["eval_prompts_sha256"]
        ):
            raise ValueError("Frozen evaluation prompts SHA256 integrity check failed")
        for key in ("steps", "world_size", "num_generations", "generation_batch_size",
                    "gradient_accumulation_steps", "per_device_train_batch_size"):
            _positive(key, schedule[key])
            if schedule[key] != manifest["parameters"][key]:
                raise ValueError(f"Frozen schedule parameter mismatch: {key}")
        if schedule["generation_batch_size"] != (schedule["world_size"] *
                schedule["per_device_train_batch_size"] * schedule["gradient_accumulation_steps"]):
            raise ValueError("Frozen schedule has an inconsistent generation batch size")
        if (schedule["num_generations"] < 2 or
                schedule["generation_batch_size"] % schedule["num_generations"] or
                schedule["seed"] != manifest["parameters"]["seed"]):
            raise ValueError("Frozen schedule generation groups or seed are invalid")
        expected_count = schedule["steps"] * schedule["generation_batch_size"] // schedule["num_generations"]
        if len(rows) != expected_count or len(rows) != manifest["train_count"]:
            raise ValueError("Frozen training row count does not match the schedule")
        if schedule["prompt_ids"] != [row["comparison_sample_id"] for row in rows]:
            raise ValueError("Frozen schedule prompt IDs do not match dataset row order")
        if len(evaluation["prompts"]) != manifest["eval_count"] or manifest["eval_count"] <= 0:
            raise ValueError("Frozen evaluation prompt count mismatch")
        training_keys = (set(), set())
        for phase, prompts, ids, count in (
            ("train", [row["prompt"] for row in rows], schedule["prompt_ids"], manifest["train_count"]),
            ("eval", evaluation["prompts"], evaluation["prompt_ids"], manifest["eval_count"]),
        ):
            metadata = manifest[phase]
            if len(ids) != count or len(metadata["records"]) != count or metadata["selected_count"] != count:
                raise ValueError(f"Frozen {phase} IDs or provenance count mismatch")
            if any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != count:
                raise ValueError(f"Frozen {phase} prompt IDs must be unique nonempty strings")
            seen = (set(), set())
            source_indices = set()
            for prompt, sample_id, record in zip(prompts, ids, metadata["records"]):
                index = record["source_row_index"]
                if type(index) is not int or not 0 <= index < metadata["source_rows"] or index in source_indices:
                    raise ValueError(f"Frozen {phase} source indices are invalid")
                source_indices.add(index)
                keys = _prompt_keys(prompt)
                if (sample_id != _sample_id(phase, index, prompt) or record["prompt_id"] != sample_id or
                        record["prompt_sha256"] != digest(prompt) or
                        record["normalized_prompt_sha256"] != keys[0] or record["first_user_sha256"] != keys[1]):
                    raise ValueError(f"Frozen {phase} prompt provenance mismatch")
                if any(key in seen[position] for position, key in enumerate(keys)):
                    raise ValueError(f"Frozen {phase} contains duplicate prompts or first-user questions")
                if phase == "eval" and any(key in training_keys[position] for position, key in enumerate(keys)):
                    raise ValueError("Frozen evaluation leaks training prompts or first-user questions")
                for position, key in enumerate(keys):
                    seen[position].add(key)
            if phase == "train":
                training_keys = seen
        if (manifest["train"]["source"] == manifest["eval"]["source"] and
                manifest["train"]["split"] == manifest["eval"]["split"]):
            raise ValueError("Frozen training and evaluation source splits must be independent")
    except (KeyError, TypeError, AttributeError, OSError) as error:
        raise ValueError(f"Invalid frozen data artifacts: {error}") from error
    return manifest
