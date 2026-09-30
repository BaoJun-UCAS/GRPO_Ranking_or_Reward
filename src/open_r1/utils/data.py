"""Dataset loading and the prompt contract shared by GRPO experiments.

Adapters return a mapping containing ``prompt`` (text or chat messages), and may
add reward inputs such as ``solution``. Original columns are preserved. Register
an adapter or set ``dataset_adapter: package.module:function`` in a recipe to
experiment with a new dataset without modifying the trainer.
"""

from __future__ import annotations

from contextlib import nullcontext
from importlib import import_module
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets

if TYPE_CHECKING:
    from ..configs import ScriptArguments


logger = logging.getLogger(__name__)


def _load_dataset(source, config=None, *, default_split="train") -> DatasetDict:
    """Load Hub datasets, saved Arrow datasets, or a local data file."""
    path = Path(source).expanduser()
    if path.is_dir() and ((path / "dataset_dict.json").exists() or (path / "state.json").exists()):
        loaded = datasets.load_from_disk(str(path))
    elif path.is_file():
        file_format = {".json": "json", ".jsonl": "json", ".parquet": "parquet", ".csv": "csv"}.get(
            path.suffix.lower()
        )
        if file_format is None:
            raise ValueError(f"Unsupported dataset file {path}; use JSON, JSONL, Parquet, or CSV")
        loaded = datasets.load_dataset(file_format, data_files={default_split: str(path)})
    else:
        loaded = datasets.load_dataset(source, config)
    if isinstance(loaded, Dataset):
        return DatasetDict({default_split: loaded})
    return loaded


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        return _load_dataset(
            args.dataset_name, args.dataset_config, default_split=getattr(args, "dataset_train_split", "train")
        )
    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
            if Path(dataset_config.id).expanduser().exists():
                loaded = _load_dataset(dataset_config.id, dataset_config.config, default_split=dataset_config.split)
                if dataset_config.split not in loaded:
                    raise ValueError(f"Dataset {dataset_config.id!r} has no split {dataset_config.split!r}")
                ds = loaded[dataset_config.split]
            else:
                ds = datasets.load_dataset(dataset_config.id, dataset_config.config, split=dataset_config.split)
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                if not 0 < dataset_config.weight <= 1:
                    raise ValueError("Dataset mixture weights are sampling fractions and must be in (0, 1]")
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )
                return combined_dataset
            else:
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")



def _chat_prompt(value, system_prompt):
    if not isinstance(value, list) or not value:
        raise ValueError("A conversational prompt must be a non-empty list of role/content messages")
    messages = []
    for message in value:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Each chat message must have a supported role and string content")
        if not isinstance(message.get("content"), str):
            raise ValueError("Only text chat content is supported; use a custom dataset adapter for other schemas")
        messages.append(dict(message))
    # A supervised dataset's final answer is a target, never a rollout input.
    if messages[-1]["role"] == "assistant":
        messages.pop()
    if not messages or messages[-1]["role"] not in {"user", "tool"}:
        raise ValueError("A GRPO conversation must end with a user/tool message after removing the final answer")
    if system_prompt is not None:
        if messages[0]["role"] == "system":
            messages[0] = {"role": "system", "content": system_prompt}
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def text_adapter(example, *, prompt_column, system_prompt):
    value = example[prompt_column]
    if not isinstance(value, str):
        raise ValueError(f"Text prompt column {prompt_column!r} must contain strings")
    return {"prompt": _chat_prompt([{"role": "user", "content": value}], system_prompt)}


def raw_adapter(example, *, prompt_column, system_prompt):
    """Use plain text directly for base models without a chat template."""
    value = example[prompt_column]
    if not isinstance(value, str):
        raise ValueError(f"Raw prompt column {prompt_column!r} must contain strings")
    if system_prompt is not None:
        value = f"{system_prompt}\n\n{value}"
    return {"prompt": value}


def chat_adapter(example, *, prompt_column, system_prompt):
    return {"prompt": _chat_prompt(example[prompt_column], system_prompt)}


def auto_adapter(example, *, prompt_column, system_prompt):
    if prompt_column not in example and prompt_column == "prompt" and "messages" in example:
        prompt_column = "messages"
    if prompt_column not in example:
        raise ValueError(f"Missing dataset prompt column {prompt_column!r}; available columns: {sorted(example)}")
    adapter = text_adapter if isinstance(example[prompt_column], str) else chat_adapter
    return adapter(example, prompt_column=prompt_column, system_prompt=system_prompt)


DATASET_ADAPTERS: dict[str, Callable] = {
    "auto": auto_adapter, "text": text_adapter, "chat": chat_adapter, "raw": raw_adapter
}


def register_dataset_adapter(name: str, adapter: Callable):
    """Register ``adapter(example, *, prompt_column, system_prompt) -> dict``."""
    if not name or ":" in name or name in DATASET_ADAPTERS:
        raise ValueError(f"Dataset adapter name is empty, reserved, or already registered: {name!r}")
    if not callable(adapter):
        raise TypeError("A dataset adapter must be callable")
    DATASET_ADAPTERS[name] = adapter


def get_dataset_adapter(name: str) -> Callable:
    if name in DATASET_ADAPTERS:
        return DATASET_ADAPTERS[name]
    if ":" in name:
        module, attribute = name.rsplit(":", 1)
        adapter = getattr(import_module(module), attribute)
        if not callable(adapter):
            raise TypeError(f"Dataset adapter {name!r} is not callable")
        return adapter
    raise ValueError(f"Unknown dataset adapter {name!r}; use {sorted(DATASET_ADAPTERS)} or module:function")


def truncate_conversation_prompt(prompt, tokenizer, max_prompt_length):
    """Left-truncate the last user turn while keeping chat and reward inputs aligned."""
    if max_prompt_length is None:
        return prompt
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")

    from trl.data_utils import maybe_apply_chat_template

    def rendered_length(messages):
        rendered = maybe_apply_chat_template({"prompt": messages}, tokenizer)["prompt"]
        return len(tokenizer(text=rendered, add_special_tokens=False)["input_ids"])

    if rendered_length(prompt) <= max_prompt_length:
        return prompt
    user_index = next((i for i in range(len(prompt) - 1, -1, -1) if prompt[i]["role"] == "user"), None)
    if user_index is None:
        raise ValueError("Cannot truncate an overlong conversational prompt without a user message")
    content_ids = tokenizer.encode(prompt[user_index]["content"], add_special_tokens=False)
    low, high, best = 0, len(content_ids), None
    while low <= high:
        keep = (low + high) // 2
        content = tokenizer.decode(
            content_ids[-keep:] if keep else [], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        candidate = [dict(message) for message in prompt]
        candidate[user_index]["content"] = content
        if rendered_length(candidate) <= max_prompt_length:
            best, low = candidate, keep + 1
        else:
            high = keep - 1
    if best is None:
        raise ValueError(
            f"Chat-template/history overhead exceeds max_prompt_length={max_prompt_length}; "
            "increase the limit or shorten history with a custom dataset adapter"
        )
    return best


def prepare_grpo_dataset(dataset, script_args, training_args, tokenizer) -> DatasetDict:
    """Validate and prepare only the splits used by the run, with optional sample caps."""
    adapter = get_dataset_adapter(getattr(script_args, "dataset_adapter", "auto"))
    train_split = script_args.dataset_train_split
    eval_split = script_args.dataset_test_split
    use_eval = training_args.do_eval or training_args.eval_strategy != "no"
    required = [(train_split, getattr(script_args, "max_train_samples", None))]
    if use_eval:
        if train_split == eval_split:
            raise ValueError("Training and evaluation must use different dataset splits")
        required.append((eval_split, getattr(script_args, "max_eval_samples", None)))
    selected = DatasetDict()
    for split, limit in required:
        if split not in dataset:
            raise ValueError(f"Required dataset split {split!r} is missing; available splits: {list(dataset)}")
        subset = dataset[split]
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ValueError("max_train_samples and max_eval_samples must be positive integers")
            subset = subset.select(range(min(limit, len(subset))))
        if not len(subset):
            raise ValueError(f"Dataset split {split!r} is empty")
        selected[split] = subset

    generation_batch_size = getattr(training_args, "generation_batch_size", None)
    num_generations = getattr(training_args, "num_generations", None)
    if generation_batch_size and num_generations:
        minimum_prompts = generation_batch_size // num_generations
        if len(selected[train_split]) < minimum_prompts:
            raise ValueError(
                f"Training split has {len(selected[train_split])} samples, but GRPO requires at least "
                f"generation_batch_size / num_generations = {minimum_prompts} unique prompt rows per rollout. "
                "Increase max_train_samples or reduce generation_batch_size."
            )

    # Capture only rank-independent preprocessing options. Capturing the whole
    # TrainingArguments object changes the datasets cache fingerprint per rank.
    prompt_column = script_args.dataset_prompt_column
    system_prompt = training_args.system_prompt
    max_prompt_length = training_args.max_prompt_length

    def prepare(example):
        output = adapter(example, prompt_column=prompt_column, system_prompt=system_prompt)
        if not isinstance(output, dict) or "prompt" not in output:
            raise ValueError("A dataset adapter must return a dictionary containing 'prompt'")
        output = dict(output)
        prompt = output["prompt"]
        if isinstance(prompt, list):
            if not tokenizer.chat_template:
                raise ValueError("Conversational prompts require a tokenizer chat template; set chat_template in the recipe")
            output["prompt"] = truncate_conversation_prompt(prompt, tokenizer, max_prompt_length)
        elif isinstance(prompt, str):
            limit = max_prompt_length
            if limit is not None:
                if limit <= 0:
                    raise ValueError("max_prompt_length must be positive")
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(ids) > limit:
                    output["prompt"] = tokenizer.decode(ids[-limit:], skip_special_tokens=False)
        else:
            raise ValueError("An adapter's prompt must be a string or a list of chat messages")
        return output

    main_process_first = getattr(training_args, "main_process_first", None)
    context = main_process_first(desc="Prepare GRPO dataset") if main_process_first else nullcontext()
    with context:
        for split in selected:
            # A leftover messages column can cause TRL to infer a second chat input.
            remove_columns = ["messages"] if "messages" in selected[split].column_names else None
            selected[split] = selected[split].map(prepare, remove_columns=remove_columns, desc=f"Prepare {split} prompts")
    return selected
