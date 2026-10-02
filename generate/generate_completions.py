#!/usr/bin/env python3
"""
Generation-only CLI that produces a reusable completions artifact JSON
containing single model completions {prompt, completion}.

example usage:
python generate/generate_completions.py \
    --model "HF directory" \
    --num-prompts 100 \
    --seed 42
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any, Optional
import math
import sys
from pathlib import Path
from importlib.metadata import version

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_r1.evaluation import (
    ARTIFACT_VERSION, digest, model_fingerprint, valid_generation_cache, prompt_token_ids,
    load_frozen_prompts, frozen_prompt_messages, frozen_prompt_text,
)


def load_validation_prompts(num_prompts: int, seed: int, model_path: str = None, dataset_override: str = None,
                            dataset_id=None, split=None, prompt_column_override=None, revision="main"):
    from datasets import load_dataset
    print(f"Loading {num_prompts} validation prompts...")

    # Determine dataset based on model names
    dataset_name = "HuggingFaceH4/ultrachat_200k"  # default
    split_name = "test_sft"
    prompt_column = "messages"
    dataset_type = "ultrachat"

    # If dataset is explicitly specified, use it (highest priority)
    if dataset_override:
        dataset_override_lower = dataset_override.lower()
        if dataset_override_lower == "if":
            dataset_name = dataset_id or f"{os.environ['HF_USERNAME']}/IF-Datasets-Tulu-IFEval"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "if"
            print(f"Using explicitly specified IF dataset: {dataset_name}")
        elif dataset_override_lower == "tldr":
            dataset_name = "trl-lib/tldr"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "tldr"
            print(f"Using explicitly specified tldr dataset: {dataset_name}")
        elif dataset_override_lower == "chat" or dataset_override_lower == "ultrachat":
            dataset_name = "HuggingFaceH4/ultrachat_200k"
            split_name = "test_sft"
            prompt_column = "messages"
            dataset_type = "ultrachat"
            print(f"Using explicitly specified ultrachat dataset: {dataset_name}")
        elif dataset_override_lower == "storygen" or dataset_override_lower == "sg":
            dataset_name = dataset_id or f"{os.environ['HF_USERNAME']}/RUCAIBox-Story-Generation-test"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "storygen"
            print(f"Using explicitly specified StoryGen dataset: {dataset_name}")
        else:
            raise ValueError(f"Unknown dataset override: {dataset_override}. Supported values: if, tldr, chat/ultrachat, storygen/sg")
    # Otherwise, determine dataset based on model name (default behavior)
    elif model_path:
        if "tldr" in model_path.lower():
            dataset_name = "trl-lib/tldr"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "tldr"
            print("Detected tldr model, using trl-lib/tldr dataset")
        elif "if" in model_path.lower():
            dataset_name = dataset_id or f"{os.environ['HF_USERNAME']}/IF-Datasets-Tulu-IFEval"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "if"
            print(f"Detected IF model, using {dataset_name}")
        elif "sg" in model_path.lower():
            dataset_name = dataset_id or f"{os.environ['HF_USERNAME']}/RUCAIBox-Story-Generation-test"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "storygen"
            print(f"Detected StoryGen model, using {dataset_name}")
        else:
            print("Using default ultrachat_200k dataset")

    dataset_name = dataset_id or dataset_name
    split_name = split or split_name
    prompt_column = prompt_column_override or prompt_column
    dataset = load_dataset(dataset_name, split=split_name, revision=revision)
    dataset_info = {"id": dataset_name, "revision": revision, "split": split_name,
                    "fingerprint": dataset._fingerprint, "prompt_column": prompt_column}
    print(f"Using {split_name} split; verify separation from your training data")

    golden_completions: Optional[List[str]] = None

    if prompt_column == "messages":
        # For ultrachat format
        prompts = dataset["messages"]
        validation_prompts: List[str] = []
        for example in prompts:
            for message in example:
                if message["role"] == "user":
                    validation_prompts.append(message["content"])
                    break
    else:
        # For prompt-based formats (tldr, if, storygen)
        validation_prompts = list(dataset[prompt_column])

        # For tldr, also load golden completions
        if dataset_type == "tldr":
            golden_completions = list(dataset["completion"])
            # Create pairs to keep prompts and completions aligned
            pairs = list(zip(validation_prompts, golden_completions))

            # Seed and shuffle pairs together to maintain alignment
            random.Random(seed).shuffle(pairs)
            if num_prompts > len(pairs):
                raise ValueError("Requested more prompts than available")

            # Select N pairs and unpack
            pairs = pairs[:num_prompts]
            validation_prompts, golden_completions = zip(*pairs)
            validation_prompts = list(validation_prompts)
            golden_completions = list(golden_completions)
            print(f"Loaded {len(validation_prompts)} validation prompts with golden completions from {split_name} split of {dataset_name}")
            return validation_prompts, golden_completions, dataset_type, split_name, dataset_info

    # For non-tldr datasets, shuffle and select as before
    if num_prompts > len(validation_prompts):
        raise ValueError("Requested more prompts than available")
    random.Random(seed).shuffle(validation_prompts)
    validation_prompts = validation_prompts[:num_prompts]
    print(f"Loaded {len(validation_prompts)} validation prompts from {split_name} split of {dataset_name}")
    return validation_prompts, golden_completions, dataset_type, split_name, dataset_info


def format_prompt(user_message: Any, tokenizer, dataset_type: str = "ultrachat", system_prompt: Optional[str] = "", enable_thinking: bool = False) -> str:
    # None omits the system message; an empty string preserves the training
    # recipe's explicit empty system message. Do not invent task instructions.
    messages = frozen_prompt_messages(user_message, system_prompt)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    return prompt


def generate_completion(prompt, model, tokenizer, max_new_tokens: int = 2048, use_vllm: bool = False, sampling_params: Any = None, temperature: float = 0.7, top_p: float = 0.9) -> str:
    import torch
    if use_vllm:
        outputs = model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    input_ids = torch.tensor([prompt], dtype=torch.long, device=model.device)
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            **({"temperature": temperature, "top_p": top_p} if temperature > 0 else {}),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_completions_vllm(prompts: List[str], vllm_model, sampling_params: Any, desc: str) -> List[List[str]]:
    """Generate completions using vLLM. Returns list of completion lists (one list per prompt)."""
    completions: List[List[str]] = []
    outputs = vllm_model.generate(prompts, sampling_params)
    for output in outputs:
        # Extract all n completions for this prompt
        prompt_completions = [sample.text for sample in output.outputs]
        completions.append(prompt_completions)
    return completions


def write_completions_artifact(path: str, meta: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
    output_directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_directory, exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": items}, f, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)
    print(f"Completions artifact saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate completions artifact for single model")
    parser.add_argument("--model", required=True, help="Path to model directory (HF path or local directory)")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompts-file", help="Frozen held-out JSON v1; preserve its order and require --num-prompts to match")
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-split")
    parser.add_argument("--prompt-column")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--training-config", help="Resolved training YAML; inherit system_prompt")
    parser.add_argument("--system-prompt", default=None, help="Explicit override; default is training config or empty system prompt")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse only an exact validated artifact; mismatches fail without overwrite")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of validation prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-vllm", action="store_true", help="Disable VLLM and use transformers")
    parser.add_argument("--output", required=False, help="Output file or directory for completions JSON. If omitted, saves to completions/<auto-name>.json")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum number of new tokens to generate (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling, higher values = more random (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold (default: 0.9)")
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.85, help="GPU memory utilization for vLLM (default: 0.85)")
    parser.add_argument("--n-completions", type=int, default=1, help="Number of completions to generate per prompt (default: 1)")
    parser.add_argument("--dataset", type=str, default=None, help="Explicitly specify dataset (if, tldr, chat/ultrachat, storygen/sg). If not specified, auto-detects from model path.")
    args = parser.parse_args()
    if min(args.num_prompts, args.n_completions, args.max_prompt_length, args.max_new_tokens) < 1:
        parser.error("Prompt/completion counts and token limits must be positive")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature or not 0 < args.top_p <= 1 or not 0 < args.vllm_gpu_memory <= 1:
        parser.error("Invalid temperature, top-p or GPU memory fraction")
    if args.prompts_file and any((args.dataset, args.dataset_id, args.dataset_split, args.prompt_column)):
        parser.error("--prompts-file cannot be combined with dataset selection arguments")
    frozen_prompts = frozen_identity = None
    if args.prompts_file:
        frozen_prompts, frozen_identity = load_frozen_prompts(args.prompts_file, args.num_prompts)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    import torch
    import numpy as np

    use_vllm = not args.no_vllm
    # Resolve mutable Hub names to a local snapshot before hashing/loading.
    model_path = args.model
    if not Path(model_path).is_dir():
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(args.model, revision=args.revision)
    identity = model_fingerprint(model_path)
    system_prompt = ""
    if args.training_config:
        import yaml
        system_prompt = yaml.safe_load(Path(args.training_config).read_text()).get("system_prompt")
    if args.system_prompt is not None:
        system_prompt = args.system_prompt
    if system_prompt is not None and not isinstance(system_prompt, str):
        parser.error("system_prompt must be text or null")

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load prompts
    if frozen_prompts is not None:
        prompts, golden_completions = frozen_prompts, None
        dataset_type, split_name = "frozen", "heldout"
        dataset_info = {"frozen_prompts": frozen_identity}
    else:
        prompts, golden_completions, dataset_type, split_name, dataset_info = load_validation_prompts(
            args.num_prompts, args.seed, args.model, args.dataset, args.dataset_id, args.dataset_split,
            args.prompt_column, args.dataset_revision)
    artifact_prompts = [frozen_prompt_text(prompt, system_prompt) for prompt in prompts]
    if frozen_identity is not None and len(set(artifact_prompts)) != len(artifact_prompts):
        raise ValueError("Frozen prompts become duplicates after the system-prompt override")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    formatted_prompts = [format_prompt(p, tokenizer, dataset_type, system_prompt, args.enable_thinking) for p in prompts]
    token_prompts = prompt_token_ids(tokenizer, formatted_prompts, args.max_prompt_length)
    contract = {"version": ARTIFACT_VERSION, "model_sha256": identity, "model_revision": args.revision,
                "dataset": dataset_info, "prompts_sha256": digest(prompts), "tokens_sha256": digest(token_prompts),
                "system_prompt": system_prompt, "enable_thinking": args.enable_thinking,
                "max_prompt_length": args.max_prompt_length, "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
                "n_completions": args.n_completions, "backend": "vllm" if use_vllm else "transformers",
                "versions": {name: version(name) for name in ("torch", "transformers", "vllm") if name != "vllm" or use_vllm}}
    if frozen_identity is not None:
        contract["frozen_prompts"] = frozen_identity
    if args.reuse_existing:
        if not args.output or not args.output.endswith(".json"):
            parser.error("--reuse-existing requires an explicit --output .json file")
        if Path(args.output).exists():
            if valid_generation_cache(args.output, contract, artifact_prompts, args.n_completions):
                print(f"Reusing validated completions: {args.output}")
                return
            raise ValueError("Existing completions do not match this model/data/configuration. Use a new evaluation directory; old evidence was preserved.")

    print("Running in single model mode")

    # Initialize model
    completions = []

    if use_vllm:
        from vllm import LLM, SamplingParams
        print("Using VLLM for fast inference...")

        try:
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                n=args.n_completions,  # Generate n completions per prompt
                seed=args.seed,
            )

            # Load model directly from the specified directory
            print(f"Loading model from: {args.model}")
            vllm_model = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=args.vllm_gpu_memory,
                max_model_len=args.max_prompt_length + args.max_new_tokens,
                enforce_eager=True,
                dtype="bfloat16",
                seed=args.seed,
            )

            completions = generate_completions_vllm(
                [{"prompt_token_ids": ids} for ids in token_prompts], vllm_model, sampling_params, desc="Generating completions")

        except Exception as e:
            print(f"ERROR: vLLM failed to initialize")
            print(f"Exception: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise the exception to stop execution

    else:
        print("Using standard transformers inference...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )

        for p in tqdm(token_prompts, desc="Generating completions"):
            prompt_completions = []
            for _ in range(args.n_completions):
                c = generate_completion(p, model, tokenizer, max_new_tokens=args.max_new_tokens, use_vllm=False, temperature=args.temperature, top_p=args.top_p)
                prompt_completions.append(c)
            completions.append(prompt_completions)

    items: List[Dict[str, Any]] = []
    if len(completions) != len(prompts) or any(len(c) != args.n_completions for c in completions):
        raise ValueError("Generation returned an incomplete result; artifact was not published")
    for index, (p, c) in enumerate(zip(artifact_prompts, completions)):
        item = {"prompt": p, "completions": c}
        if frozen_identity is not None:
            item["prompt_id"] = frozen_identity["prompt_ids"][index]
        items.append(item)

    meta: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "contract": contract,
        "items_sha256": digest(items),
        "dataset": dataset_type,
        "split": split_name,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "use_vllm": use_vllm,
        "n_completions": args.n_completions,
        "generation_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens" if use_vllm else "max_new_tokens": args.max_new_tokens
        },
        "model_path": args.model,
    }

    # Determine output path and auto-generate a descriptive filename without org prefix
    def repo_name(model_path: str) -> str:
        # Use only the repository/directory name (strip organization like "org/" or get last dir)
        return model_path.rstrip("/").split("/")[-1]

    def extract_base_model_name(model_path: str, temperature: float = None) -> str:
        # Extract base model name by removing checkpoint suffix
        # e.g., "model-checkpoint25" -> "model"
        import re
        name = repo_name(model_path)
        # Remove patterns like "-checkpoint25", "-checkpoint-25", "_checkpoint25", etc.
        base_name = re.sub(r'[-_]checkpoint[-_]?\d+$', '', name, flags=re.IGNORECASE)
        # Append temperature if provided
        if temperature is not None:
            base_name = f"{base_name}_temp{temperature:.1f}"
        return base_name

    model_name = repo_name(args.model)
    base_model_name = extract_base_model_name(args.model, args.temperature)
    auto_filename = f"{model_name}_{args.num_prompts}prompts_{args.n_completions}completions_seed{args.seed}_temp{args.temperature:.1f}.json"

    if args.output:
        # If output looks like a file (endswith .json), use it; otherwise treat as directory
        output_path = args.output if args.output.endswith(".json") else os.path.join(args.output, auto_filename)
    else:
        # Choose directory based on n_completions
        if args.n_completions == 1:
            output_dir = "completions"
        else:
            output_dir = f"completions_n{args.n_completions}"

        # Add subfolder with base model name
        output_dir = os.path.join(output_dir, base_model_name)

        output_path = os.path.join(output_dir, auto_filename)

    write_completions_artifact(output_path, meta, items)

    # Save golden completions for tldr dataset
    if dataset_type == "tldr" and golden_completions is not None:
        golden_items: List[Dict[str, Any]] = []
        for p, g in zip(prompts, golden_completions):
            golden_items.append({"prompt": p, "completions": [g]})

        golden_meta: Dict[str, Any] = {
            "is_golden": True,
            "dataset": dataset_type,
            "split": split_name,
            "num_prompts": args.num_prompts,
            "seed": args.seed,
            "model_path": None,
        }

        # Generate golden filename: golden_{num_prompts}prompts_seed{seed}.json
        golden_filename = f"golden_{args.num_prompts}prompts_seed{args.seed}.json"

        # Use same directory as model completions
        golden_output_path = os.path.join(os.path.dirname(output_path), golden_filename)

        write_completions_artifact(golden_output_path, golden_meta, golden_items)


if __name__ == "__main__":
    main()
