#!/usr/bin/env python3
"""Compare V0 scheduler concurrency caps with one model and a fixed KV allocation.

Example (run only after the training launcher releases this GPU):
  CUDA_VISIBLE_DEVICES=4 python scripts/benchmark_vllm_scheduler.py \
    --run-dir grpo_runs/<run> --output /tmp/vllm-scheduler.json

This intentionally changes only the scheduler admission cap after the engine
has initialized at 256. It does not compare separately initialized deployments,
whose activation profiling, KV allocation, and CUDA graph captures can differ.
Fixed per-prompt sampling seeds do not guarantee identical tokens across caps.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
import time


INITIAL_CAP = 256


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caps", type=positive_int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-tokens", type=positive_int, default=8)
    parser.add_argument("--model", help="Override the run's policy model ID/path")
    parser.add_argument("--revision", help="Override the recorded policy revision")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=positive_int)
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if any(cap > INITIAL_CAP for cap in args.caps):
        parser.error("--caps cannot exceed the initialized/captured capacity of 256")
    if len(set(args.caps)) != len(args.caps):
        parser.error("--caps must be distinct")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.output.exists():
        parser.error("--output already exists; choose a new file to preserve previous evidence")
    return args


def read_run_value(run_dir, name):
    """Read a literal launcher value without sourcing or evaluating run.env."""
    path = run_dir / "run.env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key == name:
                words = shlex.split(value)
                if len(words) == 1:
                    return words[0]
    return None


def restore_step_zero_prompts(run_dir, group_size, expected_completions):
    """Recover complete global sampler order, validating every G-sized group."""
    paths = sorted((run_dir / "reward_data").glob("reward_data_train_step_000000_*.json"))
    if not paths:
        raise ValueError("No saved training-step-0 reward records found")
    indexed = {}
    ranks = set()
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a reward-record list: {path}")
        for row in rows:
            if row.get("training_step") != 0:
                raise ValueError(f"Nonzero training step in {path}")
            key = (row["process_index"], row["rollout_index"])
            if key in indexed:
                raise ValueError(f"Duplicate rank/rollout index {key}; cannot disambiguate multiple rollouts")
            indexed[key] = row["prompt"]
            ranks.add(key[0])
    if sorted(ranks) != list(range(len(ranks))):
        raise ValueError("Reward records do not contain consecutive process ranks starting at zero")
    for rank in ranks:
        indices = sorted(index for current_rank, index in indexed if current_rank == rank)
        if indices != list(range(len(indices))):
            raise ValueError(f"Rank {rank} has missing rollout indices")
    if len(indexed) != expected_completions or len(indexed) % group_size:
        raise ValueError(
            f"Expected {expected_completions} completions in complete groups of {group_size}; found {len(indexed)}"
        )
    ordered = [indexed[key] for key in sorted(indexed)]
    prompts = []
    for start in range(0, len(ordered), group_size):
        group = ordered[start:start + group_size]
        if any(prompt != group[0] for prompt in group):
            raise ValueError(f"Prompt mismatch within generation group {start // group_size}")
        prompt = group[0]
        # _collect_reward_data historically stores conversations via str(list).
        if isinstance(prompt, str) and prompt.lstrip().startswith("["):
            try:
                candidate = ast.literal_eval(prompt)
            except (SyntaxError, ValueError):
                candidate = None
            if isinstance(candidate, list) and all(isinstance(item, dict) for item in candidate):
                prompt = candidate
        if not isinstance(prompt, (str, list)):
            raise ValueError("Saved prompts must be raw text or serialized chat messages")
        prompts.append(prompt)
    return prompts, [str(path) for path in paths]


def token_digest(token_lists):
    encoded = json.dumps(token_lists, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_idle_shared_scheduler(engine):
    """Reject incompatible runtimes instead of editing private queues/state."""
    if engine.has_unfinished_requests():
        raise RuntimeError("Cannot change the scheduler cap while requests are pending")
    config = engine.scheduler_config
    if config.num_scheduler_steps != 1:
        raise RuntimeError("This benchmark supports only the pinned single-step V0 scheduler")
    schedulers = engine.scheduler
    if len(schedulers) != 1:
        raise RuntimeError("This benchmark requires one scheduler and pipeline-parallel size 1")
    for scheduler in schedulers:
        if scheduler.scheduler_config is not config:
            raise RuntimeError("Schedulers do not share the engine config; dynamic cap changes are unsafe")
        if scheduler.waiting or scheduler.running or scheduler.swapped:
            raise RuntimeError("Scheduler queues must be empty before changing the cap")
    # Keep explicit assertions as executable documentation of the identity
    # guarantee verified in vLLM 0.8.5's LLMEngine constructor.
    assert all(scheduler.scheduler_config is config for scheduler in schedulers)
    return config, schedulers


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    import yaml

    config = yaml.safe_load((run_dir / "config/resolved_training_config.yaml").read_text(encoding="utf-8"))
    group_size = int(config["num_generations"])
    if group_size != 8:
        raise ValueError("This scheduler benchmark requires the recorded G=8 experiment")
    if float(config.get("temperature", 1.0)) != 1.0 or int(config["max_completion_length"]) != 3072:
        raise ValueError("Expected the temperature=1.0, max_completion_length=3072 experiment")
    prompts, reward_files = restore_step_zero_prompts(run_dir, group_size, int(config["generation_batch_size"]))
    model_name = args.model or config["model_name_or_path"]
    revision = args.revision or config.get("model_revision", "main")
    max_model_len = args.max_model_len or int(read_run_value(run_dir, "VLLM_MAX_MODEL_LEN") or (
        int(config["max_prompt_length"]) + int(config["max_completion_length"])
    ))
    if max_model_len < int(config["max_prompt_length"]) + 3072:
        raise ValueError("The vLLM context must cover recorded prompt and completion limits")

    # Set engine selection before importing any CUDA-aware library.
    os.environ["VLLM_USE_V1"] = "0"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    # Match training's Qwen3 enable_thinking=False convention exactly.
    from open_r1.utils.data_utils import maybe_apply_chat_template

    tokenizer_name = config.get("tokenizer_name_or_path") or model_name
    tokenizer_revision = config.get("tokenizer_revision")
    if tokenizer_revision is None and tokenizer_name == model_name:
        tokenizer_revision = revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, revision=tokenizer_revision,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    if config.get("chat_template") is not None:
        tokenizer.chat_template = config["chat_template"]
    texts = [maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts]
    tokenized = tokenizer(texts, add_special_tokens=False)["input_ids"]
    prompt_lengths = [len(tokens) for tokens in tokenized]
    if not all(0 < length <= int(config["max_prompt_length"]) for length in prompt_lengths):
        raise ValueError(
            "Recorded prompts were already truncated. Rendering now exceeds the recorded limit "
            "or is empty; verify tokenizer revision/template instead of truncating them again."
        )
    vllm_prompts = [{"prompt_token_ids": tokens} for tokens in tokenized]

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if not vllm.__version__.startswith("0.8.5"):
        raise RuntimeError("Dynamic scheduling comparison is supported only on inspected vLLM 0.8.5")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU via CUDA_VISIBLE_DEVICES")
    llm = LLM(
        model=model_name, revision=revision, tokenizer=tokenizer_name, tokenizer_revision=tokenizer_revision,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        tensor_parallel_size=1, pipeline_parallel_size=1, dtype=config.get("torch_dtype", "bfloat16"),
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=max_model_len,
        max_num_seqs=INITIAL_CAP, enforce_eager=False, seed=args.seed,
        enable_prefix_caching=args.prefix_caching,
    )
    engine = llm.llm_engine
    scheduler_config, schedulers = assert_idle_shared_scheduler(engine)
    if scheduler_config.max_num_seqs != INITIAL_CAP or engine.model_config.enforce_eager:
        raise RuntimeError("Engine must initialize with capacity 256 and CUDA graph capture enabled")
    kv_blocks = int(engine.cache_config.num_gpu_blocks)
    sampling_kwargs = {
        "n": 8, "temperature": 1.0, "max_tokens": 3072, "detokenize": False,
        "top_p": config.get("top_p", 1.0), "top_k": config.get("top_k") or -1,
        "min_p": config.get("min_p") or 0.0,
        "repetition_penalty": config.get("repetition_penalty", 1.0),
    }
    seeds = [args.seed + index * group_size for index in range(len(prompts))]
    report = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "reward_files": reward_files,
        "model": model_name, "revision": revision,
        "resolved_model_commit": getattr(engine.model_config.hf_config, "_commit_hash", None),
        "vllm_version": vllm.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(0),
        "initialized_max_num_seqs": INITIAL_CAP, "max_model_len": max_model_len,
        "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
        "chunked_prefill": scheduler_config.chunked_prefill_enabled,
        "prefix_caching": engine.cache_config.enable_prefix_caching,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_blocks": kv_blocks, "kv_block_size": engine.cache_config.block_size,
        "prompt_count": len(prompts), "prompt_token_lengths": prompt_lengths,
        "prompt_token_sha256": token_digest(tokenized), "per_prompt_seeds": seeds,
        "sampling": sampling_kwargs, "caps": args.caps, "repeats": args.repeats,
        "comparison": "one initialized engine, fixed KV allocation and CUDA graphs; only scheduler cap changes",
        "token_equality_guaranteed_across_caps": False, "measurements": [],
    }
    write_report(args.output, report)
    try:
        llm.reset_prefix_cache()
        llm.generate(vllm_prompts[:1], sampling_params=SamplingParams(
            **{**sampling_kwargs, "max_tokens": args.warmup_tokens, "seed": seeds[0]},
        ), use_tqdm=False)
        for repeat in range(args.repeats):
            for cap in args.caps:
                assert_idle_shared_scheduler(engine)
                scheduler_config.max_num_seqs = cap
                reset_result = llm.reset_prefix_cache()
                assert_idle_shared_scheduler(engine)
                if int(engine.cache_config.num_gpu_blocks) != kv_blocks:
                    raise RuntimeError("KV cache allocation changed; the comparison is no longer controlled")
                before = sum(scheduler.num_cumulative_preemption for scheduler in schedulers)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                outputs = llm.generate(vllm_prompts, sampling_params=[
                    SamplingParams(**sampling_kwargs, seed=seed) for seed in seeds
                ], use_tqdm=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                assert_idle_shared_scheduler(engine)
                token_lists = [list(sample.token_ids) for output in outputs for sample in output.outputs]
                if len(outputs) != len(prompts) or any(len(output.outputs) != group_size for output in outputs):
                    raise RuntimeError("Generation did not return all prompt groups and completions")
                lengths = [len(tokens) for tokens in token_lists]
                measurement = {
                    "repeat": repeat, "max_num_seqs": cap, "wall_seconds": elapsed,
                    "completion_tokens": sum(lengths), "tokens_per_second": sum(lengths) / elapsed,
                    "completion_lengths": lengths, "mean_completion_length": statistics.mean(lengths),
                    "output_token_sha256": token_digest(token_lists), "prefix_cache_reset": reset_result,
                    "preemption_delta": sum(s.num_cumulative_preemption for s in schedulers) - before,
                    "kv_blocks": int(engine.cache_config.num_gpu_blocks),
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "cuda_memory_scope": "current single-GPU engine process",
                }
                report["measurements"].append(measurement)
                write_report(args.output, report)
                print(json.dumps({k: measurement[k] for k in (
                    "repeat", "max_num_seqs", "wall_seconds", "completion_tokens", "tokens_per_second", "preemption_delta",
                )}), flush=True)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # A failed generation may leave requests pending. Do not change any
        # scheduler state in that case; this standalone process will exit.
        if not engine.has_unfinished_requests():
            assert_idle_shared_scheduler(engine)
            scheduler_config.max_num_seqs = INITIAL_CAP
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(args.output, report)


if __name__ == "__main__":
    main()
