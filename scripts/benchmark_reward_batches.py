#!/usr/bin/env python3
"""Benchmark QRM batch caps on one complete saved rollout, without HTTP services.

Example (choose one idle physical GPU explicitly):
    CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 python scripts/benchmark_reward_batches.py \
        --run-dir grpo_runs/grpo-baseline --step 0 \
        --revision 23c6db70a35875248f45b3cfbe6d237ba8ac3e6d \
        --batch-sizes 4 8 --repeats 3 --output /tmp/qrm-batch-benchmark.json

The first measured reward vector for batch size 4 is the comparison reference
(or the first requested size if 4 is absent). Timing includes template rendering,
tokenization, padding, device transfer, and scoring, matching the server scorer.
No network or cache environment variables are changed by this script.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
FILE_PATTERN = re.compile(r"^reward_data_(?:train_)?step_(\d+)_micro_(\d+)_proc_(\d+)(?:_[^.]+)?\.json$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0, help="Zero-based training_step stored in reward_data.")
    parser.add_argument("--model", default="friendshipkim/QRM-Llama3.1-8B-v2")
    parser.add_argument("--revision", default="main", help="Use a commit SHA for reproducible comparisons.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--max-batch-tokens", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured full-rollout passes per batch size.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.step < 0 or args.warmup < 1 or args.repeats < 1:
        parser.error("step must be non-negative; warmup and repeats must be positive")
    if not args.batch_sizes or min(args.batch_sizes) < 1 or len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("batch-sizes must be distinct positive integers")
    if args.max_length < 1 or args.max_batch_tokens < args.max_length:
        parser.error("max-batch-tokens must be at least max-length, and max-length must be positive")
    return args


def _chat_or_text(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError("Saved prompt/completion must be text or a list of chat messages")
    # Saved GRPO chat fields are Python repr strings, not JSON. Never use eval.
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    return parsed if isinstance(parsed, list) else value


def _messages(row):
    prompt = _chat_or_text(row["prompt"])
    completion = _chat_or_text(row["completion"])
    if isinstance(prompt, str) and isinstance(completion, str):
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
    elif isinstance(prompt, list) and isinstance(completion, list):
        messages = prompt + completion
    else:
        raise ValueError("Saved prompt/completion use inconsistent text and chat formats")
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Each saved conversation must end with an assistant completion")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)):
            raise ValueError("Saved messages must match the QRM role/content request contract")
    return messages


def load_rollout(run_dir: Path, step: int):
    """Reconstruct rank order and reject missing/duplicate/truncated rollout files."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("run_manifest.json is required to verify the complete generation batch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training = manifest["training_arguments"]
    expected_total = training.get("generation_batch_size")
    group_size = training.get("num_generations")
    if not isinstance(expected_total, int) or expected_total < 1:
        raise ValueError("run_manifest.json must record a positive generation_batch_size")
    if not isinstance(group_size, int) or group_size < 2 or expected_total % group_size:
        raise ValueError("Manifest num_generations must divide generation_batch_size")
    files_by_rank = {}
    micro_steps = set()
    for path in sorted((run_dir / "reward_data").glob("*.json")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None or int(match[1]) != step:
            continue
        rank = int(match[3])
        if rank in files_by_rank:
            raise ValueError(f"More than one training rollout file for step {step}, rank {rank}")
        files_by_rank[rank] = path
        micro_steps.add(int(match[2]))
    if not files_by_rank:
        raise ValueError(f"No training reward files found for step {step}")
    if set(files_by_rank) != set(range(len(files_by_rank))) or len(micro_steps) != 1:
        raise ValueError("Selected rollout has missing ranks or inconsistent micro-step indices")
    messages, prompts, selected_files, rank_counts = [], [], [], {}
    for rank, path in sorted(files_by_rank.items()):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Empty or malformed reward file: {path}")
        for index, row in enumerate(rows):
            if (row.get("training_step") != step or row.get("process_index") != rank
                    or row.get("rollout_index") != index):
                raise ValueError(f"Inconsistent step/process/rollout index in {path.name}, row {index}")
            messages.append(_messages(row))
            prompts.append(row["prompt"])
        rank_counts[rank] = len(rows)
        selected_files.append(str(path))
    if len(messages) != expected_total or len(set(rank_counts.values())) != 1:
        raise ValueError(f"Incomplete rollout: expected {expected_total} rows, got per-rank counts {rank_counts}")
    for start in range(0, len(prompts), group_size):
        if any(prompt != prompts[start] for prompt in prompts[start:start + group_size]):
            raise ValueError(f"Prompt group at global row {start} mixes different prompts")
    digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return messages, {
        "run_dir": str(run_dir), "training_step": step, "micro_step": next(iter(micro_steps)),
        "source_files": selected_files, "rank_counts": rank_counts, "examples": len(messages),
        "num_generations": group_size, "messages_sha256": digest,
    }


def differences(values, reference):
    absolute = [abs(value - expected) for value, expected in zip(values, reference)]
    return {"max_abs_error": max(absolute), "mean_abs_error": statistics.mean(absolute),
            "exact_match": values == reference}


def main(argv=None):
    args = parse_args(argv)
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) != 1 or visible[0] == "-1":
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one available GPU before running this benchmark")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing benchmark: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    messages, source = load_rollout(args.run_dir, args.step)
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from open_r1.reward_server import TransformersRewardScorer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    device = "cuda:0"
    # Match reward_server.main, including tokenizer truncation and remote model code.
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, num_labels=1, trust_remote_code=True,
        attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(device)
    model.eval()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": source,
        "model": args.model, "requested_revision": args.revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "dtype": "bfloat16", "attention_implementation": "flash_attention_2",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "gpu": torch.cuda.get_device_name(0),
        "max_length": args.max_length, "max_batch_tokens": args.max_batch_tokens,
        "warmup_passes": args.warmup, "repeats": args.repeats, "batch_results": [],
        "package_versions": {},
    }
    for package in ("torch", "transformers", "trl", "flash-attn"):
        try:
            report["package_versions"][package] = version(package)
        except PackageNotFoundError:
            report["package_versions"][package] = None
    for batch_size in args.batch_sizes:
        scorer = TransformersRewardScorer(model, tokenizer, device, batch_size, args.max_length, args.max_batch_tokens)
        print(f"batch_size={batch_size}: warming up {args.warmup} full-rollout pass(es)", flush=True)
        for _ in range(args.warmup):
            scorer.score(messages)
        torch.cuda.synchronize()
        trials = []
        reference_rewards = None
        for repeat in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            rewards = scorer.score(messages)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if len(rewards) != len(messages) or not all(math.isfinite(value) for value in rewards):
                raise RuntimeError("Scorer returned incomplete or non-finite rewards")
            if reference_rewards is None:
                reference_rewards = rewards
            trial = {
                "repeat": repeat, "seconds": elapsed, "examples_per_second": len(messages) / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "stats": dict(scorer.last_stats), "repeat_error_vs_first": differences(rewards, reference_rewards),
            }
            trials.append(trial)
            print(f"batch_size={batch_size} repeat={repeat + 1}: {elapsed:.4f}s; "
                  f"peak_allocated={trial['peak_allocated_bytes'] / 2**30:.3f} GiB; "
                  f"forward_batches={trial['stats']['batches']}", flush=True)
        report["batch_results"].append({
            "batch_size": batch_size, "median_seconds": statistics.median(row["seconds"] for row in trials),
            "min_seconds": min(row["seconds"] for row in trials),
            "max_seconds": max(row["seconds"] for row in trials),
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in trials),
            "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in trials),
            "trials": trials, "rewards": reference_rewards,
        })
    reference_size = 4 if 4 in args.batch_sizes else args.batch_sizes[0]
    reference = next(row for row in report["batch_results"] if row["batch_size"] == reference_size)
    report["reference_batch_size"] = reference_size
    for row in report["batch_results"]:
        row["reward_difference_vs_reference"] = differences(row["rewards"], reference["rewards"])
        row["speedup_vs_reference"] = reference["median_seconds"] / row["median_seconds"]
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "reference_batch_size": reference_size,
                      "summary": [{key: row[key] for key in ("batch_size", "median_seconds", "peak_allocated_bytes",
                                  "reward_difference_vs_reference", "speedup_vs_reference")}
                                  for row in report["batch_results"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
