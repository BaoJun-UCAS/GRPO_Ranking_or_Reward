#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model for vLLM evaluation."""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Directory containing adapter_config.json")
    parser.add_argument("--output", required=True, help="Directory for the merged model")
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    adapter_path = Path(args.adapter).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No adapter_config.json found in {adapter_path}")
    if output_path == adapter_path:
        raise ValueError("The merged output directory must differ from the adapter directory")

    temporary_output_path = output_path.parent / f".{output_path.name}.tmp-{os.getpid()}"
    backup_output_path = output_path.parent / f".{output_path.name}.backup-{os.getpid()}"
    if temporary_output_path.exists() or backup_output_path.exists():
        raise FileExistsError("Temporary merge path already exists; remove stale hidden merge directories first")

    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    base_model_name = peft_config.base_model_name_or_path
    device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = model.merge_and_unload(safe_merge=True)

    temporary_output_path.mkdir(parents=True, exist_ok=False)
    merged_model.save_pretrained(
        str(temporary_output_path),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
    tokenizer.save_pretrained(str(temporary_output_path))

    manifest = {
        "adapter_path": str(adapter_path),
        "base_model": base_model_name,
        "output_path": str(output_path),
        "torch_dtype": "bfloat16",
        "merged_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = temporary_output_path / "merge_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)

    # Publish the merged directory as one rename. If an earlier merged model exists,
    # keep it as a recoverable backup until the new directory is in place.
    if output_path.exists():
        os.replace(output_path, backup_output_path)
    try:
        os.replace(temporary_output_path, output_path)
    except Exception:
        if backup_output_path.exists() and not output_path.exists():
            os.replace(backup_output_path, output_path)
        raise
    if backup_output_path.exists():
        shutil.rmtree(backup_output_path)
    print(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    main()
