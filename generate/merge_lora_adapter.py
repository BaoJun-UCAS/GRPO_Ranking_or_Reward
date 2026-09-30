#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model for vLLM evaluation."""

import argparse
import json
import os
import shutil
import re
from datetime import datetime, timezone
from pathlib import Path

def resolve_base_revision(adapter_path: Path, peft_config, explicit_revision: str | None = None):
    """Resolve base provenance before loading weights; checkpoint runs inherit it."""
    manifest_path = adapter_path / "run_manifest.json"
    if not manifest_path.is_file() and re.fullmatch(r"checkpoint-[0-9]+", adapter_path.name):
        manifest_path = adapter_path.parent / "run_manifest.json"
    model_args = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_args = manifest.get("model_arguments", {})
        if not isinstance(model_args, dict):
            raise ValueError(f"Invalid model_arguments in {manifest_path}")
        recorded_base = model_args.get("model_name_or_path")
        adapter_base = peft_config.base_model_name_or_path
        # Compare local paths canonically; Hub repository IDs remain literal.
        def canonical(name):
            path = Path(name).expanduser()
            return str(path.resolve()) if path.exists() else str(name).rstrip("/")
        if not recorded_base or canonical(recorded_base) != canonical(adapter_base):
            raise ValueError(
                f"Run manifest base model {recorded_base!r} does not match adapter base {adapter_base!r}"
            )
    candidates = (
        (explicit_revision, "cli"),
        (model_args.get("model_revision"), "run_manifest"),
        (getattr(peft_config, "revision", None), "adapter_config"),
    )
    for revision, source in candidates:
        if revision is not None:
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError(f"Invalid base model revision from {source}: {revision!r}")
            return revision, source
    # Older external adapters can omit both provenance sources. Make the
    # Hugging Face default explicit in the resulting merge manifest.
    return "main", "default"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Directory containing adapter_config.json")
    parser.add_argument("--output", required=True, help="Directory for the merged model")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--revision", help="Explicit base revision; overrides training and adapter provenance")
    args = parser.parse_args()

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    base_revision, revision_source = resolve_base_revision(adapter_path, peft_config, args.revision)
    device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        revision=base_revision,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    )
    base_commit = getattr(base_model.config, "_commit_hash", None)
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
            base_model_name, revision=base_revision, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
    tokenizer.save_pretrained(str(temporary_output_path))

    manifest = {
        "adapter_path": str(adapter_path),
        "base_model": base_model_name,
        "base_model_revision": base_revision,
        "revision_source": revision_source,
        "base_model_commit": base_commit,
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
