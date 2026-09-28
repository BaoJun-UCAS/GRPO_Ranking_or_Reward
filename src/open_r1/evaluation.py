"""CPU-only contracts shared by generation, cache validation and reporting."""

import hashlib
import json
from pathlib import Path


ARTIFACT_VERSION = 2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_fingerprint(directory):
    """Hash actual local weights and configuration, not mutable paths or mtimes.

    This intentionally reads model files once; a same-path re-merge must invalidate
    the generation cache. No model is imported or loaded onto a GPU.
    """
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file()
                   and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja", ".py"}
                   and ".cache" not in p.relative_to(directory).parts)
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError(f"No model weights found in {directory}")
    manifest = []
    for path in files:
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        manifest.append((str(path.relative_to(directory)), checksum.hexdigest()))
    return digest(manifest)


def valid_generation_cache(path, contract, prompts, n_completions):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = payload["meta"]
        items = payload["items"]
        return (
            meta.get("artifact_version") == ARTIFACT_VERSION
            and meta.get("contract") == contract
            and meta.get("items_sha256") == digest(items)
            and len(items) == len(prompts)
            and all(item["prompt"] == prompt and isinstance(item["completions"], list)
                    and len(item["completions"]) == n_completions
                    and all(isinstance(text, str) for text in item["completions"])
                    for item, prompt in zip(items, prompts))
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def prompt_token_ids(tokenizer, formatted_prompts, max_prompt_length):
    """Both inference backends see exactly the same left-truncated token IDs."""
    return [tokenizer(text, add_special_tokens=False)["input_ids"][-max_prompt_length:]
            for text in formatted_prompts]
