#!/usr/bin/env python3
"""Deployment shortcuts. Help, doctor, cache and dry-run never load a model."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("Qwen/Qwen3-1.7B", "friendshipkim/QRM-Llama3.1-8B-v2")
PINNED_RUNTIME = {
    "torch": "2.6.0",
    "vllm": "0.8.5.post1",
    "transformers": "4.52.3",
    "trl": "0.18.0",
    "accelerate": "1.4.0",
    "deepspeed": "0.16.8",
    "flash-attn": "2.7.4.post1",
    "numpy": "1.26.4",
}
REQUIRED_PACKAGES = (
    "grpo", "peft", "datasets", "huggingface-hub", "fastapi", "uvicorn", "pydantic", "PyYAML",
)


def paths(env=None):
    """Resolve the same portable paths as the launcher, without creating them."""
    env = os.environ if env is None else env
    home = Path(env.get("HOME", str(Path.home()))).expanduser()
    if env.get("GRPO_CACHE_ROOT"):
        cache = Path(env["GRPO_CACHE_ROOT"]).expanduser()
    elif env.get("XDG_CACHE_HOME"):
        cache = Path(env["XDG_CACHE_HOME"]).expanduser() / "grpo"
    else:
        data_root = Path(env.get("GRPO_DATA_ROOT", f"/data/{home.name}")).expanduser()
        cache = data_root / "cache/grpo" if data_root.is_dir() and os.access(data_root, os.W_OK | os.X_OK) else home / ".cache/grpo"
    hf_home = Path(env.get("HF_HOME", str(cache / "huggingface"))).expanduser()
    hub = Path(env.get("HF_HUB_CACHE", env.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))).expanduser()
    output = Path(env.get("GRPO_OUTPUT_ROOT", str(ROOT / "grpo_runs"))).expanduser()
    selected = {"HF_HOME": hf_home, "HF_HUB_CACHE": hub, "GRPO_OUTPUT_ROOT": output}
    for key, folder in (("TORCH_EXTENSIONS_DIR", "torch_extensions"), ("TRITON_CACHE_DIR", "triton"),
                        ("VLLM_CACHE_ROOT", "vllm")):
        selected[key] = Path(env.get(key, str(cache / folder))).expanduser()
    return selected


def existing_parent(path):
    path = Path(path).absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def gib(size):
    return f"{size / (1024 ** 3):.2f} GiB"


def doctor(args):
    failures = []

    def report(ok, name, detail):
        print(f"{'OK' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failures.append(name)

    report(sys.version_info[:2] == (3, 11), "Python", f"{sys.version.split()[0]} ({sys.executable}); recipe expects 3.11")
    print(f"Conda prefix: {os.environ.get('CONDA_PREFIX', '(not activated)')}")
    for package, expected in list(PINNED_RUNTIME.items()) + [(name, None) for name in REQUIRED_PACKAGES]:
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report(False, package, "missing; use environment.yml, then install FlashAttention separately")
        else:
            report(expected is None or installed.split('+')[0] == expected, package,
                   installed + (f" (expected {expected})" if expected else ""))

    for command in ("bash", "curl", "setsid", "nvidia-smi"):
        location = shutil.which(command)
        report(location is not None, command, location or "not on PATH")
    compiler = os.environ.get("CXX", "c++")
    report(shutil.which(compiler) is not None, "C++ compiler", shutil.which(compiler) or f"{compiler} is not on PATH")
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidate = Path(cuda_home) / "bin/nvcc"
        nvcc = str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    else:
        nvcc = shutil.which("nvcc")
    report(nvcc is not None, "CUDA compiler", nvcc or "not found; DeepSpeed/FlashAttention need the CUDA 12.4 development toolchain")
    print(f"CUDA_HOME: {cuda_home or '(unset; a PATH nvcc may allow automatic detection)'}")
    selected = paths()
    selected["TMPDIR"] = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    selected["PIP_CACHE_DIR"] = Path(os.environ.get("PIP_CACHE_DIR", str(Path.home() / ".cache/pip")))
    for key, path in selected.items():
        parent = existing_parent(path)
        report(os.access(parent, os.W_OK | os.X_OK), key, f"{path}; free {gib(shutil.disk_usage(parent).free)}")
    if os.environ.get("TMPDIR"):
        report(selected["TMPDIR"].is_dir(), "TMPDIR exists", str(selected["TMPDIR"]))
    for key in ("VLLM_GPUS", "QRM_GPU", "TRAIN_GPUS", "DATASET_NAME"):
        print(f"{key}: {os.environ.get(key, '(not set)')}")
    if not os.environ.get("DATASET_NAME") and not os.environ.get("HF_USERNAME"):
        print("NOTE set DATASET_NAME or HF_USERNAME before smoke/train/dry-run.")
    print("NOTE package metadata and pip check cannot validate compiled CUDA extensions.")
    if args.cuda:
        code = """import torch, flash_attn, vllm, deepspeed
print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available(), 'device count:', torch.cuda.device_count())
assert torch.cuda.is_available(), 'CUDA is not available'
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
"""
        try:
            cuda_env = os.environ.copy()
            cuda_env.update({key: str(value) for key, value in selected.items()})
            result = subprocess.run([sys.executable, "-c", code], env=cuda_env, timeout=60, check=False)
            report(result.returncode == 0, "CUDA imports", f"exit {result.returncode}")
        except subprocess.TimeoutExpired:
            report(False, "CUDA imports", "timed out after 60s")
    else:
        print("CUDA was not initialized. Run doctor --cuda when appropriate to check GPU imports.")
    print(f"Doctor completed: {len(failures)} failed check(s).")
    return 1 if failures else 0


def cache(_args):
    hub = paths()["HF_HUB_CACHE"]
    print(f"Hub cache: {hub}")
    if not hub.is_dir():
        print("No cache directory yet.")
        return 0
    for repo in sorted(hub.glob("models--*")):
        blobs = repo / "blobs"
        files = list(blobs.iterdir()) if blobs.is_dir() else []
        complete_bytes = partial_bytes = partial_count = 0
        for path in files:
            try:
                if not path.is_file():
                    continue
                size = path.stat().st_size
            except FileNotFoundError:  # A downloader may rename a file during the scan.
                continue
            if path.name.endswith(".incomplete"):
                partial_bytes += size
                partial_count += 1
            else:
                complete_bytes += size
        print(f"{repo.name.removeprefix('models--').replace('--', '/')}: "
              f"completed blobs {gib(complete_bytes)}, partial files {partial_count} ({gib(partial_bytes)})")
    print("Blob sizes are not a download percentage; Xet chunks may be cached separately.")
    return 0


def download(args):
    selected = paths()
    for key in ("HF_HOME", "HF_HUB_CACHE"):
        os.environ.setdefault(key, str(selected[key]))
    # Import after configuring the cache: HF constants are resolved at import time.
    from huggingface_hub import snapshot_download

    models = args.model or [os.environ.get("MODEL_NAME", DEFAULT_MODELS[0]), DEFAULT_MODELS[1]]
    for model in dict.fromkeys(models):
        if Path(model).is_dir():
            print(f"Local model: {Path(model).resolve()}")
            continue
        print(f"Downloading {model}@{args.revision} to {selected['HF_HUB_CACHE']}", flush=True)
        location = snapshot_download(repo_id=model, revision=args.revision, cache_dir=str(selected["HF_HUB_CACHE"]))
        print(f"Ready: {location}")
    return 0


def launch(args):
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    if args.command == "smoke":
        # Keep explicit user overrides. The launcher derives the generation batch
        # from GPU count, per-device batch and accumulation if it is unspecified.
        defaults = {
            "MAX_STEPS": "2",
            "GRADIENT_ACCUMULATION_STEPS": "8",
            "MAX_PROMPT_LENGTH": "2048",
            "MAX_COMPLETION_LENGTH": "3072",
            "VLLM_MAX_MODEL_LEN": "6144",
            "QRM_MAX_LENGTH": "6144",
            "REWARD_BATCH_SIZE": "1",
            "MERGE_AFTER_TRAINING": "0",
        }
        for key, value in defaults.items():
            env.setdefault(key, value)
    command = ["bash", str(ROOT / "train_scripts/qwen3_1.7_grpo_chat.sh")]
    if args.dry_run:
        command.append("--dry-run")
    # Replace this process so Ctrl-C reaches the launcher's cleanup traps directly.
    os.execvpe(command[0], command, env)


def logs(args):
    if args.run_dir:
        directory = Path(args.run_dir).expanduser()
    elif os.environ.get("RUN_DIR"):
        directory = Path(os.environ["RUN_DIR"]).expanduser()
    elif os.environ.get("RUN_NAME"):
        directory = paths()["GRPO_OUTPUT_ROOT"] / os.environ["RUN_NAME"]
    else:
        output = paths()["GRPO_OUTPUT_ROOT"]
        directories = [p for p in output.iterdir() if p.is_dir() and (p / "logs").is_dir()] if output.exists() else []
        if not directories:
            raise ValueError(f"No runs under {output}; use --run-dir for a custom path")
        directory = max(directories, key=lambda p: p.stat().st_mtime_ns)
    filename = {"vllm": "vllm_server.log", "qrm": "qrm_server.log", "training": "training.log", "merge": "merge_lora.log"}[args.service]
    log_path = directory / "logs" / filename
    print(f"Log: {log_path}", flush=True)
    command = ["tail", "-n", str(args.lines)]
    if args.follow:
        command.append("-F")
    elif not log_path.is_file():
        raise ValueError(f"Log not present yet: {log_path}")
    command += ["--", str(log_path)]
    os.execvp(command[0], command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("help", help="show commands").set_defaults(func=lambda args: parser.print_help() or 0)
    check = subparsers.add_parser("doctor", help="offline metadata, tools and disk checks; no model load")
    check.add_argument("--cuda", action="store_true", help="also import compiled CUDA packages in a bounded subprocess")
    check.set_defaults(func=doctor)
    subparsers.add_parser("cache", help="show downloaded blobs and partial files, without network access").set_defaults(func=cache)
    fetch = subparsers.add_parser("download", help="prefetch policy and reward models with HF progress bars")
    fetch.add_argument("--model", action="append", help="HF repo ID or local path; repeat for several models")
    fetch.add_argument("--revision", default="main", help="revision for requested repositories; use a commit for repeatability")
    fetch.set_defaults(func=download)
    for name in ("smoke", "train"):
        run = subparsers.add_parser(name, help="two-step smoke defaults" if name == "smoke" else "run the training recipe")
        run.add_argument("--dry-run", action="store_true", help="validate and print configuration without loading a model")
        run.set_defaults(func=launch)
    log = subparsers.add_parser("logs", help="read a selected run's logs (latest run by default)")
    log.add_argument("--run-dir")
    log.add_argument("--service", choices=("vllm", "qrm", "training", "merge"), default="vllm")
    log.add_argument("--follow", action="store_true")
    log.add_argument("--lines", type=int, default=60)
    log.set_defaults(func=logs)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
