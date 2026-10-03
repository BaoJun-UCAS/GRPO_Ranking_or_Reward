#!/usr/bin/env python3
"""Prepare/train two paired 200-step controls using only existing local caches.

prepare writes configs; train --dry-run validates without using GPUs/services.
train runs the selected arms sequentially through the existing GRPO launcher.
"""

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_r1.comparison import (
    file_sha256, read_json, runtime_versions, training_code_digest, training_environment, write_json,
)
from open_r1.evaluation import digest

ARMS = ("robust_scaled", "weight_only_scaled")
DEFAULT_CONFIG = ROOT / "recipes/Qwen3-1.7B/advantage_controls_200.yaml"
DEFAULT_DIRECTORY = ROOT / "grpo_runs/advantage-controls-200-delta0.002-k2.46"
OFFLINE = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "WANDB_MODE": "offline", "HF_HUB_DISABLE_TELEMETRY": "1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def cached_snapshot(hub_cache, repo_id, revision):
    """Inspect a pinned local snapshot; never call a download API."""
    require(not Path(repo_id).is_dir(), "Source must identify the original pinned Hub model")
    path = hub_cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        require((path / name).is_file(), f"Missing cached model file: {path / name}; no download attempted")
    shards = set(read_json(path / "model.safetensors.index.json")["weight_map"].values())
    require(bool(shards), f"Empty model weight index: {path}")
    for name in shards:
        require(Path(name).name == name, "Invalid shard filename")
        require((path / name).is_file() and (path / name).stat().st_size > 0,
                f"Missing cached shard: {path / name}; no download attempted")
    for value in read_json(path / "config.json").get("auto_map", {}).values():
        for target in value if isinstance(value, list) else [value]:
            if target:
                module = target.split("--")[-1].rsplit(".", 1)[0]
                require((path / (module.replace(".", "/") + ".py")).is_file(),
                        f"Missing custom model code: {module} in {path}")
    return path.resolve()


def snapshot_inventory(path):
    # Metadata content hashes plus weight size/mtime; not a weight-content hash.
    return {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                     "sha256": None if p.suffix in {".safetensors", ".bin"} else file_sha256(p)}
            for p in sorted(path.iterdir()) if p.is_file()}


def recipe_for_arm(source, directory, arm, config, options):
    recipe = deepcopy(source)
    recipe.update(
        model_name_or_path=config["model_name"], model_revision=config["model_revision"],
        dataset_name=str(directory / "data/dataset"),
        training_schedule_path=str(directory / "data/training_schedule.json"),
        output_dir=str(directory / arm), reward_data_save_path=str(directory / arm / "reward_data"),
        run_name=arm, advantage=f"open_r1.advantage_controls:{arm}", advantage_kwargs=deepcopy(options),
        max_steps=200, resume_from_checkpoint=None, overwrite_output_dir=False,
    )
    return recipe


def code_inventory():
    return {"training_code_sha256": training_code_digest(),
            **{name: file_sha256(ROOT / name) for name in (
                "scripts/run_advantage_controls.py", "scripts/grpo_control_train.py")}}


def prepare(config_path, directory):
    from open_r1.paired_data import verify_frozen_data

    spec = yaml.safe_load(config_path.read_text())
    require(spec["steps"] == 200, "This control recipe requires 200 steps")
    options = {key: spec[key] for key in ("delta", "c", "k", "epsilon")}
    for name, value in options.items():
        require(type(value) in (int, float) and math.isfinite(value)
                and (value >= 0 if name == "delta" else value > 0), f"Invalid {name}")
    source_dir = absolute(spec["source_experiment"])
    source_path = source_dir / "baseline/config/resolved_training_config.yaml"
    source = yaml.safe_load(source_path.read_text())
    original = read_json(source_dir / "experiment.json")
    require(original["manifest_sha256"] == digest({k: v for k, v in original.items() if k != "manifest_sha256"}),
            "Source experiment manifest failed integrity validation")
    config = deepcopy(original["config"])
    data = verify_frozen_data(source_dir / "data")
    require(file_sha256(source_dir / "data/data_manifest.json") == original["data_manifest_sha256"],
            "Frozen data differs from source experiment")
    required = {"max_steps": 200, "advantage": "grpo", "scale_rewards": True, "num_iterations": 1,
                "loss_type": "bnpo", "token_broadcast": "uniform", "beta": .04, "learning_rate": 1e-6,
                "shuffle_dataset": False, "remove_unused_columns": False, "save_reward_data": True,
                "do_eval": False, "eval_strategy": "no", "resume_from_checkpoint": None}
    for key, expected in required.items():
        require(source.get(key) == expected, f"Pilot recipe must have {key}={expected!r}")
    require(source["advantage_kwargs"] == {}, "Pilot must use default standard GRPO")
    for key in ("seed", "num_generations", "generation_batch_size", "gradient_accumulation_steps", "per_device_train_batch_size"):
        require(source[key] == config[key] == data["parameters"][key], f"Source schedule mismatch: {key}")
    require(config["steps"] == data["parameters"]["steps"] == 200, "Source plan must have 200 steps")
    require(source["model_name_or_path"] == config["model_name"] and
            source["model_revision"] == config["model_revision"], "Source model mismatch")
    hub_cache = absolute(spec["hub_cache"])
    policy = cached_snapshot(hub_cache, config["model_name"], config["model_revision"])
    qrm = cached_snapshot(hub_cache, config["qrm_model"], config["qrm_revision"])
    config.update(model_name=str(policy), qrm_model=str(qrm), delta=options["delta"], c=options["c"])
    require(not directory.exists(), f"Directory already exists: {directory}; use train --dry-run or a new directory")
    directory.mkdir(parents=True)
    (directory / "configs").mkdir()
    # Reuse the frozen dataset/plan in place, without resampling or copying data.
    (directory / "data").symlink_to(source_dir / "data", target_is_directory=True)
    files = {}
    for arm in ARMS:
        name = f"configs/{arm}.yaml"
        (directory / name).write_text(yaml.safe_dump(recipe_for_arm(source, directory, arm, config, options), sort_keys=False))
        files[name] = file_sha256(directory / name)
    shutil.copyfile(source_dir / "baseline/config/accelerate_config.yaml", directory / "configs/accelerate.yaml")
    files["configs/accelerate.yaml"] = file_sha256(directory / "configs/accelerate.yaml")
    manifest = {
        "version": 1, "directory": str(directory), "source_experiment": str(source_dir),
        "source_recipe_sha256": file_sha256(source_path), "config": config, "options": options,
        "hub_cache": str(hub_cache), "files": files, "code": code_inventory(),
        "data_manifest_sha256": file_sha256(source_dir / "data/data_manifest.json"),
        "runtime_versions": runtime_versions(), "numerical_environment": original["numerical_environment"],
        "local_snapshots": {str(path): snapshot_inventory(path) for path in (policy, qrm)},
        "formulas": {"robust_scaled": "k * A_robust", "weight_only_scaled": "k * RMS(A_robust)/RMS(A_GRPO) * A_GRPO; zero denominator -> zero"},
    }
    manifest["manifest_sha256"] = digest(manifest)
    write_json(directory / "controls.json", manifest)
    print(f"Prepared local-only controls: {directory}")
    print(f"Policy: {policy}\nQRM: {qrm}\nFrozen training groups: {data['train_count']}")
    return manifest


def verify(directory):
    from open_r1.paired_data import verify_frozen_data

    manifest = read_json(directory / "controls.json")
    require(manifest["manifest_sha256"] == digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}),
            "Control manifest changed; prepare a new directory")
    require(manifest["directory"] == str(directory), "Control directory moved")
    for name, checksum in manifest["files"].items():
        require(file_sha256(directory / name) == checksum, f"Frozen config changed: {name}")
    require(manifest["code"] == code_inventory(), "Training/control code changed; prepare a new directory")
    require(manifest["runtime_versions"] == runtime_versions(), "Runtime package versions changed")
    require(file_sha256(directory / "data/data_manifest.json") == manifest["data_manifest_sha256"], "Frozen data changed")
    verify_frozen_data(directory / "data")
    for path, inventory in manifest["local_snapshots"].items():
        require(snapshot_inventory(Path(path)) == inventory, f"Local snapshot files changed: {path}")
    return manifest


def environment(directory, manifest, arm):
    env = training_environment(directory, manifest, arm)
    hub_cache = Path(manifest["hub_cache"])
    env.update(OFFLINE)
    env.update({"HF_HUB_CACHE": str(hub_cache), "HF_HOME": str(hub_cache.parent),
                "TRAINING_ENTRYPOINT": str(ROOT / "scripts/grpo_control_train.py")})
    return env


def train(directory, arms, dry_run):
    manifest = verify(directory)
    # Reject all occupied destinations before starting the first expensive arm.
    if not dry_run:
        for arm in arms:
            require(not (directory / arm).exists(), f"Run already exists: {directory / arm}; no overwrite/resume")
    for arm in arms:
        env = environment(directory, manifest, arm)
        command = ["bash", str(ROOT / "train_scripts/qwen3_1.7_grpo_chat.sh")]
        if dry_run:
            command.append("--dry-run")
        print(f"{'Validate' if dry_run else 'Train'} {arm}: {manifest['formulas'][arm]}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--arm", choices=(*ARMS, "both"), default="both")
    parser.add_argument("--dry-run", action="store_true", help="For train: validate configs without GPU processes")
    args = parser.parse_args()
    if args.action == "prepare" and args.dry_run:
        parser.error("prepare only writes configs; --dry-run belongs to train")
    os.environ.update(OFFLINE)
    directory = absolute(args.directory)
    if args.action == "prepare":
        prepare(absolute(args.config), directory)
    else:
        train(directory, ARMS if args.arm == "both" else (args.arm,), args.dry_run)


if __name__ == "__main__":
    main()
