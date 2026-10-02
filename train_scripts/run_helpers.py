"""CPU-only YAML configuration and port checks for the GRPO Bash launcher.

Parsing before substitution preserves paths containing YAML punctuation as
strings. Only documented launch variables are substituted; unrelated environment
values are never expanded into the configuration.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import socket

import yaml


INTEGER_VARIABLES = {
    "VLLM_HTTP_PORT", "VLLM_GROUP_PORT", "QRM_HTTP_PORT", "QRM_REQUEST_TIMEOUT", "QRM_MAX_BATCH_TOKENS",
    "MAX_PROMPT_LENGTH", "MAX_COMPLETION_LENGTH", "GENERATION_BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS",
    "MAX_STEPS", "NUM_GENERATIONS", "PER_DEVICE_TRAIN_BATCH_SIZE",
}
STRING_VARIABLES = {"RUN_NAME", "OUTPUT_DIR", "DATASET_NAME", "MODEL_NAME", "MODEL_REVISION"}
PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}")


# Optional overrides apply after parsing, preserving YAML strings and types.
STRING_OVERRIDES = {
    "ADVANTAGE": "advantage",
    "DATASET_CONFIG": "dataset_config",
    "DATASET_ADAPTER": "dataset_adapter",
    "DATASET_PROMPT_COLUMN": "dataset_prompt_column",
    "DATASET_TRAIN_SPLIT": "dataset_train_split",
    "DATASET_TEST_SPLIT": "dataset_test_split",
    "SYSTEM_PROMPT": "system_prompt",
    "ATTN_IMPLEMENTATION": "attn_implementation",
    "EVAL_STRATEGY": "eval_strategy",
    "LOSS_TYPE": "loss_type",
}
INTEGER_OVERRIDES = {
    "MAX_TRAIN_SAMPLES": "max_train_samples",
    "MAX_EVAL_SAMPLES": "max_eval_samples",
    "PER_DEVICE_EVAL_BATCH_SIZE": "per_device_eval_batch_size",
    "EVAL_STEPS": "eval_steps",
}
BOOLEAN_OVERRIDES = {
    "DO_EVAL": "do_eval",
    "GRADIENT_CHECKPOINTING": "gradient_checkpointing",
    "LOG_COMPLETIONS": "log_completions",
    "SAVE_REWARD_DATA": "save_reward_data",
    "OVERLAP_QRM_REFERENCE": "overlap_qrm_reference",
}


def apply_training_overrides(config, env):
    for variable, field in STRING_OVERRIDES.items():
        if variable in env:
            config[field] = env[variable]
    for variable, field in INTEGER_OVERRIDES.items():
        if variable in env:
            config[field] = positive_int(variable, env)
    for variable, field in BOOLEAN_OVERRIDES.items():
        if variable in env:
            value = env[variable].lower()
            if value not in ("0", "1", "false", "true"):
                raise ValueError(f"{variable} must be 0, 1, false or true")
            config[field] = value in ("1", "true")
    if "ADVANTAGE_KWARGS" in env:
        try:
            options = json.loads(env["ADVANTAGE_KWARGS"])
        except json.JSONDecodeError as error:
            raise ValueError("ADVANTAGE_KWARGS must be a JSON object") from error
        if not isinstance(options, dict):
            raise ValueError("ADVANTAGE_KWARGS must be a JSON object")
        config["advantage_kwargs"] = options
    if config.get("eval_strategy", "no") not in ("no", "steps", "epoch"):
        raise ValueError("EVAL_STRATEGY must be no, steps or epoch")


def substitute(value, variables):
    if isinstance(value, dict):
        return {key: substitute(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, variables) for item in value]
    if not isinstance(value, str):
        return value

    def lookup(match):
        name = match.group(1)
        if name not in variables:
            raise ValueError(f"Unsupported or unset template variable: {name}")
        return variables[name]

    match = PLACEHOLDER.fullmatch(value)
    if match:
        return lookup(match)
    return PLACEHOLDER.sub(lambda match: str(lookup(match)), value)


def positive_int(name, env):
    value = env[name]
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def resolve(config_path, accelerate_path, env):
    variables = {key: env[key] for key in STRING_VARIABLES}
    variables.update({key: positive_int(key, env) for key in INTEGER_VARIABLES})
    gpu_text = env["TRAIN_GPUS"]
    vllm_gpu_text = env["VLLM_GPUS"]
    qrm_gpu_text = env["QRM_GPU"]
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", gpu_text):
        raise ValueError("TRAIN_GPUS must be a comma-separated list of GPU indices")
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", vllm_gpu_text):
        raise ValueError("VLLM_GPUS must be a comma-separated list of GPU indices")
    if not re.fullmatch(r"[0-9]+", qrm_gpu_text):
        raise ValueError("QRM_GPU must be one GPU index")
    gpu_ids = [int(item) for item in gpu_text.split(",")]
    vllm_gpu_ids = [int(item) for item in vllm_gpu_text.split(",")]
    qrm_gpu_id = int(qrm_gpu_text)
    qrm_max_length = positive_int("QRM_MAX_LENGTH", env)
    if "VLLM_MAX_NUM_SEQS" in env:
        positive_int("VLLM_MAX_NUM_SEQS", env)
    positive_int("REWARD_BATCH_SIZE", env)
    if variables["QRM_MAX_BATCH_TOKENS"] < qrm_max_length:
        raise ValueError("QRM_MAX_BATCH_TOKENS must be at least QRM_MAX_LENGTH")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("TRAIN_GPUS contains duplicate GPU indices")
    if len(set(vllm_gpu_ids)) != len(vllm_gpu_ids):
        raise ValueError("VLLM_GPUS contains duplicate GPU indices")
    overlap = sorted((set(vllm_gpu_ids) & set(gpu_ids)) | ({qrm_gpu_id} & (set(vllm_gpu_ids) | set(gpu_ids))))
    if overlap:
        raise ValueError(f"VLLM_GPUS, QRM_GPU and TRAIN_GPUS must be disjoint; overlap: {overlap}")
    if positive_int("NUM_VLLM_GPUS", env) != len(vllm_gpu_ids):
        raise ValueError("NUM_VLLM_GPUS must match the number of VLLM_GPUS")
    expected_batch = len(gpu_ids) * variables["PER_DEVICE_TRAIN_BATCH_SIZE"] * variables["GRADIENT_ACCUMULATION_STEPS"]
    if variables["GENERATION_BATCH_SIZE"] != expected_batch:
        raise ValueError(
            "GENERATION_BATCH_SIZE must equal training GPU count × per-device batch "
            f"× gradient accumulation ({expected_batch})"
        )
    if variables["NUM_GENERATIONS"] < 2 or expected_batch % variables["NUM_GENERATIONS"]:
        raise ValueError("NUM_GENERATIONS must be at least 2 and divide GENERATION_BATCH_SIZE")
    if variables["MAX_PROMPT_LENGTH"] + variables["MAX_COMPLETION_LENGTH"] > positive_int("VLLM_MAX_MODEL_LEN", env):
        raise ValueError("VLLM_MAX_MODEL_LEN must cover MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH")
    memory = float(env["VLLM_GPU_MEMORY_UTILIZATION"])
    if not math.isfinite(memory) or not 0 < memory <= 1:
        raise ValueError("VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]")
    ports = [positive_int(key, env) for key in ("VLLM_HTTP_PORT", "QRM_HTTP_PORT", "PORT", "VLLM_GROUP_PORT")]
    if max(ports) > 65535 or len(set(ports)) != len(ports):
        raise ValueError("VLLM_HTTP_PORT, QRM_HTTP_PORT, PORT and VLLM_GROUP_PORT must be distinct ports in 1..65535")
    for key in ("DRY_RUN", "MERGE_AFTER_TRAINING"):
        if env[key] not in ("0", "1"):
            raise ValueError(f"{key} must be 0 or 1")
    if env.get("VLLM_USE_V1", "0") != "0":
        raise ValueError("This pinned TRL/vLLM service requires VLLM_USE_V1=0")
    if env.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn") != "spawn":
        raise ValueError("VLLM_WORKER_MULTIPROC_METHOD must be spawn for CUDA workers")

    with config_path.open(encoding="utf-8") as stream:
        config = substitute(yaml.safe_load(stream), variables)
    with accelerate_path.open(encoding="utf-8") as stream:
        accelerate = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(accelerate, dict):
        raise ValueError("Training and Accelerate YAML files must contain mappings")
    apply_training_overrides(config, env)
    # The policy service and trainer must load the same revision, even with
    # a custom recipe. MODEL_REVISION controls both launch components.
    config["model_revision"] = variables["MODEL_REVISION"]
    # A custom template must match the launch plan, so server and trainer cannot
    # silently load different models or write into unrelated directories.
    expected = {
        "model_name_or_path": variables["MODEL_NAME"],
        "dataset_name": variables["DATASET_NAME"],
        "output_dir": variables["OUTPUT_DIR"],
        "generation_batch_size": expected_batch,
        "per_device_train_batch_size": variables["PER_DEVICE_TRAIN_BATCH_SIZE"],
        "gradient_accumulation_steps": variables["GRADIENT_ACCUMULATION_STEPS"],
        "num_generations": variables["NUM_GENERATIONS"],
        "max_prompt_length": variables["MAX_PROMPT_LENGTH"],
        "max_completion_length": variables["MAX_COMPLETION_LENGTH"],
        "max_steps": variables["MAX_STEPS"],
        "use_vllm": True,
        "vllm_mode": "server",
        "vllm_server_base_url": f"http://127.0.0.1:{ports[0]}",
        "vllm_group_port": ports[3],
        "reward_server_url": f"http://127.0.0.1:{ports[1]}",
        "reward_server_timeout": variables["QRM_REQUEST_TIMEOUT"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Training template {key} must resolve to {value!r}; got {config.get(key)!r}")
    if config.get("reward_funcs") != ["qrm_server"]:
        raise ValueError("This split launcher requires reward_funcs: [qrm_server]")
    if accelerate.get("num_machines", 1) != 1:
        raise ValueError("This launcher supports a single machine; use a separate launcher for multi-node training")
    if accelerate.get("distributed_type") not in ("DEEPSPEED", "MULTI_GPU"):
        raise ValueError("This launcher expects an Accelerate DEEPSPEED or MULTI_GPU configuration")
    if accelerate.get("distributed_type") == "MULTI_GPU":
        if len(gpu_ids) < 2:
            raise ValueError("MULTI_GPU requires at least two TRAIN_GPUS")
        # LoRA's frozen backbone has no gradients. Avoid unused-parameter
        # traversal, which also conflicts with reentrant checkpointing.
        if config.get("use_peft"):
            config.setdefault("ddp_find_unused_parameters", False)
    evaluation_enabled = config.get("do_eval") or config.get("eval_strategy", "no") != "no"
    if evaluation_enabled:
        eval_batch = config.get("per_device_eval_batch_size", 8) * len(gpu_ids)
        if eval_batch % variables["NUM_GENERATIONS"]:
            raise ValueError(
                "Evaluation requires PER_DEVICE_EVAL_BATCH_SIZE × training GPU count "
                "to be divisible by NUM_GENERATIONS"
            )
    if accelerate.get("machine_rank", 0) != 0:
        raise ValueError("Single-machine training requires machine_rank=0")
    if accelerate.get("use_cpu", False):
        raise ValueError("Accelerate use_cpu must be false for this CUDA training launcher")
    if (
        config.get("use_peft")
        and config.get("gradient_checkpointing")
        and accelerate.get("deepspeed_config", {}).get("zero_stage") == 3
        and not (config.get("gradient_checkpointing_kwargs") or {}).get("use_reentrant", True)
    ):
        raise ValueError("PEFT + ZeRO-3 requires gradient_checkpointing_kwargs.use_reentrant=true with the pinned runtime")
    checkpoint = env.get("RESUME_FROM_CHECKPOINT")
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_dir():
            raise ValueError(f"RESUME_FROM_CHECKPOINT must be an existing directory: {checkpoint_path}")
        config["resume_from_checkpoint"] = str(checkpoint_path)
    accelerate["num_processes"] = len(gpu_ids)
    accelerate["gpu_ids"] = "all"  # CUDA_VISIBLE_DEVICES selects physical GPUs.
    return config, accelerate


def check_ports(ports):
    # Detect any listener, including one that never responds to /health/.
    # Probes are short-lived; the real server still reports races after this.
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise ValueError(f"Local port {port} is unavailable: {error}. Choose another port; no process was stopped.") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    config_parser = commands.add_parser("resolve")
    config_parser.add_argument("--config", type=Path, required=True)
    config_parser.add_argument("--accelerate-config", type=Path, required=True)
    config_parser.add_argument("--write", action="store_true")
    ports_parser = commands.add_parser("check-ports")
    ports_parser.add_argument("ports", type=int, nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "check-ports":
            check_ports(args.ports)
            return
        config, accelerate = resolve(args.config, args.accelerate_config, os.environ)
        if args.write:
            destination = Path(os.environ["OUTPUT_DIR"]) / "config"
            for name, content in (("resolved_training_config.yaml", config), ("accelerate_config.yaml", accelerate)):
                with (destination / name).open("w", encoding="utf-8") as stream:
                    yaml.safe_dump(content, stream, sort_keys=False, allow_unicode=True)
        else:
            print(f"Plan: vLLM GPUs {os.environ['VLLM_GPUS']} (TP={os.environ['NUM_VLLM_GPUS']}); "
                  f"QRM GPU {os.environ['QRM_GPU']}; training GPUs {os.environ['TRAIN_GPUS']}")
            print(f"Training processes: {accelerate['num_processes']}; generation batch: {config['generation_batch_size']}")
            print(f"vLLM scheduling: max concurrent sequences={os.environ.get('VLLM_MAX_NUM_SEQS', 'vLLM default')}")
            print(f"Ports: vLLM HTTP={os.environ['VLLM_HTTP_PORT']}, QRM HTTP={os.environ['QRM_HTTP_PORT']}, "
                  f"training={os.environ['PORT']}, weight sync={os.environ['VLLM_GROUP_PORT']}")
            print(f"QRM batching: max examples={os.environ['REWARD_BATCH_SIZE']}; "
                  f"padded-token budget={os.environ['QRM_MAX_BATCH_TOKENS']}; "
                  f"max length={os.environ['QRM_MAX_LENGTH']}")
            print(f"Output: {os.environ['OUTPUT_DIR']}")
            print(f"HF_HOME: {os.environ['HF_HOME']}; HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")
            if os.environ["DRY_RUN"] == "1":
                print("Resolved training configuration:")
                print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), end="")
                print("Resolved Accelerate configuration:")
                print(yaml.safe_dump(accelerate, sort_keys=False, allow_unicode=True), end="")
    except (KeyError, ValueError, OSError, yaml.YAMLError) as error:
        parser.exit(2, f"Configuration error: {error}\n")


if __name__ == "__main__":
    main()
