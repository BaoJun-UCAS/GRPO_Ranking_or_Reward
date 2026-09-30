# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from copy import deepcopy
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.grpo_trainer import GRPOTrainer
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.data import prepare_grpo_dataset, truncate_conversation_prompt
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


def _redact_secrets(value):
    """Return a JSON-serializable configuration snapshot without credentials."""
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            secret_key = normalized_key in {
                "token",
                "api_key",
                "apikey",
                "secret",
                "password",
                "hf_token",
                "huggingface_token",
                "wandb_api_key",
            } or normalized_key.endswith(("_api_key", "_access_token", "_auth_token", "_password", "_secret"))
            if secret_key:
                redacted[str(key)] = "<redacted>" if item else item
            else:
                redacted[str(key)] = _redact_secrets(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item) for item in value]
    return value


def _arguments_to_dict(arguments):
    if hasattr(arguments, "to_dict"):
        return arguments.to_dict()
    if is_dataclass(arguments):
        return asdict(arguments)
    return vars(arguments)


def save_run_manifest(output_dir, script_args, training_args, model_args):
    """Persist the exact parsed configuration and relevant package versions."""
    package_versions = {}
    for package in ("accelerate", "datasets", "deepspeed", "peft", "torch", "transformers", "trl", "vllm"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_arguments": _arguments_to_dict(script_args),
        "training_arguments": _arguments_to_dict(training_args),
        "model_arguments": _arguments_to_dict(model_args),
        "package_versions": package_versions,
    }
    manifest = _redact_secrets(manifest)
    output_path = Path(output_dir) / "run_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    temporary_path.replace(output_path)


def main(script_args, training_args, model_args):
    if training_args.do_eval or training_args.eval_strategy != "no":
        eval_batch = training_args.per_device_eval_batch_size * training_args.world_size
        if eval_batch % training_args.num_generations:
            raise ValueError("Global evaluation batch size must be divisible by num_generations")
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = get_dataset(script_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    dataset = prepare_grpo_dataset(dataset, script_args, training_args, tokenizer)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    #############################
    # Initialize the GRPO trainer
    #############################
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        peft_config.revision = model_args.model_revision
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.do_eval or training_args.eval_strategy != "no"
            else None
        ),
        peft_config=peft_config,
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )
    if trainer.accelerator.is_main_process:
        save_run_manifest(training_args.output_dir, script_args, training_args, model_args)

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["grpo"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Save an inference-friendly config without mutating the live policy
        # before evaluation (or making rank 0 differ from the other ranks).
        inference_config = deepcopy(trainer.model.config)
        inference_config.use_cache = True
        inference_config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
