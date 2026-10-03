# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available, is_rich_available

from .utils.data_utils import apply_chat_template, maybe_apply_chat_template, is_conversational
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from .configs import GRPOConfig
from .advantages import compute_advantages, configure_advantage
from .rolling_advantages import RollingQuantilePairwise
from .objectives import reduce_policy_loss
from .training_schedule import TrainingSchedule, append_schedule_trace, permute_rollout_payload
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def trim_padded_microbatch(
    inputs: dict[str, Optional[torch.Tensor]],
) -> dict[str, Optional[torch.Tensor]]:
    """Drop columns that are padding for every sample in one policy micro-batch."""

    trimmed = dict(inputs)
    prompt_lengths = inputs.get("_prompt_lengths")
    completion_lengths = inputs.get("_completion_lengths")
    if prompt_lengths is not None:
        prompt_width = max(1, int(prompt_lengths.max().item()))
        for key in ("prompt_ids", "prompt_mask"):
            if inputs.get(key) is not None:
                trimmed[key] = inputs[key][:, -prompt_width:]
    if completion_lengths is not None:
        completion_width = max(1, int(completion_lengths.max().item()))
        for key in ("completion_ids", "completion_mask", "old_per_token_logps", "ref_per_token_logps"):
            if inputs.get(key) is not None:
                trimmed[key] = inputs[key][:, :completion_width]
    trimmed.pop("_prompt_lengths", None)
    trimmed.pop("_completion_lengths", None)
    return trimmed


def completion_lists_from_tensors(
    completion_ids: torch.Tensor, completion_lengths: torch.Tensor
) -> tuple[list[list[int]], list[list[int]], torch.Tensor]:
    """Transfer token IDs/lengths in bulk, retaining the first EOS in each prefix.

    Lengths come from the original contiguous EOS mask, before optionally
    masking truncated responses out of the loss. Return the padded rows too:
    decoding must preserve its existing full-row semantics, and can reuse this
    CPU copy instead of transferring the same CUDA tensor a second time.
    """
    rows = completion_ids.detach().cpu().tolist()
    lengths = completion_lengths.detach().cpu()
    valid_rows = [row[:length] for row, length in zip(rows, lengths.tolist())]
    return rows, valid_rows, lengths


def summarize_timing_samples(samples: dict[str, list[float]]) -> dict[str, float]:
    """Summarize local micro-step wall times once per optimizer step."""

    counts = {len(values) for values in samples.values()}
    if not samples or counts == {0} or len(counts) != 1:
        raise ValueError("Timing samples must contain equally sized, non-empty series")
    summary = {}
    for name, values in samples.items():
        summary[f"{name}_total"] = sum(values)
        summary[f"{name}_mean"] = sum(values) / len(values)
    return summary


def shuffle_tensor_dict(tensor_dict: dict[str, Optional[torch.Tensor]]) -> dict[str, Optional[torch.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.

    Example:
        >>> x = torch.arange(6).reshape(3, 2)
        >>> y = torch.arange(3).reshape(3, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> shuffle_tensor_dict(tensor_dict)
        {'x': tensor([[2, 3],
                      [0, 1],
                      [4, 5]]),
         'y': tensor([[1],
                      [0],
                      [2]])}
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = torch.randperm(batch_size)
    return {key: tensor[permutation] if tensor is not None else None for key, tensor in tensor_dict.items()}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def run_reward_reference_overlap(reward_call, reference_call):
    """Run only the remote request in a worker; model scoring stays on this thread.

    Executor exit joins the worker even when either callable raises. Timings are
    independent wall intervals and overlap, so they must not be added together.
    """
    def timed_reward():
        started = time.perf_counter()
        rewards = reward_call()
        return rewards, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qrm-reference") as executor:
        reward_future = executor.submit(timed_reward)
        reference_started = time.perf_counter()
        reference_logps = reference_call()
        reference_elapsed = time.perf_counter() - reference_started
        rewards, reward_elapsed = reward_future.result()
    return rewards, reference_logps, {
        "qrm_service": reward_elapsed,
        "reference_precompute": reference_elapsed,
    }


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return None when the reward is not applicable to those samples. This is useful for
                  multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns None for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`]. A
            padding token, `processing_class.pad_token`, must be set. If the processing class has not set a padding
            token, `processing_class.eos_token` will be used as the default.
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Resolve custom estimators before loading any large model weights.
        self.advantage_estimator, self.advantage_kwargs = configure_advantage(
            args.advantage,
            getattr(args, "advantage_kwargs", None),
            scale_rewards=args.scale_rewards,
        )
        if isinstance(self.advantage_estimator, RollingQuantilePairwise):
            from .rolling_advantage_state import RollingAdvantageStateCallback
            callbacks = list(callbacks or []) + [RollingAdvantageStateCallback(self.advantage_estimator)]

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        # init args for reward functions
        reward_init_kwargs = {
            'revision': 'main',
            'trust_remote_code': True,
            'attn_implementation': 'flash_attention_2',
            'torch_dtype': torch.bfloat16,
            'use_cache': False,
        }

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **reward_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions
        self.use_cpu_object_collectives = args.use_cpu_object_collectives
        self.trim_unused_padding = args.trim_unused_padding
        self.profile_stage_timings = args.profile_stage_timings

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        # Reward batch size for neural network models
        self.reward_batch_size = args.reward_batch_size

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Python-object collectives otherwise use the NCCL process group and can
        # make an idle rank look GPU-busy while rank 0 waits for an HTTP service.
        # Keep the legacy path as a fallback for builds without Gloo.
        self._object_collective_group = None
        if self.use_cpu_object_collectives and self.accelerator.num_processes > 1:
            gloo_available = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.is_gloo_available()
            )
            if gloo_available:
                try:
                    self._object_collective_group = torch.distributed.new_group(backend="gloo")
                except RuntimeError as error:
                    warnings.warn(f"Could not create the Gloo control group; falling back to NCCL objects: {error}")
            else:
                warnings.warn("Gloo object collectives requested but unavailable; falling back to the default backend")

        self.training_schedule = None
        schedule_path = getattr(args, "training_schedule_path", None)
        if schedule_path is not None:
            self.training_schedule = TrainingSchedule.load(schedule_path, args, self.accelerator.num_processes)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled() or self.is_fsdp_enabled:
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger loss
        if self.use_liger_loss:
            if args.token_broadcast != "uniform":
                raise ValueError("use_liger_loss only supports token_broadcast='uniform'")
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._pending_policy_timings = defaultdict(list)
        self._last_prepare_wall_seconds = 0.0
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                if args.vllm_server_base_url is not None:
                    base_url = args.vllm_server_base_url
                else:
                    base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                self.vllm_client = VLLMClient(
                    base_url=base_url,
                    connection_timeout=args.vllm_server_timeout,
                    group_port=args.vllm_group_port,
                )
                self.vllm_client.init_communicator()

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=(self.max_prompt_length + self.max_completion_length) * 2,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                )

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

        # Initialize reward data collection if enabled
        self.save_reward_data = args.save_reward_data
        if self.save_reward_data:
            import json
            import os

            # Set up save path
            if args.reward_data_save_path is not None:
                self.reward_data_save_path = args.reward_data_save_path
            else:
                self.reward_data_save_path = os.path.join(args.output_dir, "reward_data")

            # Create directory if it doesn't exist
            os.makedirs(self.reward_data_save_path, exist_ok=True)

            # Initialize data collection structures
            self.reward_data_buffer = []  # Buffer for current training step batch

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        if isinstance(self.advantage_estimator, RollingQuantilePairwise) and resume_from_checkpoint:
            checkpoint = resume_from_checkpoint
            if checkpoint is True:
                from transformers.trainer_utils import get_last_checkpoint
                checkpoint = get_last_checkpoint(self.args.output_dir)
                if checkpoint is None:
                    raise ValueError("No checkpoint found for rolling advantage resume")
            self.advantage_estimator.load(checkpoint)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial,
                             ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)

    def _collect_reward_data(self, prompts, completions, rewards, global_step):
        """Collect reward data for saving to files."""
        import json

        # Convert tensors to lists for JSON serialization
        if isinstance(rewards, torch.Tensor):
            rewards_list = rewards.cpu().tolist()
        else:
            rewards_list = list(rewards)

        # `rewards` has already been gathered across all processes, while prompts and
        # completions are local. Select this rank's matching reward slice before saving.
        local_count = len(prompts)
        if len(rewards_list) != local_count:
            start = self.accelerator.process_index * local_count
            end = start + local_count
            if end > len(rewards_list):
                raise ValueError(
                    "Cannot align gathered rewards with local prompt/completion data: "
                    f"rank={self.accelerator.process_index}, local_count={local_count}, "
                    f"reward_count={len(rewards_list)}"
                )
            rewards_list = rewards_list[start:end]

        # Collect data for this batch
        batch_data = []
        for i, (prompt, completion, reward) in enumerate(zip(prompts, completions, rewards_list)):
            # Convert conversational format to string if needed
            if isinstance(prompt, list):  # conversational format
                prompt_text = str(prompt)  # convert to string representation
            else:
                prompt_text = prompt

            if isinstance(completion, list):  # conversational format
                completion_text = str(completion)
            else:
                completion_text = completion

            batch_data.append({
                "prompt": prompt_text,
                "completion": completion_text,
                "reward": float(reward),
                "training_step": int(global_step),
                "process_index": self.accelerator.process_index,
                "rollout_index": i,
            })

        # Save immediately since generation_batch_size generates fresh data periodically.
        self.reward_data_buffer.extend(batch_data)
        self._save_reward_data_buffer(global_step)

    def _save_reward_data_buffer(self, global_step):
        """Save buffered reward data to file."""
        if not self.reward_data_buffer:
            return

        import json

        import uuid

        # Eval does not advance the training micro-step. Separate phases and
        # uniquely identify every rollout, including repeated evaluate() calls.
        phase = "train" if self.model.training else "eval"
        record_id = uuid.uuid4().hex
        filename = (
            f"reward_data_{phase}_step_{int(global_step):06d}_micro_{int(self._step):08d}_"
            f"proc_{self.accelerator.process_index}_{record_id}.json"
        )
        filepath = os.path.join(self.reward_data_save_path, filename)
        temporary_filepath = f"{filepath}.tmp"

        # Atomic replacement prevents a partially written JSON file after interruption.
        with open(temporary_filepath, "w", encoding="utf-8") as f:
            json.dump(self.reward_data_buffer, f, ensure_ascii=False)
        os.replace(temporary_filepath, filepath)

        # Clear buffer
        self.reward_data_buffer.clear()

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        checkpoint_kwargs = dict(args.gradient_checkpointing_kwargs or {})
        checkpoint_kwargs.setdefault("use_reentrant", True)
        # Frozen PEFT weights can be repartitioned to shape [0] during
        # non-reentrant recomputation in the pinned DeepSpeed 0.16.8 stack.
        # https://github.com/deepspeedai/DeepSpeed/issues/4332
        if is_peft_model(model) and is_deepspeed_zero3_enabled() and not checkpoint_kwargs["use_reentrant"]:
            raise ValueError(
                "PEFT + DeepSpeed ZeRO-3 requires gradient_checkpointing_kwargs.use_reentrant=true "
                "with the pinned runtime. Non-reentrant recomputation can see partitioned, empty weights."
            )
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)

        if checkpoint_kwargs["use_reentrant"]:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        last_hidden_state = unwrapped_model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1, use_cache=False,
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            if self.temperature != 1.0:
                logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return all_logps[0] if len(all_logps) == 1 else torch.cat(all_logps, dim=0)

    def _can_overlap_qrm_reference(self, mode: str) -> bool:
        """Conservative guards keep caching inside one unchanged policy step."""
        if not getattr(self.args, "overlap_qrm_reference", False) or mode != "train":
            return False
        reasons = []
        if self.beta <= 0:
            reasons.append("beta must be positive")
        if self.reward_func_names != ["qrm_server"]:
            reasons.append("exactly one qrm_server reward is required")
        if not self.args.disable_dropout or self.args.sync_ref_model:
            reasons.append("dropout must be disabled and reference synchronization off")
        # Some attention implementations use F.dropout/FlashAttention's float
        # argument rather than nn.Dropout, which disable_dropout_in_model does
        # not necessarily change. Inspect the actual reference configuration.
        reference_model = self.ref_model if self.ref_model is not None else self.model
        reference_model = self.accelerator.unwrap_model(reference_model)
        if is_peft_model(reference_model):
            reference_model = reference_model.get_base_model()
        reference_config = getattr(reference_model, "config", None)
        # Forward ordering has only been validated for these dense models.
        # Other architectures can consume RNG or update state even without
        # nn.Dropout (for example, MoE router jitter).
        if getattr(reference_config, "model_type", None) not in {"qwen2", "qwen3"}:
            reasons.append("reference overlap supports only dense Qwen2/Qwen3 models")
        reference_configs = [reference_config]
        if reference_config is not None and hasattr(reference_config, "get_text_config"):
            reference_configs.append(reference_config.get_text_config())
        for config in reference_configs:
            rope_scaling = getattr(config, "rope_scaling", None)
            if rope_scaling and (
                not isinstance(rope_scaling, dict)
                or rope_scaling.get("rope_type", rope_scaling.get("type", "default")) != "default"
            ):
                reasons.append("reference overlap requires default RoPE scaling")
                break
        functional_dropout_fields = (
            "attention_dropout", "attn_pdrop", "attention_probs_dropout_prob",
            "attention_dropout_prob", "attn_dropout", "attention_dropout_rate",
        )
        for name in functional_dropout_fields:
            if any(getattr(config, name, 0) not in (None, 0) for config in reference_configs):
                reasons.append(f"reference config {name} must be zero")
        if self.args.per_device_train_batch_size != 1 or not self.trim_unused_padding:
            reasons.append("microbatch=1 and trim_unused_padding=true are required")
        if self.use_liger_loss:
            reasons.append("the overlap path currently requires the ordinary policy loss")
        if self.is_fsdp_enabled:
            reasons.append("FSDP is unsupported")
        plugin = self.accelerator.state.deepspeed_plugin
        if self.is_deepspeed_enabled and (plugin is None or plugin.zero_stage != 2):
            reasons.append("only DeepSpeed ZeRO-2 is supported")
        if self.accelerator.num_processes < 2:
            reasons.append("distributed QRM scoring is required")
        if (
            self.num_iterations != 1
            or self.args.gradient_accumulation_steps % self.args.steps_per_generation != 0
        ):
            reasons.append("one iteration and generation batches aligned within optimizer steps are required")
        if reasons:
            if not getattr(self, "_reference_overlap_fallback_warned", False):
                if self.accelerator.is_main_process:
                    warnings.warn("QRM/reference overlap disabled: " + "; ".join(reasons))
                self._reference_overlap_fallback_warned = True
            return False
        return True

    def _precompute_reference_logps(
        self, prompt_ids, prompt_mask, completion_ids, completion_mask, prompt_lengths, completion_lengths
    ):
        """Score the same individually trimmed inputs used by policy microbatch 1.

        No policy update, RNG sampling, train/eval mode change, or optimizer call
        occurs here. The detached cache follows ordinary shuffle/split/trim.
        """
        prompt_widths = prompt_lengths.detach().cpu().tolist()
        completion_widths = completion_lengths.detach().cpu().tolist()
        reference_model = self.ref_model if self.ref_model is not None else self.model
        adapter_context = (
            nullcontext() if self.ref_model is not None
            else self.accelerator.unwrap_model(self.model).disable_adapter()
        )
        cached_logps = None
        with torch.no_grad(), self.compute_loss_context_manager(), adapter_context:
            for index, (prompt_width, completion_width) in enumerate(zip(prompt_widths, completion_widths)):
                prompt_width, completion_width = max(1, int(prompt_width)), max(1, int(completion_width))
                input_ids = torch.cat((
                    prompt_ids[index : index + 1, -prompt_width:],
                    completion_ids[index : index + 1, :completion_width],
                ), dim=1)
                attention_mask = torch.cat((
                    prompt_mask[index : index + 1, -prompt_width:],
                    completion_mask[index : index + 1, :completion_width],
                ), dim=1)
                logps = self._get_per_token_logps(reference_model, input_ids, attention_mask, completion_width)
                if cached_logps is None:
                    cached_logps = logps.new_zeros(completion_ids.shape)
                cached_logps[index : index + 1, :completion_width] = logps.detach()
        # The remote request continues in its worker while this rank finishes
        # queued scoring kernels. Count completed reference work, not just CUDA
        # launch time, and surface asynchronous failures before error exchange.
        if cached_logps is not None and cached_logps.is_cuda:
            torch.cuda.synchronize(cached_logps.device)
        return cached_logps

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
            else:
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def _gather_python_objects(self, objects: list[Any]) -> list[Any]:
        if self._object_collective_group is None:
            return gather_object(objects)
        gathered = [None for _ in range(self.accelerator.num_processes)]
        torch.distributed.all_gather_object(gathered, objects, group=self._object_collective_group)
        return [item for rank_items in gathered for item in rank_items]

    def _broadcast_python_objects(self, objects: list[Any], from_process: int = 0) -> list[Any]:
        if self._object_collective_group is None:
            return broadcast_object_list(objects, from_process=from_process)
        torch.distributed.broadcast_object_list(
            objects,
            src=from_process,
            group=self._object_collective_group,
            device=torch.device("cpu"),
        )
        return objects

    def _record_phase_timings(self, mode: str, timings: dict[str, float]) -> None:
        if not self.profile_stage_timings:
            return
        names = tuple(timings)
        values = [timings[name] for name in names]
        if self._object_collective_group is not None:
            gathered = [None for _ in range(self.accelerator.num_processes)]
            torch.distributed.all_gather_object(gathered, values, group=self._object_collective_group)
            all_ranks = torch.tensor(gathered, dtype=torch.float32)
        else:
            local = torch.tensor([values], device=self.accelerator.device, dtype=torch.float32)
            all_ranks = self.accelerator.gather(local).reshape(-1, len(names)).cpu()
        for column, name in enumerate(names):
            values = all_ranks[:, column]
            self._metrics[mode][f"timing/{name}_max_s"].append(values.max().item())
            self._metrics[mode][f"timing/{name}_min_s"].append(values.min().item())
            self._metrics[mode][f"timing/{name}_rank_spread_s"].append((values.max() - values.min()).item())
            for rank, value in enumerate(values.tolist()):
                self._metrics[mode][f"timing/{name}_rank{rank}_s"].append(value)

    def _prepare_training_schedule_trace(self, generation_batch):
        """All ranks agree on validity before any generation collectives run."""
        trace, local_error = None, None
        try:
            trace = self.training_schedule.prepare_trace(
                generation_batch, self.state.global_step, self._step, self.accelerator.process_index
            )
        except Exception as error:
            local_error = f"rank {self.accelerator.process_index}: {type(error).__name__}: {error}"
        errors = self._gather_python_objects([local_error])
        if any(error is not None for error in errors):
            raise ValueError("Training schedule validation failed: " + "; ".join(error for error in errors if error))
        return trace

    def _apply_training_schedule_trace(self, generation_batch, trace):
        """Record the permutation that is actually consumed by this rollout."""
        local_error = None
        try:
            generation_batch = permute_rollout_payload(generation_batch, trace["permutation"])
            append_schedule_trace(self.args.output_dir, trace)
        except Exception as error:
            local_error = f"rank {self.accelerator.process_index}: {type(error).__name__}: {error}"
        errors = self._gather_python_objects([local_error])
        if any(error is not None for error in errors):
            raise ValueError("Training schedule trace failed: " + "; ".join(error for error in errors if error))
        return generation_batch

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        prepare_started = time.perf_counter()
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                schedule_trace = None
                if getattr(self, "training_schedule", None) is not None:
                    schedule_trace = self._prepare_training_schedule_trace(generation_batch)
                generation_batch = self._generate_and_score_completions(generation_batch)
                if schedule_trace is None:
                    generation_batch = shuffle_tensor_dict(generation_batch)
                else:
                    generation_batch = self._apply_training_schedule_trace(generation_batch, schedule_trace)
                self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            buffer_index = self._step % self.args.steps_per_generation
            inputs = self._buffered_inputs[buffer_index]
            if self.trim_unused_padding:
                inputs = trim_padded_microbatch(inputs)
                self._buffered_inputs[buffer_index] = inputs
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
            if self.trim_unused_padding:
                inputs = trim_padded_microbatch(inputs)
        self._last_prepare_wall_seconds = time.perf_counter() - prepare_started
        return inputs

    def training_step(self, model, inputs, num_items_in_batch=None):
        step_started = time.perf_counter()
        self._last_prepare_wall_seconds = 0.0
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.profile_stage_timings:
            total = time.perf_counter() - step_started
            policy = max(0.0, total - self._last_prepare_wall_seconds)
            self._pending_policy_timings["training_step"].append(total)
            self._pending_policy_timings["policy_train"].append(policy)
            if self.accelerator.sync_gradients:
                timing_summary = summarize_timing_samples(self._pending_policy_timings)
                self._record_phase_timings("train", timing_summary)
                self._pending_policy_timings.clear()
        return loss

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        generation_started = time.perf_counter()
        phase_timings = {
            "weight_sync": 0.0,
            "rollout_gather": 0.0,
            "vllm_service": 0.0,
            "rollout_broadcast": 0.0,
            "rollout_total": 0.0,
            "qrm_gather": 0.0,
            "qrm_service": 0.0,
            "qrm_broadcast": 0.0,
            "qrm_total": 0.0,
            "reference_precompute": 0.0,
            "external_sync_wait": 0.0,
            "generation_score_total": 0.0,
        }

        prompts = [x["prompt"] for x in inputs]

        # remove "completion" from inputs if exists
        if "completion" in inputs[0]:
            inputs = [{"prompt": x["prompt"]} for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_lengths_for_trimming = prompt_inputs["attention_mask"].sum(dim=1)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompt_lengths_for_trimming = prompt_lengths_for_trimming.clamp(max=self.max_prompt_length)

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                weight_sync_started = time.perf_counter()
                self._move_model_to_vllm()
                phase_timings["weight_sync"] = time.perf_counter() - weight_sync_started
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                rollout_started = time.perf_counter()
                gather_started = time.perf_counter()
                all_prompts_text = self._gather_python_objects(prompts_text)
                phase_timings["rollout_gather"] = time.perf_counter() - gather_started
                if self.accelerator.is_main_process:
                    # Since prompts contain num_generations duplicates, generate once per unique prompt.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    service_started = time.perf_counter()
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                    phase_timings["vllm_service"] = time.perf_counter() - service_started
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # With the optional Gloo group, non-main ranks wait on CPU instead of spinning an NCCL GPU kernel.
                broadcast_started = time.perf_counter()
                completion_ids = self._broadcast_python_objects(completion_ids, from_process=0)
                phase_timings["rollout_broadcast"] = time.perf_counter() - broadcast_started
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]
                phase_timings["rollout_total"] = time.perf_counter() - rollout_started

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1,  # vLLM on each GPU generates only 1 in colocate mode
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)
        # Avoid a CUDA scalar synchronization for every mask/token. Reuse the
        # bulk CPU copy for decoding as well as token-based reward functions.
        completion_rows_cpu, completion_ids_list, completion_lengths_for_trimming = completion_lists_from_tensors(
            completion_ids, completion_lengths
        )

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_rows_cpu, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        cached_ref_per_token_logps = None
        overlap_reference = self._can_overlap_qrm_reference(mode)
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]

                    # Process in batches if reward_batch_size is specified
                    print(f"Calling reward function with batch size {self.reward_batch_size}")
                    if self.reward_batch_size is not None and len(texts) > self.reward_batch_size:
                        batch_rewards = []
                        for batch_start in range(0, len(texts), self.reward_batch_size):
                            batch_end = min(batch_start + self.reward_batch_size, len(texts))
                            batch_texts = texts[batch_start:batch_end]

                            batch_reward_inputs = reward_processing_class(
                                text=batch_texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                            )
                            batch_reward_inputs = super()._prepare_inputs(batch_reward_inputs)

                            with torch.inference_mode():
                                batch_outputs = reward_func(**batch_reward_inputs)
                                batch_rewards.append(batch_outputs.logits[:, 0])  # Shape (batch_size,)

                        # Concatenate all batch results
                        rewards_per_func[:, i] = torch.cat(batch_rewards, dim=0)
                    else:
                        # Process all at once (original behavior)
                        reward_inputs = reward_processing_class(
                            text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                        )
                        reward_inputs = super()._prepare_inputs(reward_inputs)
                        with torch.inference_mode():
                            outputs = reward_func(**reward_inputs)
                            rewards_per_func[:, i] = outputs.logits[:, 0]  # Shape (B*G,)
                else:
                    if reward_func_name == "qrm_server" and self.accelerator.num_processes > 1:
                        # The external QRM owns one GPU. Gather each rank's local
                        # samples so it scores every completion exactly once, then
                        # broadcast and restore each rank's original slice.
                        qrm_started = time.perf_counter()
                        gather_started = time.perf_counter()
                        rank_payloads = self._gather_python_objects([{
                            "prompts": prompts,
                            "completions": completions,
                            "completion_ids": completion_ids_list,
                        }])
                        phase_timings["qrm_gather"] += time.perf_counter() - gather_started
                        counts = [len(payload["prompts"]) for payload in rank_payloads]
                        if self.accelerator.is_main_process:
                            all_prompts = [item for payload in rank_payloads for item in payload["prompts"]]
                            all_completions = [item for payload in rank_payloads for item in payload["completions"]]
                            all_completion_ids = [
                                item for payload in rank_payloads for item in payload["completion_ids"]
                            ]

                        if overlap_reference:
                            def reference_call():
                                return self._precompute_reference_logps(
                                    prompt_ids, prompt_mask, completion_ids, completion_mask,
                                    prompt_lengths_for_trimming, completion_lengths_for_trimming,
                                )

                            overlap_exception = None
                            try:
                                if self.accelerator.is_main_process:
                                    def reward_call():
                                        return reward_func(
                                            prompts=all_prompts, completions=all_completions,
                                            completion_ids=all_completion_ids,
                                        )

                                    output_reward_func, cached_ref_per_token_logps, overlap_timings = (
                                        run_reward_reference_overlap(reward_call, reference_call)
                                    )
                                    for name, elapsed in overlap_timings.items():
                                        phase_timings[name] += elapsed
                                else:
                                    reference_started = time.perf_counter()
                                    cached_ref_per_token_logps = reference_call()
                                    phase_timings["reference_precompute"] += time.perf_counter() - reference_started
                                    output_reward_func = [None] * sum(counts)
                            except Exception as error:
                                overlap_exception = error
                                output_reward_func = [None] * sum(counts)
                            # Every rank reaches this control collective even if
                            # its own scoring failed; no peer waits for a reward
                            # broadcast that the main rank abandoned on error.
                            errors = self._gather_python_objects([
                                None if overlap_exception is None else
                                f"rank {self.accelerator.process_index}: {type(overlap_exception).__name__}: {overlap_exception}"
                            ])
                            if any(error is not None for error in errors):
                                raise RuntimeError(
                                    "QRM/reference overlap failed: " + "; ".join(error for error in errors if error is not None)
                                ) from overlap_exception
                        elif self.accelerator.is_main_process:
                            service_started = time.perf_counter()
                            output_reward_func = reward_func(
                                prompts=all_prompts,
                                completions=all_completions,
                                completion_ids=all_completion_ids,
                            )
                            phase_timings["qrm_service"] += time.perf_counter() - service_started
                        else:
                            output_reward_func = [None] * sum(counts)
                        broadcast_started = time.perf_counter()
                        output_reward_func = self._broadcast_python_objects(output_reward_func, from_process=0)
                        phase_timings["qrm_broadcast"] += time.perf_counter() - broadcast_started
                        start = sum(counts[: self.accelerator.process_index])
                        output_reward_func = output_reward_func[start : start + counts[self.accelerator.process_index]]
                        phase_timings["qrm_total"] += time.perf_counter() - qrm_started
                    else:
                        output_reward_func = reward_func(
                            prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                        )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        phase_timings["external_sync_wait"] = (
            phase_timings["rollout_broadcast"] + phase_timings["qrm_broadcast"]
        )
        phase_timings["generation_score_total"] = time.perf_counter() - generation_started
        self._record_phase_timings(mode, phase_timings)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather rewards before group-wise advantage computation: one prompt's
        # completions may be distributed across processes.
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Save reward data if enabled
        if self.save_reward_data:
            self._collect_reward_data(prompts, completions, rewards, self.state.global_step)

        # Advantage estimators always receive complete global groups, including
        # groups whose completions straddle distributed rank boundaries.
        advantage_options = dict(self.advantage_kwargs)
        if isinstance(self.advantage_estimator, RollingQuantilePairwise):
            advantage_options.update(step=self.state.global_step, update_history=(mode == "train"))
        advantage_result = compute_advantages(
            rewards,
            self.num_generations,
            method=self.advantage_estimator,
            method_kwargs=advantage_options,
            rewards_per_func=rewards_per_func,
            reward_weights=self.reward_weights.to(device),
        )
        mean_grouped_rewards = advantage_result.group_mean
        std_grouped_rewards = advantage_result.group_std
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))
        advantages = advantage_result.advantages
        if isinstance(self.advantage_estimator, RollingQuantilePairwise):
            for name, value in self.advantage_estimator.last_metrics.items():
                self._metrics[mode][f"advantage/{name}"].append(value)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        # Both numerator and denominator must cover all ranks. Dividing the
        # global EOS count by a local batch size can produce negative ratios.
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Gathering complete conversations is expensive and only serves the
        # optional completion table. Scalar metrics/reward persistence are
        # independent and must not pay for two extra object collectives.
        if self.log_completions:
            self._textual_logs["prompt"].extend(self._gather_python_objects(prompts_text))
            self._textual_logs["completion"].extend(self._gather_python_objects(completions_text))
            for i, name in enumerate(self.reward_func_names):
                self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        result = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": cached_ref_per_token_logps,
        }
        if self.trim_unused_padding:
            result["_prompt_lengths"] = prompt_lengths_for_trimming
            result["_completion_lengths"] = completion_lengths_for_trimming
        return result

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs.get("ref_per_token_logps")
        if self.beta != 0.0 and ref_per_token_logps is None:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, logits_to_keep
                        )

        # get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(unwrapped_model, input_ids, attention_mask, logits_to_keep)

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs["old_per_token_logps"],
            ref_per_token_logps=ref_per_token_logps,
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs.get("ref_per_token_logps")
            if ref_per_token_logps is None:
                with torch.no_grad():
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps(
                            self.ref_model, input_ids, attention_mask, logits_to_keep
                        )
                    else:
                        with self.accelerator.unwrap_model(self.model).disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(
                                self.model, input_ids, attention_mask, logits_to_keep
                            )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        # Build per-token weighting of the scalar advantage per completion according to token_broadcast strategy
        token_broadcast = getattr(self.args, "token_broadcast", "uniform")
        T = completion_ids.size(1)
        B = completion_ids.size(0)
        device = per_token_logps.device
        if token_broadcast == "uniform":
            weighted_advantages = advantages.unsqueeze(1)
        else:
            positions = torch.arange(1, T + 1, device=device, dtype=per_token_logps.dtype).unsqueeze(0).expand(B, T)
            if token_broadcast == "log_decay":
                decay = 1.0 / torch.log1p(positions)
            elif token_broadcast == "poly_decay":
                decay = 1.0 / torch.sqrt(positions)
            elif token_broadcast == "loglog_decay":
                decay = 1.0 / torch.log(torch.log(positions + 5))
            else:
                raise ValueError(
                    f"Invalid token_broadcast: {token_broadcast}. Must be 'uniform', 'log_decay', 'poly_decay', or 'loglog_decay'"
                )
            weighted_advantages = advantages.unsqueeze(1) * decay

        per_token_loss1 = coef_1 * weighted_advantages
        per_token_loss2 = coef_2 * weighted_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = reduce_policy_loss(per_token_loss, completion_mask, self.loss_type, self.max_completion_length)
        valid_tokens = completion_mask.sum().clamp(min=1)

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / valid_tokens
        high_clip = (is_high_clipped * completion_mask).sum() / valid_tokens
        clip_ratio = (is_region_clipped * completion_mask).sum() / valid_tokens

        # One collective per micro-step, with rows preserving rank boundaries.
        # Every rank uses the same columns because beta is shared configuration.
        local_metrics = [low_clip, high_clip, clip_ratio]
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / valid_tokens
            local_metrics.append(mean_kl.detach())
        # Move this tiny rank-by-metric matrix once; individual scalar/boolean
        # reductions below must not each synchronize the policy GPU.
        gathered_metrics = self.accelerator.gather(torch.stack(local_metrics).unsqueeze(0)).detach().cpu()
        gathered_low_clip, gathered_high_clip, gathered_clip_ratio = gathered_metrics[:, :3].unbind(dim=1)
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(gathered_metrics[:, 3].nanmean().item())
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        # Trainer.evaluate returns this same mapping and passes it to callbacks.
        # Update it so reward metrics also reach eval_results.json and model selection.
        logs.update(metrics)
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self._textual_logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                    "advantage": self._textual_logs["advantages"],
                }
                df = pd.DataFrame(table)
                wandb.log({f"completions_step_{self.state.global_step}": wandb.Table(dataframe=df)})

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
