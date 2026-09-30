import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """Load the policy tokenizer or an explicitly configured compatible tokenizer.

    A custom tokenizer must use the same vocabulary/token IDs as the policy (and
    rollout server). ``chat_template`` overrides rendering without changing IDs.
    """
    tokenizer_path = getattr(training_args, "tokenizer_name_or_path", None) or model_args.model_name_or_path
    tokenizer_revision = getattr(training_args, "tokenizer_revision", None)
    if tokenizer_revision is None and tokenizer_path == model_args.model_name_or_path:
        tokenizer_revision = model_args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        revision=tokenizer_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token; configure one before training")
        # Reusing EOS avoids changing the vocabulary of a pretrained policy.
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model
