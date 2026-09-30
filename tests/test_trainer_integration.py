"""A real tiny CPU training/evaluation loop; skipped without the training stack."""

import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")


@pytest.mark.parametrize("advantage, advantage_kwargs", [
    ("grpo", {}),
    ("robust_pairwise", {"delta": 0.02, "c": 0.2}),
    ("rank_reward", {"rank_weight": 0.3}),
])
def test_tiny_lora_train_evaluate_and_reload(tmp_path, advantage, advantage_kwargs):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from open_r1.configs import GRPOConfig
    from open_r1.grpo_trainer import GRPOTrainer

    torch.manual_seed(42)
    torch.set_num_threads(2)
    vocab = {word: i for i, word in enumerate(["[PAD]", "[EOS]", "[UNK]", "hello", "world", "a", "b", "c"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]", padding_side="left",
    )
    config = Qwen2Config(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=1, bos_token_id=1,
    )
    model = Qwen2ForCausalLM(config)
    args = GRPOConfig(
        output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False, use_vllm=False,
        report_to=[], max_steps=2, per_device_train_batch_size=2, gradient_accumulation_steps=2,
        per_device_eval_batch_size=2, num_generations=2, max_prompt_length=8, max_completion_length=4,
        beta=0.04, learning_rate=1e-3, advantage=advantage, advantage_kwargs=advantage_kwargs,
        loss_type="grpo", save_strategy="no", logging_steps=1, disable_tqdm=True,
        gradient_checkpointing=False, trim_unused_padding=True,
    )
    dataset = Dataset.from_dict({"prompt": ["hello", "hello world", "a", "b"]})

    def reward(completions, **kwargs):
        return [float(i % 2) for i in range(len(completions))]

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, args=args, reward_funcs=reward,
        train_dataset=dataset, eval_dataset=dataset.select(range(2)),
        peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    before = {name: value.detach().clone() for name, value in trainer.model.named_parameters() if value.requires_grad}
    result = trainer.train()
    assert trainer.state.global_step == 2
    assert math.isfinite(result.training_loss)
    assert any(not torch.equal(before[name], value) for name, value in trainer.model.named_parameters() if name in before)
    # Scoring must disable KV cache even if an inference config enables it.
    trainer.model.config.use_cache = True
    metrics = trainer.evaluate()
    assert math.isfinite(metrics["eval_loss"])
    assert math.isfinite(metrics["eval_reward"])
    trainer.save_model(str(tmp_path / "adapter"))
    reloaded = PeftModel.from_pretrained(Qwen2ForCausalLM(config), tmp_path / "adapter")
    assert reloaded.peft_config["default"].r == 2


@pytest.mark.parametrize("advantage, options", [
    ("rank_reward", {"rank_weight": 0.3}),
    ("robust_pairwise", {"delta": 0.02, "c": 0.2}),
    ("grpo", {}),
])
def test_advantage_cli_json_is_parsed(tmp_path, advantage, options):
    import json
    from open_r1.configs import GRPOConfig
    from trl import TrlParser
    (args,) = TrlParser(GRPOConfig).parse_args_into_dataclasses([
        "--output_dir", str(tmp_path), "--use_cpu", "true", "--bf16", "false",
        "--per_device_train_batch_size", "2", "--num_generations", "2",
        "--advantage", advantage, "--advantage_kwargs", json.dumps(options),
        "--report_to", "none",
    ])
    assert args.advantage == advantage
    assert args.advantage_kwargs == options
