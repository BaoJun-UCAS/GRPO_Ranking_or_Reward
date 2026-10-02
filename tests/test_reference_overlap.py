"""Reference-cache equivalence and reward/reference scheduling without any GPU."""

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import warnings

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/open_r1/grpo_trainer.py"


def overlap_helper():
    function = next(node for node in ast.parse(SOURCE.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_reward_reference_overlap")
    namespace = {"ThreadPoolExecutor": ThreadPoolExecutor, "time": time}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name]


def test_http_runs_in_worker_while_reference_stays_on_calling_thread():
    calling_thread = threading.get_ident()
    reward_started = threading.Event()
    reference_started = threading.Event()

    def reward():
        assert threading.get_ident() != calling_thread
        reward_started.set()
        assert reference_started.wait(timeout=2), "Reference work never overlapped the request"
        return [0.25, 0.75]

    sentinel = object()

    def reference():
        assert threading.get_ident() == calling_thread
        assert reward_started.wait(timeout=2), "HTTP work was not submitted before reference work"
        reference_started.set()
        return sentinel

    rewards, cached, timings = overlap_helper()(reward, reference)
    assert rewards == [0.25, 0.75]
    assert cached is sentinel
    assert timings["qrm_service"] >= 0
    assert timings["reference_precompute"] >= 0


def test_http_failure_is_propagated_after_reference_work():
    failure = RuntimeError("HTTP fixture failed")
    reference_finished = threading.Event()

    def reward():
        raise failure

    def reference():
        reference_finished.set()
        return object()

    with pytest.raises(RuntimeError) as caught:
        overlap_helper()(reward, reference)
    assert caught.value is failure
    assert reference_finished.is_set()


def test_reference_failure_joins_the_http_worker_before_returning():
    failure = ValueError("Reference fixture failed")
    reward_started = threading.Event()
    reference_failed = threading.Event()
    reward_finished = threading.Event()

    def reward():
        reward_started.set()
        assert reference_failed.wait(timeout=2)
        # Give the caller a chance to exit incorrectly before this worker does.
        time.sleep(0.02)
        reward_finished.set()
        return [0.5]

    def reference():
        assert reward_started.wait(timeout=2)
        reference_failed.set()
        raise failure

    with pytest.raises(ValueError) as caught:
        overlap_helper()(reward, reference)
    assert caught.value is failure
    assert reward_finished.is_set()


@pytest.fixture
def tiny_trainer(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from datasets import Dataset
    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from open_r1.configs import GRPOConfig
    from open_r1.grpo_trainer import GRPOTrainer

    torch.manual_seed(52)
    torch.set_num_threads(2)
    vocabulary = {word: i for i, word in enumerate(["[PAD]", "[EOS]", "[UNK]", "a", "b", "c", "d", "e"])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]",
                                       unk_token="[UNK]", padding_side="left")
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=1, bos_token_id=1, attention_dropout=0.0,
    ))
    args = GRPOConfig(
        output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False, use_vllm=False,
        report_to=[], max_steps=1, per_device_train_batch_size=1, gradient_accumulation_steps=2,
        num_generations=2, max_prompt_length=8, max_completion_length=4, beta=0.04,
        loss_type="bnpo", save_strategy="no", disable_tqdm=True, gradient_checkpointing=False,
        trim_unused_padding=True, disable_dropout=True,
    )

    def qrm_server(completions, **kwargs):
        return [0.5] * len(completions)

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, args=args, reward_funcs=qrm_server,
        train_dataset=Dataset.from_dict({"prompt": ["a", "b"]}),
        peft_config=LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                               target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    # Nonzero adapters make accidentally cached policy scores distinguishable
    # from the frozen base model's reference scores.
    with torch.no_grad():
        for name, value in trainer.model.named_parameters():
            if "lora_B" in name:
                value.normal_(mean=0.0, std=0.2)
    trainer.model.train()
    return trainer


def rollout_tensors(torch):
    return {
        "prompt_ids": torch.tensor([[0, 0, 3, 4], [0, 5, 6, 7], [0, 0, 0, 3]]),
        "prompt_mask": torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]]),
        "completion_ids": torch.tensor([[5, 1, 0, 0], [4, 6, 7, 1], [1, 0, 0, 0]]),
        "completion_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]]),
        "_prompt_lengths": torch.tensor([2, 3, 1]),
        "_completion_lengths": torch.tensor([2, 4, 1]),
        "advantages": torch.tensor([0.75, -0.25, -0.5]),
        "old_per_token_logps": None,
    }


def precompute(trainer, tensors):
    return trainer._precompute_reference_logps(
        tensors["prompt_ids"], tensors["prompt_mask"], tensors["completion_ids"], tensors["completion_mask"],
        tensors["_prompt_lengths"], tensors["_completion_lengths"],
    )


@pytest.mark.parametrize("mask_last_response", [False, True])
def test_reference_cache_preserves_lora_loss_gradients_and_trimmed_token_alignment(
    tiny_trainer, monkeypatch, mask_last_response,
):
    import torch
    from open_r1.grpo_trainer import trim_padded_microbatch

    trainer = tiny_trainer
    tensors = rollout_tensors(torch)
    if mask_last_response:
        tensors["completion_mask"][-1].zero_()
    cached = precompute(trainer, tensors)
    assert cached.shape == (3, 4)
    assert not cached.requires_grad
    assert cached.grad_fn is None
    assert torch.isfinite(cached).all()
    assert trainer.model.training
    assert not any(module.disable_adapters for module in trainer.model.modules()
                   if hasattr(module, "disable_adapters") and isinstance(module.disable_adapters, bool))
    original_logps = trainer._get_per_token_logps
    calls = []

    def counted(*args, **kwargs):
        calls.append(torch.is_grad_enabled())
        return original_logps(*args, **kwargs)

    monkeypatch.setattr(trainer, "_get_per_token_logps", counted)
    for row in range(3):
        original = {key: value[row:row + 1] if value is not None else None for key, value in tensors.items()}
        cached_input = {**original, "ref_per_token_logps": cached[row:row + 1]}
        uncached_input = trim_padded_microbatch(original)
        cached_input = trim_padded_microbatch(cached_input)
        assert cached_input["ref_per_token_logps"].shape == cached_input["completion_ids"].shape
        ids = torch.cat((uncached_input["prompt_ids"], uncached_input["completion_ids"]), dim=1)
        mask = torch.cat((uncached_input["prompt_mask"], uncached_input["completion_mask"]), dim=1)
        with torch.no_grad(), trainer.model.disable_adapter():
            expected_reference = original_logps(trainer.model, ids, mask, uncached_input["completion_ids"].shape[1])
        active = uncached_input["completion_mask"].bool()
        torch.testing.assert_close(cached_input["ref_per_token_logps"][active], expected_reference[active],
                                   atol=1e-7, rtol=1e-6)
        trainer.model.zero_grad(set_to_none=True)
        calls.clear()
        uncached_loss = trainer._compute_loss(trainer.model, uncached_input)
        assert calls == [True, False]
        uncached_loss.backward()
        expected_gradients = {name: value.grad.detach().clone() for name, value in trainer.model.named_parameters()
                              if value.requires_grad and value.grad is not None}
        trainer.model.zero_grad(set_to_none=True)
        calls.clear()
        cached_loss = trainer._compute_loss(trainer.model, cached_input)
        assert calls == [True], "Cached loss unexpectedly recomputed the reference model"
        cached_loss.backward()
        torch.testing.assert_close(cached_loss.detach(), uncached_loss.detach(), atol=1e-7, rtol=1e-6)
        for name, value in trainer.model.named_parameters():
            if name in expected_gradients:
                torch.testing.assert_close(value.grad, expected_gradients[name], atol=1e-7, rtol=1e-6)


def test_liger_path_forwards_cached_reference_without_recomputing(tiny_trainer, monkeypatch):
    import torch
    from open_r1.grpo_trainer import trim_padded_microbatch

    trainer = tiny_trainer
    tensors = rollout_tensors(torch)
    cached = precompute(trainer, tensors)
    inputs = {key: value[:1] if value is not None else None for key, value in tensors.items()}
    inputs["ref_per_token_logps"] = cached[:1]
    inputs = trim_padded_microbatch(inputs)
    seen = {}

    def liger(**kwargs):
        seen.update(kwargs)
        return torch.tensor(0.125), (torch.tensor(0.0), torch.tensor(0.0))

    monkeypatch.setattr(trainer, "liger_grpo_loss", liger, raising=False)
    monkeypatch.setattr(trainer, "_get_last_hidden_state", lambda *args, **kwargs: torch.zeros(1, 2, 16))
    monkeypatch.setattr(trainer, "_get_per_token_logps", lambda *args, **kwargs: pytest.fail("Reference recomputed"))
    assert trainer.compute_liger_loss(trainer.model, inputs).item() == 0.125
    assert seen["ref_per_token_logps"] is inputs["ref_per_token_logps"]



def reference_overlap_guard():
    trainer = next(node for node in ast.parse(SOURCE.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    method = next(node for node in trainer.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_can_overlap_qrm_reference")
    namespace = {"warnings": warnings, "is_peft_model": lambda model: getattr(model, "fixture_peft", False)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[method.name]


def guard_fixture():
    return SimpleNamespace(
        args=SimpleNamespace(overlap_qrm_reference=True, disable_dropout=True, sync_ref_model=False,
                             per_device_train_batch_size=1, gradient_accumulation_steps=128, steps_per_generation=128),
        beta=0.04, reward_func_names=["qrm_server"], trim_unused_padding=True, use_liger_loss=False,
        is_fsdp_enabled=False, is_deepspeed_enabled=True, num_iterations=1,
        model=SimpleNamespace(config=SimpleNamespace(model_type="qwen2", attention_dropout=0.0)), ref_model=None,
        accelerator=SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=2)),
                                    num_processes=2, is_main_process=True, unwrap_model=lambda model: model),
    )


def test_reference_overlap_guard_accepts_supported_configuration():
    assert reference_overlap_guard()(guard_fixture(), "train") is True


@pytest.mark.parametrize("attribute,value", [
    ("beta", 0.0), ("reward_func_names", ["qrm_server", "other_reward"]),
    ("args.disable_dropout", False), ("args.sync_ref_model", True),
    ("args.per_device_train_batch_size", 2), ("trim_unused_padding", False),
    ("use_liger_loss", True), ("is_fsdp_enabled", True),
    ("accelerator.state.deepspeed_plugin.zero_stage", 3),
    ("accelerator.state.deepspeed_plugin.zero_stage", 1),
    ("accelerator.state.deepspeed_plugin", None), ("accelerator.num_processes", 1),
    ("num_iterations", 2), ("args.gradient_accumulation_steps", 129),
])
def test_reference_overlap_guard_falls_back_for_unsupported_settings(attribute, value):
    trainer = guard_fixture()
    target = trainer
    pieces = attribute.split(".")
    for name in pieces[:-1]:
        target = getattr(target, name)
    setattr(target, pieces[-1], value)
    with pytest.warns(UserWarning, match="overlap disabled"):
        assert reference_overlap_guard()(trainer, "train") is False


@pytest.mark.parametrize("enabled,mode", [(False, "train"), (True, "eval")])
def test_disabled_or_eval_overlap_needs_no_other_trainer_state(enabled, mode):
    trainer = SimpleNamespace(args=SimpleNamespace(overlap_qrm_reference=enabled))
    assert reference_overlap_guard()(trainer, mode) is False


@pytest.mark.parametrize("main_process", [False, True])
def test_overlap_fallback_warns_only_once_and_only_on_main_process(main_process):
    trainer = guard_fixture()
    trainer.beta = 0.0
    trainer.accelerator.is_main_process = main_process
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert reference_overlap_guard()(trainer, "train") is False
        assert reference_overlap_guard()(trainer, "train") is False
    assert len(recorded) == int(main_process)


@pytest.mark.parametrize("field", ["attention_dropout", "attn_pdrop", "attention_probs_dropout_prob",
                                   "attention_dropout_prob", "attn_dropout", "attention_dropout_rate"])
@pytest.mark.parametrize("value,expected", [(0.0, True), (0.1, False)])
def test_functional_reference_dropout_is_checked_even_when_module_dropout_is_disabled(field, value, expected):
    trainer = guard_fixture()
    setattr(trainer.model.config, field, value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert reference_overlap_guard()(trainer, "train") is expected


@pytest.mark.parametrize("policy_dropout,reference_dropout,expected", [(0.1, 0.0, True), (0.0, 0.1, False)])
def test_explicit_reference_model_configuration_takes_precedence(policy_dropout, reference_dropout, expected):
    trainer = guard_fixture()
    trainer.model.config.attention_dropout = policy_dropout
    trainer.ref_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2", attention_dropout=reference_dropout))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert reference_overlap_guard()(trainer, "train") is expected


def test_peft_guard_reads_base_model_reference_config():
    trainer = guard_fixture()
    base = SimpleNamespace(config=SimpleNamespace(model_type="qwen2", attention_dropout=0.1))
    trainer.model.fixture_peft = True
    trainer.model.get_base_model = lambda: base
    with pytest.warns(UserWarning, match="reference config attention_dropout"):
        assert reference_overlap_guard()(trainer, "train") is False


def test_guard_checks_nested_text_model_attention_dropout():
    trainer = guard_fixture()
    trainer.model.config.get_text_config = lambda: SimpleNamespace(model_type="qwen2", attention_dropout=0.1)
    with pytest.warns(UserWarning, match="reference config attention_dropout"):
        assert reference_overlap_guard()(trainer, "train") is False



def test_reference_failure_restores_real_peft_adapter_and_training_state(tiny_trainer, monkeypatch):
    import torch

    trainer = tiny_trainer
    failure = RuntimeError("Injected reference forward failure")
    was_training = trainer.model.training

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(trainer, "_get_per_token_logps", fail)
    with pytest.raises(RuntimeError) as caught:
        precompute(trainer, rollout_tensors(torch))
    assert caught.value is failure
    assert trainer.model.get_model_status().enabled is True
    assert trainer.model.training == was_training



@pytest.mark.parametrize("model_type,expected", [("qwen2", True), ("qwen3", True),
                                                ("mixtral", False), ("qwen3_moe", False)])
def test_reference_overlap_only_enables_validated_dense_qwen_architectures(model_type, expected):
    trainer = guard_fixture()
    trainer.model.config.model_type = model_type
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert reference_overlap_guard()(trainer, "train") is expected


@pytest.mark.parametrize("rope_scaling,expected", [(None, True), ({}, True),
    ({"rope_type": "default"}, True), ({"type": "default"}, True),
    ({"rope_type": "dynamic", "factor": 2.0}, False), ({"type": "linear", "factor": 2.0}, False)])
def test_reference_overlap_rejects_scaled_or_stateful_rope(rope_scaling, expected):
    trainer = guard_fixture()
    trainer.model.config.rope_scaling = rope_scaling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert reference_overlap_guard()(trainer, "train") is expected
