"""Frozen prompt plans and independent microbatch ordering for paired training."""

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_r1.training_schedule import (
    TrainingSchedule, append_schedule_trace, permute_rollout_payload, prompt_digest,
)


def schedule_fixture(tmp_path, **overrides):
    payload = dict(version=1, steps=2, world_size=2, num_generations=2, generation_batch_size=8,
                   gradient_accumulation_steps=2, per_device_train_batch_size=2, seed=42,
                   prompt_ids=[f"sample-{index}" for index in range(8)])
    payload.update(overrides)
    args = SimpleNamespace(max_steps=2, num_generations=2, generation_batch_size=8,
                           gradient_accumulation_steps=2, per_device_train_batch_size=2, seed=42,
                           num_iterations=1, steps_per_generation=2, shuffle_dataset=False,
                           remove_unused_columns=False, resume_from_checkpoint=None)
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, args, payload


def load_schedule(tmp_path):
    path, args, _ = schedule_fixture(tmp_path)
    return TrainingSchedule.load(path, args, world_size=2)


def input_rows(schedule, step=0, rank=0):
    return [{"comparison_sample_id": value, "prompt": [{"role": "user", "content": f"问题 {value}"}]}
            for value in schedule.expected_sample_ids(step, rank)]


def test_schedule_expected_rank_rows_match_real_repeat_sampler_and_accelerate_shards(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from accelerate.data_loader import BatchSamplerShard
    from open_r1.grpo_trainer import RepeatSampler

    schedule = load_schedule(tmp_path)
    sampler = RepeatSampler(schedule.prompt_ids, mini_repeat_count=schedule.num_generations,
                            batch_size=schedule.generation_batch_size // schedule.num_generations,
                            repeat_count=schedule.gradient_accumulation_steps, shuffle=False, seed=schedule.seed)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=schedule.local_batch_size, drop_last=False)
    for rank in range(schedule.world_size):
        shard = BatchSamplerShard(batch_sampler, num_processes=schedule.world_size, process_index=rank,
                                  split_batches=False)
        batches = list(shard)
        assert len(batches) == schedule.steps * schedule.gradient_accumulation_steps
        for micro_step, batch in enumerate(batches):
            step = micro_step // schedule.gradient_accumulation_steps
            actual = [schedule.prompt_ids[index] for index in batch]
            assert actual == schedule.expected_sample_ids(step, rank)


def test_group_spanning_two_ranks_uses_global_generation_index(tmp_path):
    path, args, _ = schedule_fixture(tmp_path, num_generations=4, generation_batch_size=12,
                                   gradient_accumulation_steps=3, prompt_ids=[f"id-{i}" for i in range(6)])
    args.num_generations, args.generation_batch_size = 4, 12
    args.gradient_accumulation_steps = args.steps_per_generation = 3
    schedule = TrainingSchedule.load(path, args, world_size=2)
    assert schedule.expected_sample_ids(0, 0) == ["id-0"] * 4 + ["id-1"] * 2
    assert schedule.expected_sample_ids(0, 1) == ["id-1"] * 2 + ["id-2"] * 4
    assert schedule.expected_sample_ids(1, 0) == ["id-3"] * 4 + ["id-4"] * 2


def test_permutations_do_not_read_or_change_global_torch_rng(tmp_path):
    torch = pytest.importorskip("torch")
    schedule = load_schedule(tmp_path)
    torch.manual_seed(78)
    before = torch.random.get_rng_state().clone()
    first = schedule.planned_permutation(1, 1)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(2001)
    before = torch.random.get_rng_state().clone()
    assert first == schedule.planned_permutation(1, 1)
    assert torch.equal(before, torch.random.get_rng_state())
    generator = torch.Generator().manual_seed(42 + 1 * 2 + 1)
    assert first == torch.randperm(4, generator=generator).tolist()


def test_trace_snapshots_unmutated_unicode_prompt_with_canonical_json_digest(tmp_path):
    pytest.importorskip("torch")
    schedule = load_schedule(tmp_path)
    rows = input_rows(schedule)
    original = copy.deepcopy(rows)
    trace = schedule.prepare_trace(rows, step=0, micro_step=0, rank=0)
    rows[0]["prompt"].append({"role": "assistant", "content": "later mutation"})
    assert trace["sample_ids"] == schedule.expected_sample_ids(0, 0)
    assert trace["processed_prompt_sha256"] == [
        hashlib.sha256(json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        for row in original
    ]
    assert trace["permutation"] == schedule.planned_permutation(0, 0)
    assert trace["permutation_seed"] == 42
    assert prompt_digest("你好") == hashlib.sha256(json.dumps("你好", sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@pytest.mark.parametrize("failure", ["wrong_order", "missing_id", "missing_row", "extra_row", "resumed", "outside_step"])
def test_invalid_rollout_is_rejected_before_generation(tmp_path, failure):
    schedule = load_schedule(tmp_path)
    rows = input_rows(schedule)
    step = micro_step = 0
    if failure == "wrong_order":
        rows[0], rows[-1] = rows[-1], rows[0]
    elif failure == "missing_id":
        rows[0].pop("comparison_sample_id")
    elif failure == "missing_row":
        rows.pop()
    elif failure == "extra_row":
        rows.append(rows[-1])
    elif failure == "resumed":
        step = 1
        rows = input_rows(schedule, step=1)
    elif failure == "outside_step":
        step = 2
    with pytest.raises(ValueError, match="Training schedule"):
        schedule.prepare_trace(rows, step=step, micro_step=micro_step, rank=0)


def test_entire_rollout_payload_is_permuted_together_including_reference_and_lengths(tmp_path):
    torch = pytest.importorskip("torch")
    schedule = load_schedule(tmp_path)
    indices = schedule.planned_permutation(0, 0)
    values = torch.arange(4, dtype=torch.float32, requires_grad=True)
    payload = {
        "prompt_ids": torch.arange(4).unsqueeze(1),
        "prompt_mask": torch.ones(4, 1, dtype=torch.long),
        "completion_ids": torch.arange(4).unsqueeze(1) + 10,
        "completion_mask": torch.ones(4, 1, dtype=torch.long),
        "advantages": values,
        "old_per_token_logps": None,
        "ref_per_token_logps": values.detach().unsqueeze(1) + 20,
        "_prompt_lengths": torch.arange(4) + 1,
        "_completion_lengths": torch.arange(4) + 2,
    }
    shuffled = permute_rollout_payload(payload, indices)
    assert set(shuffled) == set(payload)
    for key, value in payload.items():
        if value is None:
            assert shuffled[key] is None
        else:
            torch.testing.assert_close(shuffled[key], value[indices])
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    (shuffled["advantages"] * weights).sum().backward()
    expected_gradient = torch.empty(4)
    expected_gradient[indices] = weights
    torch.testing.assert_close(values.grad, expected_gradient)


@pytest.mark.parametrize("payload,permutation", [
    ({"row": [1, 2]}, [0, 1]), ({"row": None}, [1, 1]),
])
def test_invalid_payload_or_permutation_is_rejected(payload, permutation):
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        permute_rollout_payload(payload, permutation)


def test_short_tensor_field_is_rejected():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="ref_per_token_logps"):
        permute_rollout_payload({"ref_per_token_logps": torch.ones(3, 4)}, [2, 0, 1, 3])


def test_trace_is_one_complete_json_line_per_rank_and_optimizer_step(tmp_path):
    pytest.importorskip("torch")
    schedule = load_schedule(tmp_path)
    for step in range(schedule.steps):
        for rank in range(schedule.world_size):
            trace = schedule.prepare_trace(input_rows(schedule, step, rank), step,
                                           step * schedule.gradient_accumulation_steps, rank)
            append_schedule_trace(tmp_path, trace)
    for rank in range(schedule.world_size):
        traces = [json.loads(line) for line in (tmp_path / "data_order" / f"rank_{rank}.jsonl").read_text().splitlines()]
        assert [trace["step"] for trace in traces] == [0, 1]
        assert [trace["micro_step"] for trace in traces] == [0, 2]
        assert all(trace["rank"] == rank for trace in traces)


@pytest.mark.parametrize("field,value", [
    ("max_steps", 3), ("num_generations", 4), ("generation_batch_size", 16),
    ("gradient_accumulation_steps", 4), ("per_device_train_batch_size", 1), ("seed", 99),
    ("num_iterations", 2), ("steps_per_generation", 1), ("shuffle_dataset", True),
    ("remove_unused_columns", True), ("resume_from_checkpoint", "checkpoint-1"),
])
def test_runtime_must_exactly_match_frozen_schedule(tmp_path, field, value):
    path, args, _ = schedule_fixture(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="[Tt]raining schedule"):
        TrainingSchedule.load(path, args, world_size=2)


def test_world_size_is_a_runtime_constraint(tmp_path):
    path, args, _ = schedule_fixture(tmp_path)
    with pytest.raises(ValueError, match="world_size"):
        TrainingSchedule.load(path, args, world_size=1)


@pytest.mark.parametrize("overrides", [
    {"version": 2}, {"version": True}, {"steps": 0}, {"seed": -1}, {"world_size": 2.0},
    {"prompt_ids": ["duplicate"] * 8}, {"prompt_ids": [str(i) for i in range(7)]},
    {"prompt_ids": list(range(8))}, {"prompt_ids": None}, {"prompt_ids": [""] * 8},
])
def test_malformed_schedule_rejected(tmp_path, overrides):
    path, args, _ = schedule_fixture(tmp_path, **overrides)
    with pytest.raises(ValueError):
        TrainingSchedule.load(path, args, world_size=2)


def test_schedule_ignores_extra_dataset_digest_field(tmp_path):
    path, args, _ = schedule_fixture(tmp_path, dataset_sha256="root-verifies-this-separately")
    assert TrainingSchedule.load(path, args, world_size=2).steps == 2


def trainer_schedule_method(name, **namespace):
    source = Path(__file__).resolve().parents[1] / "src" / "open_r1" / "grpo_trainer.py"
    trainer = next(node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("local_error", [False, True])
def test_all_ranks_fail_when_any_rank_rejects_the_input_order(tmp_path, local_error):
    schedule = load_schedule(tmp_path)
    rows = input_rows(schedule)
    if local_error:
        rows[0]["comparison_sample_id"] = "wrong"
    gathered = []

    def gather(values):
        gathered.extend(values)
        return [values[0], "rank 1: ValueError: Training schedule sample ID mismatch"]

    trainer = SimpleNamespace(training_schedule=schedule, state=SimpleNamespace(global_step=0), _step=0,
                              accelerator=SimpleNamespace(process_index=0), _gather_python_objects=gather)
    with pytest.raises(ValueError, match="rank 1"):
        trainer_schedule_method("_prepare_training_schedule_trace")(trainer, rows)
    assert len(gathered) == 1
    assert (gathered[0] is not None) == local_error


def test_trace_write_error_is_shared_before_policy_training(tmp_path):
    torch = pytest.importorskip("torch")
    errors = []

    def fail_write(*args):
        raise OSError("disk full")

    def gather(values):
        errors.extend(values)
        return values

    trainer = SimpleNamespace(args=SimpleNamespace(output_dir=str(tmp_path)),
                              accelerator=SimpleNamespace(process_index=0), _gather_python_objects=gather)
    apply = trainer_schedule_method("_apply_training_schedule_trace", permute_rollout_payload=permute_rollout_payload,
                                    append_schedule_trace=fail_write)
    with pytest.raises(ValueError, match="disk full"):
        apply(trainer, {"advantages": torch.arange(4)}, {"permutation": [1, 0, 2, 3]})
    assert len(errors) == 1 and "rank 0" in errors[0]


def test_real_tiny_cpu_training_consumes_plan_and_eval_writes_no_extra_trace(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from open_r1.configs import GRPOConfig
    from open_r1.grpo_trainer import GRPOTrainer

    torch.set_num_threads(2)
    vocabulary = {value: index for index, value in enumerate(["[PAD]", "[EOS]", "[UNK]", "a", "b", "c", "d"])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]",
                                       unk_token="[UNK]", padding_side="left")
    config = Qwen2Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
                        pad_token_id=0, eos_token_id=1, bos_token_id=1)
    path, _, _ = schedule_fixture(tmp_path, world_size=1, generation_batch_size=4,
                                  prompt_ids=["sample-0", "sample-1", "sample-2", "sample-3"])
    args = GRPOConfig(
        output_dir=str(tmp_path / "run"), use_cpu=True, bf16=False, fp16=False, use_vllm=False,
        report_to=[], max_steps=2, per_device_train_batch_size=2, gradient_accumulation_steps=2,
        per_device_eval_batch_size=2, num_generations=2, max_prompt_length=4, max_completion_length=3,
        beta=0.0, save_strategy="no", logging_steps=1, disable_tqdm=True, gradient_checkpointing=False,
        shuffle_dataset=False, remove_unused_columns=False, training_schedule_path=str(path), seed=42,
    )
    dataset = Dataset.from_dict({"prompt": ["a", "b", "c", "d"],
                                 "comparison_sample_id": ["sample-0", "sample-1", "sample-2", "sample-3"]})

    def reward(completions, **kwargs):
        return [float(index % 2) for index in range(len(completions))]

    trainer = GRPOTrainer(model=Qwen2ForCausalLM(config), processing_class=tokenizer, args=args,
                          reward_funcs=reward, train_dataset=dataset, eval_dataset=dataset.select(range(2)))
    consumed_prompt_ids = []
    original_compute_loss = trainer._compute_loss

    def record_consumed_rows(model, inputs):
        consumed_prompt_ids.extend(inputs["prompt_ids"].detach().cpu().flatten().tolist())
        return original_compute_loss(model, inputs)

    trainer._compute_loss = record_consumed_rows
    trainer.train()
    trace_path = tmp_path / "run" / "data_order" / "rank_0.jsonl"
    traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(traces) == 2
    for step, trace in enumerate(traces):
        expected_ids = trainer.training_schedule.expected_sample_ids(step, 0)
        assert trace["sample_ids"] == expected_ids
        assert trace["permutation"] == trainer.training_schedule.planned_permutation(step, 0)
        expected_prompts = [dataset[int(sample_id.rsplit("-", 1)[1])]["prompt"] for sample_id in expected_ids]
        assert trace["processed_prompt_sha256"] == [prompt_digest(prompt) for prompt in expected_prompts]
        expected_tokens = [vocabulary[expected_prompts[index]] for index in trace["permutation"]]
        assert consumed_prompt_ids[step * 4 : (step + 1) * 4] == expected_tokens
        assert trace["micro_step"] == step * 2
    before_eval = trace_path.read_bytes()
    trainer.evaluate()
    assert trace_path.read_bytes() == before_eval
