"""Optional immutable prompt plans and RNG-independent rollout order auditing."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def prompt_digest(prompt: Any) -> str:
    """Match evaluation.digest, including its default JSON separators."""
    return hashlib.sha256(json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class TrainingSchedule:
    version: int
    steps: int
    world_size: int
    num_generations: int
    generation_batch_size: int
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    seed: int
    prompt_ids: tuple[str, ...]

    @classmethod
    def load(cls, path, args, world_size: int):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Training schedule must be a JSON object")
        names = (
            "version", "steps", "world_size", "num_generations", "generation_batch_size",
            "gradient_accumulation_steps", "per_device_train_batch_size", "seed",
        )
        for name in names:
            value = payload.get(name)
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"Training schedule {name} must be an integer >= {minimum}")
        if payload["version"] != 1:
            raise ValueError("Only training schedule version 1 is supported")
        prompt_ids = payload.get("prompt_ids")
        if not isinstance(prompt_ids, list) or any(not isinstance(value, str) or not value for value in prompt_ids):
            raise ValueError("Training schedule prompt_ids must be a list of nonempty strings")
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("Training schedule prompt_ids must be unique")
        schedule = cls(**{name: payload[name] for name in names}, prompt_ids=tuple(prompt_ids))
        schedule.validate_config(args, world_size)
        return schedule

    @property
    def local_batch_size(self) -> int:
        return self.generation_batch_size // self.world_size

    def validate_config(self, args, world_size: int) -> None:
        if self.world_size != world_size:
            raise ValueError(f"Training schedule world_size={self.world_size} does not match runtime {world_size}")
        checks = {
            "max_steps": self.steps,
            "num_generations": self.num_generations,
            "generation_batch_size": self.generation_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "seed": self.seed,
            "num_iterations": 1,
            "steps_per_generation": self.gradient_accumulation_steps,
            "shuffle_dataset": False,
            "remove_unused_columns": False,
        }
        for name, expected in checks.items():
            actual = getattr(args, name, None)
            if actual != expected:
                raise ValueError(f"Training schedule requires {name}={expected!r}; got {actual!r}")
        expected_global_batch = self.world_size * self.per_device_train_batch_size * self.gradient_accumulation_steps
        if self.generation_batch_size != expected_global_batch:
            raise ValueError("Training schedule generation_batch_size must equal world_size * microbatch * accumulation")
        if self.generation_batch_size % self.num_generations:
            raise ValueError("Training schedule generation_batch_size must be divisible by num_generations")
        expected_prompts = self.steps * self.generation_batch_size // self.num_generations
        if len(self.prompt_ids) != expected_prompts:
            raise ValueError(f"Training schedule requires {expected_prompts} prompt_ids; got {len(self.prompt_ids)}")
        if getattr(args, "resume_from_checkpoint", None):
            raise ValueError("Training schedules require a fresh run without resume_from_checkpoint")
        if self.seed + self.steps * self.world_size > 2**64 - 1:
            raise ValueError("Training schedule permutation seeds exceed torch.Generator's uint64 range")

    def _validate_position(self, step: int, rank: int) -> None:
        if not 0 <= step < self.steps:
            raise ValueError(f"Training schedule step {step} is outside [0, {self.steps})")
        if not 0 <= rank < self.world_size:
            raise ValueError(f"Training schedule rank {rank} is outside [0, {self.world_size})")

    def expected_sample_ids(self, step: int, rank: int) -> list[str]:
        self._validate_position(step, rank)
        offset = step * self.generation_batch_size + rank * self.local_batch_size
        return [self.prompt_ids[(offset + index) // self.num_generations] for index in range(self.local_batch_size)]

    def permutation_seed(self, step: int, rank: int) -> int:
        self._validate_position(step, rank)
        return self.seed + step * self.world_size + rank

    def planned_permutation(self, step: int, rank: int) -> list[int]:
        import torch

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.permutation_seed(step, rank))
        return torch.randperm(self.local_batch_size, generator=generator).tolist()

    def prepare_trace(self, inputs, step: int, micro_step: int, rank: int) -> dict:
        """Validate original local rows and snapshot prompts before generation mutates them."""
        expected = self.expected_sample_ids(step, rank)
        if micro_step != step * self.gradient_accumulation_steps:
            raise ValueError(
                f"Training schedule expects micro_step={step * self.gradient_accumulation_steps}; got {micro_step}; "
                "resumed or misaligned runs are unsupported"
            )
        actual = [row.get("comparison_sample_id") for row in inputs]
        if actual != expected:
            mismatch = next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]), None)
            detail = (f" at local row {mismatch}: got {actual[mismatch]!r}, expected {expected[mismatch]!r}"
                      if mismatch is not None else f": got {len(actual)} rows, expected {len(expected)}")
            raise ValueError(f"Training schedule sample ID mismatch for step {step}, rank {rank}{detail}")
        return {
            "version": self.version,
            "step": step,
            "micro_step": micro_step,
            "rank": rank,
            "sample_ids": actual,
            "processed_prompt_sha256": [prompt_digest(row["prompt"]) for row in inputs],
            "permutation_seed": self.permutation_seed(step, rank),
            "permutation": self.planned_permutation(step, rank),
        }


def permute_rollout_payload(payload: Mapping, permutation: list[int]) -> dict:
    """Apply one recorded row order to every tensor, preserving optional None fields."""
    import torch

    count = len(permutation)
    if sorted(permutation) != list(range(count)):
        raise ValueError("Rollout permutation must contain each local row exactly once")
    indices = torch.tensor(permutation, dtype=torch.long, device="cpu")
    result = {}
    for name, value in payload.items():
        if value is None:
            result[name] = None
        elif not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != count:
            raise ValueError(f"Rollout field {name!r} does not have {count} tensor rows")
        else:
            # Match legacy indexing, including CPU length vectors alongside CUDA payloads.
            result[name] = value[indices]
    return result


def append_schedule_trace(output_dir, trace: dict) -> None:
    directory = Path(output_dir) / "data_order"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank_{trace['rank']}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
