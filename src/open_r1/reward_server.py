"""Single-GPU HTTP service for sequence-classification reward models."""

import argparse
import asyncio
import logging
import math
import time
from typing import Annotated, Literal, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator


logger = logging.getLogger("uvicorn.error")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[Annotated[list[ChatMessage], Field(min_length=1)]] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def completions_are_present(self):
        for conversation in self.messages:
            if conversation[-1].role != "assistant":
                raise ValueError("Each conversation must end with an assistant completion")
        return self


class RewardScorer(Protocol):
    def score(self, messages: list[list[dict[str, str]]]) -> list[float]: ...


def create_app(scorer: RewardScorer, run_id: str | None = None) -> FastAPI:
    """Create an API that serializes all GPU work through one model instance."""

    app = FastAPI()
    inference_lock = asyncio.Lock()

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/health/{requested_run_id}/")
    async def owned_health(requested_run_id: str):
        if run_id is None or requested_run_id != run_id:
            raise HTTPException(status_code=404, detail="Unknown run")
        return {"status": "ok"}

    @app.post("/score/")
    async def score(request: ScoreRequest):
        messages = [[message.model_dump() for message in item] for item in request.messages]
        request_started = time.perf_counter()
        async with inference_lock:
            inference_started = time.perf_counter()
            rewards = await asyncio.to_thread(scorer.score, messages)
            inference_seconds = time.perf_counter() - inference_started
            stats = dict(getattr(scorer, "last_stats", {}))
        logger.info(
            "qrm_score examples=%d queue_s=%.4f inference_s=%.4f batches=%s input_tokens=%s "
            "padded_tokens=%s max_batch_examples=%s max_batch_padded_tokens=%s",
            len(messages),
            inference_started - request_started,
            inference_seconds,
            stats.get("batches", "unknown"),
            stats.get("input_tokens", "unknown"),
            stats.get("padded_tokens", "unknown"),
            stats.get("max_batch_examples", "unknown"),
            stats.get("max_batch_padded_tokens", "unknown"),
        )
        if len(rewards) != len(messages) or not all(math.isfinite(float(value)) for value in rewards):
            raise RuntimeError("Reward model returned an invalid result")
        return {"rewards": [float(value) for value in rewards]}

    return app


def build_length_aware_batches(
    lengths: list[int], max_batch_size: int, max_batch_tokens: int
) -> list[list[int]]:
    """Group similarly sized inputs while bounding padded tokens per batch."""

    if max_batch_size < 1 or max_batch_tokens < 1:
        raise ValueError("QRM batch limits must be positive")
    oversized = [length for length in lengths if length > max_batch_tokens]
    if oversized:
        raise ValueError(
            f"QRM input length {max(oversized)} exceeds the padded-token budget {max_batch_tokens}"
        )
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in sorted(range(len(lengths)), key=lengths.__getitem__):
        candidate_max = max(current_max, lengths[index])
        candidate_size = len(current) + 1
        if current and (candidate_size > max_batch_size or candidate_size * candidate_max > max_batch_tokens):
            batches.append(current)
            current = []
            current_max = 0
        current.append(index)
        current_max = max(current_max, lengths[index])
    if current:
        batches.append(current)
    return batches


class TransformersRewardScorer:
    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        batch_size: int,
        max_length: int,
        max_batch_tokens: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_batch_tokens = max_batch_tokens or batch_size * max_length
        self.last_stats: dict[str, int] = {}

    def score(self, messages: list[list[dict[str, str]]]) -> list[float]:
        import torch
        from trl.data_utils import apply_chat_template

        texts = [apply_chat_template({"messages": item}, self.tokenizer)["text"] for item in messages]
        encoded = self.tokenizer(
            text=texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
        batches = build_length_aware_batches(lengths, self.batch_size, self.max_batch_tokens)
        rewards = [0.0] * len(texts)
        padded_tokens = 0
        max_batch_examples = 0
        max_batch_padded_tokens = 0
        for indices in batches:
            features = [
                {name: values[index] for name, values in encoded.items()}
                for index in indices
            ]
            inputs = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            batch_padded_tokens = len(indices) * inputs["input_ids"].shape[1]
            padded_tokens += batch_padded_tokens
            max_batch_examples = max(max_batch_examples, len(indices))
            max_batch_padded_tokens = max(max_batch_padded_tokens, batch_padded_tokens)
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits[:, 0].float().cpu().tolist()
            for index, value in zip(indices, logits):
                rewards[index] = value
        self.last_stats = {
            "examples": len(texts),
            "batches": len(batches),
            "input_tokens": sum(lengths),
            "padded_tokens": padded_tokens,
            "max_batch_examples": max_batch_examples,
            "max_batch_padded_tokens": max_batch_padded_tokens,
        }
        return rewards


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=0,
        help="Maximum padded tokens per inference batch; 0 uses batch-size times max-length.",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--log-level", choices=("critical", "error", "warning", "info", "debug", "trace"), default="info"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_batch_tokens < 0:
        parser.error("--max-batch-tokens must be non-negative")
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    if args.max_batch_tokens and args.max_batch_tokens < args.max_length:
        parser.error("--max-batch-tokens must be zero or at least --max-length")
    return args


def main() -> None:
    args = parse_args()

    import torch
    import uvicorn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Preserve the completion and the most recent context if an example must
    # be shortened to the explicitly configured reward-model context window.
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        num_labels=1,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(args.device)
    model.eval()
    scorer = TransformersRewardScorer(
        model,
        tokenizer,
        args.device,
        args.batch_size,
        args.max_length,
        args.max_batch_tokens or None,
    )
    # Fail before advertising readiness if remote code, the chat template,
    # FlashAttention, or final-token scoring is incompatible.
    scorer.score([[{"role": "user", "content": "Warmup"}, {"role": "assistant", "content": "OK"}]])
    uvicorn.run(create_app(scorer, run_id=args.run_id), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
