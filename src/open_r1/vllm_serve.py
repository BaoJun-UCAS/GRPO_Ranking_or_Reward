"""Single-process vLLM server for the TRL 0.18 client protocol.

TRL 0.18's bundled server imports CUDA-aware libraries in its parent process
and then creates another ``multiprocessing.Process``.  Linux uses ``fork`` by
default, which cannot safely re-initialize CUDA.  This server keeps the same
HTTP protocol but owns the vLLM engine directly, avoiding that unsafe outer
process boundary.  vLLM remains responsible for any multiprocessing internal
to its engine.
"""

import argparse
import asyncio
import logging
import os
from concurrent.futures import Future
from contextlib import asynccontextmanager
from queue import Queue
from threading import Lock, Thread
from typing import TYPE_CHECKING, Annotated, Callable, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from vllm import LLM


# This controls only workers created internally by vLLM.  It must be set before
# importing vLLM; unlike changing Python's global start method, it does not add
# a second TRL parent/child initialization cycle.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V1", "0")


WORKER_EXTENSION = "trl.scripts.vllm_serve.WeightSyncWorkerExtension"
logger = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    """An engine failure makes further requests unsafe until restart."""


class _EngineExecutor:
    """One ordered queue for all engine calls, including fire-and-forget RPCs.

    The TRL client joins NCCL *after* receiving the HTTP reply. Waiting for an
    init/update RPC in its endpoint would deadlock. A single worker preserves
    ordering while letting those endpoints acknowledge receipt immediately.

    Python cannot interrupt a blocked native NCCL call. Shutdown therefore has
    a deadline and this worker is a daemon; the launcher's process-group cleanup
    remains responsible for terminating any stuck vLLM child processes.
    """

    def __init__(self):
        self._queue = Queue()
        self._lock = Lock()
        self._error = None
        self._closed = False
        self._worker = Thread(target=self._run, name="vllm-engine", daemon=True)
        self._worker.start()

    def check_available(self):
        with self._lock:
            self._check_available()

    def _check_available(self):
        if self._error is not None:
            raise EngineUnavailable(f"Engine operation failed: {self._error}. Restart the server; see its log.")
        if self._closed:
            raise EngineUnavailable("Server is shutting down.")

    def submit(self, operation: Callable, *args, **kwargs) -> Future:
        with self._lock:
            self._check_available()
            result = Future()
            self._queue.put((result, operation, args, kwargs))
            return result

    def _run(self):
        while (task := self._queue.get()) is not None:
            result, operation, args, kwargs = task
            # Drain pending requests with an error after the first failure;
            # never generate with weights that may be only partially updated.
            with self._lock:
                error = self._error
            if error is not None:
                result.set_exception(EngineUnavailable(str(error)))
                continue
            try:
                value = operation(*args, **kwargs)
            except Exception as exc:
                logger.exception("vLLM engine operation failed")
                with self._lock:
                    self._error = exc
                result.set_exception(EngineUnavailable(str(exc)))
            else:
                result.set_result(value)

    def shutdown(self, timeout: float):
        with self._lock:
            self._closed = True
            self._queue.put(None)
        self._worker.join(timeout)
        if self._worker.is_alive():
            logger.error("Engine did not stop within %.1fs; native NCCL calls cannot be cancelled in Python.", timeout)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    prompts: list[str] = Field(min_length=1)
    n: int = Field(default=1, ge=1)
    repetition_penalty: float = Field(default=1.0, gt=0)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=-1, ge=-1)
    min_p: float = Field(default=0.0, ge=0, le=1)
    max_tokens: int = Field(default=16, ge=1)
    guided_decoding_regex: Optional[str] = None


class InitCommunicatorRequest(BaseModel):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    world_size: int = Field(ge=2)


class UpdateWeightsRequest(BaseModel):
    name: str = Field(min_length=1)
    dtype: str
    shape: list[Annotated[int, Field(ge=0)]]


def create_app(llm: "LLM", tensor_parallel_size: int, shutdown_timeout: float = 10.0, run_id: str | None = None) -> FastAPI:
    """Build the TRL 0.18 API (one training client, one engine, TP supported).

    An acknowledged NCCL request is queued, not yet complete. Its later failure
    is reported in the server log and by HTTP 503 on health/subsequent requests;
    TRL 0.18 has no protocol for returning that failure to an already-ACKed call.
    """

    if tensor_parallel_size < 1 or not 0 < shutdown_timeout < float("inf"):
        raise ValueError("tensor_parallel_size and shutdown_timeout must be positive")
    executor = _EngineExecutor()
    communicator_requested = False

    def submit(operation, *args, **kwargs):
        try:
            return executor.submit(operation, *args, **kwargs)
        except EngineUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def wait_for(result):
        try:
            # A disconnected HTTP caller must not cancel an ordered engine job.
            return await asyncio.shield(asyncio.wrap_future(result))
        except EngineUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            if communicator_requested:
                try:
                    executor.submit(llm.collective_rpc, method="close_communicator")
                except EngineUnavailable:
                    pass  # Failed engines must not receive more model operations.
            await asyncio.to_thread(executor.shutdown, shutdown_timeout)

    async def require_healthy():
        try:
            executor.check_available()
        except EngineUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    app = FastAPI(lifespan=lifespan, dependencies=[Depends(require_healthy)])

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/health/{requested_run_id}/")
    async def owned_health(requested_run_id: str):
        if run_id is None or requested_run_id != run_id:
            raise HTTPException(status_code=404, detail="Unknown run")
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        return {"world_size": tensor_parallel_size}

    @app.post("/generate/")
    async def generate(request: GenerateRequest):
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        guided_decoding = (
            GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
            if request.guided_decoding_regex is not None
            else None
        )
        try:
            sampling_params = SamplingParams(
                n=request.n,
                repetition_penalty=request.repetition_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                max_tokens=request.max_tokens,
                guided_decoding=guided_decoding,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        outputs = await wait_for(submit(llm.generate, prompts=request.prompts, sampling_params=sampling_params))
        completion_ids = [list(output.token_ids) for result in outputs for output in result.outputs]
        return {"completion_ids": completion_ids}

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        nonlocal communicator_requested
        expected_world_size = tensor_parallel_size + 1
        if request.world_size != expected_world_size:
            raise HTTPException(
                status_code=422,
                detail=f"Expected communicator world size {expected_world_size}, got {request.world_size}.",
            )
        if communicator_requested:
            raise HTTPException(status_code=409, detail="Communicator already requested; close it before reinitializing.")
        submit(
            llm.collective_rpc,
            method="init_communicator",
            args=(request.host, request.port, expected_world_size),
        )
        communicator_requested = True
        return {"message": "Request received, initializing communicator"}

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        import torch

        if not communicator_requested:
            raise HTTPException(status_code=409, detail="Initialize the communicator before updating weights.")
        dtype_name = request.dtype.rsplit(".", maxsplit=1)[-1]
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise HTTPException(status_code=422, detail=f"Unsupported torch dtype: {request.dtype}")
        submit(llm.collective_rpc, method="update_named_param", args=(request.name, dtype, tuple(request.shape)))
        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        success = await wait_for(submit(llm.reset_prefix_cache))
        return {"message": f"Prefix cache reset: {success}"}

    @app.post("/close_communicator/")
    async def close_communicator():
        nonlocal communicator_requested
        if communicator_requested:
            result = submit(llm.collective_rpc, method="close_communicator")
            communicator_requested = False
            await wait_for(result)
        return {"message": "Communicator closed"}

    return app


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv_cache_dtype", default="auto")
    parser.add_argument("--enable_prefix_caching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enforce_eager", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log_level", default="info", choices=("critical", "error", "warning", "info", "debug", "trace"))
    parser.add_argument("--shutdown_timeout", type=float, default=10.0)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor_parallel_size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu_memory_utilization must be greater than 0 and at most 1")
    if args.max_model_len is not None and args.max_model_len < 1:
        parser.error("--max_model_len must be positive")
    if not 0 < args.shutdown_timeout < float("inf"):
        parser.error("--shutdown_timeout must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    if os.environ.get("VLLM_PORT") == str(args.port):
        raise SystemExit(
            "VLLM_PORT configures vLLM's internal communication, not HTTP. "
            "Unset it or choose a different internal port; use --port for HTTP "
            "(VLLM_HTTP_PORT in the project launcher)."
        )
    if os.environ["VLLM_USE_V1"] != "0" or os.environ["VLLM_WORKER_MULTIPROC_METHOD"] != "spawn":
        raise SystemExit(
            "This TRL 0.18/vLLM 0.8.5 server requires VLLM_USE_V1=0 and "
            "VLLM_WORKER_MULTIPROC_METHOD=spawn. Check the deployment guide before changing the engine."
        )

    import uvicorn
    from vllm import LLM

    llm = LLM(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        worker_extension_cls=WORKER_EXTENSION,
    )
    app = create_app(llm, args.tensor_parallel_size, args.shutdown_timeout, run_id=args.run_id)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
