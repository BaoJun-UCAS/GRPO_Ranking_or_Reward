"""CPU-only tests of the TRL 0.18 HTTP contract and engine operation ordering.

Run with ``python -m unittest discover -s tests -p test_vllm_serve.py -v``.
Fake model/NCCL objects deliberately block until the HTTP client receives its
ACK, so a regression that waits for NCCL in the handler fails without a GPU.
These tests do not validate real CUDA, NCCL, model loading, or training.
"""

import asyncio
import contextlib
import io
import os
import subprocess
import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import httpx

from open_r1.vllm_serve import create_app, main, parse_args


class FakeDtype:
    pass


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.generation_requests = []
        self.block_method = None
        self.fail_method = None
        self.entered = threading.Event()
        self.release = threading.Event()

    def _record(self, name):
        self.calls.append(name)
        if name == self.fail_method:
            raise RuntimeError("fake NCCL failure")
        if name == self.block_method:
            self.entered.set()
            if not self.release.wait(timeout=2):
                raise RuntimeError("HTTP ACK was not delivered before the collective")

    def collective_rpc(self, method, args=()):
        self._record(method)

    def generate(self, prompts, sampling_params, *, use_tqdm=True):
        self._record("generate")
        self.generation_requests.append((prompts, sampling_params, use_tqdm))
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(token_ids=(index, n)) for n in range(sampling_params.n)]
            )
            for index, _ in enumerate(prompts)
        ]

    def reset_prefix_cache(self):
        self._record("reset_prefix_cache")
        return True


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.dtype = FakeDtype
        fake_torch.float32 = FakeDtype()
        fake_torch.zeros = lambda: None
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = FakeSamplingParams
        fake_sampling = types.ModuleType("vllm.sampling_params")
        fake_sampling.GuidedDecodingParams = FakeSamplingParams
        self.modules = patch.dict(
            sys.modules,
            {"torch": fake_torch, "vllm": fake_vllm, "vllm.sampling_params": fake_sampling},
        )
        self.modules.start()
        self.engine = FakeEngine()
        self.app = create_app(self.engine, tensor_parallel_size=1, shutdown_timeout=0.1)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        self.engine.release.set()
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.modules.stop()

    async def init(self):
        return await asyncio.wait_for(
            self.client.post("/init_communicator/", json={"host": "0.0.0.0", "port": 51216, "world_size": 2}),
            timeout=0.5,
        )

    async def update(self):
        return await asyncio.wait_for(
            self.client.post("/update_named_param/", json={"name": "layer.weight", "dtype": "torch.float32", "shape": [2]}),
            timeout=0.5,
        )

    async def test_client_protocol_and_completion_order(self):
        self.assertEqual((await self.client.get("/health/")).json(), {"status": "ok"})
        self.assertEqual((await self.client.get("/get_world_size/")).json(), {"world_size": 1})
        response = await self.client.post("/generate/", json={"prompts": ["one", "two"], "n": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"completion_ids": [[0, 0], [0, 1], [1, 0], [1, 1]]})

    async def test_token_only_generation_preserves_sampling_and_guided_regex(self):
        response = await self.client.post("/generate/", json={
            "prompts": ["answer"], "n": 3, "max_tokens": 17,
            "temperature": 0.8, "top_p": 0.7, "top_k": 4, "min_p": 0.02,
            "repetition_penalty": 1.1, "guided_decoding_regex": "[ab]+",
        })
        self.assertEqual(response.status_code, 200)
        prompts, params, use_tqdm = self.engine.generation_requests[-1]
        self.assertEqual(prompts, ["answer"])
        self.assertFalse(use_tqdm)
        self.assertIs(params.detokenize, False)
        self.assertEqual((params.n, params.max_tokens), (3, 17))
        self.assertEqual((params.temperature, params.top_p, params.top_k, params.min_p), (0.8, 0.7, 4, 0.02))
        self.assertEqual(params.repetition_penalty, 1.1)
        self.assertEqual((params.guided_decoding.backend, params.guided_decoding.regex), ("outlines", "[ab]+"))
        # Keep vLLM's default EOS stopping; token-only output must not set
        # ignore_eos or add a text stop condition that requires detokenization.
        self.assertNotIn("ignore_eos", vars(params))
        self.assertNotIn("stop", vars(params))

    async def test_token_only_response_preserves_terminal_token_ids(self):
        # The trainer needs terminal EOS IDs to construct completion masks.
        result = [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=(11, 7, 2))])]
        with patch.object(self.engine, "generate", return_value=result):
            response = await self.client.post("/generate/", json={"prompts": ["question"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"completion_ids": [[11, 7, 2]]})

    async def test_init_ack_precedes_nccl_completion_and_health_stays_responsive(self):
        self.engine.block_method = "init_communicator"
        self.assertEqual((await self.init()).status_code, 200)
        self.assertTrue(await asyncio.to_thread(self.engine.entered.wait, 0.5))
        self.assertEqual((await self.client.get("/health/")).status_code, 200)
        self.assertFalse(self.engine.release.is_set())
        self.engine.release.set()

    async def test_weight_ack_and_serialized_generation_cache_reset_close(self):
        self.assertEqual((await self.init()).status_code, 200)
        self.engine.block_method = "update_named_param"
        self.assertEqual((await self.update()).status_code, 200)
        self.assertTrue(await asyncio.to_thread(self.engine.entered.wait, 0.5))
        generation = asyncio.create_task(self.client.post("/generate/", json={"prompts": ["test"]}))
        await asyncio.sleep(0.02)
        self.assertFalse(generation.done())
        self.assertEqual(self.engine.calls, ["init_communicator", "update_named_param"])
        self.assertEqual((await self.client.get("/health/")).status_code, 200)
        self.engine.release.set()
        self.assertEqual((await generation).status_code, 200)
        self.assertEqual((await self.client.post("/reset_prefix_cache/")).status_code, 200)
        self.assertEqual((await self.client.post("/close_communicator/")).status_code, 200)
        self.assertEqual(
            self.engine.calls,
            ["init_communicator", "update_named_param", "generate", "reset_prefix_cache", "close_communicator"],
        )

    async def test_background_exception_reaches_health_and_later_requests(self):
        self.engine.fail_method = "init_communicator"
        with self.assertLogs("open_r1.vllm_serve", level="ERROR") as logs:
            self.assertEqual((await self.init()).status_code, 200)
            response = await self.client.post("/reset_prefix_cache/")
            self.assertEqual(response.status_code, 503)
        self.assertIn("fake NCCL failure", "\n".join(logs.output))
        for endpoint in ("/health/", "/get_world_size/"):
            self.assertEqual((await self.client.get(endpoint)).status_code, 503)
        self.assertEqual((await self.client.post("/generate/", json={"prompts": ["test"]})).status_code, 503)
        self.assertEqual(self.engine.calls, ["init_communicator"])

    async def test_invalid_protocol_inputs_do_not_damage_engine(self):
        self.assertEqual((await self.update()).status_code, 409)
        response = await self.client.post("/init_communicator/", json={"host": "localhost", "port": 51216, "world_size": 3})
        self.assertEqual(response.status_code, 422)
        response = await self.client.post("/init_communicator/", json={"host": "localhost", "port": 0, "world_size": 2})
        self.assertEqual(response.status_code, 422)
        self.assertEqual((await self.init()).status_code, 200)
        self.assertEqual((await self.init()).status_code, 409)
        for dtype, shape in (("torch.zeros", [2]), ("torch.float32", [-1])):
            response = await self.client.post("/update_named_param/", json={"name": "weight", "dtype": dtype, "shape": shape})
            self.assertEqual(response.status_code, 422)
        for params in ({"n": 0}, {"max_tokens": 0}, {"temperature": -1}, {"top_p": 1.1}):
            response = await self.client.post("/generate/", json={"prompts": ["test"], **params})
            self.assertEqual(response.status_code, 422)
        self.assertEqual((await self.client.get("/health/")).status_code, 200)

    async def test_close_is_idempotent_and_can_reinitialize(self):
        await self.init()
        await self.client.post("/close_communicator/")
        await self.client.post("/close_communicator/")
        self.assertEqual(self.engine.calls, ["init_communicator", "close_communicator"])
        self.assertEqual((await self.init()).status_code, 200)
        await self.client.post("/close_communicator/")
        self.assertEqual(self.engine.calls[-2:], ["init_communicator", "close_communicator"])

    async def test_normal_shutdown_closes_communicator(self):
        await self.init()
        await self.lifespan.__aexit__(None, None, None)
        self.assertEqual(self.engine.calls, ["init_communicator", "close_communicator"])

    async def test_shutdown_has_deadline_when_native_collective_is_stuck(self):
        self.engine.block_method = "init_communicator"
        await self.init()
        self.assertTrue(await asyncio.to_thread(self.engine.entered.wait, 0.5))
        started = time.monotonic()
        with self.assertLogs("open_r1.vllm_serve", level="ERROR"):
            await self.lifespan.__aexit__(None, None, None)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual((await self.client.get("/health/")).status_code, 503)
        self.engine.release.set()


class ConfigurationTests(unittest.TestCase):
    def test_internal_port_collision_fails_before_importing_cuda(self):
        env = dict(os.environ, VLLM_PORT="8000")
        program = (
            "import sys; from open_r1.vllm_serve import main; sys.argv=['serve', '--model', 'test']; "
            "main()"
        )
        result = subprocess.run([sys.executable, "-c", program], env=env, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VLLM_PORT configures vLLM's internal communication", result.stderr)

    def test_incompatible_engine_fails_before_importing_cuda(self):
        env = dict(os.environ, VLLM_USE_V1="1")
        program = (
            "import sys; from open_r1.vllm_serve import main; sys.argv=['serve', '--model', 'test']; "
            "main()"
        )
        result = subprocess.run([sys.executable, "-c", program], env=env, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires VLLM_USE_V1=0", result.stderr)

    def test_local_default_and_invalid_arguments(self):
        self.assertEqual(parse_args(["--model", "test"]).host, "127.0.0.1")
        for option, value in (
            ("--port", "0"), ("--port", "65536"), ("--tensor_parallel_size", "0"),
            ("--gpu_memory_utilization", "0"), ("--gpu_memory_utilization", "nan"),
            ("--max_model_len", "0"), ("--max_num_seqs", "0"), ("--max_num_seqs", "-1"),
            ("--shutdown_timeout", "inf"),
        ):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args(["--model", "test", option, value])
                self.assertEqual(raised.exception.code, 2)

    def test_sequence_limit_is_optional_and_reaches_engine(self):
        self.assertIsNone(parse_args(["--model", "test"]).max_num_seqs)
        self.assertEqual(parse_args(["--model", "test", "--max_num_seqs", "64"]).max_num_seqs, 64)
        for limit in (None, 64):
            with self.subTest(limit=limit):
                fake_vllm = types.ModuleType("vllm")
                fake_vllm.LLM = Mock()
                fake_uvicorn = types.ModuleType("uvicorn")
                fake_uvicorn.run = Mock()
                argv = ["serve", "--model", "test"]
                if limit is not None:
                    argv.extend(["--max_num_seqs", str(limit)])
                with (
                    patch.dict(sys.modules, {"vllm": fake_vllm, "uvicorn": fake_uvicorn}),
                    patch.dict(os.environ, {"VLLM_PORT": "", "VLLM_USE_V1": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}),
                    patch.object(sys, "argv", argv),
                    patch("open_r1.vllm_serve.create_app", return_value=object()),
                ):
                    main()
                self.assertEqual(fake_vllm.LLM.call_args.kwargs["max_num_seqs"], limit)
                fake_uvicorn.run.assert_called_once()

    def test_import_and_help_do_not_import_cuda_libraries(self):
        program = (
            "import sys; from open_r1.vllm_serve import parse_args; "
            "assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules; "
            "parse_args(['--help'])"
        )
        result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--tensor_parallel_size", result.stdout)


if __name__ == "__main__":
    unittest.main()
