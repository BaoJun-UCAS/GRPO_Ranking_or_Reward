"""CPU-only tests for the external QRM service and its GRPO client."""

import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from open_r1.reward_server import build_length_aware_batches, create_app, parse_args
from open_r1.reward_client import get_remote_qrm_reward


class FakeScorer:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def score(self, messages):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            self.calls.append(messages)
            return [float(len(item[-1]["content"])) for item in messages]
        finally:
            with self.lock:
                self.active -= 1


class RewardServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.scorer = FakeScorer()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.scorer, run_id="run-123")),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_scoring_and_validation(self):
        self.assertEqual((await self.client.get("/health/")).json(), {"status": "ok"})
        self.assertEqual((await self.client.get("/health/run-123/")).status_code, 200)
        self.assertEqual((await self.client.get("/health/another-run/")).status_code, 404)
        payload = {
            "messages": [
                [{"role": "user", "content": "q"}, {"role": "assistant", "content": "abc"}],
                [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "hello"}],
            ]
        }
        response = await self.client.post("/score/", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"rewards": [3.0, 5.0]})
        self.assertEqual(self.scorer.calls, [payload["messages"]])
        self.assertEqual((await self.client.post("/score/", json={"messages": []})).status_code, 422)
        self.assertEqual((await self.client.post("/score/", json={"messages": [[]]})).status_code, 422)
        no_completion = {"messages": [[{"role": "user", "content": "x"}]]}
        self.assertEqual((await self.client.post("/score/", json=no_completion)).status_code, 422)

    async def test_gpu_scoring_is_serialized_across_training_ranks(self):
        payload = {"messages": [[{"role": "assistant", "content": "x"}]]}
        first, second = await asyncio.gather(
            self.client.post("/score/", json=payload),
            self.client.post("/score/", json=payload),
        )
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertEqual(self.scorer.max_active, 1)


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        return self.payload


class RewardClientTests(unittest.TestCase):
    def test_client_imports_with_only_the_python_standard_library(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        result = subprocess.run(
            [sys.executable, "-S", "-c",
             "from open_r1.reward_client import get_remote_qrm_reward; "
             "assert callable(get_remote_qrm_reward('http://localhost:8001', 30))"],
            env={**os.environ, "PYTHONPATH": str(source_root)},
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_client_combines_prompt_and_completion_without_loading_a_model(self):
        prompt = [[{"role": "user", "content": "question"}]]
        completion = [[{"role": "assistant", "content": "answer"}]]
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data)
            return FakeResponse({"rewards": [1.25]})

        with patch("open_r1.reward_client.urlopen", side_effect=fake_urlopen):
            reward = get_remote_qrm_reward("http://127.0.0.1:8001/", 30)
            self.assertEqual(reward(prompt, completion), [1.25])

        self.assertEqual(captured["url"], "http://127.0.0.1:8001/score/")
        self.assertEqual(captured["timeout"], 30)
        self.assertEqual(captured["body"]["messages"], [prompt[0] + completion[0]])

    def test_raw_text_is_wrapped_without_changing_content(self):
        with patch("open_r1.reward_client.urlopen", return_value=FakeResponse({"rewards": [1.25]})) as request:
            reward = get_remote_qrm_reward("http://127.0.0.1:8001", 30)
            self.assertEqual(reward(["raw prompt\n"], ["answer"]), [1.25])
        self.assertEqual(json.loads(request.call_args.args[0].data)["messages"], [[
            {"role": "user", "content": "raw prompt\n"}, {"role": "assistant", "content": "answer"},
        ]])
        with self.assertRaisesRegex(TypeError, "matching"):
            reward(["prompt"], [[{"role": "assistant", "content": "answer"}]])

    def test_client_rejects_non_finite_rewards(self):
        with patch("open_r1.reward_client.urlopen", return_value=FakeResponse({"rewards": [float("nan")]})):
            reward = get_remote_qrm_reward("http://127.0.0.1:8001", 30)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                reward([[{"role": "user", "content": "q"}]], [[{"role": "assistant", "content": "a"}]])

    def test_length_aware_batches_preserve_examples_and_bound_padding(self):
        lengths = [500, 100, 120, 510, 900]
        batches = build_length_aware_batches(lengths, max_batch_size=2, max_batch_tokens=1000)
        flattened = [index for batch in batches for index in batch]
        self.assertCountEqual(flattened, range(len(lengths)))
        for batch in batches:
            self.assertLessEqual(len(batch), 2)
            if len(batch) > 1:
                self.assertLessEqual(len(batch) * max(lengths[index] for index in batch), 1000)

    def test_length_aware_batches_reject_an_unenforceable_single_example(self):
        with self.assertRaisesRegex(ValueError, "exceeds the padded-token budget"):
            build_length_aware_batches([1001], max_batch_size=2, max_batch_tokens=1000)

    def test_server_argument_validation(self):
        args = parse_args(["--model", "qrm"])
        self.assertEqual((args.port, args.batch_size, args.max_batch_tokens, args.max_length), (8001, 1, 0, 4096))
        for option, value in (
            ("--port", "0"),
            ("--batch-size", "0"),
            ("--max-batch-tokens", "-1"),
            ("--max-length", "0"),
        ):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args(["--model", "qrm", option, value])
                self.assertEqual(raised.exception.code, 2)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args(["--model", "qrm", "--max-length", "1024", "--max-batch-tokens", "512"])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
