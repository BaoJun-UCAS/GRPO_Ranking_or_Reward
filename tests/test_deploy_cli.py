"""CPU-only checks for deployment shortcuts and portable cache paths."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("grpo_deploy", ROOT / "scripts/grpo.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class DeploymentCliTests(unittest.TestCase):
    def test_doctor_rejects_missing_explicit_cuda_compiler_without_loading_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            out = io.StringIO()
            with patch.dict(os.environ, {"CUDA_HOME": directory}), contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["doctor"]), 1)
            self.assertIn("FAIL CUDA compiler: not found", out.getvalue())
            self.assertIn("CUDA was not initialized", out.getvalue())

    def test_cache_precedence_and_no_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            env = {"HOME": str(base), "GRPO_CACHE_ROOT": str(base / "cache")}
            self.assertEqual(cli.paths(env)["HF_HUB_CACHE"], base / "cache/huggingface/hub")
            env["HF_HOME"] = str(base / "hf")
            self.assertEqual(cli.paths(env)["HF_HUB_CACHE"], base / "hf/hub")
            env["HF_HUB_CACHE"] = str(base / "hub")
            self.assertEqual(cli.paths(env)["HF_HUB_CACHE"], base / "hub")
            self.assertFalse((base / "cache").exists())
            self.assertEqual(cli.paths({"HOME": str(base)})["HF_HOME"], base / ".cache/grpo/huggingface")

    def test_default_cache_prefers_writable_data_root_without_creating_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data_root = base / "data"
            data_root.mkdir()
            selected = cli.paths({"HOME": str(base / "home"), "GRPO_DATA_ROOT": str(data_root)})
            self.assertEqual(selected["HF_HOME"], data_root / "cache/grpo/huggingface")
            self.assertFalse((data_root / "cache").exists())

    def test_build_caches_use_project_cache_root_and_respect_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            env = {"GRPO_CACHE_ROOT": str(base / "cache")}
            selected = cli.paths(env)
            self.assertEqual(selected["TORCH_EXTENSIONS_DIR"], base / "cache/torch_extensions")
            self.assertEqual(selected["TRITON_CACHE_DIR"], base / "cache/triton")
            self.assertEqual(selected["VLLM_CACHE_ROOT"], base / "cache/vllm")
            env["TRITON_CACHE_DIR"] = str(base / "custom-triton")
            self.assertEqual(cli.paths(env)["TRITON_CACHE_DIR"], base / "custom-triton")
            self.assertFalse((base / "cache").exists())

    def test_help_imports_no_ml_libraries(self):
        code = """import importlib.util, sys
spec = importlib.util.spec_from_file_location('deploy', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.main(['help']) == 0
assert not {'torch', 'vllm', 'transformers', 'huggingface_hub'} & sys.modules.keys()
"""
        result = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts/grpo.py")],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_smoke_uses_active_python_and_does_not_override_choices(self):
        env = {"MAX_STEPS": "3", "TRAIN_GPUS": "1,2", "DATASET_NAME": "org/data"}
        with patch.dict(os.environ, env, clear=True), patch.object(os, "execvpe") as execute:
            self.assertIsNone(cli.main(["smoke", "--dry-run"]))
        command, arguments, actual = execute.call_args.args
        self.assertEqual(command, "bash")
        self.assertIn("--dry-run", arguments)
        self.assertEqual(actual["PYTHON"], sys.executable)
        self.assertEqual(actual["MAX_STEPS"], "3")
        self.assertEqual(actual["MERGE_AFTER_TRAINING"], "0")
        self.assertNotIn("GENERATION_BATCH_SIZE", actual)

    def test_validate_uses_active_python_and_passes_merged_contract(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(subprocess, "run") as run:
            run.return_value.returncode = 7
            result = cli.main(["validate", "--run-dir", directory, "--require-merged"])
        self.assertEqual(result, 7)
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(ROOT / "scripts/validate_training_run.py"))
        self.assertEqual(command[2], directory)
        self.assertIn("--require-merged", command)

    def test_cache_report_counts_blobs_once_without_snapshot_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            blobdir = Path(directory) / "models--org--model/blobs"
            blobdir.mkdir(parents=True)
            (blobdir / "complete").write_bytes(b"123")
            (blobdir / "partial.incomplete").write_bytes(b"12")
            out = io.StringIO()
            with patch.dict(os.environ, {"HF_HUB_CACHE": directory}), contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["cache"]), 0)
            self.assertIn("org/model", out.getvalue())
            self.assertIn("partial files 1", out.getvalue())
            self.assertIn("not a download percentage", out.getvalue())

    def test_missing_run_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            err = io.StringIO()
            with patch.dict(os.environ, {"GRPO_OUTPUT_ROOT": directory}, clear=True), contextlib.redirect_stderr(err):
                self.assertEqual(cli.main(["logs"]), 1)
            self.assertIn("--run-dir", err.getvalue())


if __name__ == "__main__":
    unittest.main()


def test_download_respects_independent_model_revisions(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    download = Mock(return_value="/cached/model")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    for key, value in {"MODEL_NAME": "org/policy", "MODEL_REVISION": "policy-commit",
                       "QRM_MODEL": "org/reward", "QRM_REVISION": "reward-commit"}.items():
        monkeypatch.setenv(key, value)
    assert cli.main(["download"]) == 0
    assert [(call.kwargs["repo_id"], call.kwargs["revision"]) for call in download.call_args_list] == [
        ("org/policy", "policy-commit"), ("org/reward", "reward-commit"),
    ]
