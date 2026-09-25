"""Launcher behavior using tiny fake model/training processes, never CUDA."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest
import yaml


PROJECT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT / "train_scripts/qwen3_1.7_grpo_chat.sh"
SPEC = importlib.util.spec_from_file_location("run_helpers", PROJECT / "train_scripts/run_helpers.py")
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


@pytest.fixture
def launch_env(tmp_path):
    env = os.environ.copy()
    for name in (
        "HF_USERNAME", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE",
        "GRPO_CACHE_ROOT", "RUN_DIR", "GENERATION_BATCH_SIZE", "CONFIG_FILE", "ACCELERATE_CONFIG",
        "RESUME_FROM_CHECKPOINT", "CUDA_VISIBLE_DEVICES", "VLLM_PORT", "VLLM_HTTP_PORT",
    ):
        env.pop(name, None)
    env.update(
        PYTHON=sys.executable,
        DATASET_NAME="test/prepared-data",
        VLLM_GPU="0",
        TRAIN_GPUS="1,2",
        PER_DEVICE_TRAIN_BATCH_SIZE="1",
        NUM_GENERATIONS="8",
        GRADIENT_ACCUMULATION_STEPS="8",
        MAX_STEPS="2",
        MAX_PROMPT_LENGTH="512",
        MAX_COMPLETION_LENGTH="256",
        VLLM_MAX_MODEL_LEN="4096",
        VLLM_HTTP_PORT="8123",
        PORT="29531",
        VLLM_GROUP_PORT="51216",
        VLLM_GPU_MEMORY_UTILIZATION="0.82",
        VLLM_USE_V1="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_STARTUP_TIMEOUT="5",
        REWARD_BATCH_SIZE="2",
        MERGE_AFTER_TRAINING="0",
        DRY_RUN="0",
        RUN_NAME="test-run",
        GRPO_OUTPUT_ROOT=str(tmp_path / "runs"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
    )
    return env


def run_launcher(env, *args):
    return subprocess.run(
        ["bash", str(LAUNCHER), *args], env=env, cwd=PROJECT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20,
    )


def test_dry_run_resolves_gpu_count_batch_and_safe_strings_without_files(launch_env, tmp_path):
    launch_env["DATASET_NAME"] = "local: data # not a YAML comment"
    launch_env["HF_HUB_CACHE"] = str(tmp_path / "explicit hub")
    result = run_launcher(launch_env, "--dry-run")
    assert result.returncode == 0, result.stdout
    assert "Training processes: 2; generation batch: 16" in result.stdout
    raw_yaml = result.stdout.split("Resolved training configuration:\n", 1)[1].split(
        "Resolved Accelerate configuration:", 1
    )[0]
    config = yaml.safe_load(raw_yaml)
    assert config["dataset_name"] == launch_env["DATASET_NAME"]
    assert config["generation_batch_size"] == 16
    assert config["num_generations"] == 8
    assert launch_env["HF_HUB_CACHE"] in result.stdout
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"TRAIN_GPUS": "1,1"}, "duplicate GPU"),
        ({"TRAIN_GPUS": "0,1"}, "must not also appear"),
        ({"GENERATION_BATCH_SIZE": "32"}, "must equal training GPU count"),
        ({"NUM_GENERATIONS": "3"}, "divide GENERATION_BATCH_SIZE"),
        ({"VLLM_HTTP_PORT": "29531"}, "distinct ports"),
        ({"VLLM_GROUP_PORT": "8123"}, "distinct ports"),
        ({"VLLM_MAX_MODEL_LEN": "512"}, "must cover"),
        ({"VLLM_USE_V1": "1"}, "requires VLLM_USE_V1=0"),
    ],
)
def test_invalid_launch_plan_fails_before_writing(launch_env, tmp_path, overrides, message):
    launch_env.update(overrides)
    result = run_launcher(launch_env, "--dry-run")
    assert result.returncode != 0
    assert message in result.stdout
    assert not (tmp_path / "runs").exists()


def test_unsafe_zero3_checkpoint_mode_fails_before_launch(launch_env, tmp_path):
    config = yaml.safe_load((PROJECT / "recipes/Qwen3-1.7B/config_chat_regular_qrm_lora_5gpu.yaml").read_text())
    config["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    recipe = tmp_path / "unsafe.yaml"
    recipe.write_text(yaml.safe_dump(config))
    launch_env["CONFIG_FILE"] = str(recipe)
    result = run_launcher(launch_env, "--dry-run")
    assert result.returncode != 0
    assert "PEFT + ZeRO-3 requires" in result.stdout
    assert not (tmp_path / "runs").exists()


def test_resume_preserves_checkpoint_and_uses_new_run(launch_env, tmp_path):
    checkpoint = tmp_path / "old-run" / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    launch_env["RESUME_FROM_CHECKPOINT"] = str(checkpoint)
    result = run_launcher(launch_env, "--dry-run")
    assert result.returncode == 0, result.stdout
    assert f"resume_from_checkpoint: {checkpoint}" in result.stdout
    assert checkpoint.is_dir()
    assert not (tmp_path / "runs").exists()


def test_unknown_environment_placeholders_are_not_expanded():
    with pytest.raises(ValueError, match="Unsupported or unset"):
        HELPERS.substitute({"token": "${HF_TOKEN}"}, {"MODEL_NAME": "model"})


def test_busy_port_reports_conflict_without_http(monkeypatch):
    class OccupiedSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            raise OSError("Address already in use")

    monkeypatch.setattr(socket, "socket", lambda *args: OccupiedSocket())
    with pytest.raises(ValueError, match="Local port 8000 is unavailable"):
        HELPERS.check_ports([8000])


# This interpreter proxy executes config resolution normally, but substitutes
# harmless sleeping processes for vLLM and Accelerate. Port probes are skipped
# here so lifecycle tests also work in sandboxes that prohibit socket creation.
FAKE_PYTHON = r'''
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

args = sys.argv[1:]
state = Path(os.environ["FAKE_STATE"])
if len(args) >= 2 and args[0].endswith("run_helpers.py") and args[1] == "check-ports":
    sys.exit(0)
if args[:3] == ["-m", "pip", "freeze"]:
    print("fake-runtime==1.0")
    sys.exit(0)
if args[:2] == ["-m", "open_r1.vllm_serve"]:
    role = "server"
    assert "VLLM_PORT" not in os.environ, "HTTP port leaked into vLLM's internal TCPStore environment"
elif args[:2] == ["-m", "accelerate.commands.launch"]:
    role = "train"
else:
    os.execv(sys.executable, [sys.executable, *args])
from urllib.request import proxy_bypass
assert proxy_bypass("127.0.0.1") and proxy_bypass("localhost"), "Local service traffic must bypass proxies"
if role == "server" and os.environ.get("FAKE_SERVER_FAIL") == "1":
    print("FAKE MODEL STARTUP ERROR", flush=True)
    sys.exit(17)
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
def stop(signum, frame):
    child.terminate()
    child.wait(timeout=5)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
(state / (role + ".json")).write_text(json.dumps({
    "pid": os.getpid(), "child": child.pid, "pgid": os.getpgrp(), "args": args,
    "no_proxy": os.environ.get("no_proxy"), "NO_PROXY": os.environ.get("NO_PROXY"),
    "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
}))
print("fake " + role + " ready", flush=True)
if role == "train" and os.environ.get("FAKE_TRAIN_WAIT") != "1":
    # Deliberately leave a worker behind. The launcher's group cleanup owns it.
    sys.exit(int(os.environ.get("FAKE_TRAIN_EXIT", "0")))
while True:
    time.sleep(0.1)
'''


@pytest.fixture
def fake_runtime(launch_env, tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    scripts = {
        "python-proxy": FAKE_PYTHON,
        "curl": (
            'import os, pathlib, sys\n'
            'ready = (pathlib.Path(os.environ["FAKE_STATE"]) / "server.json").exists()\n'
            'sys.exit(0 if ready and os.environ.get("FAKE_HEALTH_FAIL") != "1" else 1)\n'
        ),
        "nvidia-smi": 'print("GPU diagnostic disabled in lifecycle test")\n',
    }
    for name, content in scripts.items():
        executable = binaries / name
        executable.write_text(f"#!{sys.executable}\n" + content, encoding="utf-8")
        executable.chmod(0o755)
    launch_env.update(
        PYTHON=str(binaries / "python-proxy"),
        PATH=f"{binaries}{os.pathsep}{os.environ['PATH']}",
        FAKE_STATE=str(state),
    )
    return launch_env, state


def process_running(pid):
    status = Path(f"/proc/{pid}/stat")
    if not status.exists():
        return False
    try:
        return status.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def assert_workers_stopped(state):
    for record in state.glob("*.json"):
        processes = json.loads(record.read_text())
        assert not process_running(processes["pid"]), processes
        assert not process_running(processes["child"]), processes


@pytest.mark.parametrize("train_exit", [0, 7])
def test_training_exit_preserves_status_and_reaps_workers(fake_runtime, train_exit):
    env, state = fake_runtime
    env["FAKE_TRAIN_EXIT"] = str(train_exit)
    result = run_launcher(env)
    assert result.returncode == train_exit, result.stdout
    run = Path(env["GRPO_OUTPUT_ROOT"]) / env["RUN_NAME"]
    status = (run / "RUN_STATUS").read_text()
    assert ("status=success" if train_exit == 0 else "status=failed") in status
    training = json.loads((state / "train.json").read_text())
    assert training["args"][training["args"].index("--num_processes") + 1] == "2"
    assert_workers_stopped(state)


@pytest.mark.parametrize("explicit_http_port", [None, "8125"])
def test_legacy_http_port_alias_is_consumed_without_leaking_to_vllm(fake_runtime, explicit_http_port):
    env, state = fake_runtime
    env.pop("VLLM_HTTP_PORT")
    env["VLLM_PORT"] = "8124"
    if explicit_http_port is not None:
        env["VLLM_HTTP_PORT"] = explicit_http_port
    expected_port = explicit_http_port or "8124"
    result = run_launcher(env)
    assert result.returncode == 0, result.stdout
    assert "VLLM_PORT is deprecated" in result.stdout
    server = json.loads((state / "server.json").read_text())
    assert server["args"][server["args"].index("--port") + 1] == expected_port
    run = Path(env["GRPO_OUTPUT_ROOT"]) / env["RUN_NAME"]
    config = yaml.safe_load((run / "config/resolved_training_config.yaml").read_text())
    assert config["vllm_server_base_url"] == f"http://127.0.0.1:{expected_port}"
    assert_workers_stopped(state)


def test_local_proxy_bypass_preserves_existing_external_proxy_rules(fake_runtime):
    env, state = fake_runtime
    env.update(HTTP_PROXY="http://127.0.0.1:9", NO_PROXY="upper.example", no_proxy="lower.example")
    result = run_launcher(env)
    assert result.returncode == 0, result.stdout
    for role in ("server", "train"):
        process = json.loads((state / (role + ".json")).read_text())
        assert process["HTTP_PROXY"] == env["HTTP_PROXY"]
        assert process["no_proxy"] == process["NO_PROXY"]
        assert set(process["NO_PROXY"].split(",")) >= {"127.0.0.1", "localhost", "::1", "upper.example", "lower.example"}
    assert_workers_stopped(state)


def test_failed_server_surfaces_original_error_without_training(fake_runtime):
    env, state = fake_runtime
    env["FAKE_SERVER_FAIL"] = "1"
    result = run_launcher(env)
    assert result.returncode != 0
    assert "FAKE MODEL STARTUP ERROR" in result.stdout
    assert not (state / "train.json").exists()


def test_startup_timeout_cleans_live_server_and_preserves_log(fake_runtime):
    env, state = fake_runtime
    env.update(FAKE_HEALTH_FAIL="1", VLLM_STARTUP_TIMEOUT="1")
    result = run_launcher(env)
    assert result.returncode != 0
    assert "startup timed out" in result.stdout
    assert "fake server ready" in result.stdout
    assert not (state / "train.json").exists()
    assert_workers_stopped(state)


@pytest.mark.parametrize("termination_signal", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_cleans_both_owned_process_groups(fake_runtime, termination_signal):
    env, state = fake_runtime
    env["FAKE_TRAIN_WAIT"] = "1"
    process = subprocess.Popen(
        ["bash", str(LAUNCHER)], env=env, cwd=PROJECT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 12
        while not (state / "train.json").exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(process.communicate()[0])
            time.sleep(0.05)
        assert (state / "train.json").exists(), "Training stub did not start"
        process.send_signal(termination_signal)
        output, _ = process.communicate(timeout=12)
        assert process.returncode == 128 + termination_signal, output
        assert_workers_stopped(state)
    finally:
        # Only the fixture's recorded groups are candidates for test cleanup.
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=12)
        for record in state.glob("*.json"):
            group = json.loads(record.read_text())["pgid"]
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
