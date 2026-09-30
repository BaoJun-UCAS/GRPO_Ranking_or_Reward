"""CPU-only validation of completed training artifacts and timing evidence."""

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_validation", ROOT / "scripts/validate_training_run.py")
VALIDATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATION)


def timing_fields(name, rank0, rank1):
    minimum, maximum = min(rank0, rank1), max(rank0, rank1)
    return {
        f"timing/{name}_min_s": minimum,
        f"timing/{name}_max_s": maximum,
        f"timing/{name}_rank_spread_s": maximum - minimum,
        f"timing/{name}_rank0_s": rank0,
        f"timing/{name}_rank1_s": rank1,
    }


def create_run(path, status=True):
    (path / "config").mkdir(parents=True)
    (path / "logs").mkdir()
    config = {
        "max_steps": 2,
        "use_peft": True,
        "profile_stage_timings": True,
        "reward_funcs": ["qrm_server"],
    }
    (path / "config/resolved_training_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (path / "config/accelerate_config.yaml").write_text(
        yaml.safe_dump({"num_processes": 2, "distributed_type": "DEEPSPEED"}), encoding="utf-8"
    )
    (path / "run.env").write_text(
        "QRM_MAX_BATCH_TOKENS=128\nREWARD_BATCH_SIZE=4\nTRAIN_GPUS=6,7\n", encoding="utf-8"
    )
    (path / "run_manifest.json").write_text(json.dumps({"package_versions": {}}), encoding="utf-8")
    timing = {"step": 2, "loss": 0.25, "reward": 1.0}
    for name, rank0, rank1 in (
        ("rollout_total", 2.0, 2.1),
        ("qrm_total", 1.0, 1.1),
        ("external_sync_wait", 0.1, 3.0),
        ("generation_score_total", 4.0, 4.2),
        ("training_step_total", 10.0, 11.0),
        ("policy_train_total", 5.0, 5.5),
    ):
        timing.update(timing_fields(name, rank0, rank1))
    state = {
        "global_step": 2,
        "max_steps": 2,
        "log_history": [timing, {"step": 2, "train_runtime": 20.0, "train_steps_per_second": 0.1}],
    }
    (path / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    (path / "train_results.json").write_text(json.dumps({"train_loss": 0.25}), encoding="utf-8")
    (path / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"fake-adapter")
    (path / "logs/training.log").write_text("training complete\n", encoding="utf-8")
    (path / "logs/vllm_server.log").write_text("generation complete\n", encoding="utf-8")
    (path / "logs/qrm_server.log").write_text(
        "INFO qrm_score examples=4 queue_s=0.0100 inference_s=2.0000 batches=2 "
        "input_tokens=90 padded_tokens=100 max_batch_examples=2 max_batch_padded_tokens=60\n",
        encoding="utf-8",
    )
    if status:
        (path / "RUN_STATUS").write_text("status=success\n", encoding="utf-8")
    return path


def test_valid_run_reports_rank_stage_shares_and_qrm_efficiency(tmp_path):
    run = create_run(tmp_path / "run")
    report = VALIDATION.validate_run(run)
    assert report["status"] == "passed", report["errors"]
    assert report["timing"]["per_rank"]["0"]["share_of_training_step"] == {
        "rollout": 0.2,
        "qrm": 0.1,
        "policy": 0.5,
        "other": 0.2,
    }
    assert report["qrm"]["padding_efficiency"] == 0.9
    assert report["qrm"]["max_batch_padded_tokens"] == 60
    output = run / "custom-report.json"
    VALIDATION.write_report(report, output)
    assert json.loads(output.read_text())["status"] == "passed"


def test_validator_rejects_early_stop_nonfinite_and_inconsistent_timing(tmp_path):
    run = create_run(tmp_path / "run")
    state_path = run / "trainer_state.json"
    state = json.loads(state_path.read_text())
    state["global_step"] = 1
    state["log_history"][0]["loss"] = float("nan")
    state["log_history"][0]["timing/qrm_total_rank_spread_s"] = 9.0
    state_path.write_text(json.dumps(state))
    report = VALIDATION.validate_run(run)
    assert report["status"] == "failed"
    assert any("global_step=1" in error for error in report["errors"])
    assert any("Non-finite metric 'loss'" in error for error in report["errors"])
    assert any("rank spread is inconsistent" in error for error in report["errors"])


def test_validator_enforces_qrm_limits_and_required_timing(tmp_path):
    run = create_run(tmp_path / "run")
    state_path = run / "trainer_state.json"
    state = json.loads(state_path.read_text())
    del state["log_history"][0]["timing/policy_train_total_rank1_s"]
    state_path.write_text(json.dumps(state))
    (run / "logs/qrm_server.log").write_text(
        "qrm_score examples=5 queue_s=0 inference_s=1 batches=1 input_tokens=100 padded_tokens=90 "
        "max_batch_examples=5 max_batch_padded_tokens=256\n"
    )
    report = VALIDATION.validate_run(run)
    assert report["status"] == "failed"
    assert any("policy_train_total_rank1_s" in error for error in report["errors"])
    assert any("padded_tokens is smaller" in error for error in report["errors"])
    assert any("exceeds token budget" in error for error in report["errors"])
    assert any("exceeds configured limit" in error for error in report["errors"])


def test_allow_running_and_merged_model_contract(tmp_path):
    run = create_run(tmp_path / "run", status=False)
    assert VALIDATION.validate_run(run)["status"] == "failed"
    assert VALIDATION.validate_run(run, allow_running=True)["status"] == "passed"
    report = VALIDATION.validate_run(run, allow_running=True, require_merged=True)
    assert report["status"] == "failed"
    (run / "merged_model").mkdir()
    (run / "merged_model/config.json").write_text("{}")
    (run / "merged_model/model.safetensors").write_bytes(b"fake-model")
    report = VALIDATION.validate_run(run, allow_running=True, require_merged=True)
    assert report["status"] == "passed"
    assert report["artifacts"]["merged_model/model.safetensors"] == len(b"fake-model")


def test_validator_rejects_nested_nonfinite_metadata_and_missing_train_loss(tmp_path):
    run = create_run(tmp_path / "run")
    (run / "run_manifest.json").write_text(
        json.dumps({"package_versions": {}, "nested": {"metric": float("nan")}})
    )
    (run / "train_results.json").write_text("{}")
    report = VALIDATION.validate_run(run)
    assert report["status"] == "failed"
    assert any("$.nested.metric" in error for error in report["errors"])
    assert any("train_loss" in error for error in report["errors"])


def test_non_peft_run_requires_full_model_weights(tmp_path):
    run = create_run(tmp_path / "run")
    config_path = run / "config/resolved_training_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["use_peft"] = False
    config_path.write_text(yaml.safe_dump(config))
    report = VALIDATION.validate_run(run)
    assert report["status"] == "failed"
    assert any("full-model weights" in error for error in report["errors"])
    (run / "model.safetensors").write_bytes(b"fake-full-model")
    assert VALIDATION.validate_run(run)["status"] == "passed"


def test_final_evaluation_requires_finite_reward_artifact(tmp_path):
    run = create_run(tmp_path / "run")
    config_path = run / "config/resolved_training_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["do_eval"] = True
    config_path.write_text(yaml.safe_dump(config))
    assert VALIDATION.validate_run(run)["status"] == "failed"
    results = run / "eval_results.json"
    results.write_text(json.dumps({"eval_loss": 0.1, "eval_samples": 2}))
    assert VALIDATION.validate_run(run)["status"] == "failed"
    results.write_text(json.dumps({"eval_loss": 0.1, "eval_reward": 1.5, "eval_samples": 2}))
    report = VALIDATION.validate_run(run)
    assert report["status"] == "passed"
    assert report["evaluation"]["eval_reward"] == 1.5
