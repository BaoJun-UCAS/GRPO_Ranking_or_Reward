#!/usr/bin/env python3
"""Validate a completed GRPO run and summarize pipeline performance without loading a model."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import yaml


REPORT_VERSION = 1
QRM_SCORE_PATTERN = re.compile(
    r"qrm_score examples=(?P<examples>\d+) queue_s=(?P<queue>\S+) "
    r"inference_s=(?P<inference>\S+) batches=(?P<batches>\S+) "
    r"input_tokens=(?P<input_tokens>\S+) padded_tokens=(?P<padded_tokens>\S+)"
    r"(?: max_batch_examples=(?P<max_batch_examples>\S+)"
    r" max_batch_padded_tokens=(?P<max_batch_padded_tokens>\S+))?"
)
REQUIRED_TIMING_STAGES = (
    "rollout_total",
    "qrm_total",
    "external_sync_wait",
    "generation_score_total",
    "training_step_total",
    "policy_train_total",
)


def _finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _add_error(report, message):
    if message not in report["errors"]:
        report["errors"].append(message)


def _add_warning(report, message):
    if message not in report["warnings"]:
        report["warnings"].append(message)


def _require_file(report, run_dir, relative):
    path = run_dir / relative
    if not path.is_file() or path.stat().st_size == 0:
        _add_error(report, f"Missing or empty artifact: {relative}")
        return None
    report["artifacts"][relative] = path.stat().st_size
    return path


def _read_json(report, path, label):
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        _add_error(report, f"Invalid {label}: {error}")
        return None
    if not isinstance(value, dict):
        _add_error(report, f"Invalid {label}: top-level value must be an object")
        return None
    return value


def _read_yaml(report, path, label):
    if path is None:
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        _add_error(report, f"Invalid {label}: {error}")
        return None
    if not isinstance(value, dict):
        _add_error(report, f"Invalid {label}: top-level value must be a mapping")
        return None
    return value


def _validate_numeric_mapping(report, value, label, required=()):
    if value is None:
        return
    for key in required:
        if not _finite_number(value.get(key)):
            _add_error(report, f"{label} is missing finite metric {key!r}")

    def walk(item, path):
        if isinstance(item, dict):
            for key, nested in item.items():
                walk(nested, f"{path}.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                walk(nested, f"{path}[{index}]")
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and not math.isfinite(item):
            _add_error(report, f"Non-finite metric at {path} in {label}")

    walk(value, "$")


def _parse_status(report, run_dir, allow_running):
    path = run_dir / "RUN_STATUS"
    if not path.is_file():
        if not allow_running:
            _add_error(report, "Missing RUN_STATUS; use --allow-running only from the active launcher")
        return None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    status = values.get("status")
    report["run_status"] = values
    if status != "success":
        _add_error(report, f"RUN_STATUS is not success: {status or 'missing'}")
    return status


def _numeric_env(path):
    values = {}
    if path is None:
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=([0-9]+)", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


def _validate_training_state(report, state, config):
    if state is None:
        return [], {}
    expected_steps = config.get("max_steps") if config else None
    if not isinstance(expected_steps, int) or isinstance(expected_steps, bool) or expected_steps < 1:
        _add_error(report, f"Resolved max_steps must be a positive integer; got {expected_steps!r}")
    global_step = state.get("global_step")
    state_max_steps = state.get("max_steps")
    if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 1:
        _add_error(report, f"trainer_state global_step must be a positive integer; got {global_step!r}")
    if isinstance(expected_steps, int) and global_step != expected_steps:
        _add_error(report, f"Training ended at global_step={global_step}; resolved max_steps={expected_steps}")
    if not isinstance(state_max_steps, int) or isinstance(state_max_steps, bool) or state_max_steps < 1:
        _add_error(report, f"trainer_state max_steps must be a positive integer; got {state_max_steps!r}")
    elif state_max_steps != expected_steps:
        _add_error(report, f"trainer_state max_steps={state_max_steps} disagrees with resolved max_steps={expected_steps}")

    history = state.get("log_history")
    if not isinstance(history, list) or not history:
        _add_error(report, "trainer_state log_history must be a non-empty list")
        return [], {}
    numeric_series = {}
    for record_index, record in enumerate(history):
        if not isinstance(record, dict):
            _add_error(report, f"log_history[{record_index}] is not an object")
            continue
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value):
                    _add_error(report, f"Non-finite metric {key!r} in log_history[{record_index}]")
                    continue
                numeric_series.setdefault(key, []).append(float(value))
    if "loss" not in numeric_series:
        _add_error(report, "No finite training loss was logged")
    report["training"] = {
        "global_step": global_step,
        "expected_steps": expected_steps,
        "history_records": len(history),
        "train_runtime": (numeric_series.get("train_runtime") or [None])[-1],
        "train_steps_per_second": (numeric_series.get("train_steps_per_second") or [None])[-1],
    }
    return history, numeric_series


def _validate_timing_record(report, record, record_index, rank_count):
    timing_keys = [key for key in record if isinstance(key, str) and key.startswith("timing/")]
    for key in timing_keys:
        value = record[key]
        if not _finite_number(value):
            _add_error(report, f"Timing metric {key!r} in log_history[{record_index}] is not finite")
        elif value < 0:
            _add_error(report, f"Timing metric {key!r} in log_history[{record_index}] is negative")

    for key in timing_keys:
        if not key.endswith("_max_s") or not _finite_number(record[key]):
            continue
        stem = key[:-6]
        minimum_key = f"{stem}_min_s"
        spread_key = f"{stem}_rank_spread_s"
        if minimum_key not in record or spread_key not in record:
            _add_error(report, f"Incomplete timing summary for {stem} in log_history[{record_index}]")
            continue
        minimum, maximum, spread = record[minimum_key], record[key], record[spread_key]
        if not all(_finite_number(value) for value in (minimum, maximum, spread)):
            continue
        tolerance = 1e-4 + 1e-4 * max(abs(minimum), abs(maximum), 1.0)
        if minimum > maximum + tolerance:
            _add_error(report, f"Timing minimum exceeds maximum for {stem} in log_history[{record_index}]")
        if abs(spread - (maximum - minimum)) > tolerance:
            _add_error(report, f"Timing rank spread is inconsistent for {stem} in log_history[{record_index}]")
        for rank in range(rank_count):
            rank_key = f"{stem}_rank{rank}_s"
            rank_value = record.get(rank_key)
            if not _finite_number(rank_value):
                _add_error(report, f"Missing finite {rank_key} in log_history[{record_index}]")
            elif not minimum - tolerance <= rank_value <= maximum + tolerance:
                _add_error(report, f"{rank_key} lies outside min/max in log_history[{record_index}]")


def _mean_series(series, key):
    values = series.get(key, [])
    return sum(values) / len(values) if values else None


def _validate_timings(report, history, numeric_series, config, accelerate):
    profiling = bool(config and config.get("profile_stage_timings"))
    rank_count = accelerate.get("num_processes", 1) if accelerate else 1
    if not isinstance(rank_count, int) or rank_count < 1:
        _add_error(report, f"Invalid Accelerate num_processes: {rank_count!r}")
        rank_count = 1
    for index, record in enumerate(history):
        if isinstance(record, dict) and any(str(key).startswith("timing/") for key in record):
            _validate_timing_record(report, record, index, rank_count)

    if not profiling:
        if any(key.startswith("timing/") for key in numeric_series):
            _add_warning(report, "Timing metrics are present although profile_stage_timings is disabled")
        return

    for stage in REQUIRED_TIMING_STAGES:
        for suffix in ("max_s", "min_s", "rank_spread_s"):
            key = f"timing/{stage}_{suffix}"
            if key not in numeric_series:
                _add_error(report, f"Required timing metric is missing: {key}")
        for rank in range(rank_count):
            key = f"timing/{stage}_rank{rank}_s"
            if key not in numeric_series:
                _add_error(report, f"Required timing metric is missing: {key}")

    bottleneck_means = {
        stage: _mean_series(numeric_series, f"timing/{stage}_max_s") for stage in REQUIRED_TIMING_STAGES
    }
    per_rank = {}
    for rank in range(rank_count):
        values = {
            stage: _mean_series(numeric_series, f"timing/{stage}_rank{rank}_s")
            for stage in REQUIRED_TIMING_STAGES
        }
        total = values.get("training_step_total")
        if total is None or total <= 0:
            continue
        rollout = values.get("rollout_total") or 0.0
        reward = values.get("qrm_total") or 0.0
        policy = values.get("policy_train_total") or 0.0
        other = total - rollout - reward - policy
        tolerance = max(0.05 * total, 1e-3)
        if other < -tolerance:
            _add_warning(
                report,
                f"Rank {rank} measured rollout+QRM+policy exceeds training-step total by {-other:.4f}s on average",
            )
        other = max(0.0, other)
        per_rank[str(rank)] = {
            "mean_seconds": {
                "training_step_total": total,
                "rollout": rollout,
                "qrm": reward,
                "policy": policy,
                "other": other,
                "external_sync_wait": values.get("external_sync_wait"),
            },
            "share_of_training_step": {
                "rollout": rollout / total,
                "qrm": reward / total,
                "policy": policy / total,
                "other": other / total,
            },
        }
    report["timing"] = {
        "logged_steps": len(numeric_series.get("timing/training_step_total_max_s", [])),
        "bottleneck_mean_seconds": bottleneck_means,
        "rank_spread_mean_seconds": {
            stage: _mean_series(numeric_series, f"timing/{stage}_rank_spread_s")
            for stage in REQUIRED_TIMING_STAGES
        },
        "per_rank": per_rank,
    }


def _parse_number(value, integer=False):
    if value is None or value == "unknown":
        return None
    parsed = int(value) if integer else float(value)
    if not math.isfinite(parsed):
        raise ValueError(value)
    return parsed


def _validate_qrm_log(report, path, config, env_values):
    if path is None:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(QRM_SCORE_PATTERN.finditer(text))
    qrm_required = bool(config and "qrm_server" in (config.get("reward_funcs") or []))
    strict_stats = bool(config and config.get("profile_stage_timings"))
    if qrm_required and not matches:
        _add_error(report, "QRM log contains no qrm_score inference records")
        return
    records = []
    for index, match in enumerate(matches):
        try:
            record = {
                "examples": _parse_number(match.group("examples"), integer=True),
                "queue_s": _parse_number(match.group("queue")),
                "inference_s": _parse_number(match.group("inference")),
                "batches": _parse_number(match.group("batches"), integer=True),
                "input_tokens": _parse_number(match.group("input_tokens"), integer=True),
                "padded_tokens": _parse_number(match.group("padded_tokens"), integer=True),
                "max_batch_examples": _parse_number(match.group("max_batch_examples"), integer=True),
                "max_batch_padded_tokens": _parse_number(match.group("max_batch_padded_tokens"), integer=True),
            }
        except (TypeError, ValueError):
            _add_error(report, f"Invalid numeric QRM statistics in record {index}")
            continue
        required = ("examples", "queue_s", "inference_s", "batches", "input_tokens", "padded_tokens")
        if strict_stats and any(record[key] is None for key in required):
            _add_error(report, f"Incomplete QRM statistics in record {index}")
            continue
        if record["examples"] is not None and record["examples"] < 1:
            _add_error(report, f"QRM record {index} has no examples")
        if record["batches"] is not None and record["examples"] is not None:
            if not 1 <= record["batches"] <= record["examples"]:
                _add_error(report, f"QRM record {index} has invalid batch count")
        if record["input_tokens"] is not None and record["padded_tokens"] is not None:
            if record["padded_tokens"] < record["input_tokens"]:
                _add_error(report, f"QRM record {index} padded_tokens is smaller than input_tokens")
        if any(record[key] is not None and record[key] < 0 for key in ("queue_s", "inference_s")):
            _add_error(report, f"QRM record {index} contains a negative duration")
        token_budget = env_values.get("QRM_MAX_BATCH_TOKENS")
        max_batch_tokens = record["max_batch_padded_tokens"]
        if token_budget and max_batch_tokens is not None and max_batch_tokens > token_budget:
            _add_error(
                report,
                f"QRM record {index} max padded batch {max_batch_tokens} exceeds token budget {token_budget}",
            )
        batch_limit = env_values.get("REWARD_BATCH_SIZE")
        max_batch_examples = record["max_batch_examples"]
        if batch_limit and max_batch_examples is not None and max_batch_examples > batch_limit:
            _add_error(
                report,
                f"QRM record {index} max batch size {max_batch_examples} exceeds configured limit {batch_limit}",
            )
        records.append(record)
    if strict_stats and records and any(
        record["max_batch_examples"] is None or record["max_batch_padded_tokens"] is None for record in records
    ):
        _add_error(report, "QRM log lacks per-batch maxima required by the current validation protocol")
    if not records:
        return
    input_tokens = sum(record["input_tokens"] or 0 for record in records)
    padded_tokens = sum(record["padded_tokens"] or 0 for record in records)
    inference_seconds = sum(record["inference_s"] or 0.0 for record in records)
    report["qrm"] = {
        "requests": len(records),
        "examples": sum(record["examples"] or 0 for record in records),
        "batches": sum(record["batches"] or 0 for record in records),
        "input_tokens": input_tokens,
        "padded_tokens": padded_tokens,
        "padding_efficiency": input_tokens / padded_tokens if padded_tokens else None,
        "inference_seconds": inference_seconds,
        "input_tokens_per_second": input_tokens / inference_seconds if inference_seconds else None,
        "max_batch_examples": max((record["max_batch_examples"] or 0) for record in records),
        "max_batch_padded_tokens": max((record["max_batch_padded_tokens"] or 0) for record in records),
    }


def _validate_weights(report, run_dir, config, require_merged):
    use_peft = bool(config and config.get("use_peft"))
    if use_peft:
        _read_json(
            report,
            _require_file(report, run_dir, "adapter_config.json"),
            "adapter_config.json",
        )
        weights = [run_dir / "adapter_model.safetensors", run_dir / "adapter_model.bin"]
        if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
            _add_error(report, "Missing non-empty adapter weights")
        else:
            for path in weights:
                if path.is_file() and path.stat().st_size > 0:
                    report["artifacts"][path.name] = path.stat().st_size
                    break
    else:
        weights = list(run_dir.glob("model*.safetensors")) + list(run_dir.glob("pytorch_model*.bin"))
        if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
            _add_error(report, "Missing non-empty full-model weights")
        else:
            selected = next(path for path in weights if path.is_file() and path.stat().st_size > 0)
            report["artifacts"][selected.name] = selected.stat().st_size
    if require_merged:
        merged = run_dir / "merged_model"
        _read_json(
            report,
            _require_file(report, run_dir, "merged_model/config.json"),
            "merged_model/config.json",
        )
        weights = list(merged.glob("*.safetensors")) + list(merged.glob("*.bin"))
        if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
            _add_error(report, "Merged model has no non-empty weights")
        else:
            selected = next(path for path in weights if path.is_file() and path.stat().st_size > 0)
            report["artifacts"][str(selected.relative_to(run_dir))] = selected.stat().st_size


def validate_run(run_dir, allow_running=False, require_merged=False):
    run_dir = Path(run_dir).expanduser().resolve()
    report = {
        "report_version": REPORT_VERSION,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "status": "failed",
        "errors": [],
        "warnings": [],
        "artifacts": {},
    }
    if not run_dir.is_dir():
        _add_error(report, f"Run directory does not exist: {run_dir}")
        return report

    _parse_status(report, run_dir, allow_running)
    config = _read_yaml(
        report,
        _require_file(report, run_dir, "config/resolved_training_config.yaml"),
        "resolved training config",
    )
    accelerate = _read_yaml(
        report,
        _require_file(report, run_dir, "config/accelerate_config.yaml"),
        "Accelerate config",
    )
    run_env = _require_file(report, run_dir, "run.env")
    manifest = _read_json(
        report, _require_file(report, run_dir, "run_manifest.json"), "run_manifest.json"
    )
    state = _read_json(report, _require_file(report, run_dir, "trainer_state.json"), "trainer_state.json")
    train_results = _read_json(
        report, _require_file(report, run_dir, "train_results.json"), "train_results.json"
    )
    _validate_numeric_mapping(report, manifest, "run_manifest.json")
    _validate_numeric_mapping(report, train_results, "train_results.json", required=("train_loss",))
    training_log = _require_file(report, run_dir, "logs/training.log")
    qrm_log = _require_file(report, run_dir, "logs/qrm_server.log")
    _require_file(report, run_dir, "logs/vllm_server.log")
    if training_log is not None and "Traceback (most recent call last)" in training_log.read_text(
        encoding="utf-8", errors="replace"
    ):
        _add_error(report, "Training log contains a Python traceback")

    _validate_weights(report, run_dir, config, require_merged)
    history, numeric_series = _validate_training_state(report, state, config)
    _validate_timings(report, history, numeric_series, config, accelerate)
    _validate_qrm_log(report, qrm_log, config, _numeric_env(run_env))
    report["status"] = "passed" if not report["errors"] else "failed"
    return report


def write_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, output_path)


def _print_summary(report):
    print(f"Run validation {report['status'].upper()}: {report['run_dir']}")
    training = report.get("training") or {}
    if training:
        print(
            f"Training: step {training.get('global_step')}/{training.get('expected_steps')}; "
            f"records={training.get('history_records')}; steps/s={training.get('train_steps_per_second')}"
        )
    timing = report.get("timing") or {}
    for rank, values in timing.get("per_rank", {}).items():
        seconds = values["mean_seconds"]
        shares = values["share_of_training_step"]
        print(
            f"Rank {rank}: total={seconds['training_step_total']:.3f}s "
            f"rollout={shares['rollout']:.1%} qrm={shares['qrm']:.1%} "
            f"policy={shares['policy']:.1%} other={shares['other']:.1%}"
        )
    qrm = report.get("qrm") or {}
    if qrm:
        efficiency = qrm.get("padding_efficiency")
        efficiency_text = f"{efficiency:.1%}" if _finite_number(efficiency) else "unavailable"
        print(
            f"QRM: requests={qrm['requests']} examples={qrm['examples']} batches={qrm['batches']} "
            f"padding_efficiency={efficiency_text}"
        )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Completed run directory")
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Do not require RUN_STATUS; intended only for the launcher's pre-success validation",
    )
    parser.add_argument("--require-merged", action="store_true", help="Require merged_model config and weights")
    parser.add_argument("--report", help="Report path; defaults to RUN_DIR/validation_report.json")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args(argv)
    report = validate_run(args.run_dir, allow_running=args.allow_running, require_merged=args.require_merged)
    output = Path(args.report).expanduser() if args.report else Path(args.run_dir).expanduser() / "validation_report.json"
    if not args.no_write_report and Path(args.run_dir).expanduser().is_dir():
        write_report(report, output)
        print(f"Report: {output.resolve()}")
    _print_summary(report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
