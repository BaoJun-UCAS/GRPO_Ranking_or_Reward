"""CPU-only tests for training metric plots."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/plot_training_metrics.py"


def write_state(path, offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "global_step": 3,
        "max_steps": 3,
        "log_history": [
            {"step": 1, "loss": 1.0 + offset, "reward": 0.1, "learning_rate": 1e-6},
            {"step": 2, "loss": 0.8 + offset, "reward": 0.2, "kl": 0.01,
             "rewards/QRM/mean": 0.2, "completions/mean_length": 20},
            {"step": 2, "loss": 0.7 + offset},
            {"step": 3, "loss": 0.5 + offset, "reward": 0.3, "grad_norm": 0.4,
             "completions/clipped_ratio": 0.25, "num_tokens": 100},
            {"step": 3, "train_loss": 0.75, "train_runtime": 12.0},
            {"step": 3, "ignored": "not numeric"},
        ],
    }), encoding="utf-8")


def run_plot(*arguments):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, arguments)],
        env={"PATH": str(Path(sys.executable).parent), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_single_run_writes_real_png_and_raw_csv(tmp_path):
    run_dir = tmp_path / "run"
    write_state(run_dir / "trainer_state.json")
    output = tmp_path / "metrics.png"
    result = run_plot(run_dir, "--output", output, "--smooth", "2")
    assert result.returncode == 0, result.stderr
    assert output.read_bytes().startswith(b"\x89PNG")
    rows = list(csv.DictReader(output.with_suffix(".csv").open()))
    loss = [row for row in rows if row["metric"] == "loss"]
    assert [(float(row["step"]), float(row["value"])) for row in loss] == [(1, 1.0), (2, 0.7), (3, 0.5)]


def test_latest_checkpoint_and_metric_listing(tmp_path):
    run_dir = tmp_path / "run"
    write_state(run_dir / "checkpoint-2/trainer_state.json")
    write_state(run_dir / "checkpoint-10/trainer_state.json")
    result = run_plot(run_dir, "--list-metrics")
    assert result.returncode == 0, result.stderr
    assert "checkpoint-10/trainer_state.json" in result.stdout
    assert "reward" in result.stdout


def test_multiple_runs_and_exact_metric_validation(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_state(first / "trainer_state.json")
    write_state(second / "trainer_state.json", offset=1)
    output = tmp_path / "comparison.svg"
    result = run_plot(first, second, "--label", "baseline", "--label", "ranking",
                      "--metrics", "loss", "reward", "--output", output)
    assert result.returncode == 0, result.stderr
    assert output.read_text().lstrip().startswith("<?xml")
    failure = run_plot(first, "--metrics", "not-a-metric", "--output", tmp_path / "bad.png")
    assert failure.returncode == 1
    assert "Metrics not found" in failure.stderr


@pytest.mark.parametrize("history", [None, [], [{"step": 1, "loss": "bad"}]])
def test_invalid_or_empty_history_fails(tmp_path, history):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {} if history is None else {"log_history": history}
    (run_dir / "trainer_state.json").write_text(json.dumps(payload))
    result = run_plot(run_dir)
    assert result.returncode == 1
    assert not (run_dir / "training_metrics.png").exists()
