#!/usr/bin/env python3
"""Plot Hugging Face Trainer/GRPO metrics from one or more completed run directories."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUPS = (
    ("Loss", ("loss", "eval_loss")),
    ("Reward", ("reward", "reward_std", "rewards/")),
    ("Policy update", ("kl", "clip_ratio/")),
    ("Completion length", ("completions/mean_length", "completions/min_length", "completions/max_length",
                           "completions/mean_terminated_length")),
    ("Completion quality", ("completions/clipped_ratio", "frac_reward_zero_std")),
    ("Gradient norm", ("grad_norm",)),
    ("Learning rate", ("learning_rate",)),
    ("Tokens processed", ("num_tokens",)),
)


def _checkpoint_number(path):
    match = re.fullmatch(r"checkpoint-(\d+)", path.parent.name)
    return int(match.group(1)) if match else -1


def resolve_state_path(candidate):
    candidate = Path(candidate).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Training run or trainer_state.json not found: {candidate}")
    direct = candidate / "trainer_state.json"
    if direct.is_file():
        return direct
    checkpoint_states = list(candidate.glob("checkpoint-*/trainer_state.json"))
    if checkpoint_states:
        return max(checkpoint_states, key=_checkpoint_number)
    raise FileNotFoundError(f"No trainer_state.json under {candidate}")


def latest_run():
    output_root = Path(os.environ.get("GRPO_OUTPUT_ROOT", PROJECT_ROOT / "grpo_runs")).expanduser()
    candidates = list(output_root.glob("*/trainer_state.json")) + list(
        output_root.glob("*/checkpoint-*/trainer_state.json")
    )
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json under {output_root}; pass a run directory explicitly")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_run(candidate, label=None):
    state_path = resolve_state_path(candidate)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {state_path}: {error}") from error
    history = state.get("log_history")
    if not isinstance(history, list):
        raise ValueError(f"Missing log_history list in {state_path}")

    series = {}
    for record in history:
        if not isinstance(record, dict):
            continue
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step):
            continue
        for metric, value in record.items():
            if metric in {"step", "epoch"} or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(value):
                series.setdefault(metric, {})[float(step)] = float(value)

    if not series:
        raise ValueError(f"No finite step-wise numeric metrics in {state_path}")
    run_dir = state_path.parent.parent if state_path.parent.name.startswith("checkpoint-") else state_path.parent
    summary = {
        key: state.get(key)
        for key in ("global_step", "max_steps", "best_metric", "best_global_step", "best_model_checkpoint")
        if state.get(key) is not None
    }
    for record in reversed(history):
        if isinstance(record, dict) and "train_runtime" in record:
            summary.update({key: record[key] for key in (
                "train_loss", "train_runtime", "train_samples_per_second", "train_steps_per_second"
            ) if key in record})
            break
    return {
        "label": label or run_dir.name,
        "run_dir": str(run_dir),
        "state_path": str(state_path),
        "series": series,
        "summary": summary,
    }


def moving_average(points, window):
    if window <= 1:
        return points
    values = [value for _, value in points]
    smoothed = []
    for index, (step, _) in enumerate(points):
        start = max(0, index - window + 1)
        subset = values[start : index + 1]
        smoothed.append((step, sum(subset) / len(subset)))
    return smoothed


def metric_matches(metric, selectors):
    return any(metric == selector or (selector.endswith("/") and metric.startswith(selector)) for selector in selectors)


def select_groups(runs, requested_metrics=None):
    available = sorted({metric for run in runs for metric in run["series"]})
    if requested_metrics:
        missing = [metric for metric in requested_metrics if metric not in available]
        if missing:
            raise ValueError(f"Metrics not found: {', '.join(missing)}. Use --list-metrics to inspect the run.")
        return [("Selected metrics", tuple(requested_metrics))]
    groups = []
    used = set()
    for title, selectors in DEFAULT_GROUPS:
        metrics = tuple(metric for metric in available if metric_matches(metric, selectors))
        if metrics:
            groups.append((title, metrics))
            used.update(metrics)
    remaining = tuple(metric for metric in available if metric not in used and not metric.startswith("train_"))
    if remaining:
        groups.append(("Other metrics", remaining))
    return groups


def write_csv(path, runs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.writer(stream)
        writer.writerow(("run", "run_dir", "state_path", "step", "metric", "value"))
        for run in runs:
            for metric in sorted(run["series"]):
                for step, value in sorted(run["series"][metric].items()):
                    writer.writerow((run["label"], run["run_dir"], run["state_path"], step, metric, value))
    os.replace(temporary, path)


def plot(runs, groups, output_path, smooth, title, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not groups:
        raise ValueError("No plottable metrics were selected")
    columns = 2 if len(groups) > 1 else 1
    rows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 3.5 * rows), squeeze=False)
    axes = axes.ravel()
    multiple_runs = len(runs) > 1
    for axis, (group_title, metrics) in zip(axes, groups):
        for run in runs:
            for metric in metrics:
                values = run["series"].get(metric)
                if not values:
                    continue
                points = moving_average(sorted(values.items()), smooth)
                label = f"{run['label']}: {metric}" if multiple_runs else metric
                axis.plot([point[0] for point in points], [point[1] for point in points],
                          marker="o" if len(points) <= 20 else None, markersize=3, linewidth=1.4, label=label)
        axis.set_title(group_title)
        axis.set_xlabel("Optimizer step")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, loc="best")
        if group_title == "Completion quality":
            axis.axhline(0, color="black", linewidth=0.7, alpha=0.4)
    for axis in axes[len(groups) :]:
        axis.set_visible(False)

    heading = title or (runs[0]["label"] if len(runs) == 1 else "Training run comparison")
    fig.suptitle(heading, fontsize=15)
    summaries = []
    for run in runs:
        details = ", ".join(f"{key}={value}" for key, value in run["summary"].items()
                            if key in {"global_step", "max_steps", "train_loss", "train_runtime"})
        summaries.append(f"{run['label']}: {details}" if details else run["label"])
    fig.text(0.01, 0.005, " | ".join(summaries), fontsize=7, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_format = output_path.suffix.lower().lstrip(".")
    if image_format not in {"png", "pdf", "svg"}:
        raise ValueError("Output extension must be .png, .pdf or .svg")
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=f".{image_format}", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        fig.savefig(temporary, format=image_format, dpi=dpi)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="Run directories or trainer_state.json files; latest run by default")
    parser.add_argument("--label", action="append", help="Display label for each input, in the same order")
    parser.add_argument("--output", help="Image path (.png/.pdf/.svg); defaults to RUN_DIR/training_metrics.png")
    parser.add_argument("--csv", dest="csv_path", help="Long-form metric CSV path; defaults beside the image")
    parser.add_argument("--metrics", nargs="+", help="Plot exact metric names in a single panel")
    parser.add_argument("--list-metrics", action="store_true", help="Print available metrics and exit")
    parser.add_argument("--smooth", type=int, default=1, help="Trailing moving-average window, default 1 (raw)")
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)
    if args.smooth < 1 or args.dpi < 1:
        parser.error("--smooth and --dpi must be positive integers")
    inputs = args.inputs or [latest_run()]
    if args.label and len(args.label) != len(inputs):
        parser.error("Repeat --label exactly once per input")
    runs = [load_run(path, args.label[index] if args.label else None) for index, path in enumerate(inputs)]
    if args.list_metrics:
        for run in runs:
            print(f"[{run['label']}] {run['state_path']}")
            for metric in sorted(run["series"]):
                print(f"  {metric}: {len(run['series'][metric])} point(s)")
        return 0

    output = Path(args.output).expanduser() if args.output else Path(runs[0]["run_dir"]) / "training_metrics.png"
    csv_path = Path(args.csv_path).expanduser() if args.csv_path else output.with_suffix(".csv")
    groups = select_groups(runs, args.metrics)
    plot(runs, groups, output, args.smooth, args.title, args.dpi)
    write_csv(csv_path, runs)
    print(f"Plot: {output.resolve()}")
    print(f"Data: {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
