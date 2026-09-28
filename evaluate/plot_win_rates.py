"""Offline plot of a validated bootstrap result, also used by legacy batch plots."""

import argparse
import json
import math
from pathlib import Path
import re


def extract_checkpoint_from_filename(filename):
    match = re.search(r"checkpoint[-_]?(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_model_name(path):
    name = Path(path).stem.lower()
    return next((kind for kind in ("ranking", "regular", "baseline") if kind in name), name)


def load_win_rate_data(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") == "invalid":
        raise ValueError(f"Refusing to plot invalid evaluation: {path}")
    config = payload["bootstrap_config"]
    result = {f"model{i}_name": parse_model_name(config[f"model{i}_path"]) for i in (1, 2)}
    observed = (payload.get("validation") or {}).get("observed", {})
    for metric in ("survey", "overall"):
        analysis = payload["bootstrap_analysis"][f"{metric}_winner_analysis"]
        if analysis["iterations_with_valid_comparisons"] < 1:
            raise ValueError(f"No valid {metric} comparisons in {path}")
        for model in (1, 2):
            stats = analysis[f"model{model}_win_rate_distribution"]
            mean = observed.get(f"{metric}_analysis", {}).get(f"model{model}_win_rate", stats["mean"])
            values = [mean, *stats["ci_95"]]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError(f"Invalid statistics in {path}")
            result[f"{metric}_mean_model{model}"] = mean
            result[f"{metric}_ci_model{model}"] = stats["ci_95"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", help="Result JSON, not judge_cache.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = load_win_rate_data(args.result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("overall", "survey")):
        means = [data[f"{metric}_mean_model{i}"] for i in (1, 2)]
        ax.bar([0, 1], means)
        # Intervals need not contain the observed estimate for small bootstrap B.
        for i in (1, 2):
            low, high = data[f"{metric}_ci_model{i}"]
            ax.vlines(i - 1, low, high, color="black")
        ax.set_xticks([0, 1], ["Model 1", "Model 2"])
        ax.set_ylim(0, 1)
        ax.set_title(metric.title() + " win rate (95% bootstrap CI)")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
