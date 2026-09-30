"""Reproduce the descriptive delta/c analysis; no GPU or training required.

Run with the training environment (NumPy, PyYAML, Matplotlib installed):
python scripts/analyze_advantage_rewards.py \
  --run-dir grpo_runs/qwen3-1.7b-grpo-p2048-c3072-800step-20260928-193332

This deliberately requires one complete training rollout per optimizer step,
complete rank files and agreement with logged reward metrics. It does not infer
reward-model noise or predictive training quality from the reward distribution.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


def quantiles(values):
    levels = [0, .01, .05, .10, .20, .25, .50, .75, .80, .90, .95, .99, 1]
    return {f"p{100*q:g}": float(v) for q, v in zip(levels, np.quantile(values, levels))}


def write_csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def analyze(run_dir, out_dir):
    config_path = run_dir / "config/resolved_training_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    accel = yaml.safe_load((run_dir / "config/accelerate_config.yaml").read_text())
    state_path = run_dir / "trainer_state.json"
    state = json.loads(state_path.read_text())
    steps = state["global_step"]
    assert steps == state["max_steps"] == cfg["max_steps"], "Incomplete training"
    assert cfg["num_iterations"] == 1, "This audit expects one rollout per step"
    assert "status=success" in (run_dir / "RUN_STATUS").read_text()
    group_size = cfg["num_generations"]
    ranks = accel["num_processes"]
    batch_size = cfg["generation_batch_size"]
    assert batch_size % ranks == 0 and batch_size % group_size == 0
    n_local = batch_size // ranks
    logs = {row["step"]: row for row in state["log_history"] if "reward_std" in row}
    assert set(logs) == set(range(1, steps + 1))
    pattern = re.compile(r"reward_data_(?:train_)?step_(\d+)_micro_(\d+)_proc_(\d+)(?:_[a-f0-9]+)?\.json")
    indexed = {}
    for path in sorted((run_dir / "reward_data").glob("*.json")):
        if "_eval_" in path.name:
            continue
        match = pattern.fullmatch(path.name)
        assert match, f"Unrecognized training file: {path.name}"
        step, micro, rank = map(int, match.groups())
        assert (step, rank) not in indexed, "More than one rollout per step/rank"
        indexed[step, rank] = (micro, path)
    assert set(indexed) == {(s, r) for s in range(steps) for r in range(ranks)}

    groups, group_steps, unique_prompts = [], [], Counter()
    ledger = []
    reward_errors, std_errors = [], []
    combined_hash = hashlib.sha256()
    duplicate_rank_vectors = 0
    for step in range(steps):
        rows, rank_rewards, micros = [], [], []
        for rank in range(ranks):
            micro, path = indexed[step, rank]
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            combined_hash.update(f"{path.name}\0{digest}\n".encode())
            local = json.loads(raw)
            assert len(local) == n_local
            assert [r["rollout_index"] for r in local] == list(range(n_local))
            assert all(r["training_step"] == step and r["process_index"] == rank for r in local)
            rows.extend(local)
            rank_rewards.append([r["reward"] for r in local])
            micros.append(micro)
        assert len(set(micros)) == 1
        for i in range(ranks):
            for j in range(i):
                duplicate_rank_vectors += int(rank_rewards[i] == rank_rewards[j])
        rewards = np.array([r["reward"] for r in rows], dtype=np.float64).reshape(-1, group_size)
        assert np.isfinite(rewards).all()
        for i in range(0, batch_size, group_size):
            prompt = rows[i]["prompt"]
            assert all(r["prompt"] == prompt for r in rows[i:i + group_size]), "Mixed prompt group"
            unique_prompts[hashlib.sha256(prompt.encode()).hexdigest()] += 1
        mean_error = abs(float(rewards.mean()) - logs[step + 1]["reward"])
        std_error = abs(float(rewards.std(axis=1, ddof=1).mean()) - logs[step + 1]["reward_std"])
        assert mean_error < 2e-6 and std_error < 2e-6, (
            f"Saved rewards disagree with trainer log at step {step + 1}: {mean_error}, {std_error}"
        )
        reward_errors.append(mean_error)
        std_errors.append(std_error)
        groups.append(rewards)
        group_steps.extend([step + 1] * len(rewards))
    rewards = np.concatenate(groups)
    group_steps = np.array(group_steps)
    difference = rewards[:, :, None] - rewards[:, None, :]
    first, second = np.triu_indices(group_size, 1)
    pair_gap = np.abs(difference[:, first, second])
    ranges = np.ptp(rewards, axis=1)
    group_std = rewards.std(axis=1, ddof=1)
    grpo = (rewards - rewards.mean(axis=1, keepdims=True)) / (group_std[:, None] + 1e-4)
    delta_candidates = [0, .005, .01, .015, .02, .03, .04, .05]
    c_candidates = [.04, .05, .08, .10, .15, .20, .30]
    scopes = {"all": np.ones(len(rewards), dtype=bool)}
    for start in range(1, steps + 1, 200):
        end = min(start + 199, steps)
        scopes[f"steps_{start:03d}_{end:03d}"] = (group_steps >= start) & (group_steps <= end)
    summaries, candidates = {}, []
    for scope, mask in scopes.items():
        gap = pair_gap[mask]
        summaries[scope] = {
            "groups": int(mask.sum()), "unordered_pairs": int(gap.size),
            "reward_quantiles": quantiles(rewards[mask]),
            "pair_gap_quantiles": quantiles(gap),
            "group_std_quantiles": quantiles(group_std[mask]),
            "group_range_quantiles": quantiles(ranges[mask]),
            "grpo_advantage_rms": float(np.sqrt(np.mean(grpo[mask] ** 2))),
            "grpo_mean_absolute_advantage": float(np.abs(grpo[mask]).mean()),
        }
        for delta in delta_candidates:
            for c in c_candidates:
                diff = difference[mask]
                advantage = (np.sign(diff) * np.clip((np.abs(diff) - delta) / c, 0, 1)).sum(2) / (group_size - 1)
                dead = float((gap <= delta).mean())
                saturated = float((gap >= delta + c).mean())
                rms = float(np.sqrt(np.mean(advantage ** 2)))
                assert np.max(np.abs(advantage.sum(1))) < 1e-12
                assert np.max(np.abs(advantage)) <= 1
                candidates.append({
                    "scope": scope, "delta": delta, "c": c,
                    "saturation_threshold": delta + c,
                    "dead_pair_fraction": dead,
                    "linear_pair_fraction": 1 - dead - saturated,
                    "saturated_pair_fraction": saturated,
                    "all_zero_group_fraction": float((ranges[mask] <= delta).mean()),
                    "zero_response_fraction": float((np.abs(advantage) < 1e-12).mean()),
                    "advantage_rms": rms,
                    "advantage_rms_ratio_to_grpo": rms / summaries[scope]["grpo_advantage_rms"],
                    "mean_absolute_advantage": float(np.abs(advantage).mean()),
                    "mean_group_max_absolute_advantage": float(np.abs(advantage).max(1).mean()),
                })
    delta, c = .01, .08
    selected_adv = (np.sign(difference) * np.clip((np.abs(difference) - delta) / c, 0, 1)).sum(2) / (group_size - 1)
    for step in range(1, steps + 1):
        mask = group_steps == step
        gap = pair_gap[mask]
        ledger.append({
            "optimizer_step": step, "reward_mean": float(rewards[mask].mean()),
            "mean_group_std": float(group_std[mask].mean()),
            "pair_p20": float(np.quantile(gap, .2)),
            "pair_median": float(np.median(gap)), "pair_p90": float(np.quantile(gap, .9)),
            "dead_fraction_delta_001": float((gap <= delta).mean()),
            "saturated_fraction_delta_001_c_008": float((gap >= delta + c).mean()),
            "all_zero_group_fraction_delta_001": float((ranges[mask] <= delta).mean()),
            "advantage_rms_delta_001_c_008": float(np.sqrt(np.mean(selected_adv[mask] ** 2))),
        })
    report = {
        "run_dir": str(run_dir.resolve()), "complete_optimizer_steps": steps,
        "audit": {
            "training_files": len(indexed), "responses": int(rewards.size),
            "prompt_group_occurrences": len(rewards), "distinct_prompt_texts": len(unique_prompts),
            "group_size": group_size, "ranks": ranks,
            "identical_reward_vectors_between_ranks": duplicate_rank_vectors,
            "max_reward_log_absolute_error": max(reward_errors),
            "max_group_std_log_absolute_error": max(std_errors),
            "all_group_prompts_and_metadata_match": True,
            "training_file_hash_ledger_sha256": combined_hash.hexdigest(),
            "resolved_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "trainer_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "selection": {
            "delta": delta, "c": c,
            "rationale": "Fixed proposal from the original 800-step Qwen3/QRM/UltraChat run: suppress about the lowest 20% of pair gaps and saturate about the highest 10%. Inspect this run's measured fractions before reusing it.",
            "not_noise_calibrated": True, "not_a_validation_of_training_quality": True,
            "scope": "Historical studentization policy rollouts with this reward model, weight and data; new-policy reward distributions may change.",
        },
        "scopes": summaries, "candidates": candidates,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(out_dir / "candidate_parameters.csv", candidates)
    write_csv(out_dir / "per_step_statistics.csv", ledger)
    np.savez_compressed(out_dir / "group_rewards.npz", rewards=rewards, optimizer_step=group_steps)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    sorted_gap = np.sort(pair_gap.ravel())
    indices = np.linspace(0, sorted_gap.size - 1, 6000).astype(int)
    axs[0, 0].plot(sorted_gap[indices], (indices + 1) / sorted_gap.size)
    axs[0, 0].axvline(.01, color="tab:green", linestyle="--", label="delta = 0.01")
    axs[0, 0].axvline(.09, color="tab:red", linestyle="--", label="delta + c = 0.09")
    axs[0, 0].set(xlim=(0, .25), ylim=(0, 1), xlabel="Absolute within-prompt reward difference", ylabel="Empirical CDF", title=f"{pair_gap.size:,} pair gaps nested in {len(rewards):,} groups")
    axs[0, 0].legend()
    options = [(.005, .08), (.01, .05), (.01, .08), (.015, .08), (.02, .2)]
    chosen = [next(r for r in candidates if r["scope"] == "all" and r["delta"] == d and r["c"] == c_) for d, c_ in options]
    bottom = np.zeros(len(chosen))
    for key, label, color in [("dead_pair_fraction", "Zero", "#a6a6a6"), ("linear_pair_fraction", "Linear", "#4c78a8"), ("saturated_pair_fraction", "Saturated", "#f58518")]:
        values = np.array([r[key] for r in chosen]) * 100
        axs[0, 1].bar(np.arange(len(chosen)), values, bottom=bottom, color=color, label=label)
        bottom += values
    axs[0, 1].set(xticks=np.arange(len(chosen)), xticklabels=[f"{d}/{c_}" for d, c_ in options], ylim=(0, 100), xlabel="delta / c", ylabel="Pair fraction (%)", title="How each candidate uses observed differences")
    axs[0, 1].legend(loc="lower right")
    window = min(25, steps)
    for field, label in [("pair_p20", "p20"), ("pair_median", "p50"), ("pair_p90", "p90")]:
        values = np.array([r[field] for r in ledger])
        axs[1, 0].plot(np.arange(window, steps + 1), np.convolve(values, np.ones(window) / window, mode="valid"), label=label)
    axs[1, 0].axhline(.01, color="tab:green", linestyle="--", alpha=.7)
    axs[1, 0].axhline(.09, color="tab:red", linestyle="--", alpha=.7)
    axs[1, 0].set(xlabel="Optimizer step", ylabel="Absolute reward difference", title=f"Per-step quantiles ({window}-step moving average)")
    axs[1, 0].legend()
    x = np.arange(len(chosen) + 1)
    values = [summaries["all"]["grpo_advantage_rms"]] + [r["advantage_rms"] for r in chosen]
    axs[1, 1].bar(x, values, color=["#a6a6a6"] + ["#4c78a8"] * len(chosen))
    axs[1, 1].set(xticks=x, xticklabels=["GRPO"] + [f"{d}/{c_}" for d, c_ in options], ylabel="Answer-level advantage RMS", title="Signal amplitude; not an effective learning-rate ratio")
    fig.suptitle(f"{steps}-step historical reward analysis — descriptive parameter choice", fontsize=15)
    fig.savefig(out_dir / "parameter_comparison.png", dpi=170)
    fig.savefig(out_dir / "parameter_comparison.pdf")
    plt.close(fig)
    print(json.dumps({"audit": report["audit"], "scopes": summaries, "selection": report["selection"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Defaults to RUN_DIR/advantage_analysis")
    args = parser.parse_args()
    analyze(args.run_dir, args.output_dir or args.run_dir / "advantage_analysis")
