#!/usr/bin/env python3
"""CPU-only audit of completed GRPO vs robust-pairwise training runs.

Defaults select the paired 200-step baseline and delta=0.002 experiment.
Requires NumPy, PyYAML and (unless --no-plots) Matplotlib. Never loads a model,
starts training, calls a reward/judge service, or modifies the source runs.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
FILE_PATTERN = re.compile(r"reward_data_(?:train_)?step_(\d+)_micro_(\d+)_proc_(\d+)(?:_[a-f0-9]+)?\.json")
METRICS = ("reward", "reward_std", "kl", "grad_norm", "loss", "learning_rate",
           "completions/mean_length", "completions/clipped_ratio",
           "frac_reward_zero_std", "clip_ratio/region_mean")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "min": None,
                "p05": None, "p25": None, "p50": None, "p75": None, "p95": None, "max": None}
    return {"count": int(len(values)), "mean": float(values.mean()), "std": float(values.std()),
            **dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                       map(float, np.quantile(values, [0, .05, .25, .5, .75, .95, 1]))))}


def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and x.std() > 0 and y.std() > 0 else None


def advantage_diagnostics(rewards, delta, c, epsilon):
    """Float64 offline reconstruction; unordered pairs exclude the diagonal.

    Each unordered pair represents two identical absolute directed gaps.
    Zero-vector cosine and zero-GRPO RMS ratio are undefined (NaN internally,
    empty CSV cell / omitted from distributions), never replaced by zero.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    require(rewards.ndim == 2 and rewards.shape[1] >= 2 and np.isfinite(rewards).all(),
            "Rewards must contain finite complete groups")
    require(math.isfinite(delta) and delta >= 0 and math.isfinite(c) and c > 0
            and math.isfinite(epsilon) and epsilon > 0, "Invalid advantage parameters")
    g = rewards.shape[1]
    centered = rewards - rewards.mean(1, keepdims=True)
    std = rewards.std(1, ddof=1)
    grpo = centered / (std[:, None] + epsilon)
    differences = rewards[:, :, None] - rewards[:, None, :]
    robust = (np.sign(differences) * np.clip((np.abs(differences) - delta) / c, 0, 1)).sum(2) / (g - 1)
    linear_check = differences.sum(2) / (g * (std[:, None] + epsilon))
    first, second = np.triu_indices(g, 1)
    gaps = np.abs(differences[:, first, second])
    grpo_rms = np.sqrt(np.mean(grpo ** 2, axis=1))
    robust_rms = np.sqrt(np.mean(robust ** 2, axis=1))
    # Classify mathematically constant reward groups directly, not by a tolerance.
    grpo_zero = np.ptp(rewards, axis=1) == 0
    robust_zero = np.ptp(rewards, axis=1) <= delta
    ratio = np.divide(robust_rms, grpo_rms, out=np.full(len(rewards), np.nan), where=~grpo_zero)
    valid_cosine = ~grpo_zero & ~robust_zero
    cosine = np.divide(np.mean(grpo * robust, axis=1), grpo_rms * robust_rms,
                       out=np.full(len(rewards), np.nan), where=valid_cosine)
    cosine = np.clip(cosine, -1, 1)
    weight_only = np.nan_to_num(ratio)[:, None] * grpo
    return {
        "reward_mean": rewards.mean(1), "reward_std": std,
        "dead_pair_fraction": (gaps <= delta).mean(1),
        "saturated_pair_fraction": (gaps >= delta + c).mean(1),
        "linear_pair_fraction": ((gaps > delta) & (gaps < delta + c)).mean(1),
        "grpo_zero": grpo_zero, "robust_zero": robust_zero,
        "grpo_rms": grpo_rms, "robust_rms": robust_rms, "rms_ratio": ratio,
        "cosine": cosine,
        "weight_only_residual_rms": np.sqrt(np.mean((robust - weight_only) ** 2, axis=1)),
        "linear_identity_max_abs_error": np.max(np.abs(grpo - linear_check), axis=1),
        "linear_fixed_c_scale": g * (std + epsilon) / ((g - 1) * c),
        "gaps": gaps,
    }


def summarize_diagnostics(d, mask):
    selected = {k: v[mask] for k, v in d.items()}
    grpo_rms = float(np.sqrt(np.mean(selected["grpo_rms"] ** 2)))
    robust_rms = float(np.sqrt(np.mean(selected["robust_rms"] ** 2)))
    return {
        "groups": int(mask.sum()), "unordered_pairs": int(selected["gaps"].size),
        "dead_pair_fraction": float(selected["dead_pair_fraction"].mean()),
        "saturated_pair_fraction": float(selected["saturated_pair_fraction"].mean()),
        "linear_pair_fraction": float(selected["linear_pair_fraction"].mean()),
        "all_zero_grpo_group_fraction": float(selected["grpo_zero"].mean()),
        "all_zero_robust_group_fraction": float(selected["robust_zero"].mean()),
        "cosine_undefined_groups": int(np.isnan(selected["cosine"]).sum()),
        "rms_ratio_undefined_groups": int(np.isnan(selected["rms_ratio"]).sum()),
        "pooled_grpo_rms": grpo_rms, "pooled_robust_rms": robust_rms,
        "pooled_rms_ratio": robust_rms / grpo_rms if grpo_rms else None,
        "rms_ratio_reward_std_correlation": correlation(selected["rms_ratio"], selected["reward_std"]),
        "linear_identity_max_abs_error": float(selected["linear_identity_max_abs_error"].max()),
        "distributions": {k: stats(selected[k]) for k in (
            "gaps", "reward_std", "grpo_rms", "robust_rms", "rms_ratio", "cosine",
            "weight_only_residual_rms", "linear_fixed_c_scale")},
    }


def load_run(run_dir, steps):
    run_dir = run_dir.resolve()
    cfg_path = run_dir / "config/resolved_training_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    state_path = run_dir / "trainer_state.json"
    state = json.loads(state_path.read_text())
    ranks = yaml.safe_load((run_dir / "config/accelerate_config.yaml").read_text())["num_processes"]
    require(state["global_step"] == state["max_steps"] == cfg["max_steps"] == steps,
            f"{run_dir}: expected a completed {steps}-step run")
    require("status=success" in (run_dir / "RUN_STATUS").read_text(), f"{run_dir}: no success marker")
    g, batch_size = cfg["num_generations"], cfg["generation_batch_size"]
    require(g >= 2 and batch_size % g == 0 and batch_size % ranks == 0, "Invalid batch/group/rank counts")
    require(cfg["num_iterations"] == 1 and batch_size == ranks * cfg["per_device_train_batch_size"]
            * cfg["gradient_accumulation_steps"], "Audit requires exactly one rollout per optimizer step")
    records = [r for r in state["log_history"] if "reward_std" in r and "eval_loss" not in r]
    logs = {r["step"]: r for r in records}
    require(len(logs) == len(records) and set(logs) == set(range(1, steps + 1)),
            f"{run_dir}: missing or duplicate training metrics")
    indexed, ignored_eval = {}, 0
    for path in sorted((run_dir / "reward_data").glob("*.json")):
        if "_eval_" in path.name:
            ignored_eval += 1
            continue
        match = FILE_PATTERN.fullmatch(path.name)
        require(match is not None, f"Unknown reward filename: {path}")
        step, micro, rank = map(int, match.groups())
        require((step, rank) not in indexed, f"Duplicate rollout for step/rank {(step, rank)}")
        indexed[step, rank] = (micro, path)
    require(set(indexed) == {(s, r) for s in range(steps) for r in range(ranks)},
            f"{run_dir}: missing or unexpected step/rank reward files")
    rewards, prompts, errors = [], [], []
    ledger = hashlib.sha256()
    for step in range(steps):
        rows, micros = [], []
        for rank in range(ranks):
            micro, path = indexed[step, rank]
            raw = path.read_bytes()
            ledger.update(f"{path.name}\0{hashlib.sha256(raw).hexdigest()}\n".encode())
            local = json.loads(raw)
            require(len(local) == batch_size // ranks, f"Incomplete rank file: {path}")
            require([r["rollout_index"] for r in local] == list(range(len(local))), f"Bad row order: {path}")
            require(all(r["training_step"] == step and r["process_index"] == rank for r in local),
                    f"Bad metadata: {path}")
            rows.extend(local)
            micros.append(micro)
        require(len(set(micros)) == 1, f"Rank micro-step mismatch: {step}")
        matrix = np.asarray([r["reward"] for r in rows], dtype=np.float64).reshape(-1, g)
        require(np.isfinite(matrix).all(), f"Non-finite reward: step {step}")
        for i in range(0, len(rows), g):
            prompt = rows[i]["prompt"]
            require(all(r["prompt"] == prompt for r in rows[i:i + g]), f"Mixed prompt group: step {step}")
            prompts.append(hashlib.sha256(prompt.encode()).hexdigest())
        error = [abs(matrix.mean() - logs[step + 1]["reward"]),
                 abs(matrix.std(1, ddof=1).mean() - logs[step + 1]["reward_std"])]
        require(max(error) < 2e-6, f"Saved rewards disagree with training log: step {step + 1}, {error}")
        errors.append(error)
        rewards.append(matrix)
    return {
        "path": str(run_dir), "config": cfg, "logs": logs,
        "rewards": np.concatenate(rewards), "prompt_hashes": np.array(prompts),
        "group_steps": np.repeat(np.arange(1, steps + 1), batch_size // g),
        "audit": {"steps": steps, "ranks": ranks, "group_size": g, "reward_files": len(indexed),
                  "ignored_eval_files": ignored_eval, "responses": steps * batch_size,
                  "groups": len(prompts), "distinct_prompts": len(set(prompts)),
                  "max_reward_log_error": float(np.max(errors, axis=0)[0]),
                  "max_reward_std_log_error": float(np.max(errors, axis=0)[1]),
                  "reward_file_ledger_sha256": ledger.hexdigest(), "config_sha256": digest(cfg_path),
                  "trainer_state_sha256": digest(state_path)},
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, (float, np.floating)) and not math.isfinite(v) else v
                             for k, v in row.items()})


def build_report(baseline, improved, steps, window):
    bcfg, icfg = baseline["config"], improved["config"]
    require(bcfg["advantage"] in {"grpo", "studentization"} and bcfg.get("scale_rewards", True) is True,
            "Baseline must use standard GRPO with reward standardization")
    require(icfg["advantage"] == "robust_pairwise", "Improved run must use robust_pairwise")
    options = icfg.get("advantage_kwargs", {})
    delta, c = options.get("delta", .02), options.get("c", .2)
    epsilon = bcfg.get("advantage_kwargs", {}).get("epsilon", 1e-4)
    require(bcfg.get("advantage_kwargs", {}).get("scale_rewards", True) is True,
            "Baseline estimator explicitly disables standardization")
    differences = {k: {"baseline": bcfg.get(k), "improved": icfg.get(k)}
                   for k in sorted(bcfg.keys() | icfg.keys()) if bcfg.get(k) != icfg.get(k)}
    same_shape = baseline["prompt_hashes"].shape == improved["prompt_hashes"].shape
    matched = int((baseline["prompt_hashes"] == improved["prompt_hashes"]).sum()) if same_shape else 0
    scopes = {"all": (1, steps), **{f"steps_{start:03d}_{min(start + window - 1, steps):03d}":
              (start, min(start + window - 1, steps)) for start in range(1, steps + 1, window)}}
    report = {
        "parameters": {"delta": delta, "c": c, "saturation_threshold": delta + c,
                       "grpo_epsilon": epsilon, "std_ddof": 1},
        "config_differences": differences,
        "prompt_alignment": {"same_number_of_groups": same_shape, "matching_group_positions": matched,
                             "all_match": same_shape and matched == len(baseline["prompt_hashes"])},
        "method": "Recompute both advantages on each run's own fixed completions; compare training metrics separately.",
        "limitations": [
            "Advantage is reconstructed in float64, not a saved final loss input tensor; small float32 differences are expected.",
            "The linear identity check verifies reward algebra only; it does not replay model loss, token masks, or parameter gradients.",
            "KL and grad_norm are observed training metrics on different on-policy completions; grad_norm includes KL and is not policy-only.",
            "No token log probabilities, masks or per-term gradients were saved in reward_data; a same-batch loss/gradient comparison cannot be recovered.",
            "RMS ratio is not an effective learning-rate ratio or a measured effective beta; no causal or judge-quality conclusion follows.",
            "Step means/quantiles are descriptive; temporally dependent training steps are not independent training seeds.",
        ], "runs": {},
    }
    per_group, per_step, metric_rows = [], [], []
    for label, run in (("baseline", baseline), ("improved", improved)):
        d = advantage_diagnostics(run["rewards"], delta, c, epsilon)
        run["diagnostics"] = d
        summaries = {}
        for scope, (start, end) in scopes.items():
            mask = (run["group_steps"] >= start) & (run["group_steps"] <= end)
            metrics = {key: stats([run["logs"][s][key] for s in range(start, end + 1)
                                  if key in run["logs"][s]]) for key in METRICS}
            for key, values in metrics.items():
                metric_rows.append({"run": label, "scope": scope, "metric": key, **values})
            summaries[scope] = {"advantage": summarize_diagnostics(d, mask), "logged_metrics": metrics}
        report["runs"][label] = {"path": run["path"], "audit": run["audit"], "scopes": summaries}
        for i, step in enumerate(run["group_steps"]):
            per_group.append({"run": label, "optimizer_step": int(step),
                              "group_in_step": i % (run["config"]["generation_batch_size"] // run["config"]["num_generations"]),
                              "prompt_sha256": run["prompt_hashes"][i],
                              **{k: v[i].item() for k, v in d.items() if k != "gaps"}})
        for step in range(1, steps + 1):
            mask = run["group_steps"] == step
            summary = summarize_diagnostics(d, mask)
            per_step.append({"run": label, "optimizer_step": step,
                             **{k: v for k, v in run["logs"][step].items()
                                if k not in {"step", "epoch"} and isinstance(v, (int, float))},
                             **{f"diagnostic/{k}": v for k, v in summary.items() if k != "distributions"},
                             "diagnostic/mean_cosine": summary["distributions"]["cosine"]["mean"],
                             "diagnostic/mean_group_rms_ratio": summary["distributions"]["rms_ratio"]["mean"]})
    report["logged_metric_comparison"] = {
        scope: {key: {"baseline_mean": report["runs"]["baseline"]["scopes"][scope]["logged_metrics"][key]["mean"],
                      "improved_mean": report["runs"]["improved"]["scopes"][scope]["logged_metrics"][key]["mean"]}
                for key in METRICS} for scope in scopes}
    for metrics in report["logged_metric_comparison"].values():
        for row in metrics.values():
            b, i = row["baseline_mean"], row["improved_mean"]
            row["improved_minus_baseline"] = i - b if b is not None and i is not None else None
            row["improved_over_baseline"] = i / b if b not in (None, 0) and i is not None else None
    return report, per_group, per_step, metric_rows


def write_markdown(path, report):
    p = report["parameters"]
    runs = report["runs"]
    lines = ["# 200-step GRPO 离线训练诊断".replace("200-step", f"{runs['baseline']['audit']['steps']}-step"), "",
             f"δ={p['delta']}，c={p['c']}，封顶阈值={p['saturation_threshold']:.6g}；无偏标准差，ε={p['grpo_epsilon']}。",
             "只读取已完成训练的日志和奖励明细；未加载模型、未启动训练或 judge。", "",
             f"标准 GRPO：`{runs['baseline']['path']}`", "",
             f"robust_pairwise：`{runs['improved']['path']}`", "",
             f"逐组 prompt 顺序一致：{report['prompt_alignment']['all_match']}；"
             f"匹配 {report['prompt_alignment']['matching_group_positions']} 组。", "",
             "两列分别使用各自训练生成的回答；每列内部在同一批奖励上重算两种 advantage。", "",
             "| 全程 advantage 诊断 | 标准 GRPO 回答 | robust 回答 |", "|---|---:|---:|"]
    summaries = [runs[label]["scopes"]["all"]["advantage"] for label in ("baseline", "improved")]
    for name, key in [("死区比例（排除对角线）", "dead_pair_fraction"), ("封顶比例", "saturated_pair_fraction"),
                      ("整组 robust advantage 为零", "all_zero_robust_group_fraction"),
                      ("合并 RMS 比 robust/GRPO", "pooled_rms_ratio")]:
        lines.append(f"| {name} | {summaries[0][key]:.6f} | {summaries[1][key]:.6f} |")
    for key in ("rms_ratio", "cosine", "reward_std"):
        a, b = [s["distributions"][key] for s in summaries]
        lines.append(f"| {key} 均值 | {fmt(a['mean'])} | {fmt(b['mean'])} |")
        lines.append(f"| {key} P05 / P50 / P95 | " + " | ".join(
            " / ".join(fmt(v[q]) for q in ("p05", "p50", "p95")) for v in (a, b)) + " |")
    lines.extend(["", "RMS 合并比值与逐组 RMS 比的均值不同；余弦只统计两向量均非零的组。", "",
                  "| 区间 | 日志指标 | GRPO 均值 | robust 均值 | robust − GRPO |", "|---|---|---:|---:|---:|"])
    for scope, metrics in report["logged_metric_comparison"].items():
        for key in ("reward", "reward_std", "kl", "grad_norm", "completions/mean_length", "completions/clipped_ratio"):
            row = metrics[key]
            lines.append(f"| {scope} | {key} | {fmt(row['baseline_mean'])} | {fmt(row['improved_mean'])} | {fmt(row['improved_minus_baseline'])} |")
    lines.extend(["", "## 数据校验", ""])
    for label, run in runs.items():
        audit = run["audit"]
        lines.append(f"- {label}：{audit['reward_files']} 个奖励文件，{audit['responses']} 个回答，{audit['groups']} 组；"
                     f"奖励均值/组标准差与日志最大误差 {audit['max_reward_log_error']:.3g} / {audit['max_reward_std_log_error']:.3g}；"
                     f"纯线性恒等式最大误差 {run['scopes']['all']['advantage']['linear_identity_max_abs_error']:.3g}。")
    lines.extend(["", "## 配置差异（完整列出）", "", "```json",
                  json.dumps(report["config_differences"], ensure_ascii=False, indent=2), "```", "",
                  "## 解读边界", "",
                  "- RMS 比衡量奖励空间中的信号幅度，不能解释为学习率比例；相关性和余弦不能单独证明性能下降的原因。",
                  "- grad_norm 是实际训练记录的整体梯度范数，包含 KL；两次训练的回答不同。日志 loss 有舍入，不能可靠拆分策略项与 KL 项。",
                  "- 纯线性检查只验证 advantage 代数恒等式，不代表实际 loss、token mask 或模型参数梯度等价测试已通过。",
                  "- 奖励明细没有保存 token logprob、mask、策略项梯度，无法据此恢复同一 batch 的实际 loss/参数梯度对照。",
                  "- 本脚本不做参数扫描、不评估 judge 可靠性、不把训练步当作独立 seed，不据此判定方法有效或无效。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value):
    return "NA" if value is None else f"{value:.6g}"


def plot_comparison(path, runs, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for label, run in runs.items():
        selected = [r for r in rows if r["run"] == label]
        for ax, key in zip(axes.flat[:4], ("reward", "kl", "grad_norm", "diagnostic/pooled_rms_ratio")):
            values = np.array([np.nan if row.get(key) is None else row[key] for row in selected], dtype=float)
            width = min(10, len(values))
            ax.plot(np.arange(width, len(values) + 1), np.convolve(values, np.ones(width) / width, mode="valid"), label=label)
            ax.set(title=f"{key} ({width}-step mean)", xlabel="Optimizer step")
        for ax, key in zip(axes.flat[4:], ("rms_ratio", "cosine")):
            values = run["diagnostics"][key]
            values = np.sort(values[np.isfinite(values)])
            if len(values):
                ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
            ax.set(title=f"Within-run {key} distribution", ylabel="Empirical CDF", xlabel=key)
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "grpo_runs/advantage-200/baseline")
    parser.add_argument("--improved", type=Path, default=ROOT / "grpo_runs/advantage-200-delta0.002/improved")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/advantage-200-delta0.002-diagnostics")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window", type=int, default=50, help="Nonoverlapping summary window in optimizer steps")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    require(args.steps > 0 and args.window > 0, "Steps and window must be positive")
    output = args.output_dir.resolve()
    require(all(not output.is_relative_to(p.resolve()) for p in (args.baseline, args.improved)),
            "Output directory must be outside source run directories")
    runs = {}
    for label, path in (("baseline", args.baseline), ("improved", args.improved)):
        print(f"Reading {label}: {path}", flush=True)
        runs[label] = load_run(path, args.steps)
    report, groups, steps, metrics = build_report(runs["baseline"], runs["improved"], args.steps, args.window)
    report["analysis_script_sha256"] = digest(Path(__file__))
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_csv(output / "per_group.csv", groups)
    write_csv(output / "per_step.csv", steps)
    write_csv(output / "metric_summary.csv", metrics)
    write_markdown(output / "report.md", report)
    if not args.no_plots:
        plot_comparison(output / "comparison.png", runs, steps)
    print(f"Wrote offline diagnostic report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
