# 方法一：滚动分位数 advantage

新 improved 方法的注册名称是 `rolling_quantile_pairwise`，实现位于
`src/open_r1/rolling_advantages.py`。历史实验的 `robust_pairwise` 和标准
`grpo` 保留原来的公式，新的方法通过独立配置选择。

对当前问题的一组 G 个回答，使用与标准 GRPO 相同的无偏组内标准差：

```text
D_g = std_unbiased(r_g) + epsilon
z_gij = (r_gi - r_gj) / D_g
ell_t = quantile(p, historical_abs_z)
b_t = quantile(1-q, historical_abs_z)
tau_t = b_t - ell_t
phi(z) = sign(z) * min(max(abs(z) - ell_t, 0), tau_t)
A_gi = sum_j phi(z_gij) / G
```

线性区间斜率为 1，比较幅度封顶到 `tau`，按 `G` 聚合。对角线贡献为零。
不除以旧参数 `c`，不乘之前实验中的 `2.46`，也不对最终 advantage 再做
组内标准化。`ell=0、tau=inf` 时与标准 GRPO 完全一致。

历史窗口只收集每组 `i<j` 的 `abs(z_gij)`，不混入对角线的零值，也不重复
收集两种比较方向。`window_size` 的单位是全局 rollout 批次；在附带的
200 step 配置中，每个 optimizer step 恰好生成一次全局 rollout。每次有
32 个问题、每题 8 个回答，单次历史包含 `32 * 28 = 896` 个分差。

当前 optimizer step 的阈值仅由更早的 rollout 估计，随后冻结；同一步内
即使生成多批回答也不重新拟合。当前分差在计算完 advantage 后才加入窗口。
没有历史的第一个 optimizer step 使用标准 GRPO，后续使用滚动分位数。
所有训练 rank 使用完整的全局 reward 分组，得到相同的阈值和历史。
评估使用已冻结阈值，不修改训练历史。

`p` 是下分位位置，`q` 是上尾分位比例，必须显式给出，满足
`p>=0、q>=0、p+q<1`。它们决定历史分布上的边界，不能保证当前批次恰好有
对应比例的比较被过滤或封顶；并列值和分布变化都会影响实际比例。
`p=q=0` 仍然使用历史最小值和最大值，并不等于关闭过滤和封顶。

若两个分位数相等，`tau=0`，按照原公式输出零 advantage，并记录
`advantage/degenerate_thresholds=1`。没有加入隐藏的下限或 GRPO 回退，以免
改变方法定义。第一个没有历史的 step 是上述明确的 warmup 例外。

训练日志包含 `advantage/ell`、`advantage/tau`、`advantage/upper_threshold`、
`advantage/dead_pair_fraction`、`advantage/saturated_pair_fraction`、
`advantage/final_rms`、`advantage/zero_group_fraction`、历史批次和分差数量。
warmup 时记录 `advantage/warmup_grpo=1`，不把无限大阈值写进 JSON 日志。
dead 和 saturated 比例只统计当前批次的非对角无序回答对。

历史、当前冻结阈值和 optimizer step 随 checkpoint 保存到
`rolling_advantage_state.json`；训练结束时输出目录也保存一份。恢复训练会
先读取 checkpoint 中的历史，并核对参数及 checkpoint step。缺失历史或
不匹配时直接报错，不静默使用空窗口。

单独在训练 YAML 中选择新方法的示例：

```yaml
advantage: rolling_quantile_pairwise
scale_rewards: true
advantage_kwargs:
  p: 0.05
  q: 0.05
  window_size: 32
  epsilon: 0.0001
```

这里的 `0.05` 仅是可运行的实验示例，尚未校准，也不是推荐最优值。
该方法适配奖励尺度和分布比例，不能仅凭这些分差识别奖励模型的评分噪声。

新增对比配置：
`recipes/Qwen3-1.7B/advantage_comparison_200_rolling_quantile.yaml`。
它保留 200 step、seed=42、G=8 和原来的训练参数，并使用本地 Qwen、QRM
快照和本地数据。`baseline` 仍使用标准 GRPO，`improved` 选择新方法；
旧配置未指定 `improved_advantage` 时继续使用历史 `robust_pairwise`。

可以先检查 `p、q、window_size`，随后准备一个新实验并仅检查启动命令：

```bash
conda activate grpo
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python scripts/compare_advantages.py prepare \
  --config recipes/Qwen3-1.7B/advantage_comparison_200_rolling_quantile.yaml \
  --experiment-dir grpo_runs/advantage-200-rolling-quantile

python scripts/compare_advantages.py train \
  --experiment-dir grpo_runs/advantage-200-rolling-quantile \
  --arm improved --dry-run
```

`prepare` 只冻结数据和配置，`--dry-run` 只打印解析后的训练命令和配置。
实现与测试过程没有启动模型训练。
