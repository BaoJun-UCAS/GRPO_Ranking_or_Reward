# 200-step advantage 尺度与组权重对照

入口为 `scripts/run_advantage_controls.py`，设置位于
`recipes/Qwen3-1.7B/advantage_controls_200.yaml`。

| arm | 最终序列 advantage |
|---|---|
| `robust_scaled`（A） | `2.46 * A_robust` |
| `weight_only_scaled`（B） | `2.46 * q_g * A_GRPO` |

其中 `q_g = RMS(A_robust_g) / RMS(A_GRPO_g)`，每个 rollout 根据**当前完整回答组**重新计算，
停止梯度；GRPO advantage 全零时令 `q_g=0`。B 保留 GRPO 的组内方向，
在同一批奖励上与 A 的每组 RMS 相等（允许浮点舍入误差）。不会读取历史 CSV 的组权重。

两组固定 `delta=0.002, c=0.08, k=2.46, epsilon=1e-4`。倍率不改变死区和封顶阈值，
也不会再做每组标准化。它只乘策略 advantage，KL 系数仍是 `beta=0.04`，学习率仍是 `1e-6`。

## 运行

从项目根目录运行，使用原有 `grpo` Python 环境：

```bash
# 首次准备：验证已有缓存和数据，创建新实验配置，不启动训练。
/data/baojun/miniconda3/envs/grpo/bin/python scripts/run_advantage_controls.py prepare

# 检查两组最终配置；不启动 GPU、模型服务或训练进程。
/data/baojun/miniconda3/envs/grpo/bin/python scripts/run_advantage_controls.py train --dry-run

# 正式训练：先 A 后 B，各 200 个 optimizer steps。
/data/baojun/miniconda3/envs/grpo/bin/python scripts/run_advantage_controls.py train --arm both
```

也可以分别启动：

```bash
/data/baojun/miniconda3/envs/grpo/bin/python scripts/run_advantage_controls.py train --arm robust_scaled
/data/baojun/miniconda3/envs/grpo/bin/python scripts/run_advantage_controls.py train --arm weight_only_scaled
```

默认目录为 `grpo_runs/advantage-controls-200-delta0.002-k2.46/`，两组输出到同名 arm 子目录。
已经准备好后直接使用 `train --dry-run` 或 `train`，无需重复 `prepare`。
脚本拒绝覆盖或恢复已有 arm；若 A 已完成而 B 未开始，指定 `--arm weight_only_scaled` 即可。
失败的实验保留原始现场，另用 `--directory grpo_runs/<新目录>` 准备和启动。
修改已冻结配置、相关代码或环境后，也应准备一个新目录。

## 与之前实验保持一致的内容

- 继承 `grpo_runs/advantage-200/baseline/config/resolved_training_config.yaml` 的训练配置，
  包括 seed 42、LoRA、BNPO、温度、优化器、warmup、KL 和保存策略。
- 每组从原始 Qwen3-1.7B checkpoint 独立初始化，**不从已经训练 200 steps 的 adapter 继续**。
- 政策模型固定 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，
  QRM 固定 revision `23c6db70a35875248f45b3cfbe6d237ba8ac3e6d`。
  在 `/data/baojun/cache/grpo/huggingface/hub` 直接找到对应 snapshot，检查 tokenizer、模型分片和自定义代码。
  缓存缺失时明确报错，不触发下载。
- 复用旧实验已经冻结的本地 `data/dataset` 和 `training_schedule.json`，
  6,400 组问题、G=8、每步 256 个回答、两卡各 microbatch=1、梯度累积 128。
  新目录的 `data` 是指向原冻结数据目录的链接；不会复制、重采样或改写数据。
- 沿用 vLLM GPU 4、QRM GPU 5、训练 GPU 6/7；两组顺序运行。
  启动器继续负责 GPU 锁、端口检查、服务清理、训练产物验证及最终 LoRA 合并。
- 强制 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和 `WANDB_MODE=offline`。
  本机 vLLM/QRM 服务仍使用 localhost 通信。

## 新增核查记录

普通的 reward、KL、grad_norm、长度等日志继续保存；另在 `trainer_state.json` 和训练日志记录：

- `advantage/final_rms`：全局完整 rollout 最终 advantage 的 pooled RMS。
- `advantage/grpo_rms`、`advantage/robust_rms`：同一批回答上的两种未乘倍率优势 RMS。
- `advantage/q_mean`、`advantage/zero_group_fraction`。

每组运行目录中的 `advantage_audit/` 保存前两个 optimizer steps 的接入核查：

- `rollout_step_001.json` / `rollout_step_002.json`：当前奖励、GRPO/robust advantage、q 和 A/B 两种结果。
- `loss_inputs_rank_0.jsonl` / `loss_inputs_rank_1.jsonl`：实际传入原始 `compute_loss` 的微批 advantage、
  预期值、对应的全局 rollout 行号、有效 completion token 数。
  审计字段经过与回答相同的排列、切分、padding 裁剪；逐值不一致时立即报错。

这是最终 advantage 的传递核查，不是策略项/KL 项梯度分解，也不测量实际参数更新范数；
不新增模型 forward/backward。两条 on-policy 训练轨迹会生成不同回答，不能把“固定奖励上逐组同 RMS”
解释成整条训练轨迹的 RMS 必然一致。

## 代码改动范围

原 `advantages.py`、`grpo_trainer.py` 和 `grpo.py` 不变。
新公式通过 `open_r1.advantage_controls:<函数名>` 接入；专用入口只在当前进程使用审计子类。
原 shell 启动器增加可选 `TRAINING_ENTRYPOINT`，缺省仍走原 `src/open_r1/grpo.py`。
