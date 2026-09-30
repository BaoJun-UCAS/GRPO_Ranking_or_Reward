# 验证与模型对比评测

这里区分两条流程：训练内 `trainer.evaluate()` 使用验证集和训练奖励；离线评测比较基础模型与训练模型在相同测试 prompt 上的回答，再交给独立 judge。两者不能互相替代，也不能把训练完成或 API 返回成功当作评测有效。

## 2026-09-30 重构验收记录

本机 GPU 4 / 5 / 6,7 分别用于 vLLM / QRM / 两 rank 策略训练。实际运行使用缓存的
Qwen3-1.7B 与 QRM-Llama3.1-8B-v2，策略 revision 固定为
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，QRM 固定为
`23c6db70a35875248f45b3cfbe6d237ba8ac3e6d`。

| `grpo_runs/` 下目录 | 已通过的实际检查 |
| --- | --- |
| `refactor-smoke-20260930` | ZeRO-2 + studentization；2 optimizer steps，2 个验证 prompt，adapter 保存，产物/timing 验收 |
| `refactor-ddp-rank-smoke-20260930` | DDP + rank_reward(0.3)；2 optimizer steps，2 个验证 prompt，LoRA 合并、重载生成、产物验收 |
| `refactor-full-length-smoke-v2-20260930` | 默认 2048/3072 prompt/completion 上限、8 次生成；2 optimizer steps + 2 个验证 prompt，最终代码验收 |

每个目录保存 `run.env`、resolved YAML、代码快照、`train_results.json`、`eval_results.json`、
`validation_report.json` 和日志；第二个目录还含 `reload_smoke.json` 与 `merged_model/`。

默认长度首次运行在末尾验证暴露了 KV cache 与右侧 completion padding 的 FlashAttention 冲突。
现已让评分 forward 显式关闭 cache，保存配置使用副本、不改变运行中的各 rank 模型；相同配置复跑通过。
失败记录保留在 `refactor-full-length-smoke-20260930`，不能作为成功产物使用。

这些都是功能 smoke：generation batch 为 16，未完成正式默认 batch 256 / 800 步训练。
默认长度复跑实际最长训练 completion 为 1045 token，验证为 1737 token，因此没有覆盖每条都达到
3072 token 的最坏显存情况，也不证明收敛或哪种 advantage 更好。两次短 smoke 的算法和 microbatch
不同，不能用其 runtime 直接宣称 DDP 提速比例。运行结束后 4–7 卡已释放。

## 2026-09-30 新 advantage 验证

`robust_pairwise` 已按附件公式接入，原始 `studentization` 及其 `grpo` 别名保留。
完整回归为 **300 passed，11 subtests passed**，包括附件三个数值例子、阈值边界、
并列奖励、异常值、低精度与跨 rank 分组；真实 CPU LoRA 测试分别运行原始 GRPO 和新方法，
检查权重更新、验证、保存与重载。4 条 warning 来自 DeepSpeed 的 Pydantic 弃用接口。

真实四卡运行目录为 `grpo_runs/robust-pairwise-smoke-20260930`，使用上面的模型 revision
及 GPU 布局，ZeRO-2、`delta=0.02`、`c=0.2`、8 次生成、generation batch 16、
2048/3072 token 上限。已完成 2 optimizer steps 和 2 个验证 prompt，保存 adapter；
`RUN_STATUS` 为 `success`，`validation_report.json` 为 `passed`，错误列表为空。
两步梯度范数分别约为 0.1022、0.0309；验证 loss 约为 0.00145，指标有限。
运行结束后 GPU 4–7 已释放。

这是功能验证，未运行完整训练或证明质量优于原始 GRPO；`delta/c` 仍是未校准的示例值。
两种方法的同配置切换命令见 [README](../README.md#改-advantage)。

## 历史 800 步日志与初步参数（2026-09-30）

参数参考来自 `grpo_runs/qwen3-1.7b-grpo-p2048-c3072-800step-20260928-193332`。
按 rollout、rank 与 `rollout_index` 恢复同一 prompt 的完整 G=8 组，共 **204,800 条回答、
25,600 组、716,800 个无序组内比较**；两 rank 的奖励向量没有重复。
相同 prompt 文本跨 rollout 出现时仍分别计组，500 步 run 未混入本统计。
重建结果与 `trainer_state.json` 的 800 条训练记录逐步对齐：奖励均值最大绝对误差
`7.16e-8`，平均组内无偏标准差最大误差 `1.05e-8`。
这验证的是奖励日志与训练标量一致；历史 `validation_report.json` 仍为 `failed`
（缺少 `qrm_score` 推理日志），本统计不改写该验收状态。

绝对奖励差 `|r_i-r_j|` 的 P20 / P50 / P90 分别为 **0.00915 / 0.02598 / 0.08410**。
主推荐起点为 **`delta=0.01, c=0.08`**：约过滤最小的 22% 比较，保留约 70% 的线性幅度，
在差值达到 `delta+c=0.09` 时封顶，覆盖约 9% 的比较。`c=0.05` 保留为下一组对照，
用于检查较早封顶、较大 advantage 幅度的影响。它们均是待验证的实验参数，不是已证实的最优值。

| `delta` | `c` | 死区 `d≤delta` | 线性区 | 饱和 `d≥delta+c` | 全零 advantage 组 |
| --- | --- | --- | --- | --- | --- |
| 0.02 | 0.20（原示例） | 40.49% | 58.33% | 1.18% | 0.97% |
| 0.01 | 0.08（主推荐） | 21.74% | 69.50% | 8.76% | 0.32% |
| 0.01 | 0.05（对照） | 21.74% | 59.85% | 18.41% | 0.32% |

表中 `d=|r_i-r_j|`，比较比例以全部无序组内 pair 为分母，全零组比例以完整 prompt 组为分母。
把 800 步分成四个连续的 200 步区间，主推荐的死区比例为 20.81%–22.31%，
线性比例 69.16%–69.78%，饱和比例 8.36%–9.41%，全零组比例 0.17%–0.41%，
说明这个分布起点在历史训练各阶段较稳定。

这些阈值依赖当前 QRM、奖励权重与数据尺度。日志没有独立的评分误差标签，
因此 `delta` 不是已校准的误差界；更换奖励模型或缩放奖励后应重新分析。
新旧 advantage 幅度不同，不应按幅度比自动调整学习率，也不能将较小更新造成的变化归因于算法优势；
对照实验需检查实际 KL 和独立评测，且新方法之后不能再做组内标准差归一化。

初步配置保存于 [robust_pairwise_initial.env](../recipes/Qwen3-1.7B/robust_pairwise_initial.env)。
在项目根目录、训练环境中使用：

```bash
source recipes/Qwen3-1.7B/robust_pairwise_initial.env
python scripts/grpo.py train
```

复现历史统计无需加载模型或占用 GPU：

```bash
python scripts/analyze_advantage_rewards.py \
  --run-dir grpo_runs/qwen3-1.7b-grpo-p2048-c3072-800step-20260928-193332
```

默认输出到该 run 的 `advantage_analysis/`，可用 `--output-dir` 指定其它目录。

## 训练运行验收

训练进程正常退出后，先做不加载模型、不占 GPU 的产物与性能证据验收：

```bash
python scripts/grpo.py validate --run-dir /path/to/training-run
# 如果本次运行约定生成 merged_model：
python scripts/grpo.py validate --run-dir /path/to/training-run --require-merged
```

未指定 `--run-dir` 时按 `RUN_DIR`、`RUN_NAME`、输出根目录下最新 run 的顺序选择。验收器检查 `RUN_STATUS`、resolved YAML、manifest、最终步数和有限 loss、adapter/完整模型权重、关键日志、每个 rank 的 timing min/max/spread，以及 QRM 动态分批是否超过 `REWARD_BATCH_SIZE` 或 `QRM_MAX_BATCH_TOKENS`。结果原子写入 `validation_report.json`，任何硬错误都返回非零退出码。

四卡 launcher 已自动执行相同验收：训练产物不完整或指标不一致时不会把 run 标成成功；开启 LoRA 合并时还会再次检查 merged model。`--allow-running` 仅供 launcher 在写最终 `RUN_STATUS` 前内部使用，不应当作手工绕过失败状态的选项。

报告中的 stage share 是计时证据的汇总，用于定位 rollout、QRM、policy 或其它同步开销；它不能证明模型质量、收敛性、奖励无偏，也不能替代真实 GPU smoke 和下方独立对比评测。


## 训练过程曲线

仓库提供完全离线的 Trainer/GRPO 指标绘图，不读取权重、不占用 GPU：

```bash
python scripts/plot_training_metrics.py /path/to/training-run
# 或：
make plot-training RUN_DIR=/path/to/training-run
```

默认读取 run 根目录的 `trainer_state.json`；如果根目录没有，会选择数字最大的 `checkpoint-*/trainer_state.json`。输出默认为同一 run 下的 `training_metrics.png` 与长表格式 `training_metrics.csv`。图中按独立纵轴展示 loss、reward、KL/clip ratio、completion 长度/截断、梯度范数、学习率和累计 token，避免把数量级不同的指标硬画在同一坐标轴。

常用选项：

```bash
# 查看某次 run 实际有哪些指标
python scripts/plot_training_metrics.py /path/to/run --list-metrics

# 只绘制指定指标，并使用 10 个日志点的移动平均
python scripts/plot_training_metrics.py /path/to/run \
  --metrics loss reward kl grad_norm --smooth 10 \
  --output /path/to/figures/train.png

# 对比两次实验；--label 必须与输入一一对应
python scripts/plot_training_metrics.py /path/to/baseline /path/to/ranking \
  --label baseline --label ranking --smooth 5 \
  --output /path/to/figures/comparison.pdf
```

脚本从 `log_history` 提取有限的数值指标，同一步同一指标重复出现时采用较后的记录；summary 中的一次性 `train_loss`、runtime 等写在图下注释，不伪装成训练曲线。`--smooth 1` 是原始数据；移动平均只改变画线，CSV 永远保留原始记录。历史两步 smoke 中的负 `completions/clipped_ratio` 来自当时已修复的统计错误，绘图会忠实保留，不应当作真实负比例解释。

## 训练内验证

配置 `do_eval: true` 可以只在训练结束后验证，保留 `eval_strategy: "no"` 即不在训练中途执行；`steps`/`epoch` 策略则用于中途验证。现在两种方式都会正确传入验证数据集。

全局验证 batch（训练进程数 × `per_device_eval_batch_size`）必须能被 `num_generations` 整除。例如 4 个训练进程、每卡验证 batch 2、8 次生成满足约束。开启验证会消耗 GPU，当前卡被占用时不要运行。

四卡 launcher 可直接使用 `DO_EVAL=1 python scripts/grpo.py train`。
默认两训练 rank、每卡 eval batch 4、8 次生成满足约束。调试可加 `MAX_EVAL_SAMPLES=8`。
验证指标会同时返回给 Trainer callbacks 并写入 `eval_results.json`；开启 `do_eval` 后运行验收器
要求该文件包含有限的 `eval_loss`、`eval_reward` 和正整数 `eval_samples`。

奖励记录文件区分 `train`/`eval`，保留 optimizer step、micro-step、进程号并添加唯一 rollout 标识。因此同一次验证的多个 batch、同一步重复验证不会覆盖；原来的训练产物保持不动。

## 离线对比评测

先准备已合并模型，激活训练环境，在项目根目录操作。下面的命令会占用所选 GPU 并调用付费 judge API，必须等 GPU 分配给你后再执行：

```bash
# 通过安全方式配置 DEEPSEEK_API_KEY，并自行选择当前可用的 judge 模型。
EVAL_GPU=0 JUDGE_MODEL=你的judge模型 \
bash evaluate/run_grpo_chat_deepseek.sh /path/to/training-run
```

每次默认创建 `training-run/evaluation/<UTC时间>-<PID>/`，不复用历史评测目录。目录内有 `completions/`、`deepseek_judge/`、`logs/`、`evaluation.env` 与 `EVALUATION_STATUS`。同一目录通过 `flock` 防止并发写入；系统需 util-linux 提供的 `flock`。

重要配置：

| 变量 | 默认 / 用途 |
| --- | --- |
| `BASE_MODEL`、`BASE_REVISION` | `Qwen/Qwen3-1.7B`、`main`；正式实验建议固定基座 commit |
| `TRAINED_MODEL` | run 目录的 `merged_model` |
| `TRAINING_CONFIG` | run 的 `config/resolved_training_config.yaml`，继承其中的 system prompt |
| `EVAL_DATASET_ID` | `HuggingFaceH4/ultrachat_200k` |
| `EVAL_DATASET_SPLIT`、`EVAL_PROMPT_COLUMN` | `test_sft`、`messages` |
| `EVAL_DATASET_REVISION` | `main`；正式实验应固定数据 revision |
| `NUM_PROMPTS`、`EVAL_SEED` | 100、42 |
| `EVAL_MAX_PROMPT_LENGTH`、`MAX_NEW_TOKENS` | 2048、1024 |
| `GENERATION_TEMPERATURE` | 0.7 |
| `BOOTSTRAP_ITERATIONS` | 1000；仅本地重采样，不是 1000 倍 API 请求 |
| `EVAL_DIR` | 指定已有目录可恢复同配置的评测；改变协议时使用新目录 |
| `PYTHON` | 当前 `python`，可显式指定环境解释器 |

若要测试自己的预处理数据集，需同时设置数据集 ID、split 与 prompt 列，不能只更改训练时的 `DATASET_NAME`：

```bash
EVAL_DATASET_ID=your_org/UltraChat-200k \
EVAL_DATASET_SPLIT=test EVAL_PROMPT_COLUMN=prompt \
EVAL_GPU=0 JUDGE_MODEL=你的judge模型 \
bash evaluate/run_grpo_chat_deepseek.sh /path/to/training-run
```

默认测试集不是训练过程中使用的 `val`。请独立检查数据来源和重复样本，避免通过测试集挑选超参数。

## 生成与缓存约定

生成脚本现在不再自动加入 `<think>/<answer>` 指令。启动器继承训练配置的 system prompt；独立调用默认使用空 system prompt，也可显式传 `--training-config` 或 `--system-prompt`。空字符串和 `null` 含义不同：前者保留空 system 消息，后者完全不添加。Qwen3 默认关闭 thinking，可独立调用时显式打开 `--enable-thinking`。

vLLM 与 Transformers 都先套用模板、以 `add_special_tokens=False` 编码，再保留相同上限的末尾 token；不再一个后端 2048 截断、另一个不截断。截断会丢失开头内容，长文本评测必须确认预算足够；两个后端的数值/采样实现仍可能不同，正式对照应固定后端，不承诺逐 token 相同。vLLM 不可用时不再静默换后端，需显式使用 `--no-vllm`。

每个生成文件包含版本化 contract：模型文件内容 SHA256、数据集 fingerprint、样本及输入 token 指纹、模板、长度、seed、采样参数、后端及依赖版本。Hub 模型先解析并下载到本地 snapshot，再按内容计算指纹；本地模型即使路径、大小和 mtime 不变，只要权重改变也会失效。哈希会读取权重，带来一次 CPU/磁盘开销，不占 GPU。

启动器总是调用生成入口的 `--reuse-existing` 校验，而非仅判断文件非空。contract 不一致、文件损坏或旧格式缺少元数据时，失败退出并保留原文件。请改用新 `EVAL_DIR`，不要为了通过校验手动改 JSON 元数据。

直接使用 judge 可以读取旧格式回答文件，但不能将旧格式与新 contract 格式混配；新格式两份回答的评测协议必须一致。历史回答不具备完整溯源信息，正式报告建议重新生成。

## 判决有效性与统计

- A/B 顺序随机化并记录，解析后映射回模型身份。
- 每个维度独立解析，严格接受 A/B（或启用平局时的 Tie），不把 `AB` 当作 A，也不跨章节借用 Winner。
- 五个维度必须全部解析成功，才产生 survey 判决；overall 单独解析。允许平局时 overall 也遵守该设置。
- 默认要求所有样本的 overall 和 survey 都有效，且至少两个有效样本。API 异常或格式错误会保留在结果中，不缓存为成功判决。
- 验证失败写入 `status=invalid` 并以非零状态退出，shell 写 `status=failed`；不会打印胜者。没有有效统计时输出 `null`，而非零胜率和零宽置信区间。
- 独立 judge CLI 可用 `--min-valid-fraction` 显式放宽完整率；此时结果是有效判决子集上的条件统计，必须披露排除比例，不可当作缺失完全随机。
- 同一批判决上进行本地 percentile bootstrap。B 增大不增加独立 prompt 数，也不证明模型收敛；有效样本数量看 `validation`，原始比例看 `validation.observed`。

缓存身份包含 endpoint、模型名、评分协议版本、解析器版本、thinking 模式、顺序和实际问答内容。修改协议或解析器必须同步更新对应版本。只有完整有效判决才能复用；重试失败样本仍可能产生 API 费用。

修复前的失败报告和解析错误缓存不应用于模型优劣结论；本轮没有删除这些历史文件，新协议不会复用旧缓存键。

## 无 GPU 绘图与回归测试

```bash
# 可选绘图依赖，不需要模型或 CUDA
python -m pip install 'matplotlib>=3.8,<4'
python evaluate/plot_win_rates.py /path/to/result_bootstrap.json \
  --output /path/to/win_rates.png

# CPU 测试建议使用独立环境，不改变已验证训练栈
python -m pip install -r requirements-test.txt
PYTHONPATH=src CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg python -m pytest -q tests/
```

绘图拒绝无效评测或缺少有效比较的报告；直接读取结果 JSON，不依赖特定模型名称或 checkpoint 文件名。旧多温度绘图入口的本地依赖也已补齐，遇到同一 checkpoint 的多份冲突结果会报错，避免按目录遍历顺序静默覆盖。

本轮回归覆盖解析异常、API/格式失败、缓存身份变化、生成主流程（模拟两种后端）、记录不覆盖、仅训练末尾验证、shell 退出状态以及实际 CPU PNG 输出。由于 GPU 被其他用户占用，没有进行修复后真实 GPU 生成或付费 API 验收；已有训练权重无需重训。
