# 验证与模型对比评测

这里区分两条流程：训练内 `trainer.evaluate()` 使用验证集和训练奖励；离线评测比较基础模型与训练模型在相同测试 prompt 上的回答，再交给独立 judge。两者不能互相替代，也不能把训练完成或 API 返回成功当作评测有效。

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
