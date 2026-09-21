# 五卡 GRPO 首轮实验配置与修改记录

## 1. 实验目标

首轮只运行 UltraChat 数据集上的原始 GRPO 优势计算，即 `studentization`，不运行 GOPO ranking。策略模型为 `Qwen/Qwen3-1.7B`，奖励模型为 `friendshipkim/QRM-Llama3.1-8B-v2`。为适配四张 24 GB RTX 4090 训练卡，策略模型采用 BF16 LoRA；奖励模型由 DeepSpeed ZeRO-3 分片。

## 2. GPU 部署

根据服务器拓扑，默认选择：

- GPU 0：独立 TRL vLLM server
- GPU 3、4、5、6：四进程 ZeRO-3 训练；四张卡同属 NUMA 1，彼此为 PIX 路径

GPU 编号没有写死在 Python 代码中。更换卡时直接设置：

```bash
VLLM_GPU=0 TRAIN_GPUS=3,4,5,6 bash train_scripts/qwen3_1.7_grpo_chat.sh
```

`TRAIN_GPUS` 必须恰好包含四张卡，且不能包含 `VLLM_GPU`。启动脚本会检查这些 GPU 是否对 `nvidia-smi` 可见。更换编号时尽量让四张训练卡位于同一 NUMA 节点；vLLM 卡可以位于另一个 NUMA 节点。

## 3. 训练配置

配置文件：`recipes/Qwen3-1.7B/config_chat_regular_qrm_lora_5gpu.yaml`

主要参数：

| 项目 | 设置 |
|---|---|
| 策略模型 | Qwen/Qwen3-1.7B |
| 数据集 | `${DATASET_NAME}`，默认 `${HF_USERNAME}/UltraChat-200k` |
| 奖励模型 | friendshipkim/QRM-Llama3.1-8B-v2 |
| 优势 | studentization |
| 损失聚合 | bnpo |
| LoRA | r=32, alpha=64, dropout=0 |
| LoRA 层 | q/k/v/o 与 gate/up/down projection |
| 训练卡 | 4 |
| 每卡 batch | 1 |
| 梯度累积 | 64 |
| 每次生成 | 256 条 completion，即 32 个 prompt × 8 generations |
| prompt/completion 上限 | 1024 / 1024 tokens |
| 学习率 | 1e-5 |
| 训练步数 | 500 |
| checkpoint | 每 100 步保存，只保留最近 3 个 |
| W&B | 默认 offline，可改为 online |

`MAX_STEPS`、`GENERATION_BATCH_SIZE`、`GRADIENT_ACCUMULATION_STEPS`、`MAX_PROMPT_LENGTH`、`MAX_COMPLETION_LENGTH` 和 `REWARD_BATCH_SIZE` 均可在启动命令前覆盖。启动器会检查 generation batch 是否等于 `4 × 每卡 batch 1 × 梯度累积`，并且能被 8 generations 整除，避免静默改变有效 batch。

这是低显存首跑配置，优化器的有效 completion batch 为 256，低于旧 recipe 的 1024。它适合先验证趋势和完整流水线，但论文最终对比应确保 GRPO 与 GOPO 使用完全相同的 LoRA、batch、步数和奖励模型配置。

## 4. 运行方法

确认已完成 README 中的安装和 UltraChat 预处理，然后执行：

```bash
export HF_USERNAME=你的HuggingFace用户名
export GOPO_OUTPUT_ROOT=/data/$USER/gopo_runs
export WANDB_MODE=offline

VLLM_GPU=0 TRAIN_GPUS=3,4,5,6 \
  bash train_scripts/qwen3_1.7_grpo_chat.sh
```

如果 `/data` 没有写权限，把 `GOPO_OUTPUT_ROOT` 改到根分区下有权限且空间充足的位置。恢复中断实验时指定原目录：

```bash
RUN_DIR=/data/$USER/gopo_runs/已有实验目录 \
VLLM_GPU=0 TRAIN_GPUS=3,4,5,6 \
  bash train_scripts/qwen3_1.7_grpo_chat.sh
```

如奖励模型阶段显存不足，优先降低：

```bash
REWARD_BATCH_SIZE=2
```

如 vLLM 初始化显存不足，优先降低：

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.75
```

## 5. 结果目录

每次新运行都会产生独立时间戳目录：

```text
run-directory/
├── config/
│   ├── resolved_training_config.yaml
│   ├── accelerate_config.yaml
│   └── pip-freeze.txt
├── logs/
│   ├── system_snapshot.txt
│   ├── vllm_server.log
│   ├── training.log
│   └── merge_lora.log
├── reward_data/
├── checkpoint-*/
├── adapter_config.json
├── run.env
├── run_manifest.json
├── RUN_STATUS
├── merged_model/
└── evaluation/
```

`RUN_STATUS` 标明成功或失败；`run_manifest.json` 保存解析后的训练、模型、数据参数和关键软件版本，但会遮盖凭据字段，同时保留 `token_broadcast` 等普通超参数。`pip-freeze.txt` 保存完整 Python 包清单。奖励记录按训练 step、micro-step 和进程编号分别原子写入，避免多进程覆盖或中断产生半个 JSON 文件。

建议单次实验至少预留 200 GB。脚本默认把 Hugging Face 模型缓存放在 `/data/$USER/gopo_cache`，可用 `GOPO_CACHE_ROOT` 改写。

## 6. DeepSeek V4.1 Flash 评估

训练成功后，脚本默认把 LoRA 合并到 `merged_model/`。评估时执行：

```bash
export DEEPSEEK_API_KEY=你的密钥
EVAL_GPU=0 bash evaluate/run_grpo_chat_deepseek.sh /data/$USER/gopo_runs/实验目录
```

评估配置：

- API base URL：`https://api.deepseek.com`
- 模型标识：`deepseek-flash`，对应 DeepSeek V4.1 Flash
- thinking：显式关闭，以控制输出量和费用
- 默认生成 100 个相同测试 prompt 的 base/model completion
- 只调用 100 次 judge API
- 1000 次 bootstrap 全部在本地完成
- 每次 API 判决立即追加到 `judge_cache.jsonl`，中断后可继续
- 最终 JSON 同时保存原始判决、token 使用量、抽样索引、win rate 和置信区间
- `evaluation.env` 和 `EVALUATION_STATUS` 记录本次评测配置与成功/失败状态，不包含 API key

API key 只通过环境变量读取，不写入 recipe、运行清单或日志命令。

### 费用估算（2026-09-21）

DeepSeek 官方当前对 `deepseek-flash` 的 cache-miss 输入报价为每百万 token 0.15 美元（非高峰）/0.30 美元（高峰），输出为 0.60/1.20 美元；cache hit 输入更低。当前脚本只有 100 次 API 判决，1000 次 bootstrap 不调用 API。若每次平均输入约 3000 token、输出约 1000 token，则总计约 30 万输入和 10 万输出 token，费用约为：

- 非高峰：`0.3 × $0.15 + 0.1 × $0.60 = $0.105`；
- 高峰：`0.3 × $0.30 + 0.1 × $1.20 = $0.21`。

实际文字长短会影响费用，建议首轮为账户准备 1 美元余额作为宽裕上限。最终 JSON 的 `api_usage` 保存服务端返回的 token 数，`cache_summary` 区分本轮 cache hit 和真实 API 调用数，可据官方实时价格复算。价格页面：https://api-docs.deepseek.com/quick_start/pricing/

## 7. 本次代码修复

1. 将训练从 `vllm_mode: colocate` 改为独立 server 模式，真正使用“一张 vLLM + 四张训练卡”。
2. 移除训练脚本原先不存在的 GPU 7 默认编号，改为可覆盖的 GPU 参数。
3. 新增 LoRA 合并工具，使训练后的 adapter 能被 vLLM 直接用于生成评估结果。
4. 将 checkpoint 保留数从最多 100 个降为 3 个，并关闭逐步 completion 表上传，控制磁盘占用。
5. 修复奖励保存时“全局 gathered reward 与本地 prompt 错位”的问题。
6. 奖励数据改为原子写入，并把 micro-step 纳入文件名，避免覆盖。
7. 生成结果改为原子写入，避免任务中断留下损坏 JSON。
8. DeepSeek 支持改为明确的 provider、base URL、模型名和 thinking 开关，不再依赖 API key 前缀猜服务商。
9. 修复原评估代码在每个 bootstrap iteration 中重复调用 judge 的高成本问题；现在先评判 N 个唯一样本，再本地 bootstrap。
10. 新增 append-only judge cache、自动重试、token 统计和最终结果原子保存。
11. 新增 `run_manifest.json`，保存无敏感信息的完整实验环境。
12. 补充 LoRA、生成和 DeepSeek 评估所需的显式 Python 依赖。
13. 修复 judge 对 `Winner: [A]`/`Winner: [B]` 格式的解析，并防止把普通文本中的字母误判为总体赢家。
14. 新增环境搭建、GitHub 管理文档、实验 issue 模板、PR 模板和轻量静态 CI。
15. 移除 UltraChat 预处理末尾遗留的交互式断点，并固定验证集抽样 seed=42，保证不同机器得到同一划分。
16. 把评测生成的 seed 显式传给 vLLM，而不再只把 seed 写进元数据。
17. 评测 completion 文件名纳入样本数、seed 和温度，避免改变配置后误复用旧文件；失败时也会写状态文件。

原来的全参数 recipes 均保留，便于以后做全参数 sanity check 或论文补充实验。
