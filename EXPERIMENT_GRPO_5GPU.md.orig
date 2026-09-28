# GRPO 首轮实验配置与部署记录

## 实验目标与当前状态

首轮目标是在 UltraChat 上完成 `studentization` 优势计算的实验 baseline，再在相同条件下研究 ranking 等方法。策略模型为 `Qwen/Qwen3-1.7B`，奖励模型为 `friendshipkim/QRM-Llama3.1-8B-v2`，策略模型采用 BF16 LoRA，训练采用 DeepSpeed ZeRO-3。

当前 recipe 使用 `loss_type: bnpo` 和仓库自定义 trainer。`studentization` 的名字本身不保证论文原始 GRPO 的完整复现；正式报告前仍需审核损失归一化、KL 项、采样/更新策略和相关实现。部署整理没有替用户变更算法定义。

2026-09-24 已完成 1+4 张 RTX 4090 的两步 GPU 集成验收，包含真实权重同步、生成、反向传播、保存、合并及合并模型重新加载生成。更长训练、不同服务器、长输入与论文原始 GRPO 算法一致性仍需独立验证。

2026-09-25，用户确认已自行完成后续训练。本文以下保留可追溯的两步集成验收记录；尚未核对该次后续训练的配置和日志，因此不补写其步数、指标或收敛结论。

## 2026-09-24 GPU 验收结果

运行标识：`deployment-acceptance-20260924-reentrant`。测试基于 `2661095` 之上的本轮提交前工作树，日志中的旧 Git SHA 不代表只运行了旧代码；本轮提交包含所验收的运行时代码。验收后另修正了一个多卡日志统计分母错误（下述），不改变 loss 或权重更新。

| 项目 | 结果 |
| --- | --- |
| 硬件 | GPU 0 生成，GPU 4/5/6/7 训练；RTX 4090 24 GB |
| 数据 | 缓存的 UltraChat-200k：train 207365、val 500、test 23110 |
| 策略 revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| QRM revision | `23c6db70a35875248f45b3cfbe6d237ba8ac3e6d` |
| 核心环境 | torch 2.6.0+cu124、vLLM 0.8.5.post1、TRL 0.18.0、DeepSpeed 0.16.8、nvcc 12.4.131 |
| 配置 | smoke 默认值，另开 `MERGE_AFTER_TRAINING=1`；本地缓存离线加载、W&B offline |
| 服务与通信 | health、权重同步、prefix reset、真实生成均成功 |
| 优化 | `global_step=2`；梯度范数约 0.374、0.434；平均 train loss 0.0000377316 |
| 保存 | `checkpoint-2`、最终 adapter、配置与训练状态均存在 |
| 权重更新 | adapter 全部张量为有限值；196 个 LoRA B 张量有非零更新 |
| 合并与加载 | safe merge 成功；合并模型重新加载到 GPU 并生成 8 个 token |
| 退出 | launcher 返回 0，`RUN_STATUS` 为 success；GPU context 和三个实验端口释放 |

训练日志出现过负的 `completions/clipped_ratio`，原因是把跨卡 EOS 数量除以本卡样本数。已修正为全局分母并补单元测试；原始日志保留，不用于解释生成截断率。该字段只用于统计，不参与 loss。每次运行的 resolved YAML、完整 `pip-freeze.txt`、硬件快照、模型和日志保留在本地 run 目录，不提交权重或原始机器日志。

短测试中训练卡显存已接近 24 GB；不能直接据此认定正式默认 1024/1024 token、reward batch 4 能安全运行。应先保持已验证的小长度/batch，再逐项扩大并观察峰值。两个 step 仅证明流水线可执行，不构成模型效果或收敛证据。

## GPU 分配与批量约束

实验最初按 1 张生成卡 + 4 张训练卡设计，因此保留文件名 `config_chat_regular_qrm_lora_5gpu.yaml`。启动器现在从 `TRAIN_GPUS` 推导训练进程数，支持调整为 1+N 布局；这不保证相同模型在更少显存上能够运行。

此前检查的八张 RTX 4090 服务器中，GPU `4,5,6,7` 同属 NUMA 1、互为 `NODE` 路径，适合作为训练组；GPU `0` 可用于 vLLM。换服务器必须重新看拓扑和空闲情况，卡号没有通用含义。vLLM 卡与训练卡不可重叠。

```bash
export DATASET_NAME=你的组织或用户名/UltraChat-200k
VLLM_GPU=0 TRAIN_GPUS=4,5,6,7 python scripts/grpo.py smoke --dry-run
```

generation batch 由以下规则计算，显式覆盖时也会验证：

```text
GENERATION_BATCH_SIZE = 训练卡数 × PER_DEVICE_TRAIN_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS
GENERATION_BATCH_SIZE 必须能被 NUM_GENERATIONS 整除
```

## 默认参数

配置文件：[`recipes/Qwen3-1.7B/config_chat_regular_qrm_lora_5gpu.yaml`](recipes/Qwen3-1.7B/config_chat_regular_qrm_lora_5gpu.yaml)。下表按 4 张训练卡计算；解析后的 YAML 是每次运行的实际配置。

| 参数 | 正式实验默认 | `python scripts/grpo.py smoke` 默认 |
| --- | --- | --- |
| 策略模型 | Qwen/Qwen3-1.7B | 相同 |
| 奖励模型 | friendshipkim/QRM-Llama3.1-8B-v2 | 相同 |
| 数据集 | `DATASET_NAME`，或 `${HF_USERNAME}/UltraChat-200k` | 相同 |
| 优势 / 损失配置 | studentization / bnpo | 相同 |
| LoRA | r=32, alpha=64, dropout=0 | 相同 |
| LoRA 层 | q/k/v/o、gate/up/down projection | 相同 |
| 每卡 batch / generations | 1 / 8 | 1 / 8 |
| 梯度累积 | 64 | 8 |
| generation batch（4 卡） | 256 completions | 32 completions |
| 训练侧 prompt 截断 / completion 上限 | 1024 / 1024 tokens | 512 / 256 tokens |
| reward batch | 4 | 2 |
| optimizer steps | 500 | 2 |
| 学习率 | 1e-5 | 相同 |
| checkpoint | 每 100 步及训练结束时；保留最近 3 个 | 本次训练结束保存 `checkpoint-2` |
| 训练后 LoRA 合并 | 开启 | 关闭，减少 smoke 成本 |
| W&B | offline | offline |

smoke 会尊重用户显式设置的环境变量；若之前导出了 `MAX_STEPS=500` 等参数，应先取消或覆盖，以免把 smoke 变成正式训练。两个 optimizer step 只是验证流水线，不能用于模型效果或收敛判断。

算法和长输入仍需另行审核：当前继承的 trainer 仅截断训练侧的 prompt token，发给 vLLM 的仍是完整文本；QRM 输入也没有跟随该截断。因此 `MAX_PROMPT_LENGTH` 不是三条路径共享的输入上限，长样本可能超出服务上下文或增加奖励模型显存。奖励模型还固定使用 BF16、FlashAttention 2 和 `revision=main`。换模型、长文本数据或进行论文级对照前，应显式调整这些语义并固定模型 revision；本次部署整理不暗中修改训练算法。

## 运行、恢复与产物

安装和数据准备见 [部署指南](docs/ENVIRONMENT_SETUP_ZH.md)。GPU 空闲后执行：

```bash
export DATASET_NAME=你的组织或用户名/UltraChat-200k
export VLLM_GPU=0
export TRAIN_GPUS=4,5,6,7
python scripts/grpo.py doctor --cuda
python scripts/grpo.py smoke
# 需要同时验收合并时可单独开启：
MERGE_AFTER_TRAINING=1 python scripts/grpo.py smoke
# smoke 验收后：
python scripts/grpo.py train
```

输出根目录默认是仓库内 `grpo_runs/`，可用 `GRPO_OUTPUT_ROOT` 指向自己的大磁盘。新 run 使用独立名称，启动器拒绝覆盖已有 run 目录。恢复训练需使用明确 checkpoint 并写入新目录，保留原日志：

```bash
RESUME_FROM_CHECKPOINT=/path/to/old-run/checkpoint-100 \
RUN_NAME=resume-studentization-seed42 \
python scripts/grpo.py train
```

恢复应复用原模型、数据与关键训练超参数。是否能恢复取决于完整 checkpoint 是否存在，而不是运行是否叫 smoke；本次两步验收保存了 `checkpoint-2`。继续训练时还需使目标总步数大于已完成步数。只有最终 adapter、没有完整 checkpoint 时，不等同于恢复优化器等训练状态。`RUN_DIR` 可以指定新输出目录，不是自动覆盖旧日志的开关。

产物示例：

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
│   └── merge_lora.log       # 开启合并后
├── reward_data/
├── checkpoint-*/           # 达到保存周期后
├── adapter_config.json     # 成功保存 adapter 后
├── run.env
├── run_manifest.json       # trainer 初始化完成后
├── RUN_STATUS
├── merged_model/           # 开启合并并成功后
└── evaluation/             # 显式执行评测后
```

早期启动失败不会生成全部文件。`RUN_STATUS` 记录 launcher 状态，`run_manifest.json` 记录训练参数与版本并遮盖凭据字段；`pip-freeze.txt` 记录完整已安装包列表。奖励记录按 step、micro-step 和进程编号原子写入，避免多个进程互相覆盖。

## 本轮部署整理

此次整理围绕可重复部署、问题定位和机器迁移：

1. 统一项目/Conda 环境名称为 `grpo`，保留现有 `open_r1` 模块路径与研究方法名称。
2. 以 `environment.yml` 为统一入口，先安装 PyTorch/vLLM/项目，再安装 FlashAttention；记录磁盘不足与缺少 CUDA Toolkit 的真实失败原因。
3. 用项目内 vLLM 服务替代旧 TRL 的外层模型进程/Pipe 桥接，维护当前 TRL 0.18 所需协议；取消按日志静默时间自动重试。
4. GPU 列表决定训练进程数，路径/端口/模型/数据与 batch 配置可覆盖，启动前验证配置；不依赖个人 `/data/用户名` 路径。
5. 增加 `doctor`、`download`、`cache`、`smoke`、`train`、`logs` 及 Make 快捷命令；dry-run 只输出计划和解析配置。
6. 明确本地服务生命周期、失败日志、状态记录与新目录恢复方式，避免覆盖上次失败证据。
7. 增加无需 GPU 的测试和部署复盘，完成两步 GPU 集成验收，并保留迁移到新服务器时应重复的验收清单。

仓库已有的实验支持继续保留：LoRA 合并；checkpoint 保留上限；奖励/生成结果原子保存；确定性的 UltraChat 验证集划分；带 seed 的评测生成；judge 判决缓存与本地 bootstrap；实验清单、GitHub 模板和轻量 CI。旧全参数 recipes 也保留，后续使用前需独立核查显存和配置兼容性。

## 可选评估与实验验收

已有评估入口会在同一批 prompt 上生成基础模型和训练模型的回答，再调用外部 judge API。模型标识、API 行为与费用会变化，运行前核对当前服务商信息并显式选择 `JUDGE_MODEL`；本文不固定服务商代际或价格。

```bash
# 预先通过安全方式设置 DEEPSEEK_API_KEY；此命令会调用外部付费 API。
JUDGE_MODEL=服务商当前可用的模型标识 \
EVAL_GPU=0 bash evaluate/run_grpo_chat_deepseek.sh /path/to/successful-run
```

默认评测规模是 100 个独立 prompt pair、1000 次本地 bootstrap；已成功缓存的判决可复用，但重试可能增加真实 API 请求数。API key 不应写入仓库、运行清单或公开日志。

新实验验收清单（每次实验分别填写）：

- [ ] 记录 commit、完整依赖版本、数据版本和解析后的 YAML。
- [ ] 选定 GPU 已分配且 vLLM/训练组无重叠。
- [ ] 服务 readiness、生成和训练权重同步均成功。
- [ ] smoke 完成两个 optimizer step，保存 adapter，`RUN_STATUS` 为 success。
- [ ] 显式开启合并的测试或正式训练产出可加载的 merged model。
- [ ] 退出后本次进程、GPU context 与监听端口已释放。
- [ ] 对原始 GRPO 的算法一致性已单独审核，不能仅凭 `studentization` 名字验收。
- [ ] baseline/ranking 对照使用一致的模型、奖励、LoRA、batch、步数、seed 和评测 prompt。
