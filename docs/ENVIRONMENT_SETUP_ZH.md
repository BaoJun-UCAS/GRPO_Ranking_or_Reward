# 五卡 GRPO 环境配置与运行清单

本文面向单机 Ubuntu 服务器、RTX 4090 24 GB、1 张独立 vLLM 卡和 4 张 ZeRO-3 训练卡。默认拓扑为 GPU 0 跑 vLLM，GPU 3/4/5/6 跑训练；卡号可以在命令行修改。

## 1. 硬件和系统前置条件

建议环境：

- Ubuntu 22.04 或兼容的 64 位 Linux；
- Python 3.11；
- NVIDIA 驱动能够运行 CUDA 12.4 构建的 PyTorch 2.6；
- 至少 5 张可用的 24 GB GPU；
- 系统内存建议 128 GB 以上；
- 单次实验预留至少 200 GB，模型缓存和多次实验建议 500 GB 以上。

先检查：

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi topo -p2p r
df -h / /data
free -h
```

本机已观察到 GPU 3/4/5/6 同属 NUMA 1 且彼此为 PIX，因此推荐把它们作为训练组，GPU 0 作为 vLLM。4090 没有 NVLink，ZeRO-3 会经过 PCIe 通信；同一 NUMA/PCIe 组能减少跨 NUMA 的额外开销。

## 2. 安装系统工具

需要有 sudo 权限的管理员执行：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential ninja-build pkg-config \
  gettext-base curl numactl libaio-dev
git lfs install
```

`gettext-base` 提供启动脚本使用的 `envsubst`，`ninja-build` 用于编译 FlashAttention，`libaio-dev` 供 DeepSpeed 使用。

## 3. 创建独立 Conda 环境

如果服务器还没有 Conda，先安装 Miniconda；已有 Conda 时从此处开始：

```bash
conda create -n gopo python=3.11 -y
conda activate gopo
python --version
python -m pip install --upgrade pip wheel packaging setuptools ninja
```

不要在同一个环境里混装另一套 torch/CUDA 版本。本项目的 vLLM 版本与 PyTorch 2.6.0 配套。

## 4. 克隆并安装项目

```bash
git clone git@github.com:YOUR_NAME/gopo-reproduction.git
cd gopo-reproduction
git checkout main
```

按固定顺序安装：

```bash
python -m pip install vllm==0.8.5.post1
MAX_JOBS=8 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
GIT_LFS_SKIP_SMUDGE=1 python -m pip install -e ".[judge]"
python -m pip check
```

如果 FlashAttention 报找不到 `nvcc`，说明只有驱动、没有 CUDA Toolkit。执行 `nvcc --version` 验证，然后让管理员安装与当前 PyTorch 构建兼容的 CUDA Toolkit；不要随意降级 torch，因为 vLLM wheel 与 torch 版本强关联。

验证关键版本和 CUDA：

```bash
python - <<'PY'
import torch, transformers, trl, vllm, deepspeed, flash_attn, peft
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("vllm", vllm.__version__)
print("deepspeed", deepspeed.__version__)
print("flash_attn", flash_attn.__version__)
print("peft", peft.__version__)
print("cuda_available", torch.cuda.is_available(), "gpu_count", torch.cuda.device_count())
PY

accelerate env
trl vllm-serve --help >/dev/null
bash -n train_scripts/qwen3_1.7_grpo_chat.sh
bash -n evaluate/run_grpo_chat_deepseek.sh
```

## 5. 准备缓存和结果目录

服务器 `/data` 空间充足，建议所有大文件放在那里：

```bash
mkdir -p /data/$USER/gopo_cache /data/$USER/gopo_runs
chmod 700 /data/$USER/gopo_cache /data/$USER/gopo_runs
export GOPO_CACHE_ROOT=/data/$USER/gopo_cache
export GOPO_OUTPUT_ROOT=/data/$USER/gopo_runs
```

启动器会把 Hugging Face 缓存放到 `$GOPO_CACHE_ROOT/huggingface`，每次运行创建一个独立时间戳目录。不要把这些目录放进 Git 仓库。

## 6. 登录 Hugging Face 与 W&B

```bash
hf auth login
wandb login
export HF_USERNAME=你的_HuggingFace_用户名
```

训练脚本默认 `WANDB_MODE=offline`，所以首跑不依赖网络上传：

```bash
export WANDB_MODE=offline
```

需要在线记录时改为 `WANDB_MODE=online`。不要把 `HF_TOKEN`、W&B key 或 DeepSeek key 写入仓库。

## 7. 准备 UltraChat 单数据集

默认训练数据是 `${HF_USERNAME}/UltraChat-200k`。首次预处理会下载原始数据并推送到你的 Hugging Face 账号：

```bash
python preprocess_data/preprocess_ultrachat_dataset.py
```

如果别人已经提供了同结构的数据集，无需重复预处理，直接设置：

```bash
export DATASET_NAME=组织名或用户名/UltraChat-200k
```

数据集必须至少包含 `train` 和 `val` split，prompt 列名为 `prompt`。训练前可快速检查：

```bash
python - <<'PY'
import os
from datasets import load_dataset
name = os.environ.get("DATASET_NAME", f"{os.environ['HF_USERNAME']}/UltraChat-200k")
ds = load_dataset(name)
print(ds)
print(ds["train"].column_names)
print(ds["train"][0]["prompt"])
PY
```

## 8. 先做两步 smoke test

等待五张目标卡空闲后运行。默认推荐 GPU 0 跑 vLLM，GPU 3/4/5/6 训练：

```bash
HF_USERNAME=你的用户名 \
VLLM_GPU=0 TRAIN_GPUS=3,4,5,6 \
MAX_STEPS=2 GENERATION_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=8 \
MAX_PROMPT_LENGTH=512 MAX_COMPLETION_LENGTH=256 REWARD_BATCH_SIZE=2 \
RUN_NAME=smoke-grpo-studentization-seed42 \
bash train_scripts/qwen3_1.7_grpo_chat.sh
```

`GENERATION_BATCH_SIZE=32` 能被每个 prompt 的 8 个 generation 整除，并与 4 卡 × 每卡 batch 1 × 梯度累积 8 对齐。smoke test 成功标准：

- vLLM health check 通过；
- 四个训练进程都启动；
- 至少完成两个 optimizer step；
- run 目录出现 `run_manifest.json`、reward data、训练日志和 `RUN_STATUS`；
- 结尾 LoRA 合并成功并产生 `merged_model/config.json`。

实时观察：

```bash
watch -n 2 nvidia-smi
tail -f /data/$USER/gopo_runs/smoke-grpo-studentization-seed42/logs/training.log
```

## 9. 启动正式 500 步实验

```bash
export HF_USERNAME=你的用户名
export GOPO_OUTPUT_ROOT=/data/$USER/gopo_runs
export GOPO_CACHE_ROOT=/data/$USER/gopo_cache
export WANDB_MODE=offline

VLLM_GPU=0 TRAIN_GPUS=3,4,5,6 \
RUN_NAME=qwen3-1.7b-ultrachat-grpo-studentization-qrm-lora-seed42 \
bash train_scripts/qwen3_1.7_grpo_chat.sh
```

默认正式配置为 1024/1024 token、8 generations、generation batch 256、梯度累积 64、500 steps、QRM batch 4。卡号无需改代码，例如换 vLLM 卡：

```bash
VLLM_GPU=2 TRAIN_GPUS=3,4,5,6 bash train_scripts/qwen3_1.7_grpo_chat.sh
```

四张训练卡尽量选择 `nvidia-smi topo -m` 中互相为 PIX/PXB 且属于同一 NUMA 节点的一组。

## 10. 中断恢复与常见问题

恢复同一个 run：

```bash
RUN_DIR=/data/$USER/gopo_runs/已有目录 \
VLLM_GPU=0 TRAIN_GPUS=3,4,5,6 \
bash train_scripts/qwen3_1.7_grpo_chat.sh
```

Trainer 会在输出目录中寻找最近 checkpoint；恢复前检查 `RUN_STATUS` 和日志。常见调整：

```bash
# 奖励模型 OOM
REWARD_BATCH_SIZE=2 bash train_scripts/qwen3_1.7_grpo_chat.sh

# vLLM OOM
VLLM_GPU_MEMORY_UTILIZATION=0.75 bash train_scripts/qwen3_1.7_grpo_chat.sh

# 通信错误时临时增加诊断信息
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL bash train_scripts/qwen3_1.7_grpo_chat.sh

# 若日志明确显示 PCIe P2P 初始化失败，再用共享内存/主机路径绕过 P2P（会变慢）
NCCL_P2P_DISABLE=1 bash train_scripts/qwen3_1.7_grpo_chat.sh

# 8000 或 29501 端口被占用
VLLM_PORT=8010 PORT=29511 bash train_scripts/qwen3_1.7_grpo_chat.sh
```

如果正式配置在 24 GB 上仍 OOM，依次降低 `REWARD_BATCH_SIZE`、`GENERATION_BATCH_SIZE`、`MAX_COMPLETION_LENGTH`。改变 generation batch 时应保持它能被 8 整除，并同步记录在实验 issue 中。不要用 `CUDA_VISIBLE_DEVICES` 包住总启动脚本；脚本会分别为 vLLM 与训练进程设置可见卡。

## 11. DeepSeek V4.1 Flash 评估

密钥只在当前 shell 设置。避免把带密钥的命令写进脚本或提交历史：

```bash
read -s DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
echo
```

训练完成后：

```bash
EVAL_GPU=0 bash evaluate/run_grpo_chat_deepseek.sh \
  /data/$USER/gopo_runs/qwen3-1.7b-ultrachat-grpo-studentization-qrm-lora-seed42
```

默认生成 base 与训练模型在同一批 100 个 prompt 上的回答，调用 `deepseek-flash` 100 次，关闭 thinking，然后在本地 bootstrap 1000 次。每个判决即时追加到 cache，中断后重跑会复用已完成项。评测产物在 run 目录的 `evaluation/` 下。

## 12. 正式实验验收清单

- [ ] `git status` 干净，记录了 commit SHA；
- [ ] 使用 `studentization`，不是 `ranking`；
- [ ] UltraChat 数据集与 split 正确；
- [ ] vLLM 和训练 GPU 不重叠；
- [ ] `run.env` 与 resolved YAML 已保存；
- [ ] `RUN_STATUS` 为 success；
- [ ] 最终 adapter 和 merged model 可加载；
- [ ] DeepSeek judge 的 parsing failure rate 可接受；
- [ ] JSON 结果、日志、W&B run 和 SHA256 已归档；
- [ ] 论文中的 GRPO/GOPO 对比使用相同 LoRA、batch、步数、seed 和评测 prompt。
