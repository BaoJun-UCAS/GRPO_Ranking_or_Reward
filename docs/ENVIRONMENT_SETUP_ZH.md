# GRPO 部署与运行指南

本文使用项目根目录下的 `environment.yml` 作为统一安装入口，面向 Linux、Python 3.11、NVIDIA CUDA GPU。当前四卡默认布局为：物理 GPU 4 运行策略 rollout（vLLM），GPU 5 运行独立 8B QRM 服务，GPU 6–7 运行两进程 DeepSpeed ZeRO-2 LoRA 策略训练；启动器不会使用 GPU 0–3。

当前状态：2026-09-24 的 1+4 卡验收属于旧布局的历史证据；新的 1+1+2 四卡拆分已完成静态与 CPU 模拟测试，仍应先运行两步真实 GPU smoke，再开始正式训练。`studentization`、`bnpo` 和自定义 trainer 仍是实验 baseline，不能直接宣称为论文原始 GRPO 的严格复现。历史记录见 [旧五卡实验说明](../EXPERIMENT_GRPO_5GPU.md)，当前排查方法见 [部署问题复盘](DEPLOYMENT_TROUBLESHOOTING_ZH.md)。

## 1. 检查账户与系统

```bash
command -v conda
conda info --envs
uname -m
nvidia-smi -L
nvidia-smi topo -m
df -h
df -i
free -h
```

如果 `conda` 命令不存在，也可能只是当前 shell 没有加载已有安装。先检查你实际使用的安装目录，例如 `test -f "$HOME/miniconda3/etc/profile.d/conda.sh"`；存在时执行 `source "$HOME/miniconda3/etc/profile.d/conda.sh"`，无需重装。不要假定别人的 `/data/用户名/miniconda3` 路径在新服务器也存在。

没有安装时，可以按 [Miniconda 官方安装说明](https://www.anaconda.com/docs/getting-started/miniconda/install)安装到自己的可写目录，无需 sudo。以下仅适用于 `uname -m` 为 `x86_64` 的 Linux；其他架构要选择对应安装包，并重新核对本项目二进制依赖是否支持：

```bash
GRPO_INSTALLER_DIR=$(mktemp -d)
curl -fL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o "$GRPO_INSTALLER_DIR/miniconda.sh"
sha256sum "$GRPO_INSTALLER_DIR/miniconda.sh"
# 对照官方下载列表的 SHA256 后，交互式安装并选择自己的安装目录。
bash "$GRPO_INSTALLER_DIR/miniconda.sh"
```

按照安装结果打开新终端，或 `source` 实际安装目录的 `etc/profile.d/conda.sh`。本项目不要求更改系统 Python。

## 2. 选择磁盘和缓存目录

依赖下载、解压、编译会同时占用临时目录和目标环境；模型还会占用单独的 Hugging Face 缓存。不能只看项目所在磁盘的剩余空间。将下面的 `GRPO_WORK_DIR` 改成自己有权限且空间充足的位置；`$HOME` 只是跨服务器可用的示例，不代表它一定有足够空间。

```bash
export GRPO_WORK_DIR="$HOME/grpo-work"
mkdir -p "$GRPO_WORK_DIR/tmp" "$GRPO_WORK_DIR/cache/pip" "$GRPO_WORK_DIR/runs"
export TMPDIR="$GRPO_WORK_DIR/tmp"
export PIP_CACHE_DIR="$GRPO_WORK_DIR/cache/pip"
export GRPO_CACHE_ROOT="$GRPO_WORK_DIR/cache/grpo"
export HF_HOME="$GRPO_CACHE_ROOT/huggingface"
export GRPO_OUTPUT_ROOT="$GRPO_WORK_DIR/runs"
df -h "$TMPDIR" "$GRPO_WORK_DIR"
df -i "$TMPDIR" "$GRPO_WORK_DIR"
```

建议初次实验预留约 100–200 GB 作为容量规划起点，实际需要受环境、模型版本、checkpoint 数量和训练步数影响；这不是精确下载量。BF16 参数量估算下，1.7B 策略模型与 8B 奖励模型的权重合计约 19.4 GB（十进制），实际仓库下载还可能包含额外文件。可以先下载再运行训练，避免把下载时间误认成初始化挂起。

模型 Hub 缓存优先级为 `HF_HUB_CACHE`、兼容旧设置的 `HUGGINGFACE_HUB_CACHE`、`HF_HOME/hub`。项目默认 `HF_HOME` 为 `${GRPO_CACHE_ROOT}/huggingface`。缓存根目录依次选择显式 `GRPO_CACHE_ROOT`、显式 `XDG_CACHE_HOME/grpo`、可写的 `/data/<用户>/cache/grpo`，最后回退到 `$HOME/.cache/grpo`；可用 `GRPO_DATA_ROOT` 改写自动探测的数据盘根目录。在不同终端运行下载和训练时，应使用同一组路径，否则可能重复下载。项目不会自动加载 `.env`；这些变量可放入自己保管的 shell 配置片段，再显式 `source`。

vLLM TP>1 会创建 Unix IPC socket。启动器默认给 vLLM 单独创建长度受控的 `/tmp/grpo-vllm.*`，退出时只删除自己创建的目录；训练过程的 `TMPDIR` 仍保存在 run 目录。若显式设置 `VLLM_TMPDIR`，路径长度不得超过 60 个字符，且启动器不会删除用户提供的目录。

训练启动器还将 Torch 扩展、Triton 和 vLLM 的缓存默认放到 `GRPO_CACHE_ROOT` 下的 `torch_extensions`、`triton`、`vllm` 子目录，避免模型在大盘、编译缓存却写满系统盘。显式设置 `TORCH_EXTENSIONS_DIR`、`TRITON_CACHE_DIR`、`VLLM_CACHE_ROOT` 时保留你的选择；`doctor` 会显示这些路径。首次安装 FlashAttention 的构建缓存与 pip 临时目录仍按安装时的环境变量管理。

## 3. 创建环境

进入已经克隆的项目根目录，依次执行；上一步失败时先处理错误，不要继续安装下一项：

```bash
conda env create -f environment.yml
conda activate grpo
python --version
python -m pip check
```

也可用 `make env-create` 创建环境。已有同名环境时先检查 `conda info --envs`。如要修复依赖安装中断的现有环境，可以在解决磁盘/网络问题后执行 `conda env update -n grpo -f environment.yml`，并重新完成下方检查；不要直接删除整个 Miniconda 或公共缓存。

关键兼容边界为 Python 3.11、PyTorch 2.6.0、vLLM 0.8.5.post1、TRL 0.18.0、Transformers 4.52.3。`environment.yml` 固定关键二进制版本，`setup.py` 声明项目依赖；它们不是所有传递依赖的完整锁文件。不要只单独升级 torch/vLLM/TRL 中的一个。实际运行时保存的 `pip-freeze.txt` 可用于比较两台机器上的依赖漂移。

环境文件同时安装 NVIDIA CUDA 12.4 编译器和开发库（不安装或替换系统驱动）。这不只是 FlashAttention 源码安装的需要：本次 GPU 实测中，缺少 `nvcc` 导致 DeepSpeed 0.16.8 在导入阶段直接失败。仅安装 PyTorch 的 CUDA wheel 或预编译 FlashAttention 不足以补齐这些工具。已有旧 `grpo` 环境可单独补充：

```bash
conda install -n grpo --override-channels -c nvidia/label/cuda-12.4.1 \
  cuda-nvcc=12.4 cuda-libraries-dev=12.4 cuda-nvtx=12.4 cuda-cupti=12.4
conda activate grpo
test -x "$CONDA_PREFIX/bin/nvcc"
export CUDA_HOME="$CONDA_PREFIX"
```

此开发包组合基于 [NVIDIA Conda 安装说明](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#conda-installation)。不要只补一个 `nvcc` 可执行文件：Torch 的扩展头文件还会引用 cuBLAS/cuSPARSE/cuSOLVER 等开发头文件。[DeepSpeed 安装说明](https://www.deepspeed.ai/tutorials/advanced-install/)也要求核对编译器 CUDA 与 PyTorch CUDA 版本。若使用管理员提供的 Toolkit，改用其真实目录，并保留编译版本检查。

`Installing pip dependencies` 阶段可能长时间只显示转圈，Conda 会汇总 pip 子进程输出。是否失败应看退出码和完整日志；第一次部署实测错误是磁盘不足，而不是转圈本身，详见故障文档。

## 4. 安装 FlashAttention 并做无 GPU 检查

默认训练 recipe 使用 `flash_attention_2`。它在 PyTorch 安装成功后单独安装：

```bash
nvcc --version
export CUDA_HOME="$CONDA_PREFIX"  # 仅适用于上面安装在当前 Conda 环境内的 Toolkit。
MAX_JOBS=8 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip check
python scripts/grpo.py doctor
```

如果 `nvcc` 不存在，先安装或加载与 torch CUDA 构建兼容的 CUDA Toolkit。集群常通过管理员提供的环境模块加载，具体路径由服务器决定。若 Toolkit 已在某目录，先验证该目录的 `bin/nvcc` 确实存在，再把 `CUDA_HOME` 指向它、把其 `bin` 加入 `PATH`。只设置一个不存在的 `CUDA_HOME` 不会解决问题。驱动和 PyTorch 自带的 CUDA 运行库并不等同于 CUDA 编译工具链；FlashAttention 的源码安装要求见 [官方 v2.7.4.post1 说明](https://github.com/Dao-AILab/flash-attention/tree/v2.7.4.post1#installation-and-features)。

系统还需 Bash、`setsid`（通常来自 util-linux）、`nvidia-smi`，源码编译需 C/C++ 编译器。`environment.yml` 已提供 git、curl、gettext/envsubst、ninja 等用户环境工具；系统缺少的工具由管理员按实际发行版补齐，无需照抄另一台服务器的 apt 命令。

`doctor` 不初始化 CUDA，也不启动服务或训练；GPU 被其他用户占用时可运行。`pip check` 只检查已经安装的软件包的依赖声明，缺少整个项目或 CUDA 编译组件时也可能通过。CUDA 可见性和编译扩展导入检查留到目标卡可用后：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/grpo.py doctor --cuda
```

这里的卡号仅是例子，应换为分配给自己的 GPU。`doctor --cuda` 不执行真实训练或生成 kernel，不能代替 smoke。仅运行无 GPU 测试时，可安装 `python -m pip install -r requirements-test.txt`，再执行 `make test`，无需安装整套训练依赖。

## 5. 预下载策略与奖励模型

```bash
python scripts/grpo.py download
python scripts/grpo.py cache
```

默认预下载 `Qwen/Qwen3-1.7B` 和 `friendshipkim/QRM-Llama3.1-8B-v2`；下载阶段不占 GPU。指定其他仓库时可重复 `--model`：

```bash
python scripts/grpo.py download --model Qwen/Qwen3-1.7B
```

要周期性查看当前缓存文件变化：

```bash
watch -n 5 'python scripts/grpo.py cache'
```

缓存体积和 `.incomplete` 文件能提示是否有下载活动，但不等于精确百分比；使用 Xet 等下载方式时，中间缓存可能位于其他子目录。显式下载命令的进度和成功退出才是更可靠的信号。缓存下载完成也不代表模型已经加载到 GPU。

QRM 奖励模型由 `open_r1.reward_server` 在物理 GPU 5 单独加载；策略训练进程不会再复制 8B QRM。训练 rank 会把待评分样本聚合到主 rank，由主 rank 发出一次 HTTP 请求，再广播并切分奖励结果。

## 6. 准备数据集

如果已有兼容数据集，直接设置完整 ID，无需 `HF_USERNAME`：

```bash
export DATASET_NAME=你的组织或用户名/UltraChat-200k
```

数据集需提供 `train`、`val` split 及 `prompt` 列，后续评测还需要 `test`。从原始 UltraChat 创建自己的版本时，预处理脚本会下载并**上传到自己的 Hugging Face 账号**，需要写权限 token：

```bash
export HF_USERNAME=你的HuggingFace用户名
read -rsp 'Hugging Face write token: ' HF_TOKEN
export HF_TOKEN
printf '\n'
python preprocess_data/preprocess_ultrachat_dataset.py
export DATASET_NAME="$HF_USERNAME/UltraChat-200k"
```

不要把 token 写入 Git。训练公共数据集或下载公共模型本身不要求写权限。默认 W&B 为 `offline`；只有切换 `WANDB_MODE=online` 才需要在线账号配置。

## 7. 预览、两步训练、正式运行

当前启动器默认使用物理卡 4–7，并设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`：GPU 4 为 vLLM、GPU 5 为 QRM、GPU 6–7 为策略训练。三个角色必须互不重叠；启动前仍要用 `nvidia-smi` 和 `nvidia-smi topo -m` 确认这些卡已分配且空闲。

```bash
export VLLM_GPUS=4
export QRM_GPU=5
export TRAIN_GPUS=6,7
python scripts/grpo.py smoke --dry-run
```

`--dry-run` 检查并打印运行计划和解析后的配置，不创建 run 目录、不下载模型、不初始化 GPU。确认配置后，等 GPU 可用再执行：

```bash
python scripts/grpo.py smoke
# 两步实验通过后，再执行配置的正式实验：
python scripts/grpo.py train
```

对应快捷命令是 `make dry-run`、`make smoke`、`make train`。也可直接使用 `bash train_scripts/qwen3_1.7_grpo_chat.sh --dry-run` 或去掉 `--dry-run` 执行训练。不要在总启动器外包一层 `CUDA_VISIBLE_DEVICES` 重新编号；启动器会分别为服务和训练设置可见 GPU。

一次实验需要四个不同的本机端口：`VLLM_HTTP_PORT`（默认 8000）、`QRM_HTTP_PORT`（默认 8001）、训练 rendezvous 的 `PORT`（默认 29501），以及权重同步的 `VLLM_GROUP_PORT`（默认 51216）。并行实验还需使用不同 GPU、run 目录和全部四个端口。例如：

```bash
VLLM_HTTP_PORT=8010 QRM_HTTP_PORT=8011 PORT=29511 \
VLLM_GROUP_PORT=51226 python scripts/grpo.py train
```

启动器在启动前检查端口，但检查不是预留；若随后发生端口竞争，仍需依据当次日志处理。

不要将 vLLM 原生的 `VLLM_PORT` 当作 HTTP 端口：它实际用于引擎内部通信。启动器兼容旧变量作为 HTTP 端口的弃用别名，但不会继续向模型进程传递它。直接调用服务模块时使用 `--port` 配置 HTTP；不要同时把 `VLLM_PORT` 设成相同值。

训练进程数由 `TRAIN_GPUS` 数量推导，vLLM tensor-parallel 大小由 `VLLM_GPUS` 数量推导。当前默认是 `VLLM_GPUS=4`、`QRM_GPU=5`、`TRAIN_GPUS=6,7`、每卡 batch 1、梯度累积 128，有效 generation batch 为 256。旧的单数变量 `VLLM_GPU` 会被明确拒绝，避免遗留的 `VLLM_GPU=0` 覆盖新布局。generation batch 自动取：

```text
训练卡数 × PER_DEVICE_TRAIN_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS
```

如果显式设置 `GENERATION_BATCH_SIZE`，它必须满足上述约束且能被 `NUM_GENERATIONS` 整除。smoke 默认两步、梯度累积 8、prompt 上限 512、completion 上限 256、QRM batch 1，并跳过合并；它验证流水线，不用于报告收敛结果。数据映射阶段会在保留聊天结构的前提下左截断最后一条用户内容，因此策略训练、vLLM rollout 和 QRM 共享同一结构化 prompt。快捷命令尊重显式环境变量，运行前注意清除旧覆盖。

两步验收需要同时满足：服务 health 成功；生成和训练侧权重同步无错误；完成两个 optimizer step；adapter 保存成功；`RUN_STATUS` 为 success。要连同合并验收，执行 `MERGE_AFTER_TRAINING=1 python scripts/grpo.py smoke` 并确认合并产物可加载。正式 `train` 默认开启合并。只看到权重下载或 `/health/` 返回 200 都不等于训练成功。

当前四卡 recipe 使用两进程 ZeRO-2；LoRA 下可通过禁用 adapter 复用基础策略作为 reference，避免 ZeRO-3 路径额外复制 reference model。recipe 仍保留重入式 gradient checkpoint。旧 ZeRO-3 配置的非重入 checkpoint 故障属于历史兼容边界，详见故障复盘。

## 8. 查看日志、停止与恢复

```bash
python scripts/grpo.py logs --service vllm --follow
python scripts/grpo.py logs --service qrm --follow
python scripts/grpo.py logs --service training --follow
# 多人或多次运行时明确指定目录，避免查看错日志：
python scripts/grpo.py logs --run-dir /path/to/run --service qrm --follow
```

默认输出根目录是仓库内的 `grpo_runs/`，可用 `GRPO_OUTPUT_ROOT` 改到大磁盘。一次运行的配置、版本、硬件快照、训练/服务日志、状态和模型都集中在该 run 目录。健康检查的作用是防止训练在生成服务尚未就绪时启动；超时只是“未就绪”的结果，原因仍需查看 `vllm_server.log` 中的首次异常。

在启动器终端按 Ctrl+C 会触发其管理的子进程清理；在 `logs --follow` 或 `tail -f` 终端按 Ctrl+C 只停止日志跟随。异常终止后应检查端口、进程归属和命令行，避免按模糊名称杀掉其他实验；具体检查见 [故障排查](DEPLOYMENT_TROUBLESHOOTING_ZH.md#停止后的进程与端口检查)。

恢复时指定原 checkpoint，复用原来的模型、数据和训练超参数，并输出到一个新 run：

```bash
RESUME_FROM_CHECKPOINT=/path/to/old-run/checkpoint-100 \
RUN_NAME=resume-studentization-seed42 \
python scripts/grpo.py train
```

启动器拒绝覆盖已有 run 目录，原始失败日志和 checkpoint 会保留。`RUN_DIR` 可指定新输出位置；它不负责覆盖或复用旧目录。恢复需要完整 checkpoint，而非仅最终 adapter；smoke 若保存了完整 `checkpoint-2` 也可恢复，继续训练时目标总步数应大于已完成步数。没有 checkpoint 的启动失败目录不能恢复训练；固定 `RUN_NAME` 也不能替代检查 checkpoint。首次正式运行建议使用默认的时间戳名称。

## 9. 扩展与评测

验证配置、独立评测目录、缓存有效性、失败状态和无 GPU 绘图见 [验证与评测指南](EVALUATION_ZH.md)。修复后的评测不再直接复用旧格式生成文件，旧判决缓存也会因协议版本更新失效；保留历史证据并新建评测目录。

更换服务器主要改 GPU 分配和缓存/结果路径；更换模型、数据集或奖励方法则需要相应 recipe 与代码适配。部署变量通过环境传入，实验超参数以解析后的 YAML 为准，奖励函数扩展放在 `src/open_r1/rewards.py` 对应注册逻辑，避免继续在启动脚本堆叠机器专用分支。

训练结束后可用已有 `generate/` 和 `evaluate/` 脚本比较同一批 prompt 上的基础模型与训练模型。`evaluate/run_grpo_chat_deepseek.sh` 会调用外部付费 judge API；运行前自行核对可用模型标识、API 参数与当前价格，设置正确的密钥和 judge 配置。GPU smoke、离线单元测试、付费评估是不同步骤，部署快捷命令不会自动启动付费评估。

提交实验报告时至少保留 commit SHA、resolved YAML、`pip-freeze.txt`、硬件信息、数据版本、seed、checkpoint 和验收日志；密钥、模型权重和缓存不应提交到 Git。
