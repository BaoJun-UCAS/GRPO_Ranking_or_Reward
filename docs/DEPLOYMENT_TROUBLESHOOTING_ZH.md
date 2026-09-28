# 部署问题复盘与排查

本页记录 2026-09-23–24 部署中实际遇到的现象、能支持的结论，以及可在其他服务器复用的检查方法。原始记录包含本地终端输出和诊断日志；用户路径、临时 PID、凭据及完整机器日志不放入仓库。新服务器安装步骤见 [部署指南](ENVIRONMENT_SETUP_ZH.md)。

## 已观察到的问题

| 现象 | 证据与结论 | 现在的处理方式 |
| --- | --- | --- |
| `Installing pip dependencies` 长时间只转圈 | Conda 正在等待 pip 子进程；转圈本身不能证明卡死 | 安装前检查磁盘；保留完整输出和退出码，必要时查看 pip 子进程 |
| 下载 torch 时报 `ProtocolError`、`CondaEnvException: Pip failed` | 完整 traceback 的底层错误是 `OSError: [Errno 28] No space left on device`，发生在临时文件写入；不能按网络中断处理 | 将 `TMPDIR`、pip 缓存、环境目标放在空间足够的位置，同时检查 inode |
| 安装失败后 `pip check` 仍通过 | 当时 torch/项目依赖未安装完整；该命令只检查已安装包的依赖声明 | 用 `doctor` 检查关键包是否存在，再做明确的 CUDA/训练验收 |
| FlashAttention 安装失败 | 第二次环境已成功创建；额外安装明确报 `nvcc was not found` 与 `CUDA_HOME ... is not set` | 环境创建与 FlashAttention 编译分两步；配置真实存在的兼容 CUDA Toolkit |
| `gopo`/`grpo`、包名和旧说明混杂 | 早期环境、启动器与文档使用了不同命名 | 当前项目/环境统一为 `grpo`，保留 `open_r1` Python 模块路径以避免无关导入迁移 |
| 服务一直停在 health 检查 | 诊断中捕获到旧 TRL 服务子进程的 CUDA fork 异常，父进程继续等 Pipe 的 ready 消息 | 使用项目内服务入口，直接在服务进程初始化模型，启动失败传播错误 |
| 改成 spawn 并重试两次仍失败 | 失败日志停在 engine 初始化；未取得对应现场的阻塞栈，因此不能确证剩余问题是线程池竞争 | 删除“日志静默即重启”策略；保留有界等待、阶段日志和失败输出，不把重试计作修复 |
| GPU 验收时模型加载、CUDA 图编译完成，但 HTTP 报 `address already in use` | 项目导出的 `VLLM_PORT` 与 vLLM 原生变量重名：内部 TCPStore 先绑定该端口，HTTP 再次绑定失败；不是另一个用户的服务 | HTTP 配置改名 `VLLM_HTTP_PORT`，不再向引擎传递旧 HTTP 别名；内部通信、HTTP、训练集合和权重同步分离 |
| health 成功后 DeepSpeed 报 `CUDA_HOME does not exist` | GPU 验收进入训练端后，DeepSpeed 0.16.8 导入时检查本地 Toolkit；环境没有 `nvcc`，尽管 torch 和 FlashAttention wheel 能导入 | 环境文件加入匹配的 CUDA 12.4 开发工具；`doctor` 将缺失编译器报告为失败，`doctor --cuda` 增加 DeepSpeed 导入检查 |
| 模型和通信初始化完成，权重更新 HTTP 请求报 502 | 当次环境的 Python requests 确实把 `127.0.0.1` 请求交给 HTTP 代理；服务日志只有已到达请求的 200，没有对应 502，指向代理链路问题 | 为启动器所有子进程统一补充 loopback `NO_PROXY`/`no_proxy`，保留原有外网代理和绕过列表；仅让 health 的 curl 绕过代理不够 |
| 生成、奖励打分成功，首次反向传播报 `CheckpointError`，权重 shape 变成 `[0]` | PEFT 冻结参数在 ZeRO-3 非重入 checkpoint 重算中被释放；本机 Torch/DeepSpeed 源码与上游报告对应 | 固定版本 recipe 使用 `use_reentrant: true`；正确传递 checkpoint kwargs 并启用输入梯度；启动前拒绝该不兼容组合，不关闭 metadata 检查 |
| 进程结束后难判断是否有残留 | 旧父进程可能仍活着；`tail -f ...vllm...` 也会被宽泛关键词匹配 | 启动器管理自己创建的进程；排查时核对 UID、PID、父子关系和实际监听端口 |
| 不知道模型下载到哪里、奖励模型在哪里运行 | 缓存变量和多个终端可能不一致；vLLM 日志不代表 QRM 服务状态 | `download` 与 `cache` 使用同一缓存解析规则；默认 QRM 在物理 GPU 5 独立加载，并写入 `qrm_server.log` |

## vLLM 健康检查失败：已知原因与验证边界

旧入口 `trl vllm-serve` 在 TRL 0.18.0 中先加载 Torch/vLLM，再创建额外模型进程。本次诊断捕获过 `Cannot re-initialize CUDA in forked subprocess`：父进程已有 CUDA 状态后使用 `fork`，子进程无法按此路径重新初始化 CUDA。父进程同时阻塞在等待子进程 ready 消息的 Pipe 读取，导致服务没有完成启动，健康检查持续超时。上游也报告过同类错误，见 [TRL issue #3450](https://github.com/huggingface/trl/issues/3450)。

`VLLM_WORKER_MULTIPROC_METHOD=spawn` 只约束 vLLM 内部进程，不足以控制旧 TRL 那一层额外 `multiprocessing.Process`。曾增加全局 spawn 包装器后，有些诊断启动成功，但用户终端仍出现两次 engine 初始化停滞。没有捕获到这两次停滞现场的精确堆栈，因此不能把“线程池竞争”等推测写成已确认根因，也不能说最初的 fork 异常解释了每一次后续失败。

后续空闲 GPU 验收发现了另一个可复现的代码错误：把 HTTP 端口导出为 `VLLM_PORT` 会让 vLLM 的内部 TCPStore 先占用相同端口，再导致 HTTP 绑定失败。这个变量的内部通信含义在 [vLLM 0.8.5 官方文档](https://docs.vllm.ai/en/v0.8.5/serving/env_vars.html)中已有明确警告。改为 `VLLM_HTTP_PORT` 并隔离旧别名后，本次实测已收到 HTTP 200 并进入训练阶段。两个问题属于不同的启动层次，不能只用“显卡忙”或“再等一会”概括。

当前入口是 [`python -m open_r1.vllm_serve`](../src/open_r1/vllm_serve.py)。它移除 TRL 的额外模型进程和 Pipe 桥接，让服务进程直接持有 LLM，保留 TRL 0.18 客户端需要的生成、权重同步与通信接口。启动器让服务在本地监听、使用这一依赖组合对应的 V0 协议；它不是任意新版 vLLM/TRL 的通用替代品。升级时应共同检查服务与 trainer 客户端协议，并重做集成测试。

此前服务初版的 GPU 测试越过原卡点并到达权重加载，随后出现显存不足；当时 GPU 已被其他用户占用。该结果本身只能证明初版越过原卡点。用户确认 GPU 空闲后，2026-09-24 的后续验收才完成了完整 health、生成、NCCL 权重同步、两步训练、保存、合并及合并模型重新加载生成，见 [验收记录](../EXPERIMENT_GRPO_5GPU.md#2026-09-24-gpu-验收结果)。最初失败时的 GPU 空闲记录，不能用之后的 GPU 占用来反推解释。

## 根据故障阶段检查

### 环境安装和磁盘

```bash
command -v python
python -m pip --version
conda info --envs
python scripts/grpo.py doctor
df -h "${TMPDIR:-/tmp}" "$CONDA_PREFIX"
df -i "${TMPDIR:-/tmp}" "$CONDA_PREFIX"
python -m pip cache info
```

上面涉及 `CONDA_PREFIX` 的命令应在已激活 Conda 环境后执行。日志内的第一条异常通常比最后的 `Pip failed` 更有信息。`No space left` 时还要检查 inode、配额以及安装目录，不要只换镜像源；网络超时、DNS 或证书错误才分别检查网络、代理和证书。

删除旧环境前先确认其名称和路径，并退出该环境：

```bash
conda info --envs
conda deactivate
conda env remove -n gopo
conda info --envs
```

这仅适用于确认不再需要的旧 `gopo` 环境，会删除该环境的软件；不会删除独立模型缓存、实验结果或整个 Conda 安装。缓存可能被多个环境复用，不应把“卸载环境”扩展为批量删除账号目录。当前 `grpo` 环境无需因为历史改名而重建。

### 下载与缓存

```bash
python scripts/grpo.py cache
python scripts/grpo.py download
```

两个命令会显示解析后的缓存位置。首次下载可能耗时较长，可先完成下载，再启动训练。缓存的 `.incomplete` 文件大小不增长只是一条线索；不能只凭它判定死锁，尤其在 Xet 下载、解压或 CPU 编译阶段。判断完成以下载命令成功退出和目标 snapshot 为准。

### 启动到健康检查

```bash
python scripts/grpo.py logs --run-dir /path/to/run --service vllm
python scripts/grpo.py logs --run-dir /path/to/run --service qrm
curl --noproxy '*' --connect-timeout 1 --max-time 3 -i \
  "http://127.0.0.1:${VLLM_HTTP_PORT:-8000}/health/"
curl --noproxy "*" --connect-timeout 1 --max-time 3 -i \
  "http://127.0.0.1:${QRM_HTTP_PORT:-8001}/health/"
ss -ltnp
```

本地检查明确绕过代理。`curl` 超时表示本次请求在限定时间内没有成功，不直接指明下载、CUDA、编译、通信还是端口问题。按日志的第一个失败阶段处理：

- 下载阶段：检查模型权限、缓存路径、磁盘和网络。
- CUDA/权重加载阶段：检查 traceback、GPU 显存和依赖版本。
- 编译/图捕获阶段：初次启动可能较慢；有持续进展时才考虑延长 `VLLM_STARTUP_TIMEOUT`。
- 服务启动前退出：查看 launcher 打印的错误尾部与完整服务日志。
- 端口已经监听：查明进程归属，或分别配置 `VLLM_HTTP_PORT`、`QRM_HTTP_PORT`、训练 `PORT` 和 `VLLM_GROUP_PORT`。四者必须互不相同；不要自动复用未知服务。直接运行 vLLM 服务时，原生 `VLLM_PORT` 属于引擎内部通信，不能与 HTTP 的 `--port` 相同。

“120 秒没写新日志”不等于停止工作：下载、编译或加载时日志可能稀疏。重复重启会中断本来仍在进行的工作，因此不再用该条件自动重启。若持续停滞，保留当次 run、命令、进程树和完整日志，再诊断阻塞位置；不要只增大等待时间。

### GPU、显存和通信

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

先确认物理 GPU 4–7 已分配给当前实验且没有冲突。改变 `VLLM_GPU_MEMORY_UTILIZATION` 不能替代获得空闲卡；QRM 独占 GPU 5，不再占用 GPU 6–7 的训练显存。真实 OOM 时按失败角色调整 vLLM 利用率、QRM batch/长度或策略 batch/长度，并记录解析后的 YAML。

`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1` 等不应成为所有服务器的默认设置。只有日志、拓扑和最小通信测试指向相关路径时才作有记录的对照实验；盲目关闭可能掩盖配置问题或降低性能。

### 生成成功后的反向传播错误

当 `CheckpointError` 明确显示冻结权重由正常形状变成 `[0]` 时，检查是否同时启用 PEFT、DeepSpeed ZeRO-3、gradient checkpointing 且 `use_reentrant: false`。本次固定版本组合中，ZeRO-3 的 forward 后置 hook 释放权重，而非重入 checkpoint 仍保留冻结参数对象，导致重算时读取到已清空的参数。机制见 [DeepSpeed #8130](https://github.com/deepspeedai/DeepSpeed/pull/8130)，上游 TRL 的兼容处理见 [TRL #6356](https://github.com/huggingface/trl/pull/6356)。

当前四卡默认 recipe 使用 ZeRO-2，因此不会进入上述 ZeRO-3 reference-model 路径；它仍使用 `gradient_checkpointing_kwargs: {use_reentrant: true}`。启动器保留对自定义 ZeRO-3 配置的兼容校验。升级依赖后仍应重新验证，不能用关闭 metadata 校验或忽略异常来掩盖空权重问题。

## 停止后的进程与端口检查

正常情况下，在启动器所在终端 Ctrl+C，由启动器清理它创建的进程。在只跟随日志的终端 Ctrl+C 不会停止训练。

```bash
pgrep -u "$USER" -af "open_r1[.](vllm_serve|reward_server)|src/open_r1/grpo[.]py|accelerate launch|accelerate[.]commands[.]launch"
ss -ltnp "sport = :8000 or sport = :8001"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

将 `8000` 换成该 run 的端口。对于列出的 PID，先用 `ps -p PID -o user,pid,ppid,pgid,stat,etime,args` 检查，再对确认属于本次实验的进程发送 TERM。只有进程不响应且确认目标后才考虑 KILL；僵尸进程需要父进程回收，不能靠重复 KILL 清除。不要使用 `pkill python`、`pkill -f vllm` 等宽泛规则，也不要照抄历史 PID——PID 会被系统复用。

判断清理完成要同时检查目标进程、相关子进程、端口和 GPU context；`ps -p 单个PID` 无结果仅说明那个 PID 已不存在。共享服务器上其他用户占用 GPU 是正常情况。

## 在新服务器重复 GPU 验收

```bash
export DATASET_NAME=你的组织或用户名/UltraChat-200k
export VLLM_GPUS=4
export QRM_GPU=5
export TRAIN_GPUS=6,7
python scripts/grpo.py smoke --dry-run
# 以下只在物理卡 4–7 已分配且空闲时执行：
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/grpo.py doctor --cuda
python scripts/grpo.py smoke
```

把卡号改成实际分配。`doctor --cuda` 只检查 CUDA 可见性和扩展导入；实际生成与训练由 smoke 验证。验收结果应包含：使用的 commit 和软件版本；health 成功；至少一次生成；训练端到 vLLM 的权重同步；两个 optimizer step；adapter 保存；`RUN_STATUS=success`；退出后本次服务/训练进程与监听端口已释放。smoke 默认不合并 LoRA；需要验收该阶段时用 `MERGE_AFTER_TRAINING=1 python scripts/grpo.py smoke`。本机通过不意味着其他机器自动通过，迁移后应重新完成这些检查。
