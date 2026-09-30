# GRPO

GRPO training framework for language-model ranking and reward experiments.

Deployment starts with a CPU-only environment check and a two-step smoke test.
The default launcher uses four physical GPUs: GPU 4 for policy rollout with
vLLM, GPU 5 for the external 8B QRM server, and GPUs 6–7 for two-process
DeepSpeed ZeRO-2 LoRA policy training.

The current `studentization` recipe is an experimental baseline, not a verified
paper-exact reproduction of original GRPO: it uses `loss_type: bnpo` and a custom
trainer. On 2026-09-24, the 1+4 RTX 4090 smoke test passed readiness, generation,
weight synchronization, two optimizer steps, checkpoint/adapter saving, and LoRA
merging. The merged model was reloaded and generated tokens successfully.
The 2026-09-30 refactor also passed real four-GPU, two-step ZeRO-2 and DDP
smoke tests, held-out reward evaluation, and DDP adapter merge/reload. A further
smoke run with the default 2048/3072 token limits passed after fixing evaluation
KV-cache handling. Batch size was 16; these checks do not establish full-scale
memory requirements, long-run convergence, or model quality. See:

- [Experiment configuration and change log](EXPERIMENT_GRPO_5GPU.md)
- [Chinese environment setup guide](docs/ENVIRONMENT_SETUP_ZH.md)
- [Deployment failures, evidence, and troubleshooting](docs/DEPLOYMENT_TROUBLESHOOTING_ZH.md)
- [Validation, evaluation contracts, caching, and offline plots](docs/EVALUATION_ZH.md)
- [Chinese GitHub project management guide](docs/GITHUB_PROJECT_MANAGEMENT_ZH.md)

## 架构与实验扩展

主入口仍是 `python scripts/grpo.py`。训练编排、数据输入、advantage 与目标函数已经分开，
修改实验方法不需要复制整份 Trainer：

| 模块 | 职责 / 扩展入口 |
| --- | --- |
| `scripts/grpo.py`、`train_scripts/run_helpers.py` | 环境检查、服务启动、配置解析、GPU/端口检查、产物验收 |
| `src/open_r1/grpo.py` | 组合数据、模型、奖励和 Trainer；保存本次配置与依赖版本 |
| `src/open_r1/utils/data.py` | Hub / 本地数据加载、prompt 适配、截断、split 与样本数量校验 |
| `src/open_r1/utils/model_utils.py` | Transformers 模型与 tokenizer 加载，revision 和模板配置 |
| `src/open_r1/advantages.py` | 全局分组 advantage 接口及数值检查 |
| `src/open_r1/objectives.py` | `grpo` / `bnpo` / `dr_grpo` 的 loss 归约 |
| `src/open_r1/grpo_trainer.py` | rollout、全局 reward 收集、调用 advantage、反向传播与分布式协调 |
| `src/open_r1/reward_client.py` | 无模型依赖的 HTTP 奖励客户端，兼容聊天与原始文本 |
| `src/open_r1/reward_server.py`、`vllm_serve.py` | 独立 GPU 服务，保持明确的评分和权重同步协议 |

### 改 advantage

默认 `studentization` 保持原有无偏标准差与 `1e-4` epsilon；`grpo` 是它的别名，
计算原始 GRPO 的组内标准化 advantage：`(reward - group_mean) / (group_std + epsilon)`。
此别名只选择 advantage；当前四卡 recipe 仍使用 `loss_type: bnpo`，不是完整论文复现。

`robust_pairwise` 对同一 prompt 下的加权原始 reward 两两比较，先抑制微小差异，再截断过大的差异。
设 `d_ij = r_i - r_j`，组大小为 `G`：

```text
A_i = sum_{j != i} sign(d_ij) * clamp((abs(d_ij) - delta) / c, 0, 1) / (G - 1)
```

差值不超过 `delta` 时不贡献信号；超过 `delta + c` 后单个比较饱和为 ±1。
结果直接作为 advantage，**不再按组内标准差归一化**。
`delta >= 0`、`c > 0` 均使用聚合后的加权原始 reward 尺度，改变奖励模型或 reward weights 后需要重新选择。
函数默认 `delta=0.02, c=0.2` 保留为附件中的数值示例。
当前 Qwen3/QRM/UltraChat 实验的初步参数为 **`delta=0.01, c=0.08`**，已保存到
[`robust_pairwise_initial.env`](recipes/Qwen3-1.7B/robust_pairwise_initial.env)。
这组值来自完整 800 step 训练的组内奖励差统计，尚未校准评分误差或验证质量提升；
选择依据和对照参数见 [验证文档](docs/EVALUATION_ZH.md)。
参数放在 `advantage_kwargs` 中；**`advantage_kwargs.delta` 与 TRL 顶层的 PPO `delta` 是不同参数**。

先设置 `DATASET_NAME`，其余模型、数据、batch、loss 和四卡配置保持相同，切换命令如下：

```bash
# 原始 GRPO advantage（等价于 ADVANTAGE=studentization）
RUN_NAME=grpo-baseline DO_EVAL=1 ADVANTAGE=grpo ADVANTAGE_KWARGS='{}' \
  python scripts/grpo.py train

# 新方法：加载保存的初步参数 delta=0.01、c=0.08
source recipes/Qwen3-1.7B/robust_pairwise_initial.env
RUN_NAME=grpo-pairwise-d001-c008 DO_EVAL=1 python scripts/grpo.py train
```

加 `--dry-run` 可先检查最终配置；比较实验应保持其它超参数一致。
`scale_rewards: false` 仅关闭 `studentization` / `grpo` baseline 的标准差缩放
（以及 `rank_reward` 的 reward 分支缩放），保留组内中心化；它**不控制 `robust_pairwise`**。

现有 `ranking` 使用平均并列排名：相同 reward 得到相同 advantage，整组同分得到零；
没有并列时与旧 ranking 公式一致。`rank_reward` 的定义仍是
`(1-rank_weight) * studentization + rank_weight * ranking`，rank 分支范围为 `[-2, 2]`。
两者继续通过 `ADVANTAGE=ranking` 或
`ADVANTAGE=rank_reward ADVANTAGE_KWARGS='{"rank_weight":0.3,"epsilon":0.0001}'` 选择。

自定义算法放进可导入的 Python 模块，例如 `src/open_r1/my_advantage.py`：

```python
def centered_reward(batch, *, scale=1.0):
    # batch.rewards: [prompt数, num_generations]，已经包含所有训练 rank。
    # batch.rewards_per_func: [prompt数, num_generations, reward函数数]。
    return scale * (batch.rewards - batch.group_mean)
```

```yaml
advantage: open_r1.my_advantage:centered_reward
advantage_kwargs:
  scale: 0.5
```

也可使用 `register_advantage(name)` 注册。返回值必须与 `batch.rewards` 同形状、同设备且有限；
接口会 detach，再按训练 rank 切片。不要在适配函数内重新通信或按 rank 局部分组。
`loss_type` 现在在普通 loss 路径中实际生效，完全屏蔽的 batch 返回可反传的零；
Liger 路径目前要求 `token_broadcast: uniform`。
所有评分 forward 显式关闭 KV cache；保存时只修改配置副本，避免长短混合 completion 的 FlashAttention 验证报错。

### 换模型与数据

```bash
MODEL_NAME=Qwen/Qwen3-1.7B MODEL_REVISION=main \
DATASET_NAME=HuggingFaceH4/ultrachat_200k DATASET_PROMPT_COLUMN=messages \
DATASET_TRAIN_SPLIT=train_sft DATASET_TEST_SPLIT=test_sft DATASET_ADAPTER=chat \
  python scripts/grpo.py smoke --dry-run
```

`MODEL_REVISION` 同时控制训练模型、默认 tokenizer 与 vLLM，正式实验应固定 commit。
支持 Transformers/vLLM 均能加载的因果语言模型；换架构时复制 recipe，通过 `CONFIG_FILE`
设置合适的 `lora_target_modules`、精度、注意力实现、模板与上下文长度。
独立 tokenizer 可用 YAML 的 `tokenizer_name_or_path` / `tokenizer_revision` 指定，但必须与
模型和 rollout 服务保持同一词表/token ID；`chat_template` 可只覆盖渲染模板。
`QRM_MODEL` / `QRM_REVISION` 控制外部奖励模型，服务要求兼容现有 QRM 的输出协议。
当前 QRM 只接收 `system/user/assistant` 的 `role/content` 消息；含 tool 调用或额外 name 字段的数据需自定义 adapter 转换。

`DATASET_NAME` 支持 Hub ID、`save_to_disk()` 目录或 JSON/JSONL/Parquet/CSV 文件。
单文件只提供一个训练 split；开启验证时应使用有独立验证 split 的 DatasetDict 或 Hub 数据集。
`DATASET_ADAPTER=auto|text|chat|raw` 统一产生 `prompt`，保留 `solution` 等奖励字段；
chat 会移除最后一个用户问题之后的参考 assistant 回答，保留此前多轮历史，避免答案泄漏。
自定义格式使用 `dataset_adapter: my_module:my_adapter`，函数签名为
`my_adapter(example, *, prompt_column, system_prompt) -> {"prompt": ...}`。
`raw` 直接使用字符串，适合无 chat template 的基础模型；QRM 客户端将原始 prompt/回答包装为 user/assistant 评分。
`MAX_TRAIN_SAMPLES` / `MAX_EVAL_SAMPLES` 只用于调试，先限量再预处理。

### 四卡效率与运行方式

同步 GRPO 的每轮仍按“同步权重 → 生成 → 奖励 → 策略更新”执行；分工独立的 GPU 会等待其它阶段，
因此四卡拆分不意味着四卡持续满负载。已有 Gloo 对象通信、QRM 长度分桶/动态 token 预算、
训练 padding 裁剪和阶段计时继续保留。本次进一步取消 `log_completions=false` 时的两次全文 all-gather，
并将普通 loss 指标通信从每 micro-step 的 3–4 次合为 1 次，让 ranking 共用同一四卡启动流程。
数据预处理按 rank 共享缓存，只处理本次需要的 split，避免多卡重复 tokenize。

默认保留 ZeRO-2。LoRA 仅更新少量参数，可对照 DDP：

```bash
ACCELERATE_CONFIG=recipes/accelerate_configs/ddp_2gpus.yaml python scripts/grpo.py train
```

显存允许时可尝试 `PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=64`，
保持默认全局 generation batch 256；这减少 micro-step 次数，但需要在目标长度下验证显存。
不要只减少 accumulation 后将不同全局 batch 的实验称作同条件加速。
用 `validation_report.json` 中每 rank 的阶段耗时定位瓶颈，再比较同模型、数据、长度、全局 batch
和预热后多个 step 的吞吐。历史一小时监控中 GPU 4–7 平均利用率约为
24.9% / 18.1% / 43.4% / 94.1%，旧日志没有阶段计时，不能据此解释为有效训练计算或推算提速倍数。

当前主机可直接使用以下流程（其它主机替换数据路径与 GPU 编号）：

```bash
source /data/baojun/miniconda3/etc/profile.d/conda.sh
conda activate grpo
export CUDA_HOME="$CONDA_PREFIX"
export DATASET_NAME=/data/baojun/datasets/ultrachat_200k_grpo_seed42
export VLLM_GPUS=4 QRM_GPU=5 TRAIN_GPUS=6,7
python scripts/grpo.py doctor

# 短序列两步 smoke + 两条验证；不代表正式长度的显存或质量验证。
NUM_GENERATIONS=4 MAX_PROMPT_LENGTH=256 MAX_COMPLETION_LENGTH=128 \
VLLM_MAX_MODEL_LEN=1024 QRM_MAX_LENGTH=1024 QRM_MAX_BATCH_TOKENS=2048 \
MAX_TRAIN_SAMPLES=16 MAX_EVAL_SAMPLES=2 DO_EVAL=1 PER_DEVICE_EVAL_BATCH_SIZE=2 \
  python scripts/grpo.py smoke

# 正式训练；不继承上面仅作用于单条命令的样本/长度限制。
RUN_NAME=my-grpo-baseline DO_EVAL=1 python scripts/grpo.py train
python scripts/grpo.py validate --run-dir grpo_runs/my-grpo-baseline --require-merged
python scripts/plot_training_metrics.py grpo_runs/my-grpo-baseline
```

`DO_EVAL=1` 默认为训练结束后在验证 split 上计算 loss/reward，并保存 `eval_results.json`。
训练中验证可加 `EVAL_STRATEGY=steps EVAL_STEPS=100`；默认每卡 eval batch 为 4，
两训练进程的全局 batch 8 与默认 `num_generations=8` 匹配。
独立模型质量对比见 [验证与评测说明](docs/EVALUATION_ZH.md)；产物验收和训练奖励不能代替质量评测。

## Installation

Run from the repository root. Choose a writable disk with room for temporary
downloads, the Conda environment, model caches, and training outputs:

```bash
export GRPO_WORK_DIR="$HOME/grpo-work"  # Change to your large writable disk.
mkdir -p "$GRPO_WORK_DIR/tmp" "$GRPO_WORK_DIR/cache/pip" "$GRPO_WORK_DIR/runs"
export TMPDIR="$GRPO_WORK_DIR/tmp"
export PIP_CACHE_DIR="$GRPO_WORK_DIR/cache/pip"
export GRPO_CACHE_ROOT="$GRPO_WORK_DIR/cache/grpo"
export HF_HOME="$GRPO_CACHE_ROOT/huggingface"
export GRPO_OUTPUT_ROOT="$GRPO_WORK_DIR/runs"

conda env create -f environment.yml
conda activate grpo
```

The environment file installs PyTorch 2.6.0, vLLM 0.8.5.post1, CUDA 12.4
development tools, the editable GRPO package, and training/evaluation dependencies. Once that command
succeeds, install FlashAttention so its build can see PyTorch. A source build
needs the CUDA Toolkit (`nvcc`), a C++ compiler, and enough host memory;
`nvidia-smi` alone does not establish that a Toolkit is installed.

```bash
nvcc --version
export CUDA_HOME="$CONDA_PREFIX"  # For the Toolkit installed by environment.yml.
MAX_JOBS=8 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip check
python scripts/grpo.py doctor
```

Keep the pinned PyTorch/vLLM/TRL versions together. `pip check` only checks
installed dependency metadata; it does not prove that CUDA, imports, or training
work. `doctor` checks local prerequisites without initializing CUDA. Use
`python scripts/grpo.py doctor --cuda` when the selected GPUs are available.

Contributors can install the development extras with:

```shell
GIT_LFS_SKIP_SMUDGE=1 python -m pip install -e ".[dev]"
```

## Quick commands

All shortcuts use the active Python environment and work without `make` via
`python scripts/grpo.py --help`.

| Task | Command | GPU use |
| --- | --- | --- |
| Create the Conda environment | `make env-create` | None |
| Check environment, paths, and tools | `make doctor` | No CUDA initialization |
| Check CUDA explicitly | `make doctor-cuda` | CUDA initialization |
| Download policy and reward models | `make download-models` | None; uses network and disk |
| Inspect model cache and partial files | `make cache` | None |
| Preview smoke plan and resolved configuration | `make dry-run` | None; no run directory created |
| Run two optimizer steps | `make smoke` | Starts QRM, vLLM, and training |
| Run the configured experiment | `make train` | Starts QRM, vLLM, and training |
| Follow the latest service/training log | `make logs` | None |
| Validate final artifacts and timing evidence | `make validate-run RUN_DIR=/path/to/run` | None |
| Plot training metrics | `make plot-training RUN_DIR=/path/to/run` | None |
| Run lightweight tests | `make test` | None |

Install `requirements-test.txt` for lightweight CPU tests. Tensor mathematics, data adapters,
and the real tiny-model training tests run on CPU when the training stack is installed,
and are explicitly skipped without it.

Set the dataset and GPU assignment before the launch commands:

```bash
export DATASET_NAME=your_org/UltraChat-200k
export VLLM_GPUS=4
export QRM_GPU=5
export TRAIN_GPUS=6,7
python scripts/grpo.py smoke --dry-run
python scripts/grpo.py download
# Run only after the selected GPUs are available:
python scripts/grpo.py smoke
```

These defaults target the inspected eight-GPU host. The launcher sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and isolates each role itself; do not wrap it in
another `CUDA_VISIBLE_DEVICES`. Alternate physical GPU IDs remain configurable,
but must be explicit, disjoint, and checked against local scheduling rules.

## Data Preprocessing

Preprocessing scripts in `preprocess_data/` download, process, and **upload**
datasets to your Hugging Face account. Skip this section if `DATASET_NAME`
already points to a compatible dataset with `train`/`val` splits and a `prompt`
column. Set `HF_USERNAME` and a write-enabled `HF_TOKEN` before preprocessing;
keep tokens out of scripts and Git. Public model downloads do not require a
write token. W&B is offline by default; `wandb login` is needed only for online
tracking.

### UltraChat Dataset (Chat)

```shell
python preprocess_data/preprocess_ultrachat_dataset.py
```

Downloads `HuggingFaceH4/ultrachat_200k`, creates train/val/test splits, and pushes to `$HF_USERNAME/UltraChat-200k`.

### TLDR Dataset (Summarization)

```shell
python preprocess_data/preprocess_tldr_datasets.py
```

Downloads `trl-lib/tldr`, samples validation set, and pushes to `$HF_USERNAME/tldr`.

### Instruction Following Dataset

```shell
python preprocess_data/preprocess_if_datasets.py
```

Merges `allenai/tulu-3-sft-personas-instruction-following` (train) with `google/IFEval` (test), and pushes to `$HF_USERNAME/IF-Datasets-Tulu-IFEval`.

## Training

The Chat launcher first starts `python -m open_r1.reward_server` on GPU 5,
then `python -m open_r1.vllm_serve` on GPU 4, and verifies an instance-specific
health endpoint for each service before starting two-process ZeRO-2 training on
GPUs 6–7. The vLLM service preserves the TRL 0.18 weight-synchronization
protocol; a generic OpenAI-compatible endpoint is not sufficient.

The four-GPU recipe keeps Python-object gather/broadcast traffic on an optional
CPU/Gloo control group, while tensor gradients and reward tensors continue to
use the normal distributed backend. This prevents a policy rank waiting for
vLLM or QRM from appearing GPU-busy solely because an NCCL object collective is
spinning. If Gloo is unavailable, the trainer warns and falls back to the legacy
collective path.

The QRM server sorts requests by tokenized length and forms batches bounded by
both `REWARD_BATCH_SIZE` and `QRM_MAX_BATCH_TOKENS`; results are restored to
their original order. Policy micro-batches also discard prompt/completion
columns that are padding for every sample in that micro-batch. Stage timing is
enabled in this recipe: `timing/rollout_total_*`, `timing/qrm_total_*`,
`timing/external_sync_wait_*`, and `timing/policy_train_total_*` expose per-rank,
minimum, maximum, and rank-spread wall times in the normal Trainer logs. The
`*_total_*` policy metrics sum all accumulation micro-steps in one optimizer
step; the accompanying `*_mean_*` metrics retain the per-micro-step average.

### Running Training

```shell
# Studentized experimental baseline (LoRA, UltraChat + QRM)
export DATASET_NAME=your_org/UltraChat-200k
python scripts/grpo.py train

# Ranking GRPO training (ranking advantage)
bash train_scripts/qwen3_1.7_grpo_ranking_chat.sh
```

### Customizing Training

| Variable | Purpose |
| --- | --- |
| `VLLM_GPUS`, `QRM_GPU`, `TRAIN_GPUS` | Disjoint rollout, reward-server, and policy-training GPUs; defaults are `4`, `5`, and `6,7` |
| `DATASET_NAME` | Dataset ID; alternatively set `HF_USERNAME` for its `UltraChat-200k` dataset |
| `MODEL_NAME`, `QRM_MODEL`, `QRM_REVISION` | Policy/reward model selection; pin revisions for formal runs |
| `CONFIG_FILE`, `ACCELERATE_CONFIG` | Defaults to the four-GPU external-QRM recipe and two-process ZeRO-2 |
| `GRPO_OUTPUT_ROOT`, `RUN_DIR` | Output root, or an explicit new run directory; existing directories are not overwritten |
| `RESUME_FROM_CHECKPOINT` | Explicit checkpoint path for resuming into a new run directory |
| `GRPO_CACHE_ROOT`, `GRPO_DATA_ROOT`, `HF_HOME`, `HF_HUB_CACHE` | Cache location; explicit paths win, otherwise a writable `/data/<user>/cache/grpo` is preferred before the home-directory fallback |
| `VLLM_TMPDIR` | Optional short vLLM IPC directory; the launcher otherwise creates and cleans `/tmp/grpo-vllm.*` |
| `NUM_GENERATIONS`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS` | Batch configuration; generation batch is derived unless explicitly supplied |
| `MAX_STEPS`, `MAX_PROMPT_LENGTH`, `MAX_COMPLETION_LENGTH`, `REWARD_BATCH_SIZE` | Training scale and memory controls; QRM accepts at most 4 examples per dynamic batch by default |
| `QRM_MAX_LENGTH`, `QRM_MAX_BATCH_TOKENS`, `QRM_REQUEST_TIMEOUT`, `QRM_STARTUP_TIMEOUT` | Reward context, padded-token budget (default 6144), and bounded request/startup waits |
| `VLLM_HTTP_PORT`, `QRM_HTTP_PORT`, `PORT`, `VLLM_GROUP_PORT` | vLLM HTTP, QRM HTTP, training rendezvous, and weight-sync ports; all four must be distinct |

The default output root is `grpo_runs/` inside the repository. Without explicit
cache settings, models prefer `/data/<user>/cache/grpo/huggingface` when that
data root is writable, then fall back to `$HOME/.cache/grpo/huggingface`.
The launcher stores the resolved YAML, hardware snapshot, terminal logs, reward
records, checkpoints, final adapter, and merged evaluation model in one run
directory. W&B defaults to offline mode; set `WANDB_MODE=online` to upload
metrics. No `.env` file is loaded implicitly.

The smoke shortcut defaults to two steps and skips LoRA merging; use
`MERGE_AFTER_TRAINING=1 python scripts/grpo.py smoke` to include it. Successful
launcher runs are accepted by the CPU-only artifact/timing validator before they
are marked successful. Re-run the same checks with
`python scripts/grpo.py validate --run-dir /path/to/run`; add
`--require-merged` when a merged model is part of the contract. The structured
result is written to `validation_report.json`. Explicit
environment overrides are respected, so clear stale training values before a
smoke run. To resume while preserving the original run's evidence:

```bash
RESUME_FROM_CHECKPOINT=/path/to/old-run/checkpoint-100 python scripts/grpo.py train
```

Follow the log for a specific run:

```bash
python scripts/grpo.py logs --run-dir /path/to/run --service vllm --follow
python scripts/grpo.py logs --run-dir /path/to/run --service qrm --follow
python scripts/grpo.py logs --run-dir /path/to/run --service training --follow
```

### Available Recipes

Recipes are in `recipes/Qwen3-1.7B/`:

| Config | Dataset | Advantage | Reward Model |
|--------|---------|-----------|--------------|
| `config_chat_regular_qrm_lora_4gpu_split.yaml` | UltraChat | Studentization | External QRM; 1 vLLM + 2 ZeRO-2 training GPUs |
| `config_chat_regular_qrm_lora_5gpu.yaml` | UltraChat | Studentization | Legacy local QRM training recipe |
| `config_chat_regular_qrm_seed42.yaml` | UltraChat | Regular | QRM |
| `config_chat_ranking_qrm_seed42.yaml` | UltraChat | Ranking | QRM |
| `config_tldr_regular_skywork-8b_seed42.yaml` | TLDR | Regular | Skywork-8B |
| `config_tldr_ranking_skywork-8b_seed42.yaml` | TLDR | Ranking | Skywork-8B |
| `config_if_regular_skywork-8b_seed42.yaml` | IF-Datasets | Regular | Skywork-8B |
| `config_if_ranking_skywork-8b_seed42.yaml` | IF-Datasets | Ranking | Skywork-8B |

The legacy full-parameter recipes remain for research. They are not validated
interchangeably with the Chat deployment launcher; check their vLLM mode, reward
function, model-specific options, and memory requirements before adapting one.

## Generation

The `generate/` folder contains scripts for generating model completions on evaluation datasets.

### Generate Completions

`generate_completions.py` generates completions from a trained model checkpoint for later evaluation.

```shell
python generate/generate_completions.py \
    --model <path/to/model_checkpoint> \
    --dataset chat \
    --num-prompts 1000 \
    --n-completions 3 \
    --temperature 0.7 \
    --seed 42
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Path to model (HuggingFace path or local directory) |
| `--dataset` | Dataset to use: `chat`, `tldr`, `if`, or `storygen` (auto-detected from model path if omitted) |
| `--num-prompts` | Number of prompts to sample (default: 100) |
| `--n-completions` | Number of completions per prompt (default: 1) |
| `--temperature` | Sampling temperature (default: 0.7) |
| `--top-p` | Top-p nucleus sampling (default: 0.9) |
| `--max-new-tokens` | Maximum tokens to generate (default: 2048) |
| `--seed` | Random seed for reproducibility |
| `--output` | Output path (auto-generated if omitted) |
| `--no-vllm` | Disable vLLM, use transformers instead |
| `--vllm-gpu-memory` | GPU memory utilization for vLLM (default: 0.85) |

#### Output Structure

Completions are saved to `completions_n{N}/<model_name>_temp{T}/` with auto-generated filenames:

```
completions_n3/
└── Qwen3-1.7B-chat-ranking_temp0.7/
    └── Qwen3-1.7B-chat-ranking-checkpoint100_1000prompts_3completions_seed42_temp0.7.json
```


## Evaluation

The `evaluate/` folder contains scripts for comparing model outputs using LLM-as-judge evaluation.

### Bootstrap Judge Evaluation

`bootstrap_judge.py` compares two models' completions using bootstrap sampling to compute win rates with confidence intervals.

```shell
python evaluate/bootstrap_judge.py \
    --completions1 <path/to/model1_completions.json> \
    --completions2 <path/to/model2_completions.json> \
    --api-provider deepseek \
    --judge-model YOUR_AVAILABLE_JUDGE_MODEL \
    --thinking-mode disabled \
    --N 100 \
    --B 1000 \
    --seed 42 \
    --no-ties
```

This optional stage calls a paid external API. Verify the provider's current
model identifiers, API compatibility, and prices before running it. Set
`DEEPSEEK_API_KEY` in the environment rather than a recipe or script. The
evaluator judges `N` unique pairs, writes successful judgments to an append-only
cache, and performs `B` bootstrap resamples locally. Cached pairs are reused;
retries and failed responses can still cause additional API requests.

For the complete post-training workflow, including base-model and trained-model generation, run:

```shell
EVAL_GPU=0 bash evaluate/run_grpo_chat_deepseek.sh /absolute/path/to/training-run
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `--completions1` | Path to first model's completions (JSON/JSONL) |
| `--completions2` | Path to second model's completions (JSON/JSONL) |
| `--judge-model` | LLM judge model; defaults to `deepseek-flash` |
| `--api-provider` | `deepseek`, `openai`, `anthropic`, or `auto` |
| `--api-key` | Optional CLI override; environment variables are safer |
| `--N` | Number of unique prompt pairs evaluated by the API |
| `--B` | Number of local bootstrap resamples |
| `--seed` | Random seed for reproducibility |
| `--no-ties` | Force judge to pick a winner (no ties allowed) |
| `--allow-ties` | Allow ties in evaluation |
| `--completion-index` | Which completion to use from multi-completion files (default: 0) |
| `--output-dir` | Output directory for results (default: `evaluate/`) |

#### Completion File Format

The script accepts JSON files with the following structure:

```json
{
  "meta": {},
  "items": [
    {"prompt": "...", "completions": ["response1", "response2", ...]},
    ...
  ]
}
```

Or JSONL format with one item per line containing `prompt` and `responses` fields.
