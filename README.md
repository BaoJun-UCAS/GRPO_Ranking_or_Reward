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
That is historical validation of the legacy layout, not validation of the new
four-GPU split, long-run convergence, or the formal-run memory budget. See:

- [Experiment configuration and change log](EXPERIMENT_GRPO_5GPU.md)
- [Chinese environment setup guide](docs/ENVIRONMENT_SETUP_ZH.md)
- [Deployment failures, evidence, and troubleshooting](docs/DEPLOYMENT_TROUBLESHOOTING_ZH.md)
- [Validation, evaluation contracts, caching, and offline plots](docs/EVALUATION_ZH.md)
- [Chinese GitHub project management guide](docs/GITHUB_PROJECT_MANAGEMENT_ZH.md)

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

For the CPU test suite alone, install `requirements-test.txt`; the full training
stack is not required for those tests.

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
