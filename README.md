# GRPO

GRPO training framework for language-model ranking and reward experiments.

Deployment starts with a CPU-only environment check and a two-step smoke test.
The default experiment uses Qwen3-1.7B, UltraChat, and a QRM reward model with
one dedicated vLLM GPU plus configurable ZeRO-3 training GPUs.

The current `studentization` recipe is an experimental baseline, not a verified
paper-exact reproduction of original GRPO: it uses `loss_type: bnpo` and a custom
trainer. On 2026-09-24, the 1+4 RTX 4090 smoke test passed readiness, generation,
weight synchronization, two optimizer steps, checkpoint/adapter saving, and LoRA
merging. The merged model was reloaded and generated tokens successfully.
This validates the short smoke configuration, not long-run convergence or the
memory budget of larger recipes. See:

- [Experiment configuration and change log](EXPERIMENT_GRPO_5GPU.md)
- [Chinese environment setup guide](docs/ENVIRONMENT_SETUP_ZH.md)
- [Deployment failures, evidence, and troubleshooting](docs/DEPLOYMENT_TROUBLESHOOTING_ZH.md)
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
| Run two optimizer steps | `make smoke` | Starts vLLM and training |
| Run the configured experiment | `make train` | Starts vLLM and training |
| Follow latest vLLM log | `make logs` | None |
| Run lightweight tests | `make test` | None |

For the CPU test suite alone, install `requirements-test.txt`; the full training
stack is not required for those tests.

Set the dataset and GPU assignment before the launch commands:

```bash
export DATASET_NAME=your_org/UltraChat-200k
export VLLM_GPU=0
export TRAIN_GPUS=1,2,3,4
python scripts/grpo.py smoke --dry-run
python scripts/grpo.py download
# Run only after the selected GPUs are available:
python scripts/grpo.py smoke
```

GPU numbers are examples, not a reservation or a topology recommendation for
every server. Use `nvidia-smi topo -m` and local scheduling rules to select them.
For the previously inspected eight-4090 host, training GPUs `4,5,6,7` shared
NUMA 1; this fact does not apply automatically to another host.

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

The Chat launcher starts `python -m open_r1.vllm_serve`, waits for a bounded
readiness check, then starts ZeRO-3 training. The local service preserves the
TRL 0.18 client protocol and avoids that version's extra outer model process.
Do not replace it with `trl vllm-serve` or a generic OpenAI-compatible vLLM
server: the trainer also requires weight-synchronization endpoints.

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
| `VLLM_GPU`, `TRAIN_GPUS` | One server GPU and a disjoint, nonempty list of training GPUs; process count is inferred |
| `DATASET_NAME` | Dataset ID; alternatively set `HF_USERNAME` for its `UltraChat-200k` dataset |
| `MODEL_NAME`, `CONFIG_FILE`, `ACCELERATE_CONFIG` | Model and recipe selection; changing the model can also require LoRA/reward adjustments |
| `GRPO_OUTPUT_ROOT`, `RUN_DIR` | Output root, or an explicit new run directory; existing directories are not overwritten |
| `RESUME_FROM_CHECKPOINT` | Explicit checkpoint path for resuming into a new run directory |
| `GRPO_CACHE_ROOT`, `HF_HOME`, `HF_HUB_CACHE` | Cache location; explicit Hugging Face settings take precedence |
| `NUM_GENERATIONS`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS` | Batch configuration; generation batch is derived unless explicitly supplied |
| `MAX_STEPS`, `MAX_PROMPT_LENGTH`, `MAX_COMPLETION_LENGTH`, `REWARD_BATCH_SIZE` | Training scale and memory controls |
| `VLLM_HTTP_PORT`, `PORT`, `VLLM_GROUP_PORT` | HTTP, training rendezvous, and weight-sync ports (defaults: 8000, 29501, 51216); use three distinct ports per experiment |

The default output root is `grpo_runs/` inside the repository. Without explicit
cache settings, models use `${XDG_CACHE_HOME:-$HOME/.cache}/grpo/huggingface`.
The launcher stores the resolved YAML, hardware snapshot, terminal logs, reward
records, checkpoints, final adapter, and merged evaluation model in one run
directory. W&B defaults to offline mode; set `WANDB_MODE=online` to upload
metrics. No `.env` file is loaded implicitly.

The smoke shortcut defaults to two steps and skips LoRA merging; use
`MERGE_AFTER_TRAINING=1 python scripts/grpo.py smoke` to include it. Explicit
environment overrides are respected, so clear stale training values before a
smoke run. To resume while preserving the original run's evidence:

```bash
RESUME_FROM_CHECKPOINT=/path/to/old-run/checkpoint-100 python scripts/grpo.py train
```

Follow the log for a specific run:

```bash
python scripts/grpo.py logs --run-dir /path/to/run --service vllm --follow
python scripts/grpo.py logs --run-dir /path/to/run --service training --follow
```

### Available Recipes

Recipes are in `recipes/Qwen3-1.7B/`:

| Config | Dataset | Advantage | Reward Model |
|--------|---------|-----------|--------------|
| `config_chat_regular_qrm_lora_5gpu.yaml` | UltraChat | Studentization | QRM (LoRA + server vLLM) |
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
