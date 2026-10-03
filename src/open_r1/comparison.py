"""Controlled GRPO vs robust-pairwise experiment; GPU/API work is explicit.

Preparation freezes data, configuration, runtime versions and training code.
Completed stages may be reused after validation. A partial training run is never
silently resumed: --restart-failed archives it and restarts that arm from base.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import runpy
import shlex
import shutil
import subprocess
import sys

import yaml

from open_r1.evaluation import digest, model_fingerprint
from open_r1.judge_protocol import (
    DEFAULT_JUDGE_TEMPERATURE, JUDGE_CACHE_VERSION, JUDGE_PARSER_VERSION, JUDGE_PROTOCOL_VERSION,
    build_comparison_prompt, judge_settings_metadata,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'recipes/Qwen3-1.7B/advantage_comparison_200.yaml'
ARMS = {'baseline': ('grpo', {}), 'improved': ('robust_pairwise', None)}
PACKAGES = ('torch', 'transformers', 'trl', 'datasets', 'accelerate', 'deepspeed', 'peft', 'vllm')
NUMERICAL_ENV = ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_code_digest():
    files = sorted((ROOT / 'src/open_r1').rglob('*.py'))
    files += [ROOT / name for name in ('train_scripts/run_helpers.py',
              'train_scripts/qwen3_1.7_grpo_chat.sh', 'generate/merge_lora_adapter.py')]
    return digest({str(path.relative_to(ROOT)): file_sha256(path) for path in files})


def runtime_versions():
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def improved_spec(config):
    """Keep historical robust experiments intact; opt into method 1 explicitly."""
    method = config.get('improved_advantage', 'robust_pairwise')
    if method == 'robust_pairwise':
        options = config.get('improved_advantage_kwargs', {'delta': config.get('delta'), 'c': config.get('c')})
    elif method == 'rolling_quantile_pairwise':
        options = config.get('improved_advantage_kwargs')
        if not isinstance(options, dict) or not {'p', 'q'} <= options.keys():
            raise ValueError('rolling_quantile_pairwise requires improved_advantage_kwargs.p and q')
    else:
        raise ValueError('improved_advantage must be robust_pairwise or rolling_quantile_pairwise')
    from open_r1.advantages import configure_advantage
    configure_advantage(method, options, scale_rewards=True)  # Validate before preparing data/models.
    return method, deepcopy(options)


def checked_config(path, dataset_override=None):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Experiment config must be a mapping')
    if dataset_override is not None:
        config['data']['source'] = dataset_override
    for name in ('steps', 'world_size', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                 'num_generations', 'max_prompt_length', 'max_completion_length'):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(config.get('seed')) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if config['world_size'] != 2 or config['per_device_train_batch_size'] != 1:
        raise ValueError('This four-GPU comparison requires two policy ranks and microbatch 1')
    batch = config['world_size'] * config['per_device_train_batch_size'] * config['gradient_accumulation_steps']
    if config['num_generations'] < 2 or batch % config['num_generations']:
        raise ValueError('num_generations >= 2 must divide the global generation batch')
    improved_spec(config)
    for prefix in ('model', 'qrm'):
        name = config[f'{prefix}_name' if prefix == 'model' else 'qrm_model']
        if not Path(name).expanduser().is_dir() and not re.fullmatch(r'[0-9a-fA-F]{40}', str(config[f'{prefix}_revision'])):
            raise ValueError(f'{prefix}_revision must pin a 40-character commit SHA for Hub models')
    services, evaluation = config['services'], config['evaluation']
    if len(str(services['train_gpus']).split(',')) != config['world_size']:
        raise ValueError('services.train_gpus must contain world_size GPU indices')
    if services['max_model_len'] < config['max_prompt_length'] + config['max_completion_length']:
        raise ValueError('max_model_len is shorter than prompt + completion limits')
    if services['qrm_max_batch_tokens'] < services['qrm_max_length']:
        raise ValueError('qrm_max_batch_tokens must cover qrm_max_length')
    for name in ('num_prompts', 'max_new_tokens', 'bootstrap_iterations'):
        if type(evaluation.get(name)) is not int or evaluation[name] < 1:
            raise ValueError(f'evaluation.{name} must be a positive integer')
    if type(evaluation.get('seed')) is not int or evaluation['seed'] < 0:
        raise ValueError('evaluation.seed must be a nonnegative integer')
    if (not math.isfinite(evaluation['temperature']) or evaluation['temperature'] < 0
            or not 0 < evaluation['top_p'] <= 1 or not 0 < evaluation['vllm_gpu_memory'] <= 1):
        raise ValueError('Invalid evaluation sampling/memory settings')
    config['generation_batch_size'] = batch
    return config


def build_recipe(config, data_directory):
    recipe = yaml.safe_load(resolve_path(config['training_recipe']).read_text())
    recipe.update(seed=config['seed'], data_seed=config['seed'], shuffle_dataset=False,
                  remove_unused_columns=False, num_iterations=1, num_train_epochs=1,
                  training_schedule_path=str(data_directory / 'training_schedule.json'),
                  dataset_adapter='auto', dataset_prompt_column='prompt', dataset_train_split='train',
                  dataset_test_split='test', system_prompt=config['system_prompt'],
                  do_eval=False, eval_strategy='no', eval_on_start=False,
                  max_train_samples=None, max_eval_samples=None,
                  save_reward_data=True, log_completions=False, scale_rewards=True,
                  resume_from_checkpoint=None, push_to_hub=False, overwrite_output_dir=False)
    # TRL rejects setting both generation_batch_size and steps_per_generation.
    # It derives the latter from the former and the actual distributed world size.
    recipe.pop('steps_per_generation', None)
    # Both methods retain exactly the same loss, optimizer, LoRA, KL and precision.
    if recipe.get('loss_type') != 'bnpo':
        raise ValueError('The initial comparison expects the existing BNPO loss for both arms')
    return recipe


def prepare(config_path, directory, dataset_override=None):
    from open_r1.paired_data import freeze_data
    config = checked_config(config_path, dataset_override)
    directory = Path(directory).expanduser().resolve()
    if directory.exists():
        raise ValueError(f'Experiment already exists: {directory}; choose a new directory or reuse its stages')
    directory.mkdir(parents=True)
    try:
        data_dir = directory / 'data'
        freeze_data(**config['data'], directory=data_dir, system_prompt=config['system_prompt'],
                    steps=config['steps'], world_size=config['world_size'],
                    per_device_train_batch_size=config['per_device_train_batch_size'],
                    gradient_accumulation_steps=config['gradient_accumulation_steps'],
                    num_generations=config['num_generations'], seed=config['seed'],
                    eval_seed=config['evaluation']['seed'], num_eval_prompts=config['evaluation']['num_prompts'])
        (directory / 'configs').mkdir()
        shared = build_recipe(config, data_dir)
        files = {}
        for arm in ARMS:
            advantage, options = ('grpo', {}) if arm == 'baseline' else improved_spec(config)
            recipe = deepcopy(shared)
            recipe.update(advantage=advantage, advantage_kwargs=options)
            name = f'configs/{arm}.yaml'
            (directory / name).write_text(yaml.safe_dump(recipe, sort_keys=False), encoding='utf-8')
            files[name] = file_sha256(directory / name)
        accel = directory / 'configs/accelerate.yaml'
        shutil.copyfile(resolve_path(config['accelerate_config']), accel)
        accel_config = yaml.safe_load(accel.read_text())
        if accel_config.get('num_processes') != config['world_size']:
            raise ValueError('Accelerate process count differs from the frozen schedule')
        files['configs/accelerate.yaml'] = file_sha256(accel)
        local_models = {}
        for field in ('model_name', 'qrm_model'):
            if Path(config[field]).expanduser().is_dir():
                config[field] = str(Path(config[field]).expanduser().resolve())
                local_models[field] = model_fingerprint(config[field])
        manifest = {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'directory': str(directory), 'config': config, 'files': files,
                    'training_code_sha256': training_code_digest(), 'runtime_versions': runtime_versions(),
                    'numerical_environment': {name: os.environ.get(name) for name in NUMERICAL_ENV},
                    'local_model_fingerprints': local_models,
                    'data_manifest_sha256': file_sha256(data_dir / 'data_manifest.json')}
        manifest['manifest_sha256'] = digest(manifest)
        write_json(directory / 'experiment.json', manifest)
    except BaseException:
        shutil.rmtree(directory)  # Only this function's newly created directory.
        raise
    print(f'Prepared: {directory}\nTraining prompts: {config["steps"] * config["generation_batch_size"] // config["num_generations"]}; '
          f'held-out prompts: {config["evaluation"]["num_prompts"]}', flush=True)
    return manifest


def verify_experiment(directory, *, check_training_runtime=True):
    from open_r1.paired_data import verify_frozen_data
    directory = Path(directory).expanduser().resolve()
    manifest = read_json(directory / 'experiment.json')
    payload = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if manifest.get('version') != 1 or digest(payload) != manifest.get('manifest_sha256'):
        raise ValueError('Experiment manifest was modified; prepare a new experiment')
    if manifest['directory'] != str(directory):
        raise ValueError('Experiment directory moved; absolute frozen training paths would no longer match')
    for name, expected in manifest['files'].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f'Frozen configuration changed: {name}')
    if file_sha256(directory / 'data/data_manifest.json') != manifest['data_manifest_sha256']:
        raise ValueError('Data manifest changed')
    verify_frozen_data(directory / 'data')
    if check_training_runtime:
        if training_code_digest() != manifest['training_code_sha256']:
            raise ValueError('Training code changed since preparation; restore it or prepare a new experiment')
        if runtime_versions() != manifest['runtime_versions']:
            raise ValueError('Training package versions changed since preparation')
        for field, expected in manifest['local_model_fingerprints'].items():
            if model_fingerprint(manifest['config'][field]) != expected:
                raise ValueError(f'Initial model changed: {field}')
    return manifest


def training_environment(directory, manifest, arm):
    """Ignore ambient training overrides, including stale smoke/resume settings."""
    config = manifest['config']
    services = config['services']
    helpers = runpy.run_path(str(ROOT / 'train_scripts/run_helpers.py'))
    controlled = set().union(*(set(helpers[name]) for name in (
        'STRING_OVERRIDES', 'INTEGER_OVERRIDES', 'BOOLEAN_OVERRIDES', 'STRING_VARIABLES', 'INTEGER_VARIABLES')))
    controlled.update({'ADVANTAGE_KWARGS', 'RESUME_FROM_CHECKPOINT', 'RUN_DIR', 'RUN_NAME', 'CONFIG_FILE',
                       'ACCELERATE_CONFIG', 'GRPO_OUTPUT_ROOT', 'DRY_RUN', 'VLLM_GPU', 'VLLM_PORT',
                       'CUDA_VISIBLE_DEVICES', 'MERGE_AFTER_TRAINING', 'JUDGE_API_KEY',
                       'OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'ANTHROPIC_API_KEY'})
    env = stable_backend_environment(manifest)
    env = {key: value for key, value in env.items() if key not in controlled}
    env.update({
        'PYTHON': sys.executable, 'CONFIG_FILE': str(directory / f'configs/{arm}.yaml'),
        'ACCELERATE_CONFIG': str(directory / 'configs/accelerate.yaml'),
        'RUN_DIR': str(directory / arm), 'RUN_NAME': f'{directory.name}-{arm}',
        'DATASET_NAME': str(directory / 'data/dataset'), 'MODEL_NAME': config['model_name'],
        'MODEL_REVISION': config['model_revision'], 'QRM_MODEL': config['qrm_model'],
        'QRM_REVISION': config['qrm_revision'], 'MAX_STEPS': str(config['steps']),
        'GENERATION_BATCH_SIZE': str(config['generation_batch_size']),
        'PER_DEVICE_TRAIN_BATCH_SIZE': str(config['per_device_train_batch_size']),
        'GRADIENT_ACCUMULATION_STEPS': str(config['gradient_accumulation_steps']),
        'NUM_GENERATIONS': str(config['num_generations']), 'MAX_PROMPT_LENGTH': str(config['max_prompt_length']),
        'MAX_COMPLETION_LENGTH': str(config['max_completion_length']), 'MERGE_AFTER_TRAINING': '1',
        'RESUME_FROM_CHECKPOINT': '', 'VLLM_GPUS': str(services['vllm_gpus']),
        'QRM_GPU': str(services['qrm_gpu']), 'TRAIN_GPUS': str(services['train_gpus']),
        'VLLM_HTTP_PORT': str(services['vllm_http_port']), 'QRM_HTTP_PORT': str(services['qrm_http_port']),
        'PORT': str(services['training_port']), 'VLLM_GROUP_PORT': str(services['weight_sync_port']),
        'VLLM_MAX_MODEL_LEN': str(services['max_model_len']), 'QRM_MAX_LENGTH': str(services['qrm_max_length']),
        'QRM_MAX_BATCH_TOKENS': str(services['qrm_max_batch_tokens']),
        'REWARD_BATCH_SIZE': str(services['reward_batch_size']), 'VLLM_MAX_NUM_SEQS': str(services['vllm_max_num_seqs']),
        'VLLM_GPU_MEMORY_UTILIZATION': str(services['vllm_gpu_memory_utilization']),
        'VLLM_USE_V1': '0', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'WANDB_MODE': 'offline',
    })
    return env


def stable_backend_environment(manifest):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('VLLM_') or key in ('VLLM_CACHE_ROOT', 'VLLM_TMPDIR', 'VLLM_STARTUP_TIMEOUT')}
    for name in NUMERICAL_ENV:
        value = manifest['numerical_environment'].get(name)
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    env.update(VLLM_USE_V1='0', VLLM_WORKER_MULTIPROC_METHOD='spawn', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    return env


def execute(command, *, env=None):
    # Arguments are never interpreted by a shell; API keys stay in the environment.
    print('Executing:', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)


def require_completed(directory, arm, steps):
    run = directory / arm
    status = dict(line.split('=', 1) for line in (run / 'RUN_STATUS').read_text().splitlines() if '=' in line)
    if status.get('status') != 'success':
        raise ValueError(f'{arm} did not complete successfully')
    report = read_json(run / 'validation_report.json')
    state = read_json(run / 'trainer_state.json')
    if report.get('status') != 'passed' or state.get('global_step') != steps:
        raise ValueError(f'{arm} validation/optimizer-step count is incorrect')
    merged = run / 'merged_model'
    if not (merged / 'config.json').is_file() or not any(merged.glob('*.safetensors')):
        raise ValueError(f'{arm} merged model is missing')


def train(directory, arms, *, dry_run=False, restart_failed=False):
    directory = Path(directory).expanduser().resolve()
    for arm in arms:
        manifest = verify_experiment(directory)
        run = directory / arm
        if run.exists() and not dry_run:
            try:
                require_completed(directory, arm, manifest['config']['steps'])
                audit_arm(directory, manifest, arm)
            except (ValueError, OSError, KeyError) as error:
                if not restart_failed:
                    raise ValueError(f'{arm} is incomplete/invalid: {error}. Use train --restart-failed to archive and restart from base.') from error
                archive = directory / f'{arm}.failed-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}'
                run.rename(archive)
                # Preserve old answers/results as evidence, then regenerate for
                # the new weights. Never attach old judgments to a fresh run.
                for name in ('evaluation', 'comparison_result.json', 'data_order_audit.json'):
                    artifact = directory / name
                    if artifact.exists():
                        artifact.rename(archive / name)
                print(f'Archived previous run: {archive}')
            else:
                print(f'Reusing audited completed arm: {arm}')
                continue
        command = [sys.executable, ROOT / 'scripts/grpo.py', 'train']
        if dry_run:
            command.append('--dry-run')
        execute(command, env=training_environment(directory, manifest, arm))
        if not dry_run:
            require_completed(directory, arm, manifest['config']['steps'])
            audit_arm(directory, manifest, arm)
    return 0


def audit_arm(directory, manifest, arm):
    """Validate every consumed row and actual microbatch permutation, not only seeds."""
    from types import SimpleNamespace
    from open_r1.training_schedule import TrainingSchedule
    config = manifest['config']
    args = SimpleNamespace(**{**config, 'max_steps': config['steps'], 'num_iterations': 1,
                              'steps_per_generation': config['gradient_accumulation_steps'], 'shuffle_dataset': False,
                              'remove_unused_columns': False})
    schedule = TrainingSchedule.load(directory / 'data/training_schedule.json', args, config['world_size'])
    require_completed(directory, arm, config['steps'])
    run = directory / arm
    if read_json(run / 'run_manifest.json').get('package_versions') != manifest['runtime_versions']:
        raise ValueError(f'{arm}: training package versions differ from the experiment')
    recorded_env = {}
    for line in (run / 'run.env').read_text().splitlines():
        name, separator, value = line.partition('=')
        if separator:
            words = shlex.split(value)
            recorded_env[name] = words[0] if len(words) == 1 else ''
    expected_env = training_environment(directory, manifest, arm)
    for name in ('MODEL_NAME', 'MODEL_REVISION', 'QRM_MODEL', 'QRM_REVISION', 'VLLM_GPUS', 'QRM_GPU',
                 'TRAIN_GPUS', 'VLLM_MAX_MODEL_LEN', 'VLLM_MAX_NUM_SEQS', 'VLLM_GPU_MEMORY_UTILIZATION',
                 'QRM_MAX_LENGTH', 'QRM_MAX_BATCH_TOKENS', 'REWARD_BATCH_SIZE'):
        if recorded_env.get(name) != expected_env[name]:
            raise ValueError(f'{arm}: actual service configuration differs: {name}')
    actual = yaml.safe_load((run / 'config/resolved_training_config.yaml').read_text())
    expected = yaml.safe_load((directory / f'configs/{arm}.yaml').read_text())
    for key, value in expected.items():
        if isinstance(value, str) and '${' in value:
            continue
        if actual.get(key) != value:
            raise ValueError(f'{arm} actual training config differs: {key}')
    for key, value in {'max_steps': config['steps'], 'dataset_name': str(directory / 'data/dataset'),
                       'model_name_or_path': config['model_name'], 'model_revision': config['model_revision'],
                       'generation_batch_size': config['generation_batch_size'],
                       'gradient_accumulation_steps': config['gradient_accumulation_steps'],
                       'per_device_train_batch_size': config['per_device_train_batch_size'],
                       'num_generations': config['num_generations'], 'max_prompt_length': config['max_prompt_length'],
                       'max_completion_length': config['max_completion_length']}.items():
        if actual.get(key) != value:
            raise ValueError(f'{arm} actual training config differs: {key}')
    traces = []
    local_batch = config['generation_batch_size'] // config['world_size']
    for rank in range(config['world_size']):
        path = run / f'data_order/rank_{rank}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != config['steps']:
            raise ValueError(f'{arm} rank {rank}: expected exactly {config["steps"]} rollout traces')
        for step, row in enumerate(rows):
            if (row.get('version') != 1 or row.get('step') != step or row.get('rank') != rank
                    or row.get('micro_step') != step * config['gradient_accumulation_steps']):
                raise ValueError(f'{arm}: missing/duplicate/reordered step at rank {rank}, step {step}')
            if row.get('sample_ids') != schedule.expected_sample_ids(step, rank):
                raise ValueError(f'{arm}: actual training sample order differs from plan')
            permutation = schedule.planned_permutation(step, rank)
            if hasattr(permutation, 'tolist'):
                permutation = permutation.tolist()
            if row.get('permutation') != permutation or sorted(permutation) != list(range(local_batch)):
                raise ValueError(f'{arm}: optimizer microbatch order differs from plan')
            hashes = row.get('processed_prompt_sha256', [])
            if len(hashes) != local_batch or any(not isinstance(h, str) or not re.fullmatch(r'[0-9a-f]{64}', h) for h in hashes):
                raise ValueError(f'{arm}: processed prompt fingerprints are incomplete')
            traces.append(row)
    return sorted(traces, key=lambda row: (row['step'], row['rank']))


def audit(directory):
    directory = Path(directory).expanduser().resolve()
    manifest = verify_experiment(directory, check_training_runtime=False)
    traces = {arm: audit_arm(directory, manifest, arm) for arm in ARMS}
    if traces['baseline'] != traces['improved']:
        raise ValueError('The two arms consumed different prompts, order, or microbatch permutations; judging is blocked')
    report = {'status': 'passed', 'steps': manifest['config']['steps'],
              'train_prompt_count': len(read_json(directory / 'data/training_schedule.json')['prompt_ids']),
              'identical_processed_prompts': True, 'identical_microbatch_order': True,
              'trace_sha256': digest(traces['baseline']), 'manifest_sha256': manifest['manifest_sha256']}
    write_json(directory / 'data_order_audit.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def generation_command(directory, config, arm):
    evaluation = config['evaluation']
    return [sys.executable, ROOT / 'generate/generate_completions.py', '--model', directory / arm / 'merged_model',
            '--prompts-file', directory / 'data/eval_prompts.json', '--num-prompts', evaluation['num_prompts'],
            '--training-config', directory / arm / 'config/resolved_training_config.yaml',
            '--max-prompt-length', config['max_prompt_length'], '--max-new-tokens', evaluation['max_new_tokens'],
            '--temperature', evaluation['temperature'], '--top-p', evaluation['top_p'], '--seed', evaluation['seed'],
            '--n-completions', 1, '--vllm-gpu-memory', evaluation['vllm_gpu_memory'], '--reuse-existing',
            '--output', directory / f'evaluation/completions/{arm}.json']


def generate(directory):
    directory = Path(directory).expanduser().resolve()
    audit(directory)
    manifest = verify_experiment(directory, check_training_runtime=False)
    env = stable_backend_environment(manifest)
    env['CUDA_VISIBLE_DEVICES'] = str(manifest['config']['evaluation']['gpu'])
    env['VLLM_USE_V1'] = '0'
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    for arm in ARMS:
        execute(generation_command(directory, manifest['config'], arm), env=env)
    return 0


def judge_settings(config):
    settings = deepcopy(config['judge'])
    for name, variable in (('model', 'JUDGE_MODEL'), ('base_url', 'JUDGE_BASE_URL'), ('api_provider', 'JUDGE_API_PROVIDER')):
        settings[name] = os.environ.get(variable) or settings[name]
    if settings['api_provider'] not in ('openai', 'deepseek', 'anthropic') or not settings['model']:
        raise ValueError('Set JUDGE_MODEL and an api_provider of openai/deepseek/anthropic')
    value = os.environ.get('JUDGE_TEMPERATURE', settings.get('temperature', DEFAULT_JUDGE_TEMPERATURE))
    settings['temperature'] = None if value is None or str(value).lower() == 'default' else float(value)
    settings['request_settings'] = judge_settings_metadata(
        settings['api_provider'], settings['model'], settings['thinking_mode'], settings['temperature'])
    settings.update(protocol_version=JUDGE_PROTOCOL_VERSION, parser_version=JUDGE_PARSER_VERSION,
                    cache_version=JUDGE_CACHE_VERSION,
                    prompt_template_sha256=digest(build_comparison_prompt('', '', '', True)))
    key_name = {'openai': 'OPENAI_API_KEY', 'deepseek': 'DEEPSEEK_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY'}[settings['api_provider']]
    key = os.environ.get('JUDGE_API_KEY') or os.environ.get(key_name)
    if not key:
        raise ValueError(f'Set JUDGE_API_KEY (or {key_name}); it is never saved to experiment files')
    env = os.environ.copy()
    env[key_name] = key
    return settings, env


def judge(directory):
    directory = Path(directory).expanduser().resolve()
    audit(directory)
    manifest = verify_experiment(directory, check_training_runtime=False)
    config = manifest['config']
    verify_completions(directory, manifest)
    settings, env = judge_settings(config)
    output = directory / 'evaluation/judge' / digest(settings)[:12]
    write_json(output / 'judge_settings.json', settings)
    command = [sys.executable, ROOT / 'evaluate/bootstrap_judge.py',
               '--completions1', directory / 'evaluation/completions/baseline.json',
               '--completions2', directory / 'evaluation/completions/improved.json',
               '--api-provider', settings['api_provider'], '--judge-model', settings['model'],
               '--thinking-mode', settings['thinking_mode'], '--max-retries', settings['max_retries'],
               '--judge-temperature', 'default' if settings['temperature'] is None else settings['temperature'],
               '--N', config['evaluation']['num_prompts'], '--B', config['evaluation']['bootstrap_iterations'],
               '--seed', config['evaluation']['seed'], '--allow-ties', '--judge-both-orders',
               '--min-valid-fraction', 1.0, '--output-dir', output, '--cache-path', output / 'judge_cache.jsonl']
    if settings['base_url']:
        command += ['--base-url', settings['base_url']]
    execute(command, env=env)
    reports = list(output.rglob('*_both_orders_bootstrap.json'))
    if len(reports) != 1:
        raise ValueError('Expected exactly one paired-order judge report')
    report = read_json(reports[0])
    if report.get('status') != 'success' or not report['validation']['passed']:
        raise ValueError('Judge validation failed; inspect the raw report and retry missing judgments')
    observed = report['validation']['observed']['overall_analysis']
    interval = report['bootstrap_analysis']['overall_winner_analysis']['model2_score_distribution']['ci_95']
    write_json(directory / 'comparison_result.json', {
        'status': 'completed', 'model1': 'baseline: original GRPO advantage',
        'model2': f"improved: {config.get('improved_advantage', 'robust_pairwise')} advantage", 'judge_results_directory': str(output),
        'data_order_audit': str(directory / 'data_order_audit.json'),
        'raw_judge_report': str(reports[0]), 'observed': observed,
        'improved_mean_score': observed['model2_mean_score'], 'improved_score_ci95': interval,
        'neutral_score': 0.5,
        'interpretation': 'model2 mean score > 0.5 favors improved; use the prompt-level 95% CI and disagreement/failure counts. '
                          'One training seed and 200 steps are a pilot, not proof of general superiority.'})
    print(f'Comparison complete: {directory / "comparison_result.json"}')
    return 0


def verify_completions(directory, manifest):
    """Standalone judging must also reject answers from old/different weights."""
    from open_r1.evaluation import frozen_prompt_text, load_frozen_prompts, valid_generation_cache
    config = manifest['config']
    evaluation = config['evaluation']
    prompts, identity = load_frozen_prompts(directory / 'data/eval_prompts.json', evaluation['num_prompts'])
    text_prompts = [frozen_prompt_text(prompt, config['system_prompt']) for prompt in prompts]
    expected = {'frozen_prompts': identity, 'prompts_sha256': digest(prompts),
                'system_prompt': config['system_prompt'], 'enable_thinking': False,
                'max_prompt_length': config['max_prompt_length'], 'max_new_tokens': evaluation['max_new_tokens'],
                'temperature': evaluation['temperature'], 'top_p': evaluation['top_p'],
                'seed': evaluation['seed'], 'n_completions': 1, 'backend': 'vllm'}
    token_hash = None
    for arm in ARMS:
        path = directory / f'evaluation/completions/{arm}.json'
        payload = read_json(path)
        contract = payload.get('meta', {}).get('contract', {})
        expected['model_sha256'] = model_fingerprint(directory / arm / 'merged_model')
        for name, value in expected.items():
            if contract.get(name) != value:
                raise ValueError(f'{arm} completions do not match current model/data/settings: {name}')
        if not valid_generation_cache(path, contract, text_prompts, 1):
            raise ValueError(f'{arm} completions integrity check failed')
        current_hash = contract.get('tokens_sha256')
        if not isinstance(current_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', current_hash):
            raise ValueError('Completions lack actual input-token fingerprints')
        if token_hash is not None and current_hash != token_hash:
            raise ValueError('The two models received different tokenized evaluation prompts')
        token_hash = current_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'train', 'audit', 'generate', 'judge'):
        command = sub.add_parser(name)
        command.add_argument('--experiment-dir', required=True, type=Path)
        if name in ('prepare', 'run'):
            command.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
            command.add_argument('--dataset', help='Override source dataset during preparation only')
        if name in ('train', 'run'):
            command.add_argument('--restart-failed', action='store_true', help='Archive partial/invalid arms and restart them from base weights')
        if name == 'train':
            command.add_argument('--arm', choices=['both', *ARMS], default='both')
            command.add_argument('--dry-run', action='store_true', help='Print both resolved launch plans without loading models')
    args = parser.parse_args(argv)
    directory = args.experiment_dir.expanduser().resolve()
    try:
        if args.command == 'prepare':
            prepare(args.config, directory, args.dataset)
        elif args.command == 'train':
            return train(directory, ARMS if args.arm == 'both' else [args.arm],
                         dry_run=args.dry_run, restart_failed=args.restart_failed)
        elif args.command == 'audit':
            audit(directory)
        elif args.command == 'generate':
            return generate(directory)
        elif args.command == 'judge':
            return judge(directory)
        else:
            if directory.exists():
                manifest = verify_experiment(directory)
                if args.dataset is not None and args.dataset != manifest['config']['data']['source']:
                    raise ValueError('An existing experiment cannot change dataset; prepare a new directory')
            else:
                config = checked_config(args.config, args.dataset)
                judge_settings(config)  # Detect missing judge credentials before hours of training.
                manifest = prepare(args.config, directory, args.dataset)
            judge_settings(manifest['config'])
            train(directory, ARMS, restart_failed=args.restart_failed)
            generate(directory)
            return judge(directory)
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
