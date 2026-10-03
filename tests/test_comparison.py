"""Exercise paired orchestration with real CPU data and launcher dry-runs."""
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from open_r1 import comparison as comparison
from open_r1.evaluation import digest
from open_r1.training_schedule import TrainingSchedule


@pytest.fixture
def experiment(tmp_path):
    from datasets import Dataset, DatasetDict
    source = tmp_path / 'source'
    DatasetDict({'train': Dataset.from_dict({'prompt': [f'Train question {i}' for i in range(24)]}),
                 'test': Dataset.from_dict({'prompt': [f'Test question {i}' for i in range(8)]})}).save_to_disk(source)
    config = yaml.safe_load(comparison.DEFAULT_CONFIG.read_text())
    config.update(steps=2, num_generations=2, gradient_accumulation_steps=4)
    config['evaluation']['num_prompts'] = 4
    config['data']['source'] = str(source)
    recipe = tmp_path / 'experiment.yaml'
    recipe.write_text(yaml.safe_dump(config))
    directory = tmp_path / 'paired'
    manifest = comparison.prepare(recipe, directory)
    return directory, manifest


def test_prepare_freezes_only_advantage_difference_and_detects_drift(experiment):
    directory, manifest = experiment
    comparison.verify_experiment(directory)
    a, b = [yaml.safe_load((directory / f'configs/{arm}.yaml').read_text()) for arm in comparison.ARMS]
    assert {k for k in a if a[k] != b[k]} == {'advantage', 'advantage_kwargs'}
    assert a['advantage'] == 'grpo' and b['advantage_kwargs'] == {'delta': .01, 'c': .08}
    assert 'steps_per_generation' not in a  # TRL disallows setting both generation geometry fields.
    assert a['remove_unused_columns'] is False and a['shuffle_dataset'] is False
    schedule = comparison.read_json(directory / 'data/training_schedule.json')
    assert len(schedule['prompt_ids']) == 8
    (directory / 'configs/baseline.yaml').write_text('changed: true')
    with pytest.raises(ValueError, match='configuration changed'):
        comparison.verify_experiment(directory)


def test_rolling_improved_prepare_preserves_baseline_and_uses_explicit_quantiles(experiment, tmp_path):
    _, manifest = experiment
    config = manifest['config'].copy()
    config.pop('delta')
    config.pop('c')
    config.update(improved_advantage='rolling_quantile_pairwise',
                  improved_advantage_kwargs={'p': .1, 'q': .05, 'window_size': 4, 'epsilon': .0001})
    recipe = tmp_path / 'rolling.yaml'
    recipe.write_text(yaml.safe_dump(config))
    directory = tmp_path / 'rolling-paired'
    rolling_manifest = comparison.prepare(recipe, directory)
    comparison.verify_experiment(directory)
    baseline, improved = [yaml.safe_load((directory / f'configs/{arm}.yaml').read_text())
                          for arm in comparison.ARMS]
    assert {k for k in baseline if baseline[k] != improved[k]} == {'advantage', 'advantage_kwargs'}
    assert baseline['advantage'] == 'grpo' and baseline['advantage_kwargs'] == {}
    assert improved['advantage'] == 'rolling_quantile_pairwise'
    assert improved['advantage_kwargs'] == config['improved_advantage_kwargs']
    assert rolling_manifest['config']['steps'] == config['steps']
    env = comparison.training_environment(directory, rolling_manifest, 'improved')
    result = subprocess.run([env['PYTHON'], str(comparison.ROOT / 'scripts/grpo.py'), 'train', '--dry-run'],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'advantage: rolling_quantile_pairwise' in result.stdout
    assert not (directory / 'improved').exists()


def test_rolling_comparison_requires_explicit_valid_proportions():
    with pytest.raises(ValueError, match='p and q'):
        comparison.improved_spec({'improved_advantage': 'rolling_quantile_pairwise'})
    with pytest.raises(ValueError, match='less than 1'):
        comparison.improved_spec({'improved_advantage': 'rolling_quantile_pairwise',
                                  'improved_advantage_kwargs': {'p': .6, 'q': .5}})


def test_real_launcher_dry_runs_ignore_ambient_smoke_and_method_settings(experiment, monkeypatch):
    directory, manifest = experiment
    for name, value in {'ADVANTAGE': 'ranking', 'MAX_TRAIN_SAMPLES': '1', 'MAX_STEPS': '1',
                        'DO_EVAL': '1', 'RESUME_FROM_CHECKPOINT': '/bad/checkpoint',
                        'VLLM_ATTENTION_BACKEND': 'bad_backend'}.items():
        monkeypatch.setenv(name, value)
    for arm in comparison.ARMS:
        env = comparison.training_environment(directory, manifest, arm)
        assert 'MAX_TRAIN_SAMPLES' not in env and 'ADVANTAGE' not in env
        assert 'VLLM_ATTENTION_BACKEND' not in env and env['RESUME_FROM_CHECKPOINT'] == ''
        result = subprocess.run([env['PYTHON'], str(comparison.ROOT / 'scripts/grpo.py'), 'train', '--dry-run'],
                                env=env, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'max_steps: 2' in result.stdout
        assert 'training_schedule_path:' in result.stdout
        assert 'remove_unused_columns: false' in result.stdout
        assert not (directory / arm).exists()


def fabricate_completed_arm(directory, manifest, arm):
    config = manifest['config']
    run = directory / arm
    (run / 'config').mkdir(parents=True)
    (run / 'merged_model').mkdir()
    (run / 'merged_model/config.json').write_text('{}')
    (run / 'merged_model/model.safetensors').write_bytes(arm.encode())
    (run / 'RUN_STATUS').write_text('status=success\n')
    comparison.write_json(run / 'trainer_state.json', {'global_step': config['steps']})
    comparison.write_json(run / 'validation_report.json', {'status': 'passed'})
    comparison.write_json(run / 'run_manifest.json', {'package_versions': manifest['runtime_versions']})
    env = comparison.training_environment(directory, manifest, arm)
    (run / 'run.env').write_text('\n'.join(f'{k}={shlex.quote(v)}' for k, v in env.items()))
    actual = yaml.safe_load((directory / f'configs/{arm}.yaml').read_text())
    helpers = comparison.runpy.run_path(str(comparison.ROOT / 'train_scripts/run_helpers.py'))
    variables = {**env, 'OUTPUT_DIR': str(run), 'QRM_REQUEST_TIMEOUT': '900'}
    for field in helpers['INTEGER_VARIABLES']:
        if field in variables:
            variables[field] = int(variables[field])
    actual = helpers['substitute'](actual, variables)
    (run / 'config/resolved_training_config.yaml').write_text(yaml.safe_dump(actual))
    args = SimpleNamespace(**{**config, 'max_steps': config['steps'], 'num_iterations': 1,
                             'steps_per_generation': config['gradient_accumulation_steps'],
                             'shuffle_dataset': False, 'remove_unused_columns': False})
    schedule = TrainingSchedule.load(directory / 'data/training_schedule.json', args, config['world_size'])
    (run / 'data_order').mkdir()
    for rank in range(config['world_size']):
        rows = []
        for step in range(config['steps']):
            inputs = [{'comparison_sample_id': sid, 'prompt': 'prepared ' + sid}
                      for sid in schedule.expected_sample_ids(step, rank)]
            rows.append(schedule.prepare_trace(inputs, step, step * config['gradient_accumulation_steps'], rank))
        (run / f'data_order/rank_{rank}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')


def test_audit_checks_real_order_and_processed_prompts(experiment):
    directory, manifest = experiment
    for arm in comparison.ARMS:
        fabricate_completed_arm(directory, manifest, arm)
    assert comparison.audit(directory)['status'] == 'passed'
    path = directory / 'improved/data_order/rank_0.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['processed_prompt_sha256'][0] = digest('different truncated prompt')
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match='consumed different'):
        comparison.audit(directory)


def test_generation_commands_share_frozen_prompts_and_settings(experiment):
    directory, manifest = experiment
    a, b = [list(map(str, comparison.generation_command(directory, manifest['config'], arm))) for arm in comparison.ARMS]
    assert a[a.index('--prompts-file') + 1] == b[b.index('--prompts-file') + 1]
    assert a[a.index('--temperature') + 1] == '0.0'
    assert '--dataset' not in a


def test_judge_credentials_are_not_command_arguments_or_saved_settings(monkeypatch):
    config = yaml.safe_load(comparison.DEFAULT_CONFIG.read_text())
    monkeypatch.setenv('JUDGE_MODEL', 'fixture-judge')
    monkeypatch.setenv('JUDGE_API_KEY', 'fixture-secret')
    settings, env = comparison.judge_settings(config)
    assert env['OPENAI_API_KEY'] == 'fixture-secret'
    assert 'fixture-secret' not in json.dumps(settings)


def test_standalone_judging_rejects_answers_from_previous_weights(experiment):
    from open_r1.evaluation import ARTIFACT_VERSION, frozen_prompt_text, load_frozen_prompts, model_fingerprint
    directory, manifest = experiment
    config = manifest['config']
    evaluation = config['evaluation']
    prompts, identity = load_frozen_prompts(directory / 'data/eval_prompts.json', evaluation['num_prompts'])
    for arm in comparison.ARMS:
        fabricate_completed_arm(directory, manifest, arm)
        contract = {'frozen_prompts': identity, 'prompts_sha256': digest(prompts),
                    'model_sha256': model_fingerprint(directory / arm / 'merged_model'),
                    'system_prompt': config['system_prompt'], 'enable_thinking': False,
                    'max_prompt_length': config['max_prompt_length'], 'max_new_tokens': evaluation['max_new_tokens'],
                    'temperature': evaluation['temperature'], 'top_p': evaluation['top_p'],
                    'seed': evaluation['seed'], 'n_completions': 1, 'backend': 'vllm', 'tokens_sha256': digest([1, 2])}
        items = [{'prompt': frozen_prompt_text(p, config['system_prompt']), 'completions': ['fixture answer']} for p in prompts]
        comparison.write_json(directory / f'evaluation/completions/{arm}.json', {
            'meta': {'artifact_version': ARTIFACT_VERSION, 'contract': contract, 'items_sha256': digest(items)}, 'items': items})
    comparison.verify_completions(directory, manifest)
    (directory / 'improved/merged_model/model.safetensors').write_bytes(b'retrained weights')
    with pytest.raises(ValueError, match='model_sha256'):
        comparison.verify_completions(directory, manifest)
