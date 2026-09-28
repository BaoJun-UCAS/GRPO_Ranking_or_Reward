"""Evaluation regressions: CPU only, mocked API clients, no model downloads."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from open_r1.evaluation import (
    ARTIFACT_VERSION, digest, model_fingerprint, valid_generation_cache, prompt_token_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge = load_module("judge_under_test", "evaluate/bootstrap_judge.py")
generation = load_module("generation_under_test", "generate/generate_completions.py")
plotting = load_module("plotting_under_test", "evaluate/plot_win_rates.py")


def response(winner="A"):
    return "\n".join(
        f"**{index}. {name}**\n- Winner: [{winner}]\n- Justification: explanation"
        for index, name in enumerate(("Helpfulness", "Correctness", "Coherence", "Complexity", "Verbosity"), 1)
    ) + f"\n**Overall Recommendation:**\nWinner: [{winner}]\nJustification: explanation"


@pytest.mark.parametrize("winner", ["AB", "A/B", "Based", "", "A]", "[B"])
def test_parser_rejects_invalid_decision(winner):
    parsed = judge.parse_survey_response(response(winner), False, False)
    assert parsed["overall_parsing_failed"]
    assert parsed["survey_winner"] is None


def test_parser_does_not_cross_sections():
    text = response().replace("- Winner: [A]", "No verdict", 1)
    parsed = judge.parse_survey_response(text, False, False)
    assert parsed["criterion_evaluations"]["helpfulness"]["parsing_failed"]
    assert parsed["survey_winner"] is None
    assert parsed["survey_calculation"]["successful_criteria_count"] == 4


@pytest.mark.parametrize("swap,expected", [(False, "model1"), (True, "model2")])
def test_parser_swapped_order(swap, expected):
    parsed = judge.parse_survey_response(response(), swap, False)
    assert parsed["overall_winner"] == parsed["survey_winner"] == expected


def test_ties_follow_protocol():
    assert judge.parse_survey_response(response("Tie"), False, True)["overall_winner"] == "tie"
    assert judge.parse_survey_response(response("Tie"), False, False)["overall_parsing_failed"]


def test_empty_statistics_are_not_zero_success():
    failure = {"overall_parsing_failed": True, "overall_winner": None, "survey_winner": None}
    analysis = judge.analyze_bootstrap_results([{"analysis": judge._analyze_iteration([failure, failure])}])
    for section in ("overall_winner_analysis", "survey_winner_analysis"):
        assert analysis[section]["exclusion_rate_distribution"]["mean"] == 1
        assert analysis[section]["model1_win_rate_distribution"]["mean"] is None


@pytest.mark.parametrize("mode,expected", [("success", 0), ("api_failure", 1), ("parse_failure", 1)])
def test_judge_main_status_and_plot_gate(tmp_path, monkeypatch, mode, expected):
    items = [{"prompt": str(i), "completions": ["answer"]} for i in range(3)]
    for filename in ("first.json", "second.json"):
        (tmp_path / filename).write_text(json.dumps({"meta": {}, "items": items}))
    def create(**kwargs):
        if mode == "api_failure":
            raise RuntimeError("mock outage")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=response() if mode == "success" else "malformed response"))], usage=None)
    client = SimpleNamespace(base_url="https://mock.invalid/v1", chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(judge, "setup_judge_client", lambda *a: ("deepseek", client))
    monkeypatch.setattr(sys, "argv", ["judge", "--completions1", str(tmp_path / "first.json"),
        "--completions2", str(tmp_path / "second.json"), "--N", "3", "--B", "10",
        "--api-key", "fake-unused", "--max-retries", "1", "--output-dir", str(tmp_path / "out")])
    assert judge.main() == expected
    report = next((tmp_path / "out").glob("*bootstrap.json"))
    payload = json.loads(report.read_text())
    assert payload["status"] == ("success" if expected == 0 else "invalid")
    if expected:
        assert not list((tmp_path / "out").rglob("*.jsonl"))
        with pytest.raises(ValueError, match="invalid"):
            plotting.load_win_rate_data(report)
    else:
        assert plotting.load_win_rate_data(report)
        # All successful decisions can be reused with zero new API token usage.
        monkeypatch.setattr(client.chat.completions, "create", lambda **kw: pytest.fail("cache miss"))
        assert judge.main() == 0
        payload = json.loads(report.read_text())
        assert payload["cache_summary"]["hits_this_run"] == 3
        assert payload["api_usage"]["input_tokens"] == 0


def test_cache_identity_includes_endpoint_and_protocol(monkeypatch):
    item = {"prompt": "p", "completion1": "a", "completion2": "b"}
    args = (item, "openai", "model", False, False, "disabled")
    key = judge._cache_key(*args, base_url="https://a.invalid")
    assert key != judge._cache_key(*args, base_url="https://b.invalid")
    monkeypatch.setattr(judge, "JUDGE_PROTOCOL_VERSION", "changed")
    assert key != judge._cache_key(*args, base_url="https://a.invalid")


def test_model_hash_changes_even_with_same_size_and_mtime(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"aaaa")
    before = weight.stat()
    fingerprint = model_fingerprint(tmp_path)
    weight.write_bytes(b"bbbb")
    os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert model_fingerprint(tmp_path) != fingerprint


@pytest.mark.parametrize("field", ["model_sha256", "max_new_tokens", "prompts_sha256", "system_prompt", "backend"])
def test_generation_cache_rejects_contract_changes(tmp_path, field):
    path = tmp_path / "completions.json"
    contract = {field: "before"}
    items = [{"prompt": "p", "completions": ["answer"]}]
    path.write_text(json.dumps({"meta": {"artifact_version": ARTIFACT_VERSION, "contract": contract,
        "items_sha256": digest(items)}, "items": items}))
    assert valid_generation_cache(path, contract, ["p"], 1)
    assert not valid_generation_cache(path, {field: "after"}, ["p"], 1)
    payload = json.loads(path.read_text())
    payload["items"][0]["completions"] = ["tampered"]
    path.write_text(json.dumps(payload))
    assert not valid_generation_cache(path, contract, ["p"], 1)


def test_template_and_token_budget_are_explicit():
    seen = {}
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.update(messages=messages, kwargs=kwargs)
            return "text"
        def __call__(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return {"input_ids": [1, 2, 3, 4, 5]}
    tokenizer = Tokenizer()
    generation.format_prompt("p", tokenizer, system_prompt="")
    assert seen["messages"] == [{"role": "system", "content": ""}, {"role": "user", "content": "p"}]
    assert not seen["kwargs"]["enable_thinking"]
    generation.format_prompt("p", tokenizer, system_prompt=None)
    assert seen["messages"] == [{"role": "user", "content": "p"}]
    assert prompt_token_ids(tokenizer, ["text"], 3) == [[3, 4, 5]]


@pytest.mark.parametrize("field", ["system_prompt", "backend", "max_new_tokens"])
def test_judge_rejects_mismatched_generation_protocols(field):
    items = [{"prompt": "p", "completion": "answer"}]
    with pytest.raises(ValueError, match="contract mismatch"):
        judge.merge_completions_artifacts({"contract": {field: "a"}}, items,
                                          {"contract": {field: "b"}}, items)


def test_judge_checks_artifact_integrity(tmp_path):
    path = tmp_path / "artifact.json"
    items = [{"prompt": "p", "completions": ["answer"]}]
    payload = {"meta": {"artifact_version": ARTIFACT_VERSION, "items_sha256": digest(items)}, "items": items}
    path.write_text(json.dumps(payload))
    assert judge.read_single_completions_artifact(str(path))[1][0]["completion"] == "answer"
    payload["items"][0]["completions"] = ["changed"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity"):
        judge.read_single_completions_artifact(str(path))


def test_eval_reward_batches_and_phases_do_not_overwrite(tmp_path):
    source = ROOT / "src/open_r1/grpo_trainer.py"
    trainer = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "GRPOTrainer")
    method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "_save_reward_data_buffer")
    scope = {"os": os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    fake = SimpleNamespace(reward_data_save_path=str(tmp_path), _step=16, accelerator=SimpleNamespace(process_index=0))
    for i, training in enumerate((False, False, True)):
        fake.model = SimpleNamespace(training=training)
        fake.reward_data_buffer = [{"batch": i}]
        scope[method.name](fake, 2)
    assert len(list(tmp_path.glob("reward_data_eval_*.json"))) == 2
    assert len(list(tmp_path.glob("reward_data_train_*.json"))) == 1


@pytest.mark.parametrize("do_eval,strategy,expected", [(True, "no", [1]), (False, "steps", [1]), (False, "no", None)])
def test_end_only_eval_gets_dataset(do_eval, strategy, expected):
    source = ROOT / "src/open_r1/grpo.py"
    call = next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "GRPOTrainer")
    expr = next(kw.value for kw in call.keywords if kw.arg == "eval_dataset")
    assert eval(compile(ast.Expression(expr), str(source), "eval"), {
        "dataset": {"val": [1]}, "script_args": SimpleNamespace(dataset_test_split="val"),
        "training_args": SimpleNamespace(do_eval=do_eval, eval_strategy=strategy)}) == expected


@pytest.mark.parametrize("script", ["generate/generate_completions.py", "evaluate/plot_win_rates.py", "evaluate/plot_win_rates_all_temps.py"])
def test_cpu_cli_help(script):
    result = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"}, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["vllm", "transformers"])
def test_generation_main_mock_backends_and_cache(tmp_path, monkeypatch, backend):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"fake weights")
    captured = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"]
        def __call__(self, text, **kwargs):
            return {"input_ids": [10, 11, 12, 13]}
    class Auto:
        @staticmethod
        def from_pretrained(*a, **kw):
            return Tokenizer()
    class LLM:
        def __init__(self, **kwargs):
            captured.append(kwargs)
        def generate(self, prompts, sampling):
            captured.append(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="answer")]) for _ in prompts]
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Auto, AutoModelForCausalLM=Auto))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None, bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setattr(generation, "version", lambda name: "test")
    monkeypatch.setattr(generation, "load_validation_prompts", lambda *a: (["p0", "p1"], None, "ultrachat", "test", {"id": "fixture"}))
    def complete(ids, *a, **kw):
        captured.append(ids)
        return "answer"
    monkeypatch.setattr(generation, "generate_completion", complete)
    output = tmp_path / "completions.json"
    argv = ["generate", "--model", str(model_dir), "--num-prompts", "2", "--output", str(output),
            "--max-prompt-length", "2", "--max-new-tokens", "4", "--reuse-existing"]
    if backend == "transformers":
        argv.append("--no-vllm")
    monkeypatch.setattr(sys, "argv", argv)
    generation.main()
    payload = json.loads(output.read_text())
    assert payload["meta"]["contract"]["tokens_sha256"] == digest([[12, 13], [12, 13]])
    assert len(payload["items"]) == 2
    if backend == "vllm":
        assert captured[-1] == [{"prompt_token_ids": [12, 13]}, {"prompt_token_ids": [12, 13]}]
    else:
        assert captured == [[12, 13], [12, 13]]
    count = len(captured)
    generation.main()
    assert len(captured) == count  # A valid cache never loads a generation model.
    (model_dir / "model.safetensors").write_bytes(b"new weights!")
    with pytest.raises(ValueError, match="do not match"):
        generation.main()
    assert json.loads(output.read_text()) == payload


def test_plot_a_valid_report_without_gpu(tmp_path, monkeypatch):
    parsed = judge.parse_survey_response(response(), False, False)
    observed = judge._analyze_iteration([parsed, parsed])
    payload = {"status": "success", "bootstrap_config": {"model1_path": "baseline", "model2_path": "trained"},
               "bootstrap_analysis": judge.analyze_bootstrap_results([{"analysis": observed}])}
    report = tmp_path / "report.json"
    output = tmp_path / "report.png"
    report.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv", ["plot", str(report), "--output", str(output)])
    plotting.main()
    assert output.read_bytes().startswith(b"\x89PNG")


def test_launcher_propagates_judge_failure(tmp_path):
    # A fake interpreter intercepts every generation/judge command; no external
    # client or GPU process can be reached in this integration test.
    run_dir = tmp_path / "run"
    (run_dir / "merged_model").mkdir(parents=True)
    (run_dir / "merged_model/config.json").write_text("{}")
    (run_dir / "config").mkdir()
    (run_dir / "config/resolved_training_config.yaml").write_text('system_prompt: ""\n')
    fake = tmp_path / "fake-python"
    fake.write_text('#!/bin/sh\ncase "$1" in *bootstrap_judge.py) exit 1;; *) exit 0;; esac\n')
    fake.chmod(0o755)
    eval_dir = tmp_path / "evaluation"
    result = subprocess.run(["bash", str(ROOT / "evaluate/run_grpo_chat_deepseek.sh"), str(run_dir)],
        env={**os.environ, "PYTHON": str(fake), "DEEPSEEK_API_KEY": "fake-unused", "EVAL_DIR": str(eval_dir)},
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "status=failed" in (eval_dir / "EVALUATION_STATUS").read_text()
