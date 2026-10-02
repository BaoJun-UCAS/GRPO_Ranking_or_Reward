"""Frozen held-out inputs and paired-order judgments, with no GPU or API calls."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from open_r1.evaluation import (
    combine_order_judgments, digest, frozen_prompt_messages, frozen_prompt_text, load_frozen_prompts,
)

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge = load_module("paired_judge_under_test", "evaluate/bootstrap_judge.py")
generation = load_module("frozen_generation_under_test", "generate/generate_completions.py")


def write_prompts(path, prompts=None):
    if prompts is None:
        prompts = ["single question", [
            {"role": "system", "content": "original system"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "follow-up"},
        ]]
    payload = {"version": 1, "prompts": prompts, "prompt_ids": [f"test:{i}" for i in range(len(prompts))],
               "prompts_sha256": digest(prompts)}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def test_frozen_prompt_manifest_preserves_order_ids_and_full_file_hash(tmp_path):
    path = tmp_path / "heldout.json"
    payload = write_prompts(path)
    prompts, identity = load_frozen_prompts(path, 2)
    assert prompts == payload["prompts"]
    assert identity["prompt_ids"] == payload["prompt_ids"]
    assert identity["prompts_sha256"] == payload["prompts_sha256"]
    before = identity["file_sha256"]
    payload["source"] = {"split": "heldout"}
    path.write_text(json.dumps(payload))
    assert load_frozen_prompts(path, 2)[1]["file_sha256"] != before
    with pytest.raises(ValueError, match="exactly match"):
        load_frozen_prompts(path, 1)


@pytest.mark.parametrize("change,pattern", [
    (lambda p: p.update(version=2), "version"),
    (lambda p: p.update(version=True), "version"),
    (lambda p: p.update(prompt_ids=["x", "x"]), "unique"),
    (lambda p: p.update(prompt_ids=["x"]), "length"),
    (lambda p: p.update(prompt_ids=["x", ""]), "nonempty"),
    (lambda p: p.update(prompts_sha256="tampered"), "integrity"),
    (lambda p: p["prompts"].append("extra"), "exactly match"),
])
def test_frozen_manifest_rejects_changed_or_invalid_inputs(tmp_path, change, pattern):
    path = tmp_path / "heldout.json"
    payload = write_prompts(path)
    change(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=pattern):
        load_frozen_prompts(path, 2)


@pytest.mark.parametrize("prompt", [
    "", "  ", [], [{"role": "assistant", "content": "answer"}],
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}],
    [{"role": "user", "content": "q"}, {"role": "user", "content": "q2"}],
    [{"role": "user", "content": [{"type": "image"}]}],
    [{"role": "user", "content": "q", "extra": True}],
])
def test_invalid_frozen_chats_are_rejected(prompt):
    with pytest.raises(ValueError):
        frozen_prompt_messages(prompt)


def test_duplicate_frozen_prompts_are_rejected(tmp_path):
    path = tmp_path / "heldout.json"
    write_prompts(path, ["duplicate", "duplicate"])
    with pytest.raises(ValueError, match="unique"):
        load_frozen_prompts(path, 2)


def test_multiturn_context_and_system_override_match_training(tmp_path):
    payload = write_prompts(tmp_path / "heldout.json")
    chat = payload["prompts"][1]
    messages = frozen_prompt_messages(chat, "configured system")
    assert messages[0] == {"role": "system", "content": "configured system"}
    assert messages[1:] == chat[1:]
    assert frozen_prompt_messages(chat, None) == chat
    assert json.loads(frozen_prompt_text(chat, "configured system")) == messages
    assert chat[0]["content"] == "original system", "Formatting mutated the shared frozen prompts"
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda m, **kw: calls.append((m, kw)) or "formatted")
    generation.format_prompt(chat, tokenizer, system_prompt="", enable_thinking=False)
    assert calls[0][0][0] == {"role": "system", "content": ""}
    assert calls[0][0][1:] == chat[1:]
    assert calls[0][1]["enable_thinking"] is False


def test_two_trained_models_generate_from_the_same_frozen_inputs(tmp_path, monkeypatch):
    prompt_file = tmp_path / "heldout.json"
    frozen = write_prompts(prompt_file)
    seen = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages, sort_keys=True)

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(char) for char in text]}

    class Auto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Tokenizer()

    class LLM:
        def __init__(self, **kwargs):
            seen.append(("model", kwargs["model"]))

        def generate(self, prompts, sampling):
            seen.append(("tokens", prompts))
            return [SimpleNamespace(outputs=[SimpleNamespace(text=f"answer{i}")]) for i in range(len(prompts))]

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Auto, AutoModelForCausalLM=Auto))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(generation, "version", lambda name: "fixture")
    monkeypatch.setattr(generation, "load_validation_prompts", lambda *a, **kw: pytest.fail("Frozen inputs fetched a dataset"))
    payloads = []
    for model in ("trained_grpo", "trained_pairwise"):
        model_dir = tmp_path / model
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(model.encode())
        output = tmp_path / f"{model}.json"
        monkeypatch.setattr(sys, "argv", ["generate", "--model", str(model_dir), "--prompts-file", str(prompt_file),
            "--num-prompts", "2", "--max-prompt-length", "2048", "--output", str(output), "--reuse-existing"])
        generation.main()
        payloads.append(json.loads(output.read_text()))
        previous = len(seen)
        generation.main()
        assert len(seen) == previous
    assert payloads[0]["meta"]["contract"]["model_sha256"] != payloads[1]["meta"]["contract"]["model_sha256"]
    assert payloads[0]["meta"]["contract"]["tokens_sha256"] == payloads[1]["meta"]["contract"]["tokens_sha256"]
    assert payloads[0]["meta"]["contract"]["frozen_prompts"]["prompt_ids"] == frozen["prompt_ids"]
    assert payloads[0]["items"] == payloads[1]["items"]
    assert json.loads(payloads[0]["items"][1]["prompt"])[-1]["content"] == "follow-up"
    assert seen[1] == seen[3]
    meta1, items1 = judge.read_single_completions_artifact(str(tmp_path / "trained_grpo.json"))
    meta2, items2 = judge.read_single_completions_artifact(str(tmp_path / "trained_pairwise.json"))
    assert len(judge.merge_completions_artifacts(meta1, items1, meta2, items2)[1]) == 2


@pytest.mark.parametrize("field", ["tokens_sha256", "frozen_prompts", "versions"])
def test_pairing_rejects_unequal_effective_generation_contract(field):
    items = [{"prompt": "same question", "completion": "answer"}]
    with pytest.raises(ValueError, match=field):
        judge.merge_completions_artifacts({"contract": {field: "first"}}, items,
                                          {"contract": {field: "second"}}, items)


def order_result(winner, swapped):
    return {"order_swapped": swapped, "prompt": "q", "model1_completion": "first", "model2_completion": "second",
            "source_index": 0, "overall_winner": winner, "survey_winner": winner,
            "overall_parsing_failed": winner is None}


@pytest.mark.parametrize("first,second,outcome,score", [
    ("model1", "model1", "model1", 0.0), ("model2", "model2", "model2", 1.0),
    ("tie", "tie", "tie", 0.5), ("model1", "model2", "order_disagreement", 0.5),
    ("model2", "tie", "order_disagreement", 0.75), ("model1", "tie", "order_disagreement", 0.25),
    (None, "model2", None, None),
])
def test_pair_reduction_distinguishes_tie_disagreement_and_failure(first, second, outcome, score):
    combined = combine_order_judgments([order_result(first, False), order_result(second, True)])
    assert combined["overall_winner"] == outcome
    assert combined["overall_model2_score"] == score
    assert combined["overall_parsing_failed"] == (outcome is None)
    analysis = judge._analyze_iteration([combined])["overall_analysis"]
    assert analysis["ties"] == int(outcome == "tie")
    assert analysis["order_disagreements"] == int(outcome == "order_disagreement")
    assert analysis["excluded"] == int(outcome is None)


def survey_response(winner):
    return "\n".join(f"**{i}. {name}**\n- Winner: [{winner}]\n- Justification: fixture"
        for i, name in enumerate(("Helpfulness", "Correctness", "Coherence", "Complexity", "Verbosity"), 1)
    ) + f"\n**Overall Recommendation:**\nWinner: [{winner}]\nJustification: fixture"


def test_both_orders_bootstrap_clusters_and_resume_only_failed_call(tmp_path, monkeypatch):
    items = [{"prompt": str(i), "completion": f"answer{i}"} for i in range(6)]
    calls = []

    def evaluate(api_type, client, model, prompt, first, second, allow_ties, order_swapped=False, **kwargs):
        calls.append((prompt, order_swapped))
        decisions = {"0": "model2", "1": "model1", "2": "tie", "3": "model2" if order_swapped else "model1",
                     "4": "tie" if order_swapped else "model2", "5": None if order_swapped else "model2"}
        mapped = decisions[prompt]
        if mapped is None:
            return {"overall_winner": None, "overall_parsing_failed": True, "survey_winner": None, "raw_response": "bad"}
        token = "Tie" if mapped == "tie" else ("A" if (mapped == "model1") != order_swapped else "B")
        return judge.parse_survey_response(survey_response(token), order_swapped, allow_ties)

    monkeypatch.setattr(judge, "call_judge", evaluate)
    kwargs = dict(N=6, B=80, judge_model="mock", api_type="openai", judge_client=None,
                  allow_ties=True, seed=52, cache_path=str(tmp_path / "cache.jsonl"), judge_both_orders=True)
    resamples, judged, selected = judge.run_bootstrap_evaluation({}, items, {}, items, **kwargs)
    assert len(calls) == 12
    assert set(calls) == {(str(i), swapped) for i in range(6) for swapped in (False, True)}
    assert len(judged) == 6
    observed = judge._analyze_iteration(judged)["overall_analysis"]
    assert (observed["valid_comparisons"], observed["excluded"], observed["ties"], observed["order_disagreements"]) == (5, 1, 1, 2)
    assert observed["model2_mean_score"] == pytest.approx(0.55)
    for resample in resamples:
        assert len(resample["sample_positions"]) == 6
        assert resample["analysis"]["total_comparisons"] == 6
        assert max(resample["sample_positions"]) < 6
    statistics = judge.analyze_bootstrap_results(resamples)["overall_winner_analysis"]["model2_score_distribution"]
    expected = [sample["analysis"]["overall_analysis"]["model2_mean_score"] for sample in resamples]
    np.testing.assert_allclose(statistics["ci_95"], np.percentile(expected, [2.5, 97.5]))
    calls.clear()
    judge.run_bootstrap_evaluation({}, items, {}, items, **kwargs)
    assert calls == [("5", True)]


def test_main_both_orders_is_blind_and_reports_position_bias_separately(tmp_path, monkeypatch, capsys):
    items = [{"prompt": f"question{i}", "completions": ["first answer"]} for i in range(3)]
    for number in (1, 2):
        (tmp_path / f"trained_model{number}.json").write_text(json.dumps({
            "meta": {"model_path": f"private-trained-checkpoint-{number}"}, "items": items}))
    calls = []

    def create(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        assert "private-trained-checkpoint" not in prompt
        assert "**Response A:**" in prompt and "**Response B:**" in prompt
        calls.append(prompt)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=survey_response("A")))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))

    client = SimpleNamespace(base_url="https://unused.invalid", chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(judge, "setup_judge_client", lambda *args: ("deepseek", client))
    monkeypatch.setattr(sys, "argv", ["judge", "--completions1", str(tmp_path / "trained_model1.json"),
        "--completions2", str(tmp_path / "trained_model2.json"), "--N", "3", "--B", "20", "--api-key", "unused",
        "--judge-both-orders", "--allow-ties", "--output-dir", str(tmp_path / "out")])
    assert judge.main() == 0
    assert len(calls) == 6
    payload = json.loads(next((tmp_path / "out").glob("*both_orders_bootstrap.json")).read_text())
    observed = payload["validation"]["observed"]["overall_analysis"]
    assert observed["ties"] == 0 and observed["order_disagreements"] == 3
    assert observed["model2_mean_score"] == 0.5
    assert payload["bootstrap_analysis"]["overall_winner_analysis"]["model2_score_distribution"]["ci_95"] == [0.5, 0.5]
    assert payload["api_usage"]["input_tokens"] == 60
    assert payload["cache_summary"]["api_calls_this_run"] == 6
    assert len(payload["judged_results"]) == 3
    assert "Overall Winner:" not in capsys.readouterr().out
    assert judge.main() == 0
    assert len(calls) == 6
    payload = json.loads(next((tmp_path / "out").glob("*both_orders_bootstrap.json")).read_text())
    assert payload["cache_summary"]["hits_this_run"] == 6
    assert payload["api_usage"]["input_tokens"] == 0


def test_failed_order_is_not_a_tie_or_half_point(tmp_path, monkeypatch):
    items = [{"prompt": f"q{i}", "completion": "answer"} for i in range(2)]
    monkeypatch.setattr(judge, "call_judge", lambda *args, **kwargs: {
        "overall_winner": None, "overall_parsing_failed": True, "survey_winner": None, "parse_error": "fixture outage"})
    samples, results, _ = judge.run_bootstrap_evaluation({}, items, {}, items, 2, 20, "judge", "openai", None,
        True, 42, str(tmp_path / "cache.jsonl"), judge_both_orders=True)
    observed = judge._analyze_iteration(results)["overall_analysis"]
    assert observed["excluded"] == 2 and observed["ties"] == 0
    assert observed["model2_mean_score"] is None
    assert judge.analyze_bootstrap_results(samples)["overall_winner_analysis"]["model2_score_distribution"]["ci_95"] == [None, None]
    assert not (tmp_path / "cache.jsonl").exists()


def test_paired_report_plot_uses_order_averaged_score_not_consistent_win_count(tmp_path, monkeypatch):
    plotting = load_module("paired_plot_under_test", "evaluate/plot_win_rates.py")
    paired = combine_order_judgments([order_result("model1", False), order_result("model2", True)])
    observed = judge._analyze_iteration([paired, paired])
    stats = judge.analyze_bootstrap_results([{"analysis": observed}])
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"status": "success", "validation": {"observed": observed},
        "bootstrap_config": {"model1_path": "trained_grpo", "model2_path": "trained_pairwise", "judge_both_orders": True},
        "bootstrap_analysis": stats}))
    data = plotting.load_win_rate_data(path)
    assert data["metric_label"] == "paired-order score"
    assert data["overall_mean_model2"] == 0.5
    assert data["overall_ci_model2"] == [0.5, 0.5]
    output = tmp_path / "comparison.png"
    monkeypatch.setattr(sys, "argv", ["plot", str(path), "--output", str(output)])
    plotting.main()
    assert output.read_bytes().startswith(b"\x89PNG")
