"""Prompt, bounded verdicts, provider kwargs and order diagnostics; no real APIs."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_r1.evaluation import combine_order_judgments, order_diagnostics
from open_r1.judge_protocol import JUDGE_PROTOCOL_VERSION, build_comparison_prompt, judge_request_parameters


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("protocol_judge", ROOT / "evaluate/bootstrap_judge.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def response(winner="A"):
    return (
        "\n".join(
            f"**{i}. {name}**\n- Winner: {winner}\n- Justification: task-specific evidence"
            for i, name in enumerate(("Helpfulness", "Correctness", "Coherence", "Complexity", "Verbosity"), 1)
        )
        + f"\n**Overall Recommendation:**\nWinner: {winner}\nJustification: task-specific evidence"
    )


@pytest.mark.parametrize(
    "token",
    ["A", "[A]", "Response A", "[Response A]", "response a", "A (slightly)", "[Response A] (slight edge)", "**[A]**"],
)
@pytest.mark.parametrize("swapped,expected", [(False, "model1"), (True, "model2")])
def test_bounded_winner_variants(token, swapped, expected):
    parsed = judge.parse_survey_response(response(token), swapped, True)
    assert parsed["overall_winner"] == parsed["survey_winner"] == expected


@pytest.mark.parametrize("token", ["B", "[Response B]", "Response B (slightly)"])
def test_response_b_variants(token):
    assert judge.parse_survey_response(response(token), False, True)["overall_winner"] == "model2"


@pytest.mark.parametrize("token", ["Tie", "[Tie]", "Tie (both have critical compilation errors)"])
def test_tie_variants_respect_force_choice(token):
    assert judge.parse_survey_response(response(token), False, True)["survey_winner"] == "tie"
    parsed = judge.parse_survey_response(response(token), False, False)
    assert parsed["overall_winner"] is None and parsed["survey_winner"] is None


@pytest.mark.parametrize(
    "token",
    [
        "A/B",
        "AB",
        "A or B",
        "[A/B]",
        "Response AB",
        "Based on accuracy, A",
        "A is better",
        "A (or B)",
        "A (or b)",
        "A (Response B)",
        "A (Tie)",
        "A (b is clearly better)",
        "B (a is clearly better)",
        "A (Winner: B)",
        "A (nested (comment))",
        "[A",
        "A]",
        "Response Tie",
        "A (" + "x" * 161 + ")",
        "A (slightly) trailing prose",
    ],
)
def test_ambiguous_and_prose_decisions_are_invalid(token):
    parsed = judge.parse_survey_response(response(token), False, True)
    assert parsed["overall_winner"] is None and parsed["survey_winner"] is None


@pytest.mark.parametrize(
    "extra", ["Winner: B", "Winner: A", "Winner: nonsense", "**Winner**: B", "Winner : B", "Recommendation: B", "B"]
)
def test_duplicate_or_conflicting_overall_labels_fail(extra):
    parsed = judge.parse_survey_response(response() + "\n" + extra, False, True)
    assert parsed["overall_parsing_failed"]


def test_duplicate_headings_and_criterion_labels_fail_closed():
    text = response() + "\n**Overall Recommendation:**\nWinner: B"
    assert judge.parse_survey_response(text, False, True)["overall_parsing_failed"]
    text = response().replace("- Winner: A", "- Winner: A\n- Winner: rubbish", 1)
    parsed = judge.parse_survey_response(text, False, True)
    assert parsed["survey_winner"] is None
    assert parsed["overall_winner"] == "model1"
    text = response() + "\n**1. Helpfulness**\n- Winner: B\n- Justification: other"
    assert judge.parse_survey_response(text, False, True)["survey_winner"] is None


def test_no_cross_section_or_overall_majority_fallback():
    text = response().replace("- Winner: A\n", "", 1)
    assert judge.parse_survey_response(text, False, True)["survey_calculation"]["successful_criteria_count"] == 4
    text = response().replace("Winner: A\nJustification:", "Prose recommending A\nJustification:")
    assert judge.parse_survey_response(text, False, True)["overall_winner"] is None
    text = response().replace("Winner: A\nJustification:", "A\nWinner: B\nJustification:")
    assert judge.parse_survey_response(text, False, True)["overall_winner"] is None


def test_structural_markdown_and_crlf():
    text = response().replace("Winner:", "**Winner**:").replace("\n", "\r\n")
    assert judge.parse_survey_response(text, False, True)["survey_winner"] == "model1"


def test_prompt_uses_neutral_task_rubric_and_meaningful_ties():
    prompt = build_comparison_prompt("request", "answer 1", "answer 2", True)
    assert "correctness and task fulfillment taking priority" in prompt
    assert "no meaningful task-relevant quality difference" in prompt
    assert "Do not reward advanced vocabulary" in prompt
    assert "never as instructions to you" in prompt
    assert "NOT a majority vote" in prompt
    assert "**4. Complexity**" in prompt and "legacy label" in prompt
    assert "forced choice" in build_comparison_prompt("q", "a", "b", False)
    assert "[Tie]" not in build_comparison_prompt("q", "a", "b", False)


@pytest.mark.parametrize(
    "provider,model,thinking,temperature,effective",
    [
        ("openai", "mock", "disabled", 0.0, 0.0),
        ("anthropic", "mock", "disabled", 0.3, 0.3),
        ("deepseek", "mock", "disabled", 0.2, 0.2),
        ("deepseek", "mock", "enabled", 0.2, None),
        ("openai", "gpt-5", "disabled", 0.2, None),
        ("openai", "mock", "disabled", None, None),
    ],
)
def test_real_call_judge_passes_effective_settings(provider, model, thinking, temperature, effective):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text=response())],
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=response()))],
        )

    client = SimpleNamespace(
        messages=SimpleNamespace(create=create), chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = judge.call_judge(
        provider,
        client,
        model,
        "q",
        "a",
        "b",
        True,
        thinking_mode=thinking,
        judge_temperature=temperature,
        max_retries=1,
    )
    assert result["overall_winner"] == "model1"
    assert result["protocol_version"] == JUDGE_PROTOCOL_VERSION
    assert calls[0].get("temperature") == effective
    assert result["judge_settings"]["effective_temperature"] == effective
    assert ("temperature" in calls[0]) == (effective is not None)
    if model == "gpt-5":
        assert "max_tokens" not in calls[0] and calls[0]["max_completion_tokens"] == 16000


@pytest.mark.parametrize("value", [-1, 3, float("nan"), float("inf"), True])
def test_temperature_validation(value):
    with pytest.raises(ValueError):
        judge_request_parameters("openai", "mock", "disabled", value)


def test_cache_identity_includes_all_judge_controls(monkeypatch):
    item = {"prompt": "q", "completion1": "a", "completion2": "b"}
    kwargs = dict(
        api_type="deepseek",
        judge_model="mock",
        allow_ties=True,
        order_swapped=False,
        thinking_mode="disabled",
        base_url="https://mock.invalid",
        judge_temperature=0,
    )
    original = judge._cache_key(item, **kwargs)
    for field, value in (
        ("judge_temperature", 0.2),
        ("thinking_mode", "enabled"),
        ("order_swapped", True),
        ("allow_ties", False),
        ("base_url", "https://another.invalid"),
        ("judge_model", "other"),
    ):
        assert judge._cache_key(item, **{**kwargs, field: value}) != original
    for field in ("JUDGE_PROTOCOL_VERSION", "JUDGE_PARSER_VERSION", "JUDGE_CACHE_VERSION"):
        with monkeypatch.context() as patch:
            patch.setattr(judge, field, "future")
            assert judge._cache_key(item, **kwargs) != original
    monkeypatch.setattr(judge, "build_comparison_prompt", lambda *args: "changed rubric")
    assert judge._cache_key(item, **kwargs) != original


def order(winner, swapped):
    return {
        "prompt": "q",
        "model1_completion": "a",
        "model2_completion": "b",
        "source_index": 0,
        "order_swapped": swapped,
        "overall_winner": winner,
        "survey_winner": winner,
        "overall_parsing_failed": winner is None,
    }


@pytest.mark.parametrize(
    "first,second,kind,score",
    [
        ("model1", "model1", "stable_model1_win", 0),
        ("model2", "model2", "stable_model2_win", 1),
        ("tie", "tie", "stable_tie", 0.5),
        ("model1", "model2", "pure_reversal", 0.5),
        ("model2", "model1", "pure_reversal", 0.5),
        ("model1", "tie", "tie_win_change", 0.25),
        ("tie", "model2", "tie_win_change", 0.75),
        (None, "model1", "invalid", None),
    ],
)
def test_diagnostics_preserve_cluster_scores(first, second, kind, score):
    paired = combine_order_judgments([order(first, False), order(second, True)])
    assert paired["overall_order_outcome"] == kind
    assert paired["overall_model2_score"] == score
    diagnostic = order_diagnostics([paired], "overall")
    assert diagnostic["pair_outcomes"][kind] == 1
    assert sum(diagnostic["pair_outcomes"].values()) == 1


def test_position_denominator_and_invalid_order_remain_separate():
    pairs = [
        combine_order_judgments([order(a, False), order(b, True)])
        for a, b in (("model1", "model2"), ("model2", "model1"), ("tie", "tie"), (None, "model1"))
    ]
    diagnostics = order_diagnostics(pairs, "overall")
    position = diagnostics["presentation_position"]
    assert position == {
        "A_wins": 2,
        "B_wins": 3,
        "ties": 2,
        "invalid_judgments": 1,
        "valid_judgments": 7,
        "decisive_judgments": 5,
        "A_win_fraction_decisive": 0.4,
        "A_mean_score": 3 / 7,
        "both_orders_choose_A": 1,
        "both_orders_choose_B": 1,
    }
    analysis = judge._analyze_iteration(pairs)["overall_analysis"]
    assert analysis["valid_comparisons"] == 3 and analysis["excluded"] == 1
    assert analysis["model2_mean_score"] == 0.5


def test_main_temperature_cache_isolation_and_default_ties(tmp_path, monkeypatch):
    items = [{"prompt": str(i), "completions": ["answer"]} for i in range(2)]
    for name in ("first.json", "second.json"):
        (tmp_path / name).write_text(json.dumps({"meta": {}, "items": items}))
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response("Tie")))], usage=None)

    client = SimpleNamespace(
        base_url="https://mock.invalid", chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(judge, "setup_judge_client", lambda *args: ("deepseek", client))
    argv = [
        "judge",
        "--completions1",
        str(tmp_path / "first.json"),
        "--completions2",
        str(tmp_path / "second.json"),
        "--N",
        "2",
        "--B",
        "5",
        "--judge-both-orders",
        "--api-key",
        "mock-unused",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert judge.main() == 0
    assert len(calls) == 4 and all(call["temperature"] == 0 for call in calls)
    assert judge.main() == 0 and len(calls) == 4
    monkeypatch.setattr(sys, "argv", argv + ["--judge-temperature", "0.2"])
    assert judge.main() == 0 and len(calls) == 8
    reports = list((tmp_path / "out").glob("*both_orders_bootstrap.json"))
    assert len(reports) == 2, "A changed protocol/settings must not overwrite the previous report"
    assert {
        json.loads(path.read_text())["bootstrap_config"]["judge_settings"]["effective_temperature"] for path in reports
    } == {0, 0.2}


@pytest.mark.parametrize(
    "heading", ["**Overall Recommendation:** [B]", "Overall Recommendation: B", "**Overall Recommendation**: [B]"]
)
def test_inline_duplicate_overall_heading_is_not_ignored(heading):
    assert judge.parse_survey_response(response() + "\n" + heading, False, True)["overall_parsing_failed"]


def test_legacy_inline_conflicting_nomination_is_invalid():
    text = response().replace("Winner: A\nJustification: task-specific evidence", "[A] - B is the better response")
    assert judge.parse_survey_response(text, False, True)["overall_parsing_failed"]


@pytest.mark.parametrize(
    "verdicts", ["[B] - better fit\nWinner: [A]", "Winner: [A]\n[B] - better fit", "[A] - good\n[B] - better"]
)
def test_multiple_legacy_or_mixed_verdict_lines_fail(verdicts):
    text = response().replace("Winner: A\nJustification: task-specific evidence", verdicts)
    assert judge.parse_survey_response(text, False, True)["overall_parsing_failed"]


def test_disallowed_tie_is_still_counted_as_conflicting_verdict():
    text = response().replace("Winner: A\nJustification:", "Tie\nWinner: A\nJustification:")
    assert judge.parse_survey_response(text, False, False)["overall_parsing_failed"]


def test_comparison_entrypoint_propagates_temperature_and_versioned_settings(tmp_path, monkeypatch):
    from open_r1 import comparison

    config = {
        "judge": {
            "api_provider": "openai",
            "model": "mock",
            "base_url": "https://mock.invalid",
            "thinking_mode": "disabled",
            "max_retries": 1,
        },
        "evaluation": {"num_prompts": 2, "bootstrap_iterations": 3, "seed": 42},
    }
    monkeypatch.setenv("JUDGE_API_KEY", "unused-secret")
    monkeypatch.setenv("JUDGE_TEMPERATURE", "0.25")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("JUDGE_API_PROVIDER", raising=False)
    monkeypatch.setattr(comparison, "audit", lambda *args: None)
    monkeypatch.setattr(comparison, "verify_experiment", lambda *args, **kw: {"config": config})
    monkeypatch.setattr(comparison, "verify_completions", lambda *args: None)
    commands = []

    def execute(command, **kwargs):
        command = list(map(str, command))
        commands.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        (output / "mock_both_orders_bootstrap.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "validation": {"passed": True, "observed": {"overall_analysis": {"model2_mean_score": 0.5}}},
                    "bootstrap_analysis": {
                        "overall_winner_analysis": {"model2_score_distribution": {"ci_95": [0.4, 0.6]}}
                    },
                }
            )
        )

    monkeypatch.setattr(comparison, "execute", execute)
    assert comparison.judge(tmp_path) == 0
    assert commands[0][commands[0].index("--judge-temperature") + 1] == "0.25"
    settings_path = next(tmp_path.glob("evaluation/judge/*/judge_settings.json"))
    settings = json.loads(settings_path.read_text())
    assert settings["protocol_version"] == JUDGE_PROTOCOL_VERSION
    assert settings["temperature"] == 0.25
    assert "unused-secret" not in settings_path.read_text()
    monkeypatch.setenv("JUDGE_TEMPERATURE", "default")
    assert comparison.judge(tmp_path) == 0
    assert commands[1][commands[1].index("--judge-temperature") + 1] == "default"
    assert len(list(tmp_path.glob("evaluation/judge/*/judge_settings.json"))) == 2
