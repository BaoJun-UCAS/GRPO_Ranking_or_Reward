"""Offline saved-judgment reanalysis must preserve evidence and never call APIs."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reparse_under_test", ROOT / "evaluate/reparse_judgments.py")
reparse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reparse)


def survey(winner):
    sections = [
        f"**{i}. {criterion.title()}**\n- Winner: [{winner}]\n- Justification: Supported."
        for i, criterion in enumerate(reparse.CRITERIA, 1)
    ]
    return "\n\n".join(sections) + f"\n\n**Overall Recommendation:**\nWinner: [{winner}]\nJustification: Supported."


@pytest.fixture
def report():
    selected = [9, 2, 17]
    clusters = []
    for source_index, winners in zip(selected, (("A", "A"), ("A", "B"), ("B", "A"))):
        orders = []
        for swapped, winner in zip((False, True), winners):
            raw = survey(winner)
            original = reparse.judge.parse_survey_response(raw, swapped, True)
            # Deliberately saved stale decisions demonstrate actual recomputation.
            original.update(
                overall_winner="model2",
                survey_winner="model2",
                raw_response=raw,
                order_swapped=swapped,
                api_usage={"input_tokens": 123, "output_tokens": 45},
            )
            order = {
                "prompt": f"question {source_index}",
                "model1_completion": "first response",
                "model2_completion": "second response",
                "source_index": source_index,
                "order_swapped": swapped,
                "judgment": original,
                "overall_winner": "model2",
                "overall_parsing_failed": False,
                "survey_winner": "model2",
                "criterion_evaluations": original["criterion_evaluations"],
                "cache_key": f"original-{source_index}-{swapped}",
                "cache_hit": True,
            }
            orders.append(order)
        clusters.append(reparse.judge.combine_order_judgments(orders))
    return {
        "status": "invalid",
        "bootstrap_config": {
            "protocol_version": "survey-v2-strict-sections",
            "parser_version": "2",
            "allow_ties": True,
            "judge_both_orders": True,
            "subsample_size_N": 3,
            "bootstrap_iterations_B": 3,
            "seed": 31415,
            "judge_model": "original-model",
            "thinking_mode": "disabled",
            "cache_path": "/not-to-be-read/or/written/cache.jsonl",
        },
        "validation": {"required_fraction": 1.0, "passed": False},
        "selected_source_indices": selected,
        "source_metadata": {"model1": {"contract": {"frozen_prompts": {"sha256": "original"}}}},
        "judged_results": clusters,
        "bootstrap_results": [
            {"iteration": 42 + i, "sample_positions": positions, "analysis": {"original": i}}
            for i, positions in enumerate(([2, 2, 0], [1, 1, 1], [0, 2, 1]))
        ],
        "bootstrap_analysis": {"old_summary": True},
        "api_usage": {"input_tokens": 738, "output_tokens": 270},
        "cache_summary": {"api_calls_this_run": 6, "hits_this_run": 0},
    }


def test_preserves_original_protocol_decisions_and_raw_provenance(report):
    before = copy.deepcopy(report)
    result = reparse.reparse_report(report, input_path="original.json", input_sha256="filehash")
    assert report == before
    assert result["bootstrap_config"] == before["bootstrap_config"]
    assert result["source_metadata"] == before["source_metadata"]
    assert result["selected_source_indices"] == before["selected_source_indices"]
    assert result["report_type"] == "offline_parser_only_reanalysis"
    assert result["reanalysis"]["input_sha256"] == "filehash"
    assert result["reanalysis"]["original_protocol_version"] == "survey-v2-strict-sections"
    assert result["reanalysis"]["original_parser_version"] == "2"
    assert result["reanalysis"]["applied_parser_version"] == reparse.judge.JUDGE_PARSER_VERSION
    assert result["reanalysis"]["new_prompt_executed"] is False
    assert result["reanalysis"]["current_prompt_protocol_not_executed"] == reparse.judge.JUDGE_PROTOCOL_VERSION
    for field in ("status", "validation", "api_usage", "cache_summary", "bootstrap_analysis"):
        assert result["original_summary"][field] == before[field]
    for old, new in zip(before["judged_results"], result["judged_results"]):
        assert new["reanalysis"]["original_decisions"] == reparse._decisions(old)
        for old_order, new_order in zip(old["order_judgments"], new["order_judgments"]):
            provenance = new_order["reanalysis"]
            raw = old_order["judgment"]["raw_response"]
            assert new_order["judgment"]["raw_response"] == raw
            assert provenance["raw_response_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
            assert provenance["original_decisions"] == reparse._decisions(old_order)
            assert provenance["original_judgment_without_raw_response"] == {
                key: value for key, value in old_order["judgment"].items() if key != "raw_response"
            }
            assert provenance["original_cache_key"] == old_order["cache_key"]
            assert "cache_key" not in new_order
            assert new_order["judgment"]["api_usage"] == reparse.ZERO_USAGE


def test_recombines_exact_orders_and_reuses_saved_bootstrap_draws(report, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline reanalysis must not call APIs, access caches, or resample")

    for name in (
        "call_judge",
        "initialize_judge",
        "_load_judgment_cache",
        "_append_judgment_cache",
        "run_bootstrap_evaluation",
    ):
        monkeypatch.setattr(reparse.judge, name, forbidden, raising=False)
    monkeypatch.setattr(reparse.judge.np.random, "default_rng", forbidden)
    result = reparse.reparse_report(report)
    assert [row["overall_winner"] for row in result["judged_results"]] == ["order_disagreement", "model1", "model2"]
    assert [row["overall_model1_score"] for row in result["judged_results"]] == [0.5, 1, 0]
    assert result["validation"]["passed"] is True
    assert result["api_usage"] == reparse.ZERO_USAGE
    assert result["cache_summary"]["api_calls_this_run"] == 0
    assert result["cache_summary"]["cache_writes_this_run"] == 0
    for original, draw in zip(report["bootstrap_results"], result["bootstrap_results"]):
        assert draw["iteration"] == original["iteration"]
        assert draw["sample_positions"] == original["sample_positions"]
        assert draw["original_analysis"] == original["analysis"]
    scores = [draw["analysis"]["overall_analysis"]["model1_mean_score"] for draw in result["bootstrap_results"]]
    assert scores == pytest.approx([1 / 6, 1, 0.5])
    distribution = result["bootstrap_analysis"]["overall_winner_analysis"]["model1_score_distribution"]
    assert distribution["raw_values"] == scores


def test_reparse_uses_saved_tie_policy_and_swapped_mapping(report, monkeypatch):
    report["bootstrap_config"]["allow_ties"] = False
    report["judged_results"][0]["order_judgments"][0]["judgment"]["raw_response"] = survey("Tie")
    original_parse = reparse.judge.parse_survey_response
    calls = []

    def spy(raw, swapped, allow_ties):
        calls.append((swapped, allow_ties))
        return original_parse(raw, swapped, allow_ties)

    monkeypatch.setattr(reparse.judge, "parse_survey_response", spy)
    result = reparse.reparse_report(report)
    assert calls == [(False, False), (True, False)] * 3
    assert result["judged_results"][0]["overall_parsing_failed"] is True
    assert result["judged_results"][0]["survey_parsing_failed"] is True
    assert result["validation"]["passed"] is False


@pytest.mark.parametrize("missing_raw", [False, True])
def test_api_failure_is_retained_even_with_otherwise_valid_raw_text(report, monkeypatch, missing_raw):
    original = report["judged_results"][0]["order_judgments"][0]["judgment"]
    original["parse_error"] = "original request timed out"
    if missing_raw:
        original.pop("raw_response")
    original_parse = reparse.judge.parse_survey_response
    calls = []

    def spy(*args):
        calls.append(args)
        return original_parse(*args)

    monkeypatch.setattr(reparse.judge, "parse_survey_response", spy)
    result = reparse.reparse_report(report)
    failed = result["judged_results"][0]["order_judgments"][0]
    assert len(calls) == 5
    assert failed["judgment"]["parse_error"] == "original request timed out"
    assert failed["reanalysis"]["action"] == "retained_api_failure"
    assert failed["overall_winner"] is None
    assert failed["survey_winner"] is None
    assert result["cache_summary"]["failed_api_judgments"] == 1
    assert result["cache_summary"]["failed_parse_judgments"] == 0
    assert result["validation"]["overall_valid"] == 2
    assert result["status"] == "invalid"


@pytest.mark.parametrize("required_fraction, expected_pass", [(None, False), (1.0, False), (0.5, True)])
def test_validity_uses_original_fraction_or_strict_default(report, required_fraction, expected_pass):
    report["judged_results"][0]["order_judgments"][0]["judgment"]["raw_response"] = "unparseable"
    if required_fraction is None:
        report["validation"].pop("required_fraction")
    else:
        report["validation"]["required_fraction"] = required_fraction
    result = reparse.reparse_report(report)
    assert result["validation"]["passed"] is expected_pass
    assert result["validation"]["required_fraction"] == (1.0 if required_fraction is None else required_fraction)


def test_single_order_reports_are_reparsed_without_inventing_pairs(report):
    report["bootstrap_config"]["judge_both_orders"] = False
    report["judged_results"] = [cluster["order_judgments"][0] for cluster in report["judged_results"]]
    result = reparse.reparse_report(report)
    assert [row["overall_winner"] for row in result["judged_results"]] == ["model1", "model1", "model2"]
    assert all("order_judgments" not in row for row in result["judged_results"])
    assert result["cache_summary"]["logical_judgments"] == 3


@pytest.mark.parametrize("positions", [None, [], [0, 1], [0, 1, 3], [-1, 0, 1], [0, True, 1], [0, 1.0, 2]])
def test_missing_or_invalid_bootstrap_positions_are_not_reselected(report, positions):
    report["bootstrap_results"][0]["sample_positions"] = positions
    with pytest.raises(ValueError, match="sample_positions"):
        reparse.reparse_report(report)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r["bootstrap_config"].update(allow_ties="true"), "allow_ties"),
        (lambda r: r["bootstrap_config"].update(subsample_size_N=4), "subsample_size_N"),
        (lambda r: r["bootstrap_config"].update(bootstrap_iterations_B=4), "bootstrap_iterations_B"),
        (lambda r: r["selected_source_indices"].reverse(), "source_index"),
        (lambda r: r["selected_source_indices"].__setitem__(1, 9), "selected_source_indices"),
        (lambda r: r["judged_results"].__setitem__(0, None), "judged_results"),
        (lambda r: r["judged_results"][0].update(source_index=True), "source_index"),
        (lambda r: r["judged_results"][0]["order_judgments"][0].update(source_index=True), "source_index"),
        (lambda r: r["judged_results"][0]["order_judgments"][0].update(order_swapped="false"), "order_swapped"),
        (lambda r: r["judged_results"][0]["order_judgments"][0]["judgment"].update(order_swapped=True), "conflicting"),
        (lambda r: r["judged_results"][0]["order_judgments"][0].update(prompt="other question"), "prompt"),
        (lambda r: r["judged_results"][0]["order_judgments"][0]["judgment"].pop("raw_response"), "raw_response"),
        (lambda r: r["validation"].update(required_fraction=0), "required_fraction"),
        (lambda r: r["validation"].update(required_fraction=float("nan")), "required_fraction"),
    ],
)
def test_ambiguous_original_provenance_is_rejected(report, mutate, message):
    mutate(report)
    with pytest.raises(ValueError, match=message):
        reparse.reparse_report(report)


def test_file_output_hashes_original_bytes_and_leaves_source_and_cache_unchanged(report, tmp_path):
    source, output, cache = (tmp_path / name for name in ("original.json", "reparsed.json", "cache.jsonl"))
    cache.write_text("sentinel cache\n")
    report["bootstrap_config"]["cache_path"] = str(cache)
    original_bytes = json.dumps(report, indent=3, ensure_ascii=False).encode("utf-8")
    source.write_bytes(original_bytes)
    result = reparse.reparse_file(source, output)
    assert source.read_bytes() == original_bytes
    assert cache.read_text() == "sentinel cache\n"
    assert result["reanalysis"]["input_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert result["reanalysis"]["input_path"] == str(source.resolve())
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize("kind", ["same", "existing", "symlink", "dangling_symlink", "hardlink", "directory"])
def test_refuses_every_existing_output_including_symlinks(report, tmp_path, kind):
    source, destination = tmp_path / "original.json", tmp_path / "output.json"
    source.write_text(json.dumps(report))
    before = source.read_bytes()
    if kind == "same":
        destination = source
    elif kind == "existing":
        destination.write_text("do not overwrite")
    elif kind == "symlink":
        destination.symlink_to(source)
    elif kind == "dangling_symlink":
        destination.symlink_to(tmp_path / "nonexistent")
    elif kind == "hardlink":
        destination.hardlink_to(source)
    else:
        destination.mkdir()
    with pytest.raises(FileExistsError, match="new file"):
        reparse.reparse_file(source, destination)
    assert source.read_bytes() == before
    if kind == "existing":
        assert destination.read_text() == "do not overwrite"


def test_exclusive_creation_rejects_destination_created_after_preflight(report, tmp_path, monkeypatch):
    source, destination = tmp_path / "original.json", tmp_path / "output.json"
    source.write_text(json.dumps(report))
    before = source.read_bytes()
    real_open = Path.open

    def race(path, mode="r", *args, **kwargs):
        if path == destination and mode == "x":
            destination.symlink_to(source)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", race)
    with pytest.raises(FileExistsError):
        reparse.reparse_file(source, destination)
    assert source.read_bytes() == before


def test_cli_saves_invalid_evidence_with_nonzero_exit_and_labels_no_new_prompt(report, tmp_path, capsys):
    source, destination = tmp_path / "original.json", tmp_path / "invalid-reparse.json"
    report["judged_results"][0]["order_judgments"][0]["judgment"]["raw_response"] = "invalid"
    source.write_text(json.dumps(report))
    assert reparse.main(["--input", str(source), "--output", str(destination)]) == 1
    result = json.loads(destination.read_text())
    assert result["status"] == "invalid"
    output = capsys.readouterr().out
    assert "parser-only" in output and "new prompt NOT executed" in output and "API calls: 0" in output
    assert reparse.main(["--input", str(source), "--output", str(destination)]) == 2


def test_invalid_input_never_creates_an_output(tmp_path):
    source, destination = tmp_path / "broken.json", tmp_path / "output.json"
    source.write_text('{"bootstrap_config": {}}')
    assert reparse.main(["--input", str(source), "--output", str(destination)]) == 2
    assert not destination.exists()
