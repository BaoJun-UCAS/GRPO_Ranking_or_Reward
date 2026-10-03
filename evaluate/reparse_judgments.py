#!/usr/bin/env python3
"""Reparse saved judge text locally, without issuing requests or touching caches.

This is parser-only reanalysis of the *original* judging protocol. The current
judge prompt is not executed. Original decisions and provenance are retained,
and confidence intervals reuse the report's exact saved bootstrap draws.

Usage: python evaluate/reparse_judgments.py --input report.json --output NEW.json
Exit codes: 0 = validity passed, 1 = evidence saved but invalid, 2 = input/output error.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path


# Works both as a standalone script and when loaded by the CPU-only tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap_judge as judge


CRITERIA = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}


def _decisions(record):
    return copy.deepcopy(
        {
            key: value
            for key, value in record.items()
            if key.startswith(("overall_", "survey_")) or key in ("criterion_evaluations", "parse_error")
        }
    )


def _reparse_order(order, allow_ties, location):
    if not isinstance(order, dict) or not isinstance(order.get("judgment"), dict):
        raise ValueError(f"{location}: missing saved judgment object")
    saved = order["judgment"]
    swapped = order.get("order_swapped")
    if type(swapped) is not bool:
        raise ValueError(f"{location}: order_swapped must be a saved boolean")
    if "order_swapped" in saved and (type(saved["order_swapped"]) is not bool or saved["order_swapped"] != swapped):
        raise ValueError(f"{location}: conflicting saved order_swapped values")
    api_failed = "parse_error" in saved or "parse_error" in order
    raw = saved.get("raw_response", "" if api_failed else None)
    if not isinstance(raw, str):
        raise ValueError(f"{location}: missing or non-text raw_response")

    if api_failed:
        # A failed API call is not recoverable by interpreting any stale text.
        parsed = {
            "criterion_evaluations": {
                criterion: {"winner": None, "parsing_failed": True, "justification": "Original API failure retained"}
                for criterion in CRITERIA
            },
            "overall_winner": None,
            "overall_parsing_failed": True,
            "overall_justification": "Original API failure retained",
            "survey_winner": None,
            "survey_calculation": {
                "model1_wins": 0,
                "model2_wins": 0,
                "tie_count": 0,
                "model1_score": 0.0,
                "model2_score": 0.0,
                "successful_criteria_count": 0,
            },
            "parse_error": copy.deepcopy(saved.get("parse_error", order.get("parse_error"))),
        }
    else:
        parsed = judge.parse_survey_response(raw, swapped, allow_ties)
    parsed["survey_parsing_failed"] = parsed.get("survey_winner") is None
    parsed.update(raw_response=raw, order_swapped=swapped, api_usage=ZERO_USAGE.copy())

    result = copy.deepcopy(order)
    result.update(_decisions(parsed))
    # Single-order scores, if present in an older report, must not stay stale.
    for endpoint in ("overall", "survey"):
        score = {"model1": 1.0, "model2": 0.0, "tie": 0.5}.get(parsed.get(f"{endpoint}_winner"))
        result[f"{endpoint}_model1_score"] = score
        result[f"{endpoint}_model2_score"] = None if score is None else 1 - score
    result["judgment"] = parsed
    result["reanalysis"] = {
        "source_location": location,
        "action": "retained_api_failure" if api_failed else "reparsed_saved_text",
        "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_response_encoding": "utf-8",
        "original_decisions": _decisions(order),
        # Raw text is retained without modification in judgment.raw_response.
        "original_judgment_without_raw_response": copy.deepcopy(
            {key: value for key, value in saved.items() if key != "raw_response"}
        ),
        "original_cache_key": order.get("cache_key"),
        "original_cache_hit": order.get("cache_hit"),
    }
    # Never identify the new parse with an old protocol/parser cache entry.
    result.pop("cache_key", None)
    result["cache_hit"] = False
    return result


def reparse_report(report, *, input_path=None, input_sha256=None):
    """Return a separate parser-only report; never mutate the input or do I/O."""
    if not isinstance(report, dict) or not isinstance(report.get("bootstrap_config"), dict):
        raise ValueError("Report must contain bootstrap_config")
    config = report["bootstrap_config"]
    if type(config.get("allow_ties")) is not bool:
        raise ValueError("bootstrap_config.allow_ties must be a saved boolean")
    clusters = report.get("judged_results")
    draws = report.get("bootstrap_results")
    if not isinstance(clusters, list) or not clusters or any(not isinstance(row, dict) for row in clusters):
        raise ValueError("Report must contain nonempty judged_results")
    if not isinstance(draws, list) or not draws:
        raise ValueError("Report must contain nonempty bootstrap_results with sample_positions")
    n = len(clusters)
    for key, actual in (("subsample_size_N", n), ("bootstrap_iterations_B", len(draws))):
        if key in config and (type(config[key]) is not int or config[key] != actual):
            raise ValueError(f"bootstrap_config.{key} disagrees with saved report")
    selected = report.get("selected_source_indices")
    if (
        not isinstance(selected, list)
        or len(selected) != n
        or any(type(index) is not int or index < 0 for index in selected)
        or len(set(selected)) != n
    ):
        raise ValueError("selected_source_indices must identify each saved cluster exactly once")
    both_orders = config.get("judge_both_orders", "order_judgments" in clusters[0])
    if type(both_orders) is not bool:
        raise ValueError("bootstrap_config.judge_both_orders must be a boolean")

    validation = report.get("validation") or {}
    if not isinstance(validation, dict):
        raise ValueError("validation must be an object")
    required_fraction = validation.get("required_fraction", 1.0)
    if (
        type(required_fraction) not in (int, float)
        or not math.isfinite(required_fraction)
        or not 0 < required_fraction <= 1
    ):
        raise ValueError("validation.required_fraction must be in (0, 1]")

    reparsed = []
    for position, cluster in enumerate(clusters):
        location = f"judged_results[{position}]"
        if type(cluster.get("source_index")) is not int or cluster["source_index"] != selected[position]:
            raise ValueError(f"{location}: source_index disagrees with selected_source_indices")
        if both_orders:
            orders = cluster.get("order_judgments")
            if not isinstance(orders, list) or len(orders) != 2:
                raise ValueError(f"{location}: expected both saved order_judgments")
            for order in orders:
                if not isinstance(order, dict):
                    raise ValueError(f"{location}: order judgment must be an object")
                if type(order.get("source_index")) is not int:
                    raise ValueError(f"{location}: order source_index must be an integer")
                for key in ("source_index", "prompt", "model1_completion", "model2_completion"):
                    if key not in cluster or key not in order or order[key] != cluster[key]:
                        raise ValueError(f"{location}: order {key} differs from its prompt cluster")
            new_orders = [
                _reparse_order(order, config["allow_ties"], f"{location}.order_judgments[{i}]")
                for i, order in enumerate(orders)
            ]
            updated = copy.deepcopy(cluster)
            updated.update(judge.combine_order_judgments(new_orders))
            updated["reanalysis"] = {"original_decisions": _decisions(cluster)}
        else:
            if "order_judgments" in cluster:
                raise ValueError(f"{location}: paired judgments conflict with single-order configuration")
            updated = _reparse_order(cluster, config["allow_ties"], location)
        reparsed.append(updated)

    bootstrap_results = []
    for index, draw in enumerate(draws):
        positions = draw.get("sample_positions") if isinstance(draw, dict) else None
        if (
            not isinstance(positions, list)
            or len(positions) != n
            or any(type(position) is not int or not 0 <= position < n for position in positions)
        ):
            raise ValueError(f"bootstrap_results[{index}]: invalid or missing saved sample_positions")
        updated = copy.deepcopy(draw)
        updated["original_analysis"] = copy.deepcopy(draw.get("analysis"))
        updated["analysis"] = judge._analyze_iteration([reparsed[position] for position in positions])
        bootstrap_results.append(updated)

    observed = judge._analyze_iteration(reparsed)
    new_validation = {
        "required_fraction": required_fraction,
        "overall_valid": observed["overall_analysis"]["valid_comparisons"],
        "survey_valid": observed["survey_analysis"]["valid_comparisons"],
        "total": n,
        "observed": observed,
    }
    new_validation["passed"] = all(
        new_validation[key] >= max(2, required_fraction * n) for key in ("overall_valid", "survey_valid")
    )
    raw_orders = [order for cluster in reparsed for order in cluster.get("order_judgments", [cluster])]
    result = copy.deepcopy(report)
    result.update(
        report_type="offline_parser_only_reanalysis",
        status="success" if new_validation["passed"] else "invalid",
        validation=new_validation,
        judged_results=reparsed,
        bootstrap_results=bootstrap_results,
        bootstrap_analysis=judge.analyze_bootstrap_results(bootstrap_results),
        api_usage=ZERO_USAGE.copy(),
        cache_summary={
            "hits_this_run": 0,
            "api_calls_this_run": 0,
            "cache_writes_this_run": 0,
            "logical_judgments": len(raw_orders),
            "failed_api_judgments": sum("parse_error" in order["judgment"] for order in raw_orders),
            "failed_parse_judgments": sum(
                "parse_error" not in order["judgment"]
                and (order["overall_parsing_failed"] or order["survey_winner"] is None)
                for order in raw_orders
            ),
        },
        original_summary={
            key: copy.deepcopy(report.get(key))
            for key in ("status", "validation", "api_usage", "cache_summary", "bootstrap_analysis")
        },
        reanalysis={
            "mode": "offline_parser_only",
            "description": "Saved raw judgments reparsed locally; the new judge prompt was not executed.",
            "input_path": str(input_path) if input_path is not None else None,
            "input_sha256": input_sha256,
            "original_protocol_version": config.get("protocol_version"),
            "original_parser_version": config.get("parser_version"),
            "applied_parser_version": judge.JUDGE_PARSER_VERSION,
            "current_prompt_protocol_not_executed": judge.JUDGE_PROTOCOL_VERSION,
            "new_prompt_executed": False,
            "bootstrap_draws": "original saved sample_positions, unchanged",
            "api_calls": 0,
            "cache_reads": 0,
            "cache_writes": 0,
        },
    )
    return result


def reparse_file(input_path, output_path):
    """Read one report and exclusively create a new file, including symlink checks."""
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve() or os.path.lexists(destination):
        raise FileExistsError("Output must be a new file, different from the input; refusing to overwrite")
    raw = source.read_bytes()
    report = json.loads(raw)
    result = reparse_report(report, input_path=source.resolve(), input_sha256=hashlib.sha256(raw).hexdigest())
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Exclusive creation also protects against a destination created after the
    # preflight check, including a symlink pointing at the input or any cache.
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing bootstrap judge report (read only)")
    parser.add_argument("--output", required=True, help="New output file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        result = reparse_file(args.input, args.output)
    except (OSError, ValueError) as exc:
        print(f"Reparse refused: {exc}", file=sys.stderr)
        return 2
    validation = result["validation"]
    print(
        f"Offline parser-only reanalysis saved to {args.output}; new prompt NOT executed; "
        f"API calls: 0; validity: {result['status']} "
        f"(overall {validation['overall_valid']}/{validation['total']}, "
        f"survey {validation['survey_valid']}/{validation['total']})."
    )
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
