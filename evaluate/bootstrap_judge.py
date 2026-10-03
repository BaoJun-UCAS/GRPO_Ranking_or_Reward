#!/usr/bin/env python3
"""
Bootstrap-based LLM-as-judge evaluation with percentile confidence intervals.

This script judges N response pairs once, or twice in opposite presentation orders
with --judge-both-orders. Confidence intervals resample N prompt clusters locally,
keeping API calls at N or 2N before cache/retries, independent of bootstrap count B.

The output contains all low-level details from each bootstrap iteration plus statistical
analysis including confidence intervals for win rates.
"""

import os
import json
import argparse
import hashlib
import re
import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_r1.evaluation import combine_order_judgments, order_diagnostics
from open_r1.judge_protocol import (
    DEFAULT_JUDGE_TEMPERATURE, JUDGE_CACHE_VERSION, JUDGE_PARSER_VERSION, JUDGE_PROTOCOL_VERSION,
    build_comparison_prompt, judge_request_parameters, judge_settings_metadata,
)
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm


# NumPy for statistical calculations
try:
    import numpy as np
except ImportError:
    print("Warning: numpy package not found. Please install it with: pip install numpy")
    np = None

# OpenAI / Anthropic clients (mirroring existing judge implementations)
try:
    import openai
except ImportError:
    print("Warning: openai package not found. Please install it with: pip install openai")
    openai = None

try:
    import anthropic
except ImportError:
    print("Warning: anthropic package not found. Please install it with: pip install anthropic")
    anthropic = None


def read_single_completions_artifact(path: str, completion_index: int = 0) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Read a single model's completions artifact (prompt + completions/responses pairs).

    Args:
        path: Path to the completions file (JSON or JSONL)
        completion_index: Which completion to extract from multi-completion files (default: 0 = first)

    Returns:
        Tuple of (metadata dict, items list with 'completion' key)
    """
    if completion_index < 0:
        raise ValueError("completion_index must be nonnegative")
    # Handle both JSON and JSONL formats
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            # JSONL format: one JSON object per line
            lines = f.readlines()
            items = [json.loads(line) for line in lines if line.strip()]
            meta = {}  # JSONL typically doesn't have metadata
        else:
            # Standard JSON format
            data = json.load(f)
            meta = data.get("meta", {})
            items = data.get("items", [])

    if meta.get("artifact_version") is not None:
        actual_hash = hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if meta.get("items_sha256") != actual_hash:
            raise ValueError(f"Completions artifact integrity check failed: {path}")

    # Process each item to extract the requested completion
    for idx, it in enumerate(items):
        if not isinstance(it.get("prompt"), str):
            raise ValueError(f"Item {idx} missing or invalid 'prompt' field")

        # Handle multiple possible formats
        completion_extracted = False

        # Format 1: "responses" key (from n30 JSONL files)
        if "responses" in it and isinstance(it["responses"], list):
            if len(it["responses"]) <= completion_index:
                raise IndexError(f"Item {idx}: Requested completion_index={completion_index} but only {len(it['responses'])} responses available")
            it["completion"] = it["responses"][completion_index]
            completion_extracted = True

        # Format 2: "completions" key (from generate_completions.py)
        elif "completions" in it and isinstance(it["completions"], list):
            if len(it["completions"]) <= completion_index:
                raise IndexError(f"Item {idx}: Requested completion_index={completion_index} but only {len(it['completions'])} completions available")
            it["completion"] = it["completions"][completion_index]
            completion_extracted = True

        # Format 3: "completion" key (legacy single completion)
        elif "completion" in it and isinstance(it["completion"], str):
            if completion_index != 0:
                raise ValueError(f"Item {idx}: Requested completion_index={completion_index} but file only has single completion")
            # Already in correct format
            completion_extracted = True

        if not completion_extracted:
            raise ValueError(f"Item {idx} missing completion data (expected 'completion', 'completions', or 'responses' key)")
        if not isinstance(it["completion"], str):
            raise ValueError(f"Item {idx}: completion must be text")

    return meta, items


def merge_completions_artifacts(meta1: Dict[str, Any], items1: List[Dict[str, str]],
                               meta2: Dict[str, Any], items2: List[Dict[str, str]]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Merge two single completions artifacts into paired format, validating prompts match."""
    if len(items1) != len(items2):
        raise ValueError(f"Mismatch in number of items: {len(items1)} vs {len(items2)}")
    contract1, contract2 = meta1.get("contract"), meta2.get("contract")
    if bool(contract1) != bool(contract2):
        raise ValueError("Cannot compare a versioned artifact with an unverified legacy artifact")
    if contract1 and contract2:
        for field in ("version", "dataset", "prompts_sha256", "system_prompt", "enable_thinking",
                      "max_prompt_length", "max_new_tokens", "temperature", "top_p", "seed",
                      "n_completions", "backend", "tokens_sha256", "frozen_prompts", "versions"):
            if contract1.get(field) != contract2.get(field):
                raise ValueError(f"Evaluation contract mismatch: {field}")

    # Validate that prompts match exactly
    for idx, (item1, item2) in enumerate(zip(items1, items2)):
        if item1["prompt"] != item2["prompt"]:
            raise ValueError(f"Prompt mismatch at index {idx}:\n"
                           f"File 1: {item1['prompt'][:100]}...\n"
                           f"File 2: {item2['prompt'][:100]}...")

    # Merge into paired format
    merged_items = []
    for item1, item2 in zip(items1, items2):
        merged_item = {
            "prompt": item1["prompt"],  # Same for both
            "completion1": item1["completion"],
            "completion2": item2["completion"]
        }
        merged_items.append(merged_item)

    # Merge metadata, preferring non-None values
    merged_meta = {}
    for key in set(meta1.keys()) | set(meta2.keys()):
        if key in meta1 and key in meta2:
            if meta1[key] != meta2[key]:
                print(f"Warning: Metadata key '{key}' differs between files. Using value from first file.")
            merged_meta[key] = meta1[key]
        elif key in meta1:
            merged_meta[key] = meta1[key]
        else:
            merged_meta[key] = meta2[key]

    # Add information about the two source files
    merged_meta["model1_path"] = meta1.get("model_path", "model1")
    merged_meta["model2_path"] = meta2.get("model_path", "model2")
    merged_meta["merged_from_two_files"] = True

    return merged_meta, merged_items


def clean_judge_name(judge_model: str) -> str:
    return judge_model.replace("-", "").replace(".", "")


def extract_dataset_from_filename(filename: str) -> str:
    """Extract dataset type from filename (if, chat, or tldr). Returns None if not found."""
    filename_lower = filename.lower()
    if "if" in filename_lower:
        return "if"
    elif "chat" in filename_lower:
        return "chat"
    elif "tldr" in filename_lower:
        return "tldr"
    return None


def extract_type_from_filename(filename: str) -> str:
    """Extract model type from filename (ranking, regular, or baseline)."""
    filename_lower = filename.lower()
    if "ranking" in filename_lower:
        return "ranking"
    elif "regular" in filename_lower:
        return "regular"
    else:
        return "baseline"


def extract_temperature_from_filename(filename: str) -> str:
    """Extract temperature value from filename (e.g., temp0.3, _temp0.3, temperature0.3). Returns None if not found."""
    filename_lower = filename.lower()
    # Look for patterns like _temp0.3, temp0.3, temperature0.3, _temperature0.3
    # Match decimal numbers (e.g., 0.3, 0.7, 1.0)
    patterns = [
        r'_temp([0-9]+\.?[0-9]*)',  # _temp0.3 or _temp1
        r'temp([0-9]+\.?[0-9]*)',   # temp0.3 or temp1 (standalone)
        r'_temperature([0-9]+\.?[0-9]*)',  # _temperature0.3
        r'temperature([0-9]+\.?[0-9]*)',   # temperature0.3 (standalone)
    ]
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            return match.group(1)
    return None


def extract_model_size_from_filename(filename: str) -> str:
    """Extract model size from filename (e.g., Qwen3-4B -> qwen4b, Llama-3.2-3B -> llama3b). Returns None if not found."""
    # Patterns for different model naming conventions
    patterns = [
        # Qwen3-4B, Qwen3-1.7B, Qwen3-8B -> qwen4b, qwen1.7b, qwen8b
        (r'[Qq]wen\d*-(\d+\.?\d*[Bb])', lambda m: f"qwen{m.group(1).lower()}"),
        # Llama-3.2-3B, Llama-3-8B -> llama3b, llama8b
        (r'[Ll]lama[^-]*-(\d+\.?\d*[Bb])', lambda m: f"llama{m.group(1).lower()}"),
        # Generic fallback: ModelName-SizeB -> modelname-sizeb
        (r'([A-Za-z]+\d*)-(\d+\.?\d*[Bb])', lambda m: f"{m.group(1).lower()}{m.group(2).lower()}"),
    ]

    for pattern, formatter in patterns:
        match = re.search(pattern, filename)
        if match:
            return formatter(match)
    return None


def determine_output_directory(model1_path: str, model2_path: str, default_output_dir: str) -> str:
    """Determine output directory based on dataset and model types in filenames.

    Args:
        model1_path: Path to first model's completions file
        model2_path: Path to second model's completions file
        default_output_dir: Default output directory if conditions aren't met

    Returns:
        Output directory path
    """
    # Extract just the filenames (without path)
    filename1 = os.path.basename(model1_path)
    filename2 = os.path.basename(model2_path)

    # Extract dataset from both files
    dataset1 = extract_dataset_from_filename(filename1)
    dataset2 = extract_dataset_from_filename(filename2)

    # Determine dataset to use
    if dataset1 is not None and dataset2 is not None:
        # Both have dataset keywords - they must match
        if dataset1 != dataset2:
            raise ValueError(
                f"Dataset mismatch: File 1 has '{dataset1}' but File 2 has '{dataset2}'. "
                f"Both files must have the same dataset type (if, chat, or tldr)."
            )
        dataset = dataset1
    elif dataset1 is not None:
        # Only file1 has dataset keyword
        dataset = dataset1
    elif dataset2 is not None:
        # Only file2 has dataset keyword
        dataset = dataset2
    else:
        # Neither file has dataset keyword - use default directory
        return default_output_dir

    # Extract model types
    type1 = extract_type_from_filename(filename1)
    type2 = extract_type_from_filename(filename2)

    # Extract temperatures
    temp1 = extract_temperature_from_filename(filename1)
    temp2 = extract_temperature_from_filename(filename2)

    # Extract model size
    model_size1 = extract_model_size_from_filename(filename1)
    model_size2 = extract_model_size_from_filename(filename2)

    # Use model size if available (prefer first file, fallback to second)
    model_size = model_size1 or model_size2

    # Build subfolder: {model_size}-{dataset}-{type1}-vs-{type2} or {dataset}-{type1}-vs-{type2}
    if model_size:
        subfolder = f"{model_size}-{dataset}-{type1}-vs-{type2}"
    else:
        subfolder = f"{dataset}-{type1}-vs-{type2}"

    # Add temperature information if available
    if temp1 is not None and temp2 is not None:
        if temp1 == temp2:
            subfolder += f"-temp{temp1}"
        else:
            subfolder += f"-temp{temp1}-temp{temp2}"
    elif temp1 is not None:
        subfolder += f"-temp{temp1}"
    elif temp2 is not None:
        subfolder += f"-temp{temp2}"

    return os.path.join(default_output_dir, subfolder)


def generate_output_filename(model1_path: str, model2_path: str, judge_model: str,
                           N: int, B: int, seed: int, allow_ties: bool,
                           completion_index: int = 0) -> str:
    def extract_name(model_path: str) -> str:
        # Get filename without directory, remove .jsonl/.json extension
        name = model_path.split("/")[-1].replace("/", "-")
        if name.endswith(".jsonl"):
            name = name[:-6]
        elif name.endswith(".json"):
            name = name[:-5]

        # Remove training hyperparameters: bsz, lr, warmup
        name = re.sub(r'-bsz\d+-', '-', name)
        name = re.sub(r'-lr[0-9e.-]+-', '-', name)
        name = re.sub(r'-warmup\d+-', '-', name)

        # Remove prompts and completions information
        # Pattern: _NUMBERprompts_NUMBERcompletions_seedNUMBER or _NUMBERcompletions
        name = re.sub(r'_\d+prompts_\d+completions_seed\d+', '', name)
        name = re.sub(r'_\d+completions', '', name)
        name = re.sub(r'_\d+prompts', '', name)

        # Clean up any double dashes that might result
        name = re.sub(r'--+', '-', name)
        name = name.strip('-')

        return name
    model1_name = extract_name(model1_path)
    model2_name = extract_name(model2_path)
    judge_name = clean_judge_name(judge_model)

    # Extract temperatures from full paths (not just filenames, to catch temp in directory names)
    temp1 = extract_temperature_from_filename(model1_path)
    temp2 = extract_temperature_from_filename(model2_path)

    base = f"{model1_name}_vs_{model2_name}_{judge_name}"

    # Add temperature information if available
    if temp1 is not None and temp2 is not None:
        if temp1 == temp2:
            base += f"_temp{temp1}"
        else:
            base += f"_temp{temp1}-temp{temp2}"
    elif temp1 is not None:
        base += f"_temp{temp1}"
    elif temp2 is not None:
        base += f"_temp{temp2}"

    base += f"_N{N}_B{B}_seed{seed}"
    if completion_index != 0:
        base += f"_ind{completion_index}"
    if not allow_ties:
        base += "_noties"
    base += "_bootstrap"
    return f"{base}.json"


def parse_survey_response(response_text: str, order_swapped: bool, allow_ties: bool) -> Dict[str, Any]:
    """Parse the structured survey response to extract criterion evaluations."""
    response_text = response_text.replace("\r\n", "\n")
    criteria = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    criterion_evaluations = {}

    # Decisions must occupy an explicit Winner line (or the legacy leading
    # overall verdict), never a letter found in the surrounding assessment.
    label = r"^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?(?:Winner|Recommendation)(?:\*\*)?[ \t]*:(?:\*\*)?[ \t]*(.*)$"
    boundary = r"(?=^[ \t]*\*{0,2}(?:\d+\.|Overall Recommendation:)|\Z)"

    def decision(value):
        value = value.strip()
        if value.startswith("**") and value.endswith("**"):
            value = value[2:-2].strip()
        # A single, short parenthetical comment is harmless only if it does not
        # introduce another verdict or a choice between labels.
        match = re.fullmatch(
            r"(?:\[((?:Response[ \t]+)?(?:A|B)|Tie)\]|((?:Response[ \t]+)?(?:A|B)|Tie))"
            r"(?:[ \t]+\(([^()\r\n]{1,160})\))?", value, re.IGNORECASE,
        )
        if not match:
            return None
        suffix = match.group(3) or ""
        if (re.search(r"\b(?:Response\s+[AB]|Tie|Winner|Recommendation)\b", suffix, re.IGNORECASE)
                or re.search(r"\b(?:A|[Bb])\b", suffix)
                or re.search(r"\ba\s+(?:is|was|would|should|wins?|seems?|appears?|has|performs?|better|instead|overall)\b", suffix, re.I)
                or re.search(r"\b(?:prefer|choose|pick|favor|favour|winner|rather than)\s+a\b", suffix, re.I)
                or re.search(r"\b(?:or|and|versus|vs\.?)\s+[ab]\b", suffix, re.IGNORECASE)):
            return None
        token = (match.group(1) or match.group(2)).split()[-1].upper()
        if token == "TIE":
            return "tie" if allow_ties else None
        return ("model2" if order_swapped else "model1") if token == "A" else (
            "model1" if order_swapped else "model2"
        )

    def parse_section(section, allow_legacy=False):
        labels = re.findall(label, section, re.IGNORECASE | re.MULTILINE)
        if allow_legacy:
            bare_lines = [line.strip() for line in section.splitlines() if re.match(
                r"^(?:\*\*)?(?:\[(?:(?:Response[ \t]+)?[AB]|Tie)\]|(?:Response[ \t]+)?[AB]|Tie)(?![A-Za-z])",
                line.strip(), re.I,
            )]
            if (labels and bare_lines) or len(bare_lines) > 1:
                return None
        if labels:
            # Count malformed labels too; a valid first label must not hide a
            # contradictory or malformed second one.
            return decision(labels[0]) if len(labels) == 1 else None
        if allow_legacy and section.strip():
            leading = section.strip().splitlines()[0]
            # Preserve the historical "[A] - explanation" overall format.
            parts = re.split(r"[ \t]+[-—:][ \t]+", leading, maxsplit=1)
            if len(parts) == 2 and re.fullmatch(r"(?:\[)?(?:Response\s+)?(?:A|B|Tie)(?:\])?", parts[1], re.I):
                return None
            winner = decision(parts[0])
            if len(parts) == 2 and winner is not None:
                # The legacy inline justification may mention either response,
                # but may not explicitly nominate the other as the winner.
                other = "B" if winner == ("model2" if order_swapped else "model1") else "A"
                conflict = rf"\b(?:Response\s+)?{other}\s+(?:is|was|seems|appears|would be)\s+(?:(?:clearly|the|a|much|overall)\s+)*(?:better|stronger|winner|preferred)\b"
                if re.search(conflict, parts[1], re.I):
                    return None
            return winner
        return None

    for i, criterion in enumerate(criteria, 1):
        pattern = rf"^[ \t]*\*{{0,2}}{i}\.[ \t]*{criterion}\*{{0,2}}[ \t]*\n(.*?){boundary}"
        sections = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        heading_count = len(re.findall(rf"^[ \t]*\*{{0,2}}{i}\.[ \t]*{criterion}\b", response_text, re.I | re.M))
        section = sections[0] if len(sections) == 1 and heading_count == 1 else ""
        winner = parse_section(section)
        justification = re.search(r"^[ \t]*(?:[-*][ \t]+)?Justification:[ \t]*(\S[^\r\n]*)", section, re.I | re.M)
        valid = winner is not None and justification is not None
        criterion_evaluations[criterion] = {
            "winner": winner if valid else None,
            "parsing_failed": not valid,
            "justification": justification.group(1).strip() if valid else "Failed to parse criterion evaluation",
        }

    # A five-dimensional score requires all five dimensions to be valid.
    successful_criteria = [eval_result for eval_result in criterion_evaluations.values() if not eval_result.get("parsing_failed", False)]

    if len(successful_criteria) != len(criteria):
        # Incomplete surveys are invalid, not a majority over whichever parsed.
        survey_winner = None
        survey_calculation = {
            "model1_wins": 0,
            "model2_wins": 0,
            "tie_count": 0,
            "model1_score": 0.0,
            "model2_score": 0.0,
            "successful_criteria_count": len(successful_criteria)
        }
    else:
        # Count wins from successfully parsed criteria
        model1_wins = sum(1 for eval_result in successful_criteria if eval_result["winner"] == "model1")
        model2_wins = sum(1 for eval_result in successful_criteria if eval_result["winner"] == "model2")
        tie_count = sum(1 for eval_result in successful_criteria if eval_result["winner"] == "tie")

        # Calculate scores (ties count as 0.5 each)
        model1_score = model1_wins + (tie_count * 0.5)
        model2_score = model2_wins + (tie_count * 0.5)

        # Determine survey winner
        if model1_score > model2_score:
            survey_winner = "model1"
        elif model2_score > model1_score:
            survey_winner = "model2"
        else:
            survey_winner = "tie"

        survey_calculation = {
            "model1_wins": model1_wins,
            "model2_wins": model2_wins,
            "tie_count": tie_count,
            "model1_score": model1_score,
            "model2_score": model2_score,
            "successful_criteria_count": len(successful_criteria)
        }

    # Exactly one anchored overall section. Repeated headings/verdicts fail
    # closed even if one of them would otherwise be parseable.
    overall_sections = re.findall(
        rf"^[ \t]*\*{{0,2}}Overall Recommendation:\*{{0,2}}[ \t]*\n(.*?){boundary}",
        response_text, re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    overall_heading_count = len(re.findall(r"^[ \t]*\*{0,2}Overall Recommendation\b", response_text, re.I | re.M))
    overall_section = overall_sections[0] if len(overall_sections) == 1 and overall_heading_count == 1 else ""
    overall_winner = parse_section(overall_section, allow_legacy=True)
    overall_parsing_failed = overall_winner is None
    overall_justification = overall_section.strip() if overall_winner is not None else "Failed to parse overall recommendation"

    return {
        "criterion_evaluations": criterion_evaluations,
        "overall_winner": overall_winner,
        "overall_parsing_failed": overall_parsing_failed,
        "overall_justification": overall_justification,
        "survey_winner": survey_winner,
        "survey_calculation": survey_calculation
    }


def setup_judge_client(api_key: str, api_provider: str = "auto", base_url: Optional[str] = None):
    """Create a judge client without inferring DeepSeek from a secret key prefix."""
    if api_provider == "auto":
        api_provider = "anthropic" if api_key.startswith("sk-ant-") else "openai"

    if api_provider == "anthropic":
        if anthropic is None:
            raise ImportError("anthropic package is required for Claude API. Install with: pip install anthropic")
        return "anthropic", anthropic.Anthropic(api_key=api_key)
    if openai is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    if api_provider == "deepseek":
        return "deepseek", openai.OpenAI(api_key=api_key, base_url=base_url or "https://api.deepseek.com")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return "openai", openai.OpenAI(**client_kwargs)


def call_judge(api_type: str, judge_client, judge_model: str, prompt: str, response1: str, response2: str,
               allow_ties: bool, order_swapped: bool = False, thinking_mode: str = "disabled",
               max_retries: int = 5, judge_temperature: Optional[float] = DEFAULT_JUDGE_TEMPERATURE) -> Dict[str, Any]:
    """Call the judge API with survey-based evaluation across 5 criteria."""
    # Use the predetermined order (no randomization here)
    if order_swapped:
        response1, response2 = response2, response1

    comparison_prompt = build_comparison_prompt(prompt, response1, response2, allow_ties)
    request_parameters = judge_request_parameters(api_type, judge_model, thinking_mode, judge_temperature)

    last_error = None
    for attempt in range(max_retries):
        try:
            if api_type == "anthropic":
                response = judge_client.messages.create(
                    model=judge_model,
                    **request_parameters,
                    messages=[{"role": "user", "content": comparison_prompt}],
                )
                response_text = response.content[0].text.strip()
                usage = {
                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                }
            elif "gpt-5" in judge_model.lower() and api_type == "openai":
                response = judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": comparison_prompt}],
                    **request_parameters,
                )
                response_text = response.choices[0].message.content.strip()
                usage = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(response.usage, "completion_tokens", 0),
                }
            else:
                request_kwargs = {
                    "model": judge_model,
                    "messages": [{"role": "user", "content": comparison_prompt}],
                    **request_parameters,
                }
                response = judge_client.chat.completions.create(**request_kwargs)
                response_text = response.choices[0].message.content.strip()
                response_usage = getattr(response, "usage", None)
                usage = {
                    "input_tokens": getattr(response_usage, "prompt_tokens", 0),
                    "output_tokens": getattr(response_usage, "completion_tokens", 0),
                    "cached_input_tokens": getattr(
                        getattr(response_usage, "prompt_tokens_details", None), "cached_tokens", 0
                    ),
                }

            parsed_result = parse_survey_response(response_text, order_swapped, allow_ties)
            parsed_result["raw_response"] = response_text
            parsed_result["order_swapped"] = order_swapped
            parsed_result["api_usage"] = usage
            parsed_result["protocol_version"] = JUDGE_PROTOCOL_VERSION
            parsed_result["parser_version"] = JUDGE_PARSER_VERSION
            parsed_result["judge_settings"] = judge_settings_metadata(api_type, judge_model, thinking_mode, judge_temperature)
            return parsed_result
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                delay = min(2 ** attempt, 30)
                print(f"Judge request failed ({attempt + 1}/{max_retries}): {exc}; retrying in {delay}s")
                time.sleep(delay)

    print(f"Error calling judge API after {max_retries} attempts: {last_error}")
    return {
            "criterion_evaluations": {
                "helpfulness": {"winner": None, "parsing_failed": True, "justification": f"Error: {last_error}"},
                "correctness": {"winner": None, "parsing_failed": True, "justification": f"Error: {last_error}"},
                "coherence": {"winner": None, "parsing_failed": True, "justification": f"Error: {last_error}"},
                "complexity": {"winner": None, "parsing_failed": True, "justification": f"Error: {last_error}"},
                "verbosity": {"winner": None, "parsing_failed": True, "justification": f"Error: {last_error}"}
            },
            "overall_winner": None,
            "overall_parsing_failed": True,
            "overall_justification": f"Error during evaluation: {last_error}",
            "survey_winner": None,
            "survey_calculation": {
                "model1_wins": 0,
                "model2_wins": 0,
                "tie_count": 0,
                "model1_score": 0.0,
                "model2_score": 0.0,
                "successful_criteria_count": 0
            },
            "raw_response": "",
            "order_swapped": order_swapped,
            "parse_error": str(last_error),
            "api_usage": {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0},
    }


def calculate_majority_vote(individual_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate majority vote from individual judgments."""
    vote_counts = {"model1": 0, "model2": 0, "tie": 0}
    for j in individual_judgments:
        vote_counts[j["winner"]] += 1
    max_votes = max(vote_counts.values())
    majority_winners = [k for k, v in vote_counts.items() if v == max_votes]
    majority_winner = majority_winners[0] if len(majority_winners) == 1 else individual_judgments[0]["winner"]
    return {"vote_counts": vote_counts, "majority_winner": majority_winner, "final_winner": majority_winner}


def analyze_bootstrap_results(bootstrap_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Percentile intervals from prompt clusters, never independent A/B calls."""
    if not bootstrap_results:
        return {}
    if np is None:
        raise ImportError("numpy is required for statistical analysis")

    def safe_stats(values, include_ci_99=True):
        values = np.array(values, dtype=float)
        output = {
            "mean": float(np.mean(values)) if len(values) else None,
            "std": float(np.std(values)) if len(values) else None,
            "min": float(np.min(values)) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None,
            "raw_values": values.tolist(),
            "ci_95": np.percentile(values, [2.5, 97.5]).tolist() if len(values) else [None, None],
        }
        if include_ci_99:
            output["ci_99"] = np.percentile(values, [0.5, 99.5]).tolist() if len(values) else [None, None]
        return output

    output = {"bootstrap_iterations": len(bootstrap_results)}
    for endpoint in ("overall", "survey"):
        analyses = [result["analysis"][f"{endpoint}_analysis"] for result in bootstrap_results]
        valid = [analysis for analysis in analyses if analysis["valid_comparisons"] > 0]
        summary = {
            "iterations_with_valid_comparisons": len(valid),
            "iterations_excluded": len(analyses) - len(valid),
            "exclusion_rate_distribution": safe_stats([
                analysis["excluded"] / result["analysis"]["total_comparisons"]
                for analysis, result in zip(analyses, bootstrap_results)
            ], include_ci_99=False),
        }
        for model in ("model1", "model2"):
            summary[f"{model}_win_rate_distribution"] = safe_stats([analysis[f"{model}_win_rate"] for analysis in valid])
            summary[f"{model}_score_distribution"] = safe_stats([analysis[f"{model}_mean_score"] for analysis in valid])
        for metric in ("tie_rate", "order_disagreement_rate"):
            summary[f"{metric}_distribution"] = safe_stats([analysis[metric] for analysis in valid])
        output[f"{endpoint}_winner_analysis"] = summary
    return output


def _cache_key(item: Dict[str, str], api_type: str, judge_model: str, allow_ties: bool,
               order_swapped: bool, thinking_mode: str, base_url: str = "",
               judge_temperature: Optional[float] = DEFAULT_JUDGE_TEMPERATURE) -> str:
    payload = {
        "prompt": item["prompt"],
        "completion1": item["completion1"],
        "completion2": item["completion2"],
        "api_type": api_type,
        "judge_model": judge_model,
        "allow_ties": allow_ties,
        "order_swapped": order_swapped,
        "thinking_mode": thinking_mode,
        "base_url": base_url.rstrip("/"),
        "protocol_version": JUDGE_PROTOCOL_VERSION,
        "parser_version": JUDGE_PARSER_VERSION,
        "cache_version": JUDGE_CACHE_VERSION,
        "judge_settings": judge_settings_metadata(api_type, judge_model, thinking_mode, judge_temperature),
        "prompt_sha256": hashlib.sha256(build_comparison_prompt(
            item["prompt"], item["completion2"] if order_swapped else item["completion1"],
            item["completion1"] if order_swapped else item["completion2"], allow_ties,
        ).encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_judgment_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    cache = {}
    if not cache_path or not os.path.isfile(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: ignoring invalid cache line {line_number} in {cache_path}")
                continue
            if "cache_key" in record and "result" in record:
                cache[record["cache_key"]] = record["result"]
    return cache


def _append_judgment_cache(cache_path: str, cache_key: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"cache_key": cache_key, "result": result}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _analyze_iteration(iteration_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = {"total_comparisons": len(iteration_results)}
    single_scores = {"model1": 1.0, "model2": 0.0, "tie": 0.5}
    for endpoint in ("overall", "survey"):
        valid = [result for result in iteration_results
                 if result.get(f"{endpoint}_winner") in (*single_scores, "order_disagreement")
                 and not result.get(f"{endpoint}_parsing_failed", False)]
        counts = {outcome: sum(result[f"{endpoint}_winner"] == outcome for result in valid)
                  for outcome in (*single_scores, "order_disagreement")}
        model1_scores = [result.get(f"{endpoint}_model1_score", single_scores.get(result[f"{endpoint}_winner"]))
                         for result in valid]
        model1_mean = sum(model1_scores) / len(valid) if valid else None
        output[f"{endpoint}_analysis"] = {
            "valid_comparisons": len(valid),
            "excluded": len(iteration_results) - len(valid),
            "model1_wins": counts["model1"],
            "model2_wins": counts["model2"],
            "ties": counts["tie"],
            "order_disagreements": counts["order_disagreement"],
            "model1_win_rate": counts["model1"] / len(valid) if valid else None,
            "model2_win_rate": counts["model2"] / len(valid) if valid else None,
            "tie_rate": counts["tie"] / len(valid) if valid else None,
            "order_disagreement_rate": counts["order_disagreement"] / len(valid) if valid else None,
            "model1_mean_score": model1_mean,
            "model2_mean_score": 1 - model1_mean if valid else None,
            "order_diagnostics": order_diagnostics(iteration_results, endpoint),
        }
    return output


def run_bootstrap_evaluation(meta1: Dict[str, Any], items1: List[Dict[str, str]],
                             meta2: Dict[str, Any], items2: List[Dict[str, str]],
                             N: int, B: int, judge_model: str, api_type: str, judge_client,
                             allow_ties: bool, seed: int, cache_path: str,
                             thinking_mode: str = "disabled", refresh_cache: bool = False,
                             max_retries: int = 5, base_url: str = "", judge_both_orders: bool = False,
                             judge_temperature: Optional[float] = DEFAULT_JUDGE_TEMPERATURE) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """Judge each prompt once or in both orders, then bootstrap prompt clusters."""
    if np is None:
        raise ImportError("numpy is required for bootstrap sampling. Please install it with: pip install numpy")

    _, all_items = merge_completions_artifacts(meta1, items1, meta2, items2)
    total_prompts = len(all_items)
    if N > total_prompts:
        raise ValueError(f"Evaluation sample size N ({N}) cannot be larger than total prompts ({total_prompts})")
    if N < 2 or B < 1:
        raise ValueError("N must be at least 2 and B must be at least 1")

    selection_rng = random.Random(seed)
    selected_indices = selection_rng.sample(range(total_prompts), N)
    if judge_both_orders and len({all_items[index]["prompt"] for index in selected_indices}) != N:
        raise ValueError("Both-order evaluation requires unique prompts; deduplicate the held-out sample")
    cached_results = {} if refresh_cache else _load_judgment_cache(cache_path)
    judged_results = []

    calls_per_prompt = 2 if judge_both_orders else 1
    print(f"Judging {N} prompt pairs in {calls_per_prompt} order(s); {B} prompt-cluster bootstrap resamples run locally")
    for source_index in tqdm(selected_indices, desc="LLM judge prompt pairs"):
        item = all_items[source_index]
        first_order = selection_rng.choice([True, False])
        orders = [first_order, not first_order] if judge_both_orders else [first_order]
        order_results = []
        for order_swapped in orders:
            cache_key = _cache_key(item, api_type, judge_model, allow_ties, order_swapped, thinking_mode, base_url, judge_temperature)
            cached_result = cached_results.get(cache_key)
            if cached_result is None:
                judgment = call_judge(
                    api_type,
                    judge_client,
                    judge_model,
                    item["prompt"],
                    item["completion1"],
                    item["completion2"],
                    allow_ties,
                    order_swapped=order_swapped,
                    thinking_mode=thinking_mode,
                    max_retries=max_retries,
                    judge_temperature=judge_temperature,
                )
                result = {
                    "prompt": item["prompt"],
                    "model1_completion": item["completion1"],
                    "model2_completion": item["completion2"],
                    "judgment": judgment,
                    "overall_winner": judgment.get("overall_winner"),
                    "overall_parsing_failed": judgment.get("overall_parsing_failed", False),
                    "survey_winner": judgment.get("survey_winner"),
                    "survey_calculation": judgment.get("survey_calculation", {}),
                    "criterion_evaluations": judgment.get("criterion_evaluations", {}),
                    "model1_name": "model1",
                    "model2_name": "model2",
                    "order_swapped": order_swapped,
                    "source_index": source_index,
                    "cache_key": cache_key,
                }
                # Cache only valid decisions; API and parse failures are retried on
                # the next run, while their raw evidence remains in the result JSON.
                if "parse_error" not in judgment and not judgment.get("overall_parsing_failed", True) and judgment.get("survey_winner") is not None:
                    _append_judgment_cache(cache_path, cache_key, result)
                    cached_results[cache_key] = result
                result["cache_hit"] = False
            else:
                result = dict(cached_result)
                result["cache_hit"] = True
                result["source_index"] = source_index
            order_results.append(result)
        judged_results.append(combine_order_judgments(order_results) if judge_both_orders else order_results[0])

    bootstrap_rng = np.random.default_rng(seed)
    bootstrap_results = []
    for iteration in tqdm(range(B), desc="Local bootstrap iterations"):
        sample_positions = bootstrap_rng.choice(N, size=N, replace=True)
        iteration_results = [judged_results[int(position)] for position in sample_positions]
        bootstrap_results.append({
            "iteration": iteration,
            "sample_positions": sample_positions.tolist(),
            "analysis": _analyze_iteration(iteration_results),
        })

    return bootstrap_results, judged_results, selected_indices


def save_bootstrap_results(output_path: str, model1_path: str, model2_path: str, judge_model: str,
                          api_type: str, seed: int, N: int, B: int, allow_ties: bool,
                          bootstrap_results: List[Dict[str, Any]], bootstrap_analysis: Dict[str, Any],
                          judged_results: List[Dict[str, Any]], selected_indices: List[int],
                          cache_path: str, base_url: Optional[str], thinking_mode: str,
                          completion_index: int = 0, validation: Optional[Dict[str, Any]] = None,
                          judge_both_orders: bool = False, source_metadata=None,
                          judge_temperature: Optional[float] = DEFAULT_JUDGE_TEMPERATURE) -> None:
    """Save comprehensive bootstrap results to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    usage_totals = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    raw_judgments = [order for result in judged_results for order in result.get("order_judgments", [result])]
    for result in raw_judgments:
        if result.get("cache_hit"):
            continue
        usage = result.get("judgment", {}).get("api_usage", {})
        for key in usage_totals:
            usage_totals[key] += int(usage.get(key, 0) or 0)

    payload = {
        "status": "success" if validation and validation["passed"] else "invalid",
        "validation": validation,
        "bootstrap_config": {
            "model1_path": model1_path,
            "model2_path": model2_path,
            "judge_model": judge_model,
            "api_type": api_type,
            "seed": seed,
            "subsample_size_N": N,
            "bootstrap_iterations_B": B,
            "allow_ties": allow_ties,
            "completion_index_used": completion_index,
            "thinking_mode": thinking_mode,
            "base_url": base_url,
            "protocol_version": JUDGE_PROTOCOL_VERSION,
            "parser_version": JUDGE_PARSER_VERSION,
            "cache_version": JUDGE_CACHE_VERSION,
            "judge_settings": judge_settings_metadata(api_type, judge_model, thinking_mode, judge_temperature),
            "primary_endpoint": "overall",
            "survey_interpretation": "secondary equal-weight diagnostic; not the overall decision rule",
            "complexity_definition": "legacy key: task-appropriate depth, not sophistication or complexity itself",
            "cache_path": os.path.abspath(cache_path),
            "judge_both_orders": judge_both_orders,
            "aggregation_version": "paired-order-score-v1" if judge_both_orders else "single-order-v1",
            "methodology": "judge_each_prompt_in_both_orders_then_bootstrap_prompt_clusters" if judge_both_orders else "judge_N_unique_pairs_once_then_bootstrap_locally_with_replacement",
            "score_definition": "mean_across_orders_of_win_1_tie_0.5_loss_0",
            "score_denominator": "valid prompt clusters only; missing judgments are excluded, never scored as ties",
            "confidence_interval_scope": "held-out prompt sampling, conditional on these checkpoints, generated responses and judge outcomes; not training-seed variability"
        },
        "source_metadata": source_metadata,
        "selected_source_indices": selected_indices,
        "api_usage": usage_totals,
        "cache_summary": {
            "hits_this_run": sum(bool(result.get("cache_hit")) for result in raw_judgments),
            "api_calls_this_run": sum(not bool(result.get("cache_hit")) for result in raw_judgments),
            "logical_judgments": len(raw_judgments),
            "failed_api_judgments": sum("parse_error" in result.get("judgment", {}) for result in raw_judgments),
            "failed_parse_judgments": sum(
                "parse_error" not in result.get("judgment", {}) and
                (result.get("overall_parsing_failed", True) or result.get("survey_winner") is None)
                for result in raw_judgments),
        },
        "judged_results": judged_results,
        "bootstrap_analysis": bootstrap_analysis,
        "bootstrap_results": bootstrap_results
    }

    temporary_path = f"{output_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temporary_path, output_path)
    print(f"Bootstrap results saved to {output_path}")


def print_bootstrap_summary(model1_path: str, model2_path: str, judge_model: str,
                           N: int, B: int, allow_ties: bool,
                           bootstrap_analysis: Dict[str, Any]) -> None:
    """Print comprehensive summary of bootstrap evaluation results."""
    print("\n" + "=" * 100)
    print("BOOTSTRAP LLM-AS-JUDGE EVALUATION SUMMARY")
    print("=" * 100)
    print(f"\nModel 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print(f"Judge Model: {judge_model}")
    print(f"Bootstrap Configuration:")
    print(f"  - Subsample size (N): {N}")
    print(f"  - Bootstrap iterations (B): {B}")
    print(f"  - Allow ties: {allow_ties}")

    overall_analysis = bootstrap_analysis["overall_winner_analysis"]
    survey_analysis = bootstrap_analysis["survey_winner_analysis"]
    if not overall_analysis["iterations_with_valid_comparisons"] or not survey_analysis["iterations_with_valid_comparisons"]:
        print("INVALID evaluation: no valid overall or complete five-criterion judgments; no winner or confidence interval.")
        return

    # Overall winner statistics
    print(f"\n{'='*100}")
    print("OVERALL WINNER ANALYSIS (from explicit Overall Recommendation)")
    print(f"{'='*100}")

    model1_overall_dist = overall_analysis["model1_win_rate_distribution"]
    model2_overall_dist = overall_analysis["model2_win_rate_distribution"]
    overall_exclusion_dist = overall_analysis["exclusion_rate_distribution"]

    print(f"\nModel 1 Overall Win Rate:")
    print(f"  Mean: {model1_overall_dist['mean']:.3f} ± {model1_overall_dist['std']:.3f}")
    print(f"  95% CI: [{model1_overall_dist['ci_95'][0]:.3f}, {model1_overall_dist['ci_95'][1]:.3f}]")
    print(f"  99% CI: [{model1_overall_dist['ci_99'][0]:.3f}, {model1_overall_dist['ci_99'][1]:.3f}]")
    print(f"  Range: [{model1_overall_dist['min']:.3f}, {model1_overall_dist['max']:.3f}]")

    print(f"\nModel 2 Overall Win Rate:")
    print(f"  Mean: {model2_overall_dist['mean']:.3f} ± {model2_overall_dist['std']:.3f}")
    print(f"  95% CI: [{model2_overall_dist['ci_95'][0]:.3f}, {model2_overall_dist['ci_95'][1]:.3f}]")
    print(f"  99% CI: [{model2_overall_dist['ci_99'][0]:.3f}, {model2_overall_dist['ci_99'][1]:.3f}]")
    print(f"  Range: [{model2_overall_dist['min']:.3f}, {model2_overall_dist['max']:.3f}]")

    print(f"\nOverall Parsing Failure Rate:")
    print(f"  Mean: {overall_exclusion_dist['mean']:.3f} ± {overall_exclusion_dist['std']:.3f}")
    print(f"  95% CI: [{overall_exclusion_dist['ci_95'][0]:.3f}, {overall_exclusion_dist['ci_95'][1]:.3f}]")
    print(f"  Range: [{overall_exclusion_dist['min']:.3f}, {overall_exclusion_dist['max']:.3f}]")

    # Determine overall winner
    if model1_overall_dist['mean'] > model2_overall_dist['mean']:
        print(f"\n🏆 Overall Winner: Model 1 (mean win rate: {model1_overall_dist['mean']:.3f})")
    elif model2_overall_dist['mean'] > model1_overall_dist['mean']:
        print(f"\n🏆 Overall Winner: Model 2 (mean win rate: {model2_overall_dist['mean']:.3f})")
    else:
        print(f"\n🤝 Overall Result: Tie (both models: {model1_overall_dist['mean']:.3f})")

    # Survey winner statistics
    print(f"\n{'='*100}")
    print("SURVEY WINNER ANALYSIS (from criterion majority)")
    print(f"{'='*100}")

    model1_survey_dist = survey_analysis["model1_win_rate_distribution"]
    model2_survey_dist = survey_analysis["model2_win_rate_distribution"]
    survey_tie_dist = survey_analysis["tie_rate_distribution"]
    survey_exclusion_dist = survey_analysis["exclusion_rate_distribution"]

    print(f"\nModel 1 Survey Win Rate:")
    print(f"  Mean: {model1_survey_dist['mean']:.3f} ± {model1_survey_dist['std']:.3f}")
    print(f"  95% CI: [{model1_survey_dist['ci_95'][0]:.3f}, {model1_survey_dist['ci_95'][1]:.3f}]")
    print(f"  99% CI: [{model1_survey_dist['ci_99'][0]:.3f}, {model1_survey_dist['ci_99'][1]:.3f}]")
    print(f"  Range: [{model1_survey_dist['min']:.3f}, {model1_survey_dist['max']:.3f}]")

    print(f"\nModel 2 Survey Win Rate:")
    print(f"  Mean: {model2_survey_dist['mean']:.3f} ± {model2_survey_dist['std']:.3f}")
    print(f"  95% CI: [{model2_survey_dist['ci_95'][0]:.3f}, {model2_survey_dist['ci_95'][1]:.3f}]")
    print(f"  99% CI: [{model2_survey_dist['ci_99'][0]:.3f}, {model2_survey_dist['ci_99'][1]:.3f}]")
    print(f"  Range: [{model2_survey_dist['min']:.3f}, {model2_survey_dist['max']:.3f}]")

    print(f"\nSurvey Tie Rate:")
    print(f"  Mean: {survey_tie_dist['mean']:.3f} ± {survey_tie_dist['std']:.3f}")
    print(f"  95% CI: [{survey_tie_dist['ci_95'][0]:.3f}, {survey_tie_dist['ci_95'][1]:.3f}]")
    print(f"  99% CI: [{survey_tie_dist['ci_99'][0]:.3f}, {survey_tie_dist['ci_99'][1]:.3f}]")
    print(f"  Range: [{survey_tie_dist['min']:.3f}, {survey_tie_dist['max']:.3f}]")

    print(f"\nSurvey Exclusion Rate (all criteria failed):")
    print(f"  Mean: {survey_exclusion_dist['mean']:.3f} ± {survey_exclusion_dist['std']:.3f}")
    print(f"  95% CI: [{survey_exclusion_dist['ci_95'][0]:.3f}, {survey_exclusion_dist['ci_95'][1]:.3f}]")
    print(f"  Range: [{survey_exclusion_dist['min']:.3f}, {survey_exclusion_dist['max']:.3f}]")

    # Determine survey winner
    if model1_survey_dist['mean'] > model2_survey_dist['mean']:
        print(f"\n🏆 Survey Winner: Model 1 (mean win rate: {model1_survey_dist['mean']:.3f})")
    elif model2_survey_dist['mean'] > model1_survey_dist['mean']:
        print(f"\n🏆 Survey Winner: Model 2 (mean win rate: {model2_survey_dist['mean']:.3f})")
    else:
        print(f"\n🤝 Survey Result: Tie (both models: {model1_survey_dist['mean']:.3f})")

    print("=" * 100)


def print_paired_order_summary(observed, bootstrap_analysis):
    print("\nTwo-order blinded comparison (prompt is the bootstrap unit)")
    for endpoint in ("overall", "survey"):
        counts = observed[f"{endpoint}_analysis"]
        ci = bootstrap_analysis[f"{endpoint}_winner_analysis"]["model2_score_distribution"]["ci_95"]
        print(f"{endpoint}: valid={counts['valid_comparisons']}, failed={counts['excluded']}, "
              f"model1 consistent wins={counts['model1_wins']}, model2 consistent wins={counts['model2_wins']}, "
              f"explicit ties={counts['ties']}, order disagreements={counts['order_disagreements']}")
        print(f"  Model2 score={counts['model2_mean_score']:.4f}, prompt-bootstrap 95% CI={ci}; neutral=0.5")
        diagnostics = counts["order_diagnostics"]
        print(f"  Pair outcomes: {diagnostics['pair_outcomes']}")
        print(f"  Presentation-position choices: {diagnostics['presentation_position']}")
    print("Order diagnostics are descriptive; they do not establish that position caused the disagreements.")
    print("The interval describes prompt sampling conditional on these checkpoints/responses/judge outcomes, "
          "not uncertainty across training seeds. A CI crossing 0.5 is inconclusive.")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap-based LLM-as-judge evaluation with confidence intervals")
    parser.add_argument("--completions1", required=True, help="Path to first model's completions artifact JSON/JSONL")
    parser.add_argument("--completions2", required=True, help="Path to second model's completions artifact JSON/JSONL")
    parser.add_argument("--judge-model", default="deepseek-flash", help="Judge model identifier")
    parser.add_argument(
        "--api-provider", choices=["deepseek", "openai", "anthropic", "auto"], default="deepseek"
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Prefer DEEPSEEK_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY instead of this option.",
    )
    parser.add_argument("--thinking-mode", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--judge-temperature", type=lambda value: None if value.lower() == "default" else float(value),
                        default=DEFAULT_JUDGE_TEMPERATURE,
                        help="Judge sampling temperature (default: 0); use 'default' to omit. Ignored in supported reasoning-only modes.")
    parser.add_argument("--min-valid-fraction", type=float, default=1.0,
                        help="Required valid overall AND full-survey fraction (default: all judgments)")
    parser.add_argument("--N", type=int, default=100, help="Number of unique prompt pairs sent to the judge")
    parser.add_argument("--B", type=int, default=1000, help="Number of local bootstrap resamples")
    parser.add_argument("--judge-both-orders", action="store_true",
                        help="Blindly judge A/B and B/A for every prompt (2N calls before cache/retries); bootstrap N prompt clusters")
    tie_group = parser.add_mutually_exclusive_group()
    tie_group.add_argument("--allow-ties", action="store_true", help="Allow ties in evaluation")
    tie_group.add_argument("--no-ties", action="store_true", help="Disable ties in evaluation")
    parser.add_argument("--output-dir", default="evaluate", help="Directory to save results JSON")
    parser.add_argument("--cache-path", default=None, help="Append-only JSONL judgment cache")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached judgments and call the API again")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--completion-index", type=int, default=0, help="Which completion to use from multi-completion files (0=first, 1=second, etc.)")
    args = parser.parse_args()
    if not 0 < args.min_valid_fraction <= 1 or args.max_retries < 1:
        parser.error("min-valid-fraction must be in (0, 1] and max-retries must be positive")

    try:
        judge_request_parameters(args.api_provider, args.judge_model, args.thinking_mode, args.judge_temperature)
    except ValueError as exc:
        parser.error(str(exc))

    # Load both completion files
    print(f"Loading completions from: {args.completions1}")
    print(f"Using completion index: {args.completion_index} (0=first, 1=second, etc.)")
    meta1, items1 = read_single_completions_artifact(args.completions1, completion_index=args.completion_index)
    print(f"Loaded {len(items1)} completions from first file")

    print(f"Loading completions from: {args.completions2}")
    meta2, items2 = read_single_completions_artifact(args.completions2, completion_index=args.completion_index)
    print(f"Loaded {len(items2)} completions from second file")

    allow_ties = not args.no_ties

    output_dir = determine_output_directory(args.completions1, args.completions2, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    api_key_env = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_name = api_key_env.get(args.api_provider, "OPENAI_API_KEY")
    api_key = args.api_key or os.environ.get(env_name)
    if not api_key:
        raise ValueError(f"No API key provided. Set {env_name} or pass --api-key.")
    resolved_base_url = args.base_url
    if args.api_provider == "deepseek" and resolved_base_url is None:
        resolved_base_url = "https://api.deepseek.com"

    # Setup judge client
    api_type, judge_client = setup_judge_client(api_key, args.api_provider, resolved_base_url)
    resolved_base_url = str(getattr(judge_client, "base_url", resolved_base_url or ""))
    print(f"Using {api_type.upper()} API with judge model: {args.judge_model}")

    cache_path = args.cache_path or os.path.join(
        output_dir, "judge_cache", f"{clean_judge_name(args.judge_model)}_judgments.jsonl"
    )

    # Run bootstrap evaluation
    bootstrap_results, judged_results, selected_indices = run_bootstrap_evaluation(
        meta1, items1, meta2, items2,
        args.N, args.B, args.judge_model, api_type, judge_client,
        allow_ties, args.seed, cache_path,
        thinking_mode=args.thinking_mode,
        refresh_cache=args.refresh_cache,
        max_retries=args.max_retries,
        base_url=resolved_base_url,
        judge_both_orders=args.judge_both_orders,
        judge_temperature=args.judge_temperature,
    )

    # Analyze bootstrap results
    bootstrap_analysis = analyze_bootstrap_results(bootstrap_results)
    observed = _analyze_iteration(judged_results)
    validation = {
        "required_fraction": args.min_valid_fraction,
        "overall_valid": observed["overall_analysis"]["valid_comparisons"],
        "survey_valid": observed["survey_analysis"]["valid_comparisons"],
        "total": len(judged_results),
        "observed": observed,
    }
    validation["passed"] = all(validation[key] >= max(2, args.min_valid_fraction * len(judged_results))
                                for key in ("overall_valid", "survey_valid"))

    # Generate output filename and save results
    filename = generate_output_filename(
        args.completions1,
        args.completions2,
        args.judge_model,
        args.N,
        args.B,
        args.seed,
        allow_ties,
        completion_index=args.completion_index
    )
    namespace = _cache_key({"prompt": "", "completion1": "", "completion2": ""}, api_type, args.judge_model,
                           allow_ties, False, args.thinking_mode, resolved_base_url, args.judge_temperature)[:12]
    filename = filename.replace("_bootstrap.json", f"_protocol{namespace}_bootstrap.json")
    if args.judge_both_orders:
        filename = filename.replace("_bootstrap.json", "_both_orders_bootstrap.json")
    output_path = os.path.join(output_dir, filename)

    save_bootstrap_results(
        output_path,
        args.completions1,
        args.completions2,
        args.judge_model,
        api_type,
        args.seed,
        args.N,
        args.B,
        allow_ties,
        bootstrap_results,
        bootstrap_analysis,
        judged_results,
        selected_indices,
        cache_path,
        resolved_base_url,
        args.thinking_mode,
        completion_index=args.completion_index,
        validation=validation,
        judge_both_orders=args.judge_both_orders,
        judge_temperature=args.judge_temperature,
        source_metadata={"model1": meta1, "model2": meta2},
    )

    # Print summary
    if not validation["passed"]:
        print(f"INVALID evaluation: {validation}. Evidence saved; fix failures and retry. No winner declared.")
        return 1
    if args.judge_both_orders:
        print_paired_order_summary(observed, bootstrap_analysis)
        return 0
    print_bootstrap_summary(
        meta1.get("model_path", "model1"),
        meta2.get("model_path", "model2"),
        args.judge_model,
        args.N,
        args.B,
        allow_ties,
        bootstrap_analysis
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
