"""CPU-only contracts shared by generation, cache validation and reporting."""

import hashlib
import json
from pathlib import Path


ARTIFACT_VERSION = 2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_fingerprint(directory):
    """Hash actual local weights and configuration, not mutable paths or mtimes.

    This intentionally reads model files once; a same-path re-merge must invalidate
    the generation cache. No model is imported or loaded onto a GPU.
    """
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file()
                   and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja", ".py"}
                   and ".cache" not in p.relative_to(directory).parts)
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError(f"No model weights found in {directory}")
    manifest = []
    for path in files:
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        manifest.append((str(path.relative_to(directory)), checksum.hexdigest()))
    return digest(manifest)


def valid_generation_cache(path, contract, prompts, n_completions):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = payload["meta"]
        items = payload["items"]
        return (
            meta.get("artifact_version") == ARTIFACT_VERSION
            and meta.get("contract") == contract
            and meta.get("items_sha256") == digest(items)
            and len(items) == len(prompts)
            and all(item["prompt"] == prompt and isinstance(item["completions"], list)
                    and len(item["completions"]) == n_completions
                    and all(isinstance(text, str) for text in item["completions"])
                    for item, prompt in zip(items, prompts))
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def prompt_token_ids(tokenizer, formatted_prompts, max_prompt_length):
    """Both inference backends see exactly the same left-truncated token IDs."""
    return [tokenizer(text, add_special_tokens=False)["input_ids"][-max_prompt_length:]
            for text in formatted_prompts]


def frozen_prompt_messages(prompt, system_prompt=None):
    """Validate text/chat prompts and apply the same system override as training."""
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("Frozen prompts must not be empty")
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list) and prompt:
        messages = []
        expected_role = "user"
        for index, message in enumerate(prompt):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("Frozen chat messages require exactly role and content")
            role, content = message["role"], message["content"]
            if not isinstance(content, str):
                raise ValueError("Frozen chat message content must be text")
            if role == "system" and index == 0:
                messages.append(dict(message))
                continue
            if role != expected_role:
                raise ValueError("Frozen chats require alternating user/assistant messages, with an optional first system")
            messages.append(dict(message))
            expected_role = "assistant" if role == "user" else "user"
        if messages[-1]["role"] != "user" or not messages[-1]["content"].strip():
            raise ValueError("Frozen chat prompts must end with a nonempty user message")
    else:
        raise ValueError("Frozen prompts must be nonempty text or chat message lists")
    if system_prompt is not None:
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be text or null")
        system = {"role": "system", "content": system_prompt}
        if messages[0]["role"] == "system":
            messages[0] = system
        else:
            messages.insert(0, system)
    return messages


def frozen_prompt_text(prompt, system_prompt=None):
    """Human-readable judge input; retain all turns for conversational prompts."""
    if isinstance(prompt, str):
        return prompt
    return json.dumps(frozen_prompt_messages(prompt, system_prompt), ensure_ascii=False, indent=2)


def load_frozen_prompts(path, num_prompts):
    """Read an immutable ordered held-out sample shared by both candidate models."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != 1:
        raise ValueError("Frozen prompts require version=1")
    prompts, prompt_ids = payload.get("prompts"), payload.get("prompt_ids")
    if not isinstance(prompts, list) or not prompts or len(prompts) != num_prompts:
        raise ValueError("--num-prompts must exactly match the frozen prompts file length")
    if not isinstance(prompt_ids, list) or len(prompt_ids) != len(prompts):
        raise ValueError("Frozen prompt_ids must match prompts in length")
    if any(not isinstance(value, str) or not value.strip() for value in prompt_ids):
        raise ValueError("Frozen prompt_ids must be nonempty strings")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Frozen prompt_ids must be unique")
    if payload.get("prompts_sha256") != digest(prompts):
        raise ValueError("Frozen prompts_sha256 integrity check failed")
    for prompt in prompts:
        frozen_prompt_messages(prompt)
    if len({digest(prompt) for prompt in prompts}) != len(prompts):
        raise ValueError("Frozen prompts must be unique for prompt-level bootstrap")
    identity = {
        "version": 1,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "prompts_sha256": payload["prompts_sha256"],
        "prompt_ids": prompt_ids,
    }
    return prompts, identity


def combine_order_judgments(order_results):
    """Reduce two opposite-order judgments to one prompt cluster.

    Scores average wins=1, ties=0.5, losses=0 across presentation orders. A
    disagreement remains distinct from an explicit tie. Any missing verdict
    invalidates that endpoint for this prompt instead of becoming a tie.
    """
    if len(order_results) != 2 or {result.get("order_swapped") for result in order_results} != {False, True}:
        raise ValueError("Both opposite presentation orders are required")
    combined = {
        "order_judgments": order_results,
        "prompt": order_results[0]["prompt"],
        "model1_completion": order_results[0]["model1_completion"],
        "model2_completion": order_results[0]["model2_completion"],
        "source_index": order_results[0]["source_index"],
    }
    score = {"model1": 1.0, "model2": 0.0, "tie": 0.5}
    for endpoint in ("overall", "survey"):
        winners = [result.get(f"{endpoint}_winner") for result in order_results]
        valid = all(winner in score for winner in winners)
        valid = valid and not any(result.get(f"{endpoint}_parsing_failed", endpoint == "overall")
                                  for result in order_results)
        if not valid:
            outcome, model1_score = "failed", None
        else:
            outcome = winners[0] if winners[0] == winners[1] else "order_disagreement"
            model1_score = sum(score[winner] for winner in winners) / 2
        if not valid:
            order_outcome = "invalid"
        elif winners[0] == winners[1]:
            order_outcome = "stable_tie" if winners[0] == "tie" else f"stable_{winners[0]}_win"
        elif "tie" in winners:
            order_outcome = "tie_win_change"
        else:
            order_outcome = "pure_reversal"
        combined[f"{endpoint}_order_outcome"] = order_outcome
        combined[f"{endpoint}_winner"] = outcome if valid else None
        combined[f"{endpoint}_parsing_failed"] = not valid
        combined[f"{endpoint}_model1_score"] = model1_score
        combined[f"{endpoint}_model2_score"] = 1 - model1_score if valid else None
    return combined


def order_diagnostics(results, endpoint):
    """Descriptive order sensitivity and displayed-position choices, not causality.

    Cluster categories require both valid orders. Position statistics use each
    valid order judgment, including a valid half of an invalid cluster; their
    denominator is reported separately and is never used for the primary score.
    """
    categories = ("stable_model1_win", "stable_model2_win", "stable_tie", "pure_reversal", "tie_win_change", "invalid")
    counts = dict.fromkeys(categories, 0)
    position = {"A_wins": 0, "B_wins": 0, "ties": 0, "invalid_judgments": 0}
    always_a = always_b = paired = 0
    for result in results:
        orders = result.get("order_judgments", [result])
        if len(orders) == 2:
            paired += 1
            # Recompute rather than trusting stale categories in saved reports.
            combined = combine_order_judgments(orders)
            counts[combined[f"{endpoint}_order_outcome"]] += 1
        labels = []
        for order in orders:
            winner = order.get(f"{endpoint}_winner")
            if (winner not in ("model1", "model2", "tie")
                    or order.get(f"{endpoint}_parsing_failed", False)
                    or not isinstance(order.get("order_swapped"), bool)):
                position["invalid_judgments"] += 1
                labels.append(None)
            elif winner == "tie":
                position["ties"] += 1
                labels.append("tie")
            else:
                label = "A" if (winner == "model1") != order["order_swapped"] else "B"
                position[f"{label}_wins"] += 1
                labels.append(label)
        if len(orders) == 2:
            always_a += labels == ["A", "A"]
            always_b += labels == ["B", "B"]
    valid_calls = position["A_wins"] + position["B_wins"] + position["ties"]
    decisive_calls = position["A_wins"] + position["B_wins"]
    position.update({
        "valid_judgments": valid_calls,
        "decisive_judgments": decisive_calls,
        "A_win_fraction_decisive": position["A_wins"] / decisive_calls if decisive_calls else None,
        "A_mean_score": (position["A_wins"] + 0.5 * position["ties"]) / valid_calls if valid_calls else None,
        "both_orders_choose_A": always_a,
        "both_orders_choose_B": always_b,
    })
    return {
        "paired_prompt_clusters": paired,
        "pair_outcomes": counts,
        "presentation_position": position,
        "interpretation": "Descriptive sensitivity to order. Reversals, tie changes, or position preference do not establish their cause.",
    }
