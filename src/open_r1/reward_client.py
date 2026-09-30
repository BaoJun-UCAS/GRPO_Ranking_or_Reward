"""Standard-library HTTP client for the external QRM service.

Keep this module independent of training and mathematical reward dependencies so
service contracts can be tested without installing Torch or loading any model.
"""

import json
import math
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def get_remote_qrm_reward(server_url: str, timeout: int) -> Callable:
    """Score chat messages or raw text with the QRM server's chat protocol."""

    if not server_url or timeout <= 0:
        raise ValueError("qrm_server requires reward_server_url and a positive reward_server_timeout")
    endpoint = f"{server_url.rstrip('/')}/score/"

    def qrm_server_reward(prompts, completions, **kwargs) -> list[float]:
        if len(prompts) != len(completions):
            raise ValueError("QRM request has different prompt and completion counts")
        messages = []
        for prompt, completion in zip(prompts, completions):
            if isinstance(prompt, str) and isinstance(completion, str):
                messages.append([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ])
            elif isinstance(prompt, list) and isinstance(completion, list):
                messages.append(prompt + completion)
            else:
                raise TypeError("qrm_server requires matching text or conversational prompt/completion pairs")

        request = Request(
            endpoint,
            data=json.dumps({"messages": messages}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"QRM server returned HTTP {error.code}: {detail}") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError(f"QRM server returned invalid JSON: {error}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"QRM server request failed: {error}") from error

        rewards = result.get("rewards") if isinstance(result, dict) else None
        if not isinstance(rewards, list) or len(rewards) != len(messages):
            raise RuntimeError(
                f"QRM server returned {len(rewards) if isinstance(rewards, list) else 'invalid'} "
                f"rewards for {len(messages)} inputs"
            )
        try:
            converted = [float(reward) for reward in rewards]
        except (TypeError, ValueError) as error:
            raise RuntimeError("QRM server returned a non-numeric reward") from error
        if not all(math.isfinite(reward) for reward in converted):
            raise RuntimeError("QRM server returned a non-finite reward")
        return converted

    qrm_server_reward.__name__ = "qrm_server"
    return qrm_server_reward
