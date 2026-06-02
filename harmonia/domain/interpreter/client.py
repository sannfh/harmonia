"""DeepSeek API client: text prompt → MusicParameters."""

import json
import time
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from harmonia.domain.interpreter.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from harmonia.domain.interpreter.schemas import MusicParameters

_MAX_RETRIES = 3
_RETRY_DELAY_S = 1.0


def interpret(prompt: str, api_key: str, base_url: str) -> MusicParameters:
    """Call DeepSeek to convert a free-text prompt into MusicParameters.

    Retries up to ``_MAX_RETRIES`` times on malformed JSON or Pydantic
    validation errors before raising.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = _build_messages(prompt)

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,  # type: ignore[arg-type]
                temperature=0.3,
                max_tokens=256,
            )
            raw = response.choices[0].message.content or ""
            data = json.loads(raw.strip())
            return MusicParameters(**data)
        except (json.JSONDecodeError, ValidationError, KeyError) as exc:
            last_error = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY_S)

    raise ValueError(
        f"DeepSeek returned invalid MusicParameters after {_MAX_RETRIES} attempts"
    ) from last_error


def _build_messages(user_prompt: str) -> list[Any]:
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs
