"""Small bridge for the vendored AGrail detectors.

The extraction functions preserve DAS/utils.py behavior. The LLM call uses
this project's configured adapter through a per-call context variable.
"""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from typing import Callable

CHAT: ContextVar[Callable[[str], tuple[str, int]]] = ContextVar("agrail_chat")


def get_response_from_openai(prompt: str, model_name: str = "gpt-4o") -> tuple[str, int]:
    return CHAT.get()(prompt)


def detect_python_error(log: str) -> bool:
    return any(item in str(log) for item in (
        "Traceback (most recent call last):", "Error", "Exception", "SyntaxError"
    ))


def extract_json_content(text: str):
    matches = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1].strip())
    except json.JSONDecodeError:
        return None


def extract_content(text: str, content: str, n: int = -1):
    matches = re.findall(r'```' + re.escape(content) + r'\s*(.*?)\s*```',
                         text, re.DOTALL)
    return matches[n].strip() if matches else None


def capture_bool_from_string(log_str: str):
    match = re.search(r'(True|False)(?!.*(True|False))', log_str)
    return match.group(0) if match else None
