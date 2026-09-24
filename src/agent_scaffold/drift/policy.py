"""DRIFT trajectory and parameter checks, independent of the AgentDojo runner."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


UPSTREAM_REVISION = "6fd3df4763fad77398c3c273c2a4fe14dd50f7ea"


def _tag(text: str, name: str) -> str:
    match = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"DRIFT response has no <{name}> section")
    return match.group(1).strip()


def _list_value(text: str) -> list[Any]:
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(value, list):
            return value
    raise ValueError("DRIFT response must contain a list")


def parse_constraints(
    text: str, tool_names: set[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    trajectory = _list_value(_tag(text, "function_trajectory"))
    checklist = _list_value(_tag(text, "parameter_checklist"))
    return validate_constraints(trajectory, checklist, tool_names)


def validate_constraints(
    trajectory: list[Any], checklist: list[Any], tool_names: set[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    if not all(isinstance(name, str) and name in tool_names for name in trajectory):
        raise ValueError("DRIFT trajectory must contain only available tools")
    normalized: list[dict[str, Any]] = []
    if len(checklist) != len(trajectory):
        raise ValueError("DRIFT parameter checklist must match trajectory length")
    for index, name in enumerate(trajectory):
        item = checklist[index]
        if not isinstance(item, dict) or item.get("name") != name:
            raise ValueError("DRIFT checklist tool names must follow the trajectory")
        required = item.get("required parameters", item.get("required_parameters"))
        conditions = item.get("conditions")
        if required is not None and not isinstance(required, dict):
            raise ValueError("DRIFT required parameters must be an object or null")
        if conditions is not None and not isinstance(conditions, dict):
            raise ValueError("DRIFT conditions must be an object or null")
        normalized.append(
            {"name": name, "required parameters": required, "conditions": conditions}
        )
    return list(trajectory), normalized


def check_action(
    name: str,
    arguments: dict[str, Any],
    trajectory: list[str],
    checklist: list[dict[str, Any]],
    completed: list[str],
) -> str:
    position = len(completed)
    if position >= len(trajectory):
        return f"DRIFT trajectory is complete; unexpected tool {name}"
    expected = trajectory[position]
    if name != expected:
        return f"DRIFT trajectory expects {expected} at step {position + 1}, not {name}"
    required = (
        checklist[position].get("required parameters")
        if position < len(checklist)
        else None
    )
    if not required:
        return ""
    for key, expected_value in required.items():
        if expected_value is None:
            continue
        if key not in arguments:
            return f"DRIFT checklist requires argument {key} for {name}"
        # Upstream checks explicitly known values; unknown or tool-derived values
        # have no fixed value and are therefore not compared here.
        if isinstance(expected_value, str) and re.search(r"\{[^{}]*\}", expected_value):
            continue
        if arguments[key] != expected_value:
            return f"DRIFT checklist expected {key}={expected_value!r} for {name}"
    return ""


def parse_detected_instructions(text: str) -> list[str]:
    values = _list_value(_tag(text, "detected_instructions"))
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("DRIFT detected instructions must be nonempty strings")
    return values


def mask_instructions(content: str, instructions: list[str]) -> tuple[str, list[str]]:
    masked = content
    removed: list[str] = []
    for instruction in instructions:
        words = instruction.split()
        if not words:
            continue
        pattern = r"\s+".join(re.escape(word) for word in words)
        updated, count = re.subn(pattern, " ", masked, flags=re.IGNORECASE)
        if count:
            masked = updated
            removed.append(instruction)
    return masked.strip(), removed
