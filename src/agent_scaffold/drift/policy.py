"""DRIFT trajectory and parameter checks, independent of the AgentDojo runner.

Parsing mirrors DRIFTLLM.py: malformed model output degrades to an empty plan,
an unparsable checklist, or no detected instructions, never to an exception.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from typing import Any


UPSTREAM_REVISION = "6fd3df4763fad77398c3c273c2a4fe14dd50f7ea"


def parse_constraints(completion: str) -> tuple[list[str], str]:
    """Return the trajectory and raw checklist string (DRIFTLLM.py:339-378)."""
    trajectory: list[str] = []
    checklist = "None"
    if "<function_trajectory>" in completion:
        match = re.search(r"<Traj-1>(\[.*?\])</Traj-1>", completion, re.DOTALL)
        if not match:
            match = re.search(
                r"<function_trajectory>(.*?)</function_trajectory>",
                completion,
                re.DOTALL,
            )
        if match:
            trajectory = [
                func.strip()
                for func in match.group(1).strip().strip("[]").split(",")
            ]
    if "<parameter_checklist>" in completion:
        match = re.search(
            r"<parameter_checklist>(.*?)</parameter_checklist>", completion, re.DOTALL
        )
        if match:
            checklist = match.group(1)
    return trajectory, checklist


def repair_json(text: str) -> str:
    """Use upstream's json_repair when installed, else a minimal equivalent."""
    try:
        from json_repair import repair_json as upstream_repair
    except ImportError:
        upstream_repair = None
    if upstream_repair is not None:
        return upstream_repair(text)
    candidate = re.sub(r"^\s*```(?:json)?|```\s*$", "", str(text)).strip()
    starts = [i for i in (candidate.find("["), candidate.find("{")) if i >= 0]
    if starts:
        start = min(starts)
        end = max(candidate.rfind("]"), candidate.rfind("}"))
        candidate = candidate[start : end + 1] if end > start else candidate[start:]
    try:
        return json.dumps(json.loads(candidate))
    except ValueError:
        pass
    python = re.sub(r"\bnull\b", "None", candidate)
    python = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", python))
    try:
        return json.dumps(ast.literal_eval(python))
    except (ValueError, SyntaxError, TypeError):
        return ""


def insert_checklist_step(checklist: str, position: int, name: str) -> str:
    """Extend the checklist for an accepted deviation (DRIFTLLM.py:462-467, 496-499)."""
    try:
        extended: Any = json.loads(checklist)
        extended.insert(
            position, {"name": name, "required parameters": None, "conditions": None}
        )
    except Exception:
        extended = checklist
    try:
        return json.dumps(extended)
    except Exception:
        return extended


def node_check(
    checklist: str, achieved_count: int, calls: list[tuple[str, dict[str, Any]]]
) -> tuple[bool, str]:
    """Check task-known parameter values (DRIFTLLM.py:288-337).

    ``achieved_count`` already includes the calls under validation, as
    upstream's achieved trajectory does when this check runs.
    """
    try:
        items = json.loads(checklist)
    except Exception:
        return True, ""
    try:
        items = items[achieved_count - 1 :]
        for index, (name, arguments) in enumerate(calls):
            if len(items) <= index:
                return True, ""
            target = items[index]
            if name != target["name"]:
                return False, "The function name does not align with checklist."
            if target["required parameters"] is None or arguments is None:
                return True, ""
            for key, value in target["required parameters"].items():
                if value is None:
                    continue
                if re.search(r"\{[^{}]*\}", str(value)):
                    continue
                if key not in arguments:
                    return False, (
                        f"The argment of the checklist's key of '{key}' is not met "
                        f"in this function {name}."
                    )
                if str(arguments[key]) not in str(value) and str(value) not in str(
                    arguments[key]
                ):
                    return False, (
                        f"The argment of the function {name}'s '{key}' value of "
                        f"{arguments[key]} does not align with the value of '{value}' "
                        "in checklist."
                    )
        return True, ""
    except Exception:
        # Upstream wraps node_check in a bare except and accepts (DRIFTLLM.py:539-542).
        return True, ""


def parse_detected_instructions(response: str) -> list[Any] | None:
    """Return detected spans, or None when the response has no tag (DRIFTLLM.py:384-397)."""
    if "<detected_instructions>" not in response:
        return None
    match = re.search(
        r"<detected_instructions>(.*?)</detected_instructions>", response, re.DOTALL
    )
    content = match.group(1).strip() if match else "[]"
    try:
        values = ast.literal_eval(content)
    except Exception:
        return []
    return values if isinstance(values, list) else []


def remove_sentence(p: Any, t: Any) -> str:
    """Upstream mask function, verbatim (DRIFTLLM.py:410-419)."""
    if type(t) != str:
        t = ""

    words = t.split()
    escaped_words = [re.escape(word) for word in words]
    pattern = r'[\s\\]+'.join(escaped_words)

    pattern = r'\s*' + pattern + r'\s*'
    return re.sub(str(pattern), ' ', str(p), flags=re.DOTALL).strip()


def injection_isolate(response: str, content: str) -> tuple[bool, str]:
    """Mask detected spans; False asks for another detection (DRIFTLLM.py:380-438)."""
    replace_list = parse_detected_instructions(response)
    if replace_list is None:
        return False, content
    if not replace_list:
        return True, content
    length = len(content)
    masked = copy.copy(content)
    for item in replace_list:
        masked = remove_sentence(masked, item)
    if len(masked) == length:
        for item in replace_list:
            masked = remove_sentence(masked, item)
    return len(masked) != length, masked
