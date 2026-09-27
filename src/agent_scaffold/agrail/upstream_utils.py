"""Upstream DAS helpers used by the AGrail adapter and vendored detectors.

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


# DAS/utils.py:19-24
def format_dic_to_stry(dic):
    stry = "{\n"
    for key, value in dic.items():
        stry += f"    {key}: {value},\n"
    stry += "}"
    return stry


# DAS/utils.py:109-121
def extract_step_back_content(text):
    natural_language_pattern = r"Paraphrased Natural Language:\s*(.+)"
    tool_command_language_pattern = r"Paraphrased Tool Command Language:\s*(.+)"

    natural_language_match = re.search(natural_language_pattern, text)
    tool_command_language_match = re.search(tool_command_language_pattern, text)

    natural_language = natural_language_match.group(1).strip() if natural_language_match else None
    tool_command_language = tool_command_language_match.group(1).strip() if tool_command_language_match else None

    template = f"""Natural Language:{natural_language[:]}, Tool Command Language:{tool_command_language[:]}"""
    template = template.replace("#", "")
    return template


# DAS/guardrail.py:17-33
def match_in_memory_bool(text):
    match = re.search(r'\*?\*?In Memory:\*?\*?\s*(?:"(True|False)"|(True|False))?', text)
    if match:
        # Check which group matched and convert it to a boolean
        return match.group(1) == "True" or match.group(2) == "True"
    return None


# DAS/guardrail.py:36-48
def extract_json_from_text(output, index):
    # Use regex to find JSON blocks in the text
    json_pattern = re.compile(r'```json\n(.*?)\n```', re.DOTALL)
    matches = json_pattern.findall(output)

    # Parse each JSON block and return a list of parsed objects
    if index == -1:
        if matches[index][0] == "[":
            matches[index] = matches[index][1:-1]
        matches[index] = matches[index].replace("[", "{").replace("]", "}")

    return json.loads(matches[index])


# DAS/guardrail.py:51-73
def tool_call_from_react(output):
    reason_safety = []
    steps = extract_json_from_text(output, -2)
    tool_dic = {}
    for i in range(len(steps)):
        if steps[i]["Delete"] == "False":
            if steps[i]["Tool Call"] != "False":
                if steps[i]["Tool Call"] not in tool_dic:
                    tool_dic[steps[i]["Tool Call"]] = []
                tool_dic[steps[i]["Tool Call"]].append(steps[i])
            else:
                reason_safety.append(steps[i]["Result"])

    return tool_dic, reason_safety
