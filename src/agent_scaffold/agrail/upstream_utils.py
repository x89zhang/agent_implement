"""Upstream DAS helpers used by the AGrail adapter and vendored detectors.

The extraction functions preserve DAS/utils.py behavior, except for the
marked ADAPTATIONs to how current models format the same answers. The LLM call uses
this project's configured adapter through a per-call context variable.
"""

from __future__ import annotations

import ast
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
    # ADAPTATION (reply format): newer models bold the two labels
    # ("**Paraphrased Natural Language:**"); drop the emphasis so the upstream
    # patterns below read the paraphrase instead of the "**" marker.
    text = re.sub(r"\*\*\s*(Paraphrased (?:Natural|Tool Command) Language)\s*(:?)\s*\*\*\s*(:?)",
                  lambda m: m.group(1) + ":", text)
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


# ADAPTATION (reply format): upstream reads only blocks matching
# r'```json\n(.*?)\n```'. gpt-5.x replies also use ```JSON or bare ```
# fences, put the JSON on the fence line, omit the newline before the closing
# fence, or leave a block unfenced, so upstream finds too few blocks and
# raises IndexError. json_blocks() reads those shapes; the block order and
# every later step (index choice, Step 2 bracket rewrite, json.loads) are
# upstream's, so a reply that really lacks a block still raises.
_FENCE = re.compile(r"```([A-Za-z0-9_+.-]*)")


def _loads(block):
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        # The upstream prompts show single-quoted examples; accept that
        # literal form of the same data (never evaluates code).
        try:
            return ast.literal_eval(block.strip())
        except (SyntaxError, ValueError):
            pass
        raise


def _fenced(text):
    """(start, end, info, content) for each paired ``` fence, in order."""
    markers = list(_FENCE.finditer(text))
    blocks = []
    index = 0
    while index < len(markers):
        opening = markers[index]
        closing = markers[index + 1] if index + 1 < len(markers) else None
        end = closing.start() if closing else len(text)
        blocks.append((opening.start(), closing.end() if closing else len(text),
                       opening.group(1).lower(), text[opening.end():end].strip()))
        index += 2
    return blocks


def _balanced(text, start):
    """End offset of the bracketed value starting at text[start], or -1."""
    pairs = {"[": "]", "{": "}"}
    stack = []
    quote = ""
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "]}":
            if not stack or stack.pop() != char:
                return -1
            if not stack:
                return position + 1
    return -1


def _structured(value):
    return isinstance(value, dict) or (
        isinstance(value, list) and bool(value)
        and all(isinstance(item, dict) for item in value))


def _unfenced(text, start, end):
    """(start, content) for top-level JSON objects/arrays of objects in prose."""
    found = []
    position = start
    while position < end:
        if text[position] in "[{":
            stop = _balanced(text, position)
            if 0 < stop <= end:
                candidate = text[position:stop]
                try:
                    if _structured(_loads(candidate)):
                        found.append((position, candidate))
                        position = stop
                        continue
                except (json.JSONDecodeError, SyntaxError, ValueError):
                    pass
        position += 1
    return found


_STEP_KEY = re.compile(r"^\s*step\s*(\d+)\b", re.I)


def _split_steps(blocks):
    """Expand one object holding every step ({"Step 1": ..., "Step 2": ...}).

    gpt-5.x often answers a multi-step format with a single JSON object keyed
    by step instead of one block per step. The step values are exactly the
    blocks upstream expects, so they are re-serialized in step order.
    """
    expanded = []
    for block in blocks:
        try:
            value = _loads(block)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            expanded.append(block)
            continue
        steps = {}
        if isinstance(value, dict) and len(value) > 1:
            for key, item in value.items():
                match = _STEP_KEY.match(str(key))
                if not match:
                    steps = {}
                    break
                steps[int(match.group(1))] = item
        if steps:
            expanded.extend(json.dumps(steps[number], ensure_ascii=False)
                            for number in sorted(steps))
        else:
            expanded.append(block)
    return expanded


def json_blocks(output, needed=1):
    """Upstream's JSON block list, read tolerantly (see the note above)."""
    return _split_steps(_json_blocks(output, needed))


def _json_blocks(output, needed=1):
    fenced = [(start, end, info, content) for start, end, info, content in _fenced(output)
              if info == "json" or (info == "" and content[:1] in ("[", "{"))]
    blocks = [content for _, _, _, content in fenced]
    if len(blocks) >= needed:
        return blocks
    # Too few fenced blocks: add unfenced ones between the fences.
    ordered = [(start, content) for start, _, _, content in fenced]
    cursor = 0
    for start, end, _, _ in _fenced(output) + [(len(output), len(output), "", "")]:
        ordered.extend(_unfenced(output, cursor, start))
        cursor = end
    return [content for _, content in sorted(ordered)]


# DAS/guardrail.py:36-48
def extract_json_from_text(output, index, needed=None):
    # Use regex to find JSON blocks in the text. ``needed`` is how many
    # blocks the reply format has (the Executor's Step 1 and Step 2: 2).
    if needed is None:
        needed = index + 1 if index >= 0 else -index
    matches = json_blocks(output, needed)

    # Parse each JSON block and return a list of parsed objects
    if index == -1:
        if matches[index][0] == "[":
            matches[index] = matches[index][1:-1]
        matches[index] = matches[index].replace("[", "{").replace("]", "}")

    return _loads(matches[index])


# DAS/guardrail.py:51-73
def _upstream_flags(step):
    # Model-format adaptation: upstream's prompt shows "True"/"False" strings;
    # current models may emit JSON booleans, which upstream's string tests
    # would silently read as "not False" and drop every check.
    if not isinstance(step, dict):
        return step
    return {
        key: (str(value) if isinstance(value, bool) and key in ("Delete", "Tool Call") else value)
        for key, value in step.items()
    }


def tool_call_from_react(output):
    reason_safety = []
    steps = [_upstream_flags(step) for step in extract_json_from_text(output, -2)]
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
