"""Generate a conservative ROPE floor for tools absent from audited tables.

The model proposes markers, but may not omit inventory tools or invent arguments.
Unknown mutating tools without an inspectable schema are blocked outright.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from typing import Any, Callable

from ._upstream.markers import str_to_rule

_READ_PREFIXES = ("get_", "list_", "search_", "read_", "view_", "fetch_", "find_", "lookup_", "inspect_", "query_")
_MUTATION_WORDS = re.compile(r"\b(write|create|update|delete|remove|send|post|transfer|pay|refund|checkout|execute|run|install|upload|share|invite|add|set|change|submit|purchase|order)\b", re.I)
_MARKERS = {"PROMPT", "SOURCED", "EXPLICIT"}


def inventory_from_config(cfg: Any, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Use the bridge's JSON schemas, or reflect native configured functions."""
    supplied = state.get("_rope_tools")
    if isinstance(supplied, list):
        return [_normalize_tool(tool) for tool in supplied]
    tools: list[dict[str, Any]] = []
    for item in cfg.tools:
        names: list[str] = []
        complete = False
        try:
            module_name, function_name = item.import_path.split(":", 1)
            function = getattr(importlib.import_module(module_name), function_name)
            signature = inspect.signature(function)
            names = [name for name, parameter in signature.parameters.items()
                     if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
            complete = True
        except (ValueError, AttributeError, ImportError, TypeError):
            pass
        tools.append({"name": item.name, "description": item.description,
                      "arguments": names, "schema_complete": complete})
    return tools


def _normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("inputSchema") or tool.get("input_schema") or tool.get("parameters")
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            arguments = list(properties)
            complete = True
        else:
            arguments, complete = [], False
    else:
        arguments, complete = [], False
    return {"name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "arguments": arguments, "schema_complete": complete}


def _read_only_candidate(tool: dict[str, Any]) -> bool:
    name = tool["name"].lower()
    return name.startswith(_READ_PREFIXES) and not _MUTATION_WORDS.search(
        name.replace("_", " ") + " " + tool["description"]
    )


def _parse_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("ROPE floor generator returned no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("ROPE floor generator must return an object")
    return value


def generate_floor(
    tools: list[dict[str, Any]], complete: Callable[[str, str], str]
) -> tuple[dict[str, dict[str, str]], list[str], list[str]]:
    """Return (marker floor, blocked tools, read-only tools) for unknown tools.

    The model must classify every tool. Any omitted argument on a mutating tool
    is protected with PROMPT; an opaque mutating tool is blocked completely.
    """
    if not tools:
        return {}, [], []
    names = [tool["name"] for tool in tools]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("ROPE tool inventory has missing or duplicate names")
    system = (
        "You propose a global sensitive-argument table for ROPE. Read ONLY the trusted tool "
        "definitions. Return JSON exactly as {\"tools\":[{\"name\":string,\"read_only\":boolean,"
        "\"rules\":{argument:marker}}]}. Include every supplied tool exactly once. "
        "For state-changing tools, mark every argument that can redirect a target, destination, "
        "recipient, credential, amount, or side effect. Use PROMPT for request-only values, "
        "SOURCED for values from a user-named authenticated source, EXPLICIT "
        "for irreversible actions. Use only PROMPT, SOURCED, or EXPLICIT. "
        "Never use FREE, CONST, ONEOF or invented arguments. "
        "When unsure, use PROMPT. A tool is read_only only if it cannot change state."
    )
    reply = _parse_json(complete(system, json.dumps({"tools": tools}, ensure_ascii=False)))
    rows = reply.get("tools")
    if not isinstance(rows, list):
        raise ValueError("ROPE floor generator must return a tools array")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("ROPE floor generator returned an invalid tool entry")
        name = row["name"]
        if name in by_name or name not in names:
            raise ValueError(f"ROPE floor generator returned duplicate or unknown tool {name!r}")
        by_name[name] = row
    if set(by_name) != set(names):
        raise ValueError(f"ROPE floor generator omitted tools: {sorted(set(names) - set(by_name))}")
    floor: dict[str, dict[str, str]] = {}
    blocked: list[str] = []
    read_only: list[str] = []
    for tool in tools:
        name = tool["name"]
        row = by_name[name]
        if not isinstance(row.get("read_only"), bool) or not isinstance(row.get("rules"), dict):
            raise ValueError(f"ROPE floor generator returned invalid classification for {name!r}")
        args = set(tool["arguments"])
        if row["read_only"] and _read_only_candidate(tool):
            read_only.append(name)
            continue
        rules = row["rules"]
        if set(rules) - args:
            raise ValueError(f"ROPE floor generator invented arguments for {name!r}: {sorted(set(rules) - args)}")
        if not tool["schema_complete"] or not args:
            blocked.append(name)
            continue
        checked: dict[str, str] = {}
        for arg in tool["arguments"]:
            marker = rules.get(arg, "PROMPT")
            if not isinstance(marker, str) or marker not in _MARKERS:
                raise ValueError(f"ROPE floor generator returned invalid marker for {name}.{arg}")
            str_to_rule(marker)
            checked[arg] = marker
        floor[name] = checked
    return floor, blocked, read_only
