from __future__ import annotations

import importlib
import inspect
from typing import Any, get_type_hints

from pydantic import TypeAdapter

from ..config import AppConfig


def tool_definitions_from_config(cfg: AppConfig) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for tool in cfg.tools:
        properties: dict[str, Any] = {}
        required: list[str] = []
        try:
            module_name, attribute = tool.import_path.split(":", 1)
            function = getattr(importlib.import_module(module_name), attribute)
            signature = inspect.signature(function)
            type_hints = get_type_hints(function)
            for name, parameter in signature.parameters.items():
                if parameter.kind in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }:
                    continue
                annotation = (
                    type_hints[name]
                    if name in type_hints
                    else Any
                )
                properties[name] = TypeAdapter(annotation).json_schema()
                if parameter.default is inspect.Parameter.empty:
                    required.append(name)
        except Exception:
            # Tool enforcement remains useful as an allowlist when reflection fails.
            properties = {}
            required = []
        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        definitions.append(
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": schema,
            }
        )
    return definitions


def normalize_tool_definitions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        properties = schema.get("properties", schema) if isinstance(schema, dict) else {}
        normalized.append(
            {
                "name": str(tool.get("name") or ""),
                "description": str(tool.get("description") or ""),
                "args": properties if isinstance(properties, dict) else {},
            }
        )
    return [tool for tool in normalized if tool["name"]]
