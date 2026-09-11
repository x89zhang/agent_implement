"""Project guard views of Hermes' native Responses wire format.

Unchanged requests/responses remain intact, including opaque reasoning and replay IDs.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace


def guard_messages(request):
    messages = []
    if request.get("instructions"):
        messages.append({"role": "system", "content": request["instructions"]})
    items = request.get("input", [])
    if isinstance(items, str):
        return messages + [{"role": "user", "content": items}]
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            content = copy.deepcopy(item.get("content", ""))
            if isinstance(content, list):
                for part in content:
                    if part.get("type") in {"input_text", "output_text"}:
                        part["type"] = "text"
            messages.append({"role": item.get("role", "user"), "content": content})
        elif kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item["call_id"],
                            "type": "function",
                            "function": {
                                "name": item["name"],
                                "arguments": item["arguments"],
                            },
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": item["output"],
                }
            )
        elif kind == "reasoning":
            summary = item.get("summary") or []
            text = "\n".join(p.get("text", "") for p in summary)
            if text:
                messages.append({"role": "assistant", "content": text})
    return messages


def rewrite_request(request, original_messages, messages):
    if messages == original_messages:
        return request
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    # A rewritten safety view supersedes opaque replay data rather than retaining
    # hidden, potentially rejected instructions from the original history.
    updated = {**request}
    updated["instructions"] = "\n\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") == "system"
    )
    updated["input"] = _chat_messages_to_responses_input(
        [m for m in messages if m.get("role") != "system"]
    )
    updated.pop("previous_response_id", None)
    return updated


def response_view(agent, response):
    normalized = agent._get_transport().normalize_response(response)
    message = SimpleNamespace(
        content=normalized.content,
        tool_calls=[
            SimpleNamespace(
                id=t.id, function=SimpleNamespace(name=t.name, arguments=t.arguments)
            )
            for t in normalized.tool_calls or []
        ],
    )
    return SimpleNamespace(message=message, finish_reason=normalized.finish_reason)


def rewrite_response(response, original, message):
    def signature(value):
        return (
            value.content or "",
            [
                (t.id, t.function.name, json.loads(t.function.arguments))
                for t in value.tool_calls or []
            ],
        )

    if signature(original) == signature(message):
        return response
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

    output = []
    if message.content:
        output.append(
            ResponseOutputMessage(
                id="msg_project_defense",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    {"type": "output_text", "text": message.content, "annotations": []}
                ],
            )
        )
    originals = [
        item
        for item in response.output
        if getattr(item, "type", None) == "function_call"
    ]
    original_ids = [t.id for t in original.tool_calls or []]
    for call in message.tool_calls or []:
        index = original_ids.index(call.id)
        item = originals[index]
        output.append(
            ResponseFunctionToolCall(
                type="function_call",
                id=item.id,
                call_id=item.call_id,
                name=call.function.name,
                arguments=call.function.arguments,
                status="completed",
            )
        )
    response = copy.copy(response)
    response.output = output
    return response
