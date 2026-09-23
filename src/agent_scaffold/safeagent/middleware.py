"""SafeAgent Core's session-based MCP decisions at agent lifecycle boundaries."""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import yaml

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision


class SafeAgentClient:
    def __init__(self, url: str, timeout: float, api_key_env: str) -> None:
        self.url, self.timeout, self.api_key_env = url, timeout, api_key_env

    async def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            from fastmcp import Client
            from fastmcp.client.transports import StreamableHttpTransport
        except ImportError as exc:
            raise RuntimeError("SafeAgent client requires fastmcp; install requirements-safeagent.txt") from exc
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        headers = {"Authorization": f"Bearer {key}"} if key else None
        async with Client(StreamableHttpTransport(self.url, headers=headers), timeout=self.timeout) as client:
            result = await client.call_tool(tool, arguments, timeout=self.timeout)
        data = getattr(result, "data", None) or getattr(result, "structuredContent", None)
        if not isinstance(data, dict):
            raise ValueError(f"SafeAgent {tool} returned no structured mapping")
        return data

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # The project lifecycle is synchronous; Hermes may invoke it from an event loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._call(tool, arguments))
        output: list[Any] = []

        def run() -> None:
            try:
                output.append(asyncio.run(self._call(tool, arguments)))
            except Exception as exc:
                output.append(exc)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join()
        if isinstance(output[0], Exception):
            raise output[0]
        return output[0]


class SafeAgentMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.settings = cfg.safeagent
        self.client = client or SafeAgentClient(
            self.settings.mcp_url, self.settings.timeout_seconds, self.settings.api_key_env
        )
        self.developer_cfg = self._read_config(self.settings.developer_config_path)
        self.runtime_cfg = self._read_config(self.settings.runtime_config_path)

    def _read_config(self, name: str) -> dict[str, Any]:
        path = Path(name)
        if not path.is_absolute():
            path = Path(self.cfg.config_dir) / path
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"SafeAgent config must be a mapping: {path}")
        return value

    def _judge(self, state: dict[str, Any], hook: str, observation: dict[str, Any]) -> dict[str, Any]:
        session = state.setdefault("_safeagent_session_id", str(uuid.uuid4()))
        try:
            if not state.get("_safeagent_registered"):
                registration = self.client.call("safeagent_register_session", {
                    "session_id": session, "runtime_cfg": self.runtime_cfg,
                    "dev_cfg": self.developer_cfg,
                })
                if registration.get("ok") is not True:
                    raise RuntimeError(f"session registration failed: {registration.get('error')}")
                state["_safeagent_registered"] = True
            response = self.client.call("safeagent_step", {
                "session_id": session,
                "core_request": {"hook": hook, "observation": observation},
            })
            if not isinstance(response, dict) or response.get("error"):
                raise RuntimeError(f"invalid SafeAgent response: {response.get('error') if isinstance(response, dict) else response}")
            action = str(response.get("action") or "").upper()
            allowed = {
                "before_agent": {"APPROVE", "OVERRIDE", "REJECT"},
                "after_agent": {"APPROVE", "OVERRIDE", "REJECT"},
                "after_model": {"APPROVE", "REPLAN", "REJECT"},
                "before_model": {"APPROVE", "OVERRIDE", "ROLLBACK", "TERMINATE", "REJECT"},
                "tool_wrapper": {"CALL_ALLOW", "CALL_REWRITE", "CALL_BLOCK", "CALL_JIT_APPROVAL"},
            }[hook]
            if action not in allowed:
                raise ValueError(f"unexpected SafeAgent action: {action!r}")
            override = response.get("override")
            if action in {"OVERRIDE", "CALL_REWRITE"}:
                if action == "CALL_REWRITE" and not isinstance(override, dict):
                    raise ValueError("CALL_REWRITE requires a mapping override")
                if action == "OVERRIDE" and (not isinstance(override, str) or not override.strip()):
                    raise ValueError("OVERRIDE requires non-empty text")
            event = {"hook": hook, "action": action, "override": override,
                     "violations": response.get("violations") or [], "error": ""}
        except Exception as exc:
            event = {"hook": hook, "action": "ERROR", "override": None,
                     "violations": [], "error": f"{type(exc).__name__}: {exc}"}
        event["blocked"] = (
            self.settings.mode == "block"
            and (event["action"] != "ERROR" or self.settings.fail_closed)
            and event["action"] in {"ERROR", "REJECT", "REPLAN", "ROLLBACK", "TERMINATE", "CALL_BLOCK", "CALL_JIT_APPROVAL"}
        )
        state["_last_safeagent_decision"] = event
        harness = state.setdefault("harness", {}).setdefault("safeagent", {})
        harness.update({"enabled": True, "mode": self.settings.mode,
                        "status": "error" if event["action"] == "ERROR" else "active",
                        "event_count": harness.get("event_count", 0) + 1,
                        "last_action": event["action"]})
        return event

    def guard_model_input(self, state: dict[str, Any], messages: list[dict[str, Any]]) -> ModelDecision:
        if state.get("_safeagent_before_agent_done"):
            return ModelDecision(messages=messages)
        user = next((m for m in reversed(messages) if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip()), None)
        content = (user or {}).get("content") or str(state.get("_safeagent_user_request") or self.cfg.agent.task or "")
        if not content:
            return ModelDecision(messages=messages)
        event = self._judge(state, "before_agent", {"role": "user", "content": content})
        if event["action"] != "ERROR":
            state["_safeagent_before_agent_done"] = True
        if event["blocked"]:
            return ModelDecision(False, "SafeAgent rejected user input", messages=messages,
                                 content="Request blocked by SafeAgent Core.", terminate=True,
                                 decision_type="safeagent_reject")
        if self.settings.mode == "block" and event["action"] == "OVERRIDE" and isinstance(event["override"], str) and event["override"].strip() and user:
            updated = [dict(m) for m in messages]
            updated[messages.index(user)]["content"] = event["override"]
            return ModelDecision(messages=updated, decision_type="safeagent_override")
        return ModelDecision(messages=messages)

    def guard_model_output(self, state: dict[str, Any], content: str, tool_call: Any) -> ModelDecision:
        if tool_call is None:
            if not content.strip():
                return ModelDecision(content=content, tool_call=None)
            event = self._judge(state, "after_agent", {"role": "assistant", "content": content})
            if event["blocked"]:
                return ModelDecision(False, "SafeAgent rejected final output", content="Response withheld by SafeAgent Core.", tool_call=None, terminate=True, decision_type="safeagent_reject")
            if self.settings.mode == "block" and event["action"] == "OVERRIDE" and isinstance(event["override"], str):
                return ModelDecision(content=event["override"], tool_call=None, decision_type="safeagent_override")
            return ModelDecision(content=content, tool_call=None)
        name, arguments = tool_call
        user = str(state.get("_safeagent_user_request") or self.cfg.agent.task or "")
        event = self._judge(state, "after_model", {"role": "assistant", "content": content,
                            "tool_calls": [{"name": name, "args": arguments}], "last_user": user or "(no user request available)"})
        if event["blocked"]:
            if event["action"] == "REPLAN":
                return ModelDecision(content="Tool call withheld by SafeAgent Core.", tool_call=None, retry=True,
                                     feedback="SafeAgent requested a safer plan; do not repeat the rejected action.", decision_type="safeagent_replan")
            return ModelDecision(False, "SafeAgent rejected proposed tool call", content="Tool call withheld by SafeAgent Core.", tool_call=None, terminate=True, decision_type="safeagent_reject")
        return ModelDecision(content=content, tool_call=tool_call)

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        description = next((t.description for t in self.cfg.tools if t.name == name), "")
        event = self._judge(state, "tool_wrapper", {"name": name, "args": payload, "description": description})
        if event["blocked"]:
            state["_safeagent_tool_blocked"] = True
            return ToolDecision(False, f"SafeAgent {event['action']} blocked tool {name}", replacement_result="Tool blocked by SafeAgent Core.", decision_type="safeagent_block")
        if self.settings.mode == "block" and event["action"] == "CALL_REWRITE" and isinstance(event["override"], dict):
            return ToolDecision(arguments=event["override"], decision_type="safeagent_rewrite")
        return ToolDecision()

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> ResultDecision:
        if state.pop("_safeagent_tool_blocked", False):
            return ResultDecision(result=result)
        event = self._judge(state, "before_model", {"role": "tool", "name": name,
                            "tool_call_id": str(uuid.uuid4()), "content": str(result)})
        if event["blocked"]:
            return ResultDecision(False, f"SafeAgent {event['action']} rejected tool output",
                                  result="Tool output withheld by SafeAgent Core.", decision_type="safeagent_block")
        if self.settings.mode == "block" and event["action"] == "OVERRIDE" and isinstance(event["override"], str):
            return ResultDecision(result=event["override"], decision_type="safeagent_override")
        return ResultDecision(result=result)
