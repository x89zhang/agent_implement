from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import AppConfig, LLMConfig
from ..llm import LLMAdapter, _extract_usage
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision
from . import projection_generator as generator
from .upstream import (
    OMITTED_MESSAGE,
    STOP_MESSAGE,
    UPSTREAM_REVISION,
    cosine,
    masked_messages,
    transform_tool_call,
)


class MelonMiddleware(Middleware):
    """MELON: masked re-execution and tool-call comparison (Zhu et al., ICML'25).

    After each batch of tool outputs, the agent model is re-run on those outputs
    with the user task replaced by a generic "follow the file's instructions"
    task. Tool calls from the masked run accumulate in a bank. A real tool call
    whose embedding is close to a bank entry does not depend on the user task,
    so it is attributed to an indirect prompt injection.

    Hooks are ``before_tool``/``after_tool`` so the same detector runs in the
    project graphs and behind the Hermes defense bridge. Hook instances are not
    shared between graph nodes, so all detector state lives in ``state``.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.settings = cfg.melon
        self._embedder: Any = None
        self._chat_model: Any = None

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_melon_warning", "")
        return [f"MELON warning: {warning}"] if warning else []

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> ResultDecision:
        state.setdefault("_melon_outputs", []).append({"name": name, "content": str(result)})
        return ResultDecision(result=result)

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        outputs = list(state.get("_melon_outputs") or [])
        if not outputs:
            # Upstream only checks the model turn that follows a tool result.
            return self._record(state, name, payload, allowed=True, source="no_tool_output",
                                reason="no tool output observed yet")
        started = time.time()
        try:
            projection = self._projection(state)
            masked = None
            if int(state.get("_melon_masked_upto", 0)) < len(outputs):
                masked = self._masked_run(state, outputs, projection)
                state["_melon_masked_upto"] = len(outputs)
            text = transform_tool_call(name, payload, projection.get(name))
            bank = state.get("_melon_bank") or []
            best, match = -1.0, None
            if bank:
                embedding = self._embed([text])[0]
                for entry in bank:
                    similarity = cosine(embedding, entry["embedding"])
                    if similarity > best:
                        best, match = similarity, entry["text"]
                    if similarity > self.settings.threshold:
                        break
            injection = best > self.settings.threshold
            reason = (
                f"tool call {text} matches masked-run call {match} (cosine {best:.3f} > "
                f"{self.settings.threshold})" if injection else "no masked-run call matched"
            )
            return self._record(
                state, name, payload, allowed=not injection, source="comparison", reason=reason,
                detail={"tool_call": text, "max_cosine": best if bank else None, "matched": match,
                        "bank_size": len(bank), "masked_run": masked,
                        "latency_ms": int((time.time() - started) * 1000)},
            )
        except Exception as exc:
            return self._record(state, name, payload, allowed=not self.settings.fail_closed,
                                source="error", reason=f"MELON detection failed: {exc}",
                                detail={"latency_ms": int((time.time() - started) * 1000)})

    # Masked re-execution -------------------------------------------------

    def _masked_run(self, state: dict[str, Any], outputs: list[dict[str, Any]], projection: dict[str, list[str]]) -> dict[str, Any]:
        messages = masked_messages(self._system_messages(state), outputs)
        started = time.time()
        response = self._model(state).invoke(_to_langchain(messages))
        calls = [(str(call.get("name") or ""), dict(call.get("args") or {}))
                 for call in (getattr(response, "tool_calls", None) or [])]
        texts = [transform_tool_call(name, args, projection.get(name)) for name, args in calls if name]
        bank = state.setdefault("_melon_bank", [])
        known = {entry["text"] for entry in bank}
        # Upstream adds only unseen calls, so the bank keeps growing across turns.
        new = list(dict.fromkeys(text for text in texts if text not in known))
        if new:
            bank.extend({"text": text, "embedding": vector}
                        for text, vector in zip(new, self._embed(new)))
        usage = _extract_usage(response) or {}
        totals = state.setdefault("harness", {}).setdefault("melon", {}).setdefault("usage", {})
        totals["masked_runs"] = int(totals.get("masked_runs", 0)) + 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[key] = int(totals.get(key, 0)) + int(usage.get(key) or 0)
        return {"tool_output_count": len(outputs), "tool_calls": texts, "added": new,
                "usage": usage, "latency_ms": int((time.time() - started) * 1000)}

    def _system_messages(self, state: dict[str, Any]) -> list[str]:
        found = [str(message.get("content") or "") for message in state.get("messages") or []
                 if message.get("role") == "system" and isinstance(message.get("content"), str)]
        found = [text for text in found if text.strip()]
        if found:
            return found[:1]
        return [self.cfg.agent.system_prompt] if self.cfg.agent.system_prompt.strip() else []

    def _model(self, state: dict[str, Any]) -> Any:
        if self._chat_model is None:
            llm_cfg = self._merged_llm(self.settings.llm)
            if self._transport(llm_cfg) == "responses":
                model = LLMAdapter(llm_cfg).get_lc_chat_model(use_responses_api=True)
            else:
                model = LLMAdapter(llm_cfg).get_lc_chat_model()
            tools = [_function_tool(tool) for tool in self._tool_definitions(state)]
            self._chat_model = model.bind_tools(tools) if tools else model
        return self._chat_model

    def _transport(self, llm_cfg: LLMConfig) -> str:
        if self.settings.transport != "auto":
            return self.settings.transport
        if llm_cfg.provider.lower() != "openai":
            return "chat_completions"
        graph = self.cfg.graph
        hermes = getattr(getattr(self.cfg, "execution", None), "hermes", None)
        if (getattr(graph, "react_protocol", "") == "native_tool_calling"
                or getattr(graph, "openai_transport", "") == "responses"
                or getattr(hermes, "api_mode", "") == "codex_responses"
                or re.match(r"^(gpt-5|o\d)", llm_cfg.model.lower())):
            return "responses"
        return "chat_completions"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is None:
            try:
                from openai import OpenAI  # type: ignore
            except Exception as exc:  # pragma: no cover - runtime import
                raise RuntimeError("Missing dependency: openai") from exc
            settings = self.settings.embedding
            key = settings.api_key or os.environ.get(settings.api_key_env or "OPENAI_API_KEY", "")
            kwargs: dict[str, Any] = {"api_key": key or None}
            if settings.base_url:
                kwargs["base_url"] = settings.base_url
            if settings.request_timeout is not None:
                kwargs["timeout"] = settings.request_timeout
            self._embedder = OpenAI(**kwargs)
        response = self._embedder.embeddings.create(input=texts, model=self.settings.embedding.model)
        rows = sorted(response.data, key=lambda row: row.index)
        return [list(row.embedding) for row in rows]

    # Comparison-argument projection --------------------------------------

    def _projection(self, state: dict[str, Any]) -> dict[str, list[str]]:
        cached = state.get("_melon_projection")
        if isinstance(cached, dict):
            return cached
        inventory = generator.normalize_inventory(self._tool_definitions(state))
        projection: dict[str, list[str]] = {}
        sources: dict[str, str] = {}
        for tool in inventory:
            rule = generator.upstream_projection(tool)
            if rule is not None:
                projection[tool["name"]], sources[tool["name"]] = rule, "upstream"
        pending = [tool for tool in inventory if tool["name"] not in projection
                   and tool["name"] not in self.settings.projections and tool["arguments"]]
        manifest: dict[str, Any] = {"status": "upstream_only", "context_mode": "benign_only"}
        if pending and self.settings.projection_generation == "llm" and self.settings.generator.enabled:
            manifest = self._generate(pending, state)
            for name, args in manifest.pop("projection", {}).items():
                projection[name], sources[name] = args, manifest["status"]
        for name, args in self.settings.projections.items():
            projection[name], sources[name] = list(args), "configured"
        state["_melon_projection"] = projection
        event = {"step": "melon_projection_generate", "timestamp": time.time(),
                 "output": {**manifest, "upstream_revision": UPSTREAM_REVISION,
                            "projection": projection, "sources": sources,
                            "all_arguments": sorted(t["name"] for t in inventory if t["name"] not in projection)}}
        state.setdefault("trace", []).append(event)
        self._write_artifact(state, "melon_projection.json", event)
        return projection

    def _generate(self, tools: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
        settings = self.settings.generator
        llm_cfg = self._merged_llm(settings.llm)
        key = generator.fingerprint(tools, {"provider": llm_cfg.provider, "model": llm_cfg.model,
                                            "temperature": llm_cfg.temperature})
        cache = generator.batch_cache_path()
        hit = generator.load_cached(cache, key, tools)
        if hit is not None:
            return {"status": "cache", "context_mode": settings.context_mode, "projection": hit,
                    "cache_path": str(cache)}
        llm = LLMAdapter(llm_cfg)
        usage: dict[str, int] = {}

        def complete(system: str, user: str) -> str:
            response = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[field] = usage.get(field, 0) + int((response.usage or {}).get(field) or 0)
            return str(response.content)

        try:
            projection, transcript = generator.generate_projection(
                tools, complete, max_attempts=settings.max_attempts)
        except Exception as exc:
            if settings.fail_closed:
                raise
            # Upstream's behavior for an unlisted tool is to compare all arguments.
            return {"status": "fallback_all_arguments", "context_mode": settings.context_mode,
                    "error": str(exc), "usage": usage}
        generator.store_cached(cache, key, projection)
        self._write_artifact(state, "melon_projection_raw.json", transcript)
        return {"status": "llm", "context_mode": settings.context_mode, "projection": projection,
                "usage": usage, "attempts": len(transcript),
                "llm": {"provider": llm_cfg.provider, "model": llm_cfg.model}}

    # Helpers ---------------------------------------------------------------

    def _tool_definitions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        supplied = state.get("_melon_tools")
        if isinstance(supplied, list):
            return copy.deepcopy(supplied)
        from ..progent.tools import tool_definitions_from_config

        return tool_definitions_from_config(self.cfg)

    def _merged_llm(self, override: Any) -> LLMConfig:
        base = self.cfg.llm
        return replace(
            base,
            provider=override.provider or base.provider,
            model=override.model or base.model,
            temperature=override.temperature if override.temperature is not None else base.temperature,
            base_url=override.base_url or base.base_url,
            api_key=override.api_key or base.api_key,
            api_key_env=override.api_key_env or base.api_key_env,
            request_timeout=override.request_timeout if override.request_timeout is not None else base.request_timeout,
        )

    def _record(self, state: dict[str, Any], name: str, payload: dict[str, Any], *, allowed: bool,
                source: str, reason: str, detail: dict[str, Any] | None = None) -> ToolDecision:
        enforced = not allowed and self.settings.mode == "block"
        event = {"phase": "before_tool", "tool": name, "allowed": allowed, "enforced": enforced,
                 "detected": not allowed and source == "comparison", "reason": reason,
                 "source": source, "mode": self.settings.mode, **(detail or {})}
        state["_last_melon_decision"] = event
        events = state.setdefault("melon_events", [])
        events.append(event)
        harness = state.setdefault("harness", {}).setdefault("melon", {})
        harness.update({
            "enabled": True, "mode": self.settings.mode,
            "status": "failed" if source == "error" else "active",
            "threshold": self.settings.threshold, "last_decision": event,
            "event_count": len(events),
            "detections": sum(1 for item in events if item.get("detected")),
            "bank_size": len(state.get("_melon_bank") or []),
        })
        if source != "no_tool_output":
            self._write_artifact(state, "melon_events.json", events)
        if not allowed and self.settings.mode == "warn":
            state["_melon_warning"] = reason
        if not enforced:
            return ToolDecision(True, "")
        # Upstream replaces the turn with a stop message, which ends the agent loop.
        return ToolDecision(False, reason, replacement_result=f"{STOP_MESSAGE} {OMITTED_MESSAGE}",
                            decision_type="melon", terminate=True)

    def _write_artifact(self, state: dict[str, Any], filename: str, value: Any) -> None:
        run_dir = (state.get("_trace_persist") or {}).get("run_dir")
        if not run_dir:
            return
        path = Path(str(run_dir)) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("inputSchema") or tool.get("input_schema") or tool.get("parameters") or {}
    if not isinstance(schema, dict) or not schema:
        schema = {"type": "object", "properties": {}}
    return {"type": "function", "function": {
        "name": str(tool.get("name") or ""),
        "description": str(tool.get("description") or ""),
        "parameters": schema,
    }}


def _to_langchain(messages: list[dict[str, Any]]) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # type: ignore

    systems = [m["content"] for m in messages if m["role"] == "system"]
    converted: list[Any] = [SystemMessage(content="\n\n".join(systems))] if systems else []
    for message in messages:
        role = message["role"]
        if role == "user":
            converted.append(HumanMessage(content=message["content"]))
        elif role == "assistant":
            converted.append(AIMessage(content=message.get("content") or "", tool_calls=[
                {"name": call["name"], "args": call["args"], "id": call["id"], "type": "tool_call"}
                for call in message.get("tool_calls") or []
            ]))
        elif role == "tool":
            converted.append(ToolMessage(content=message["content"], tool_call_id=message["tool_call_id"]))
    return converted
