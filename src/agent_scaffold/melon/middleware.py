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
from ..tool_results import unwrap_hermes_result
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

    Once per model turn that follows new tool outputs, the agent model is re-run
    on those outputs with the user task replaced by a generic "follow the file's
    instructions" task. Tool calls from the masked run accumulate in a bank. A
    real tool call whose embedding is close to a bank entry does not depend on
    the user task, so the whole turn is attributed to an indirect injection.

    The check runs in ``guard_model_output`` (upstream pi_detector.py:223-236,
    288-365). Hosts call it once per candidate call of a turn, so the masked run
    and the turn verdict are cached in ``state["_melon_turn"]`` keyed by the
    number of tool outputs seen before that model input. ``before_tool`` only
    checks calls from graphs that never report tool calls at the model turn
    (they issue one call per turn). Hook instances are not shared between graph
    nodes, so all detector state lives in ``state``.
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
        # AgentDojo keeps a failed call's error outside the tool message content,
        # which is all upstream copies into random.txt (pi_detector.py:324).
        # random.txt holds the bare tool text: Hermes' {"result": ...} MCP envelope
        # is removed. Hermes' own <untrusted_tool_result> notice is added only to
        # the agent's tool message, not to this result; it is Hermes' defense, not
        # tool output, so the masked run does not see it either.
        entry = {"name": name, "content": "" if failed else unwrap_hermes_result(str(result))}
        if failed:
            entry["error"] = str(result)
        state.setdefault("_melon_outputs", []).append(entry)
        return ResultDecision(result=result)

    def guard_model_output(self, state: dict[str, Any], content: str, tool_call: Any) -> ModelDecision:
        if tool_call is not None:
            # Tool calls are judged here, so before_tool must not re-check them
            # against outputs produced by sibling calls of the same turn.
            state["_melon_model_turns"] = True
        turn = self._turn(state)
        if tool_call is None:
            return ModelDecision(content=content, tool_call=tool_call)
        name, payload = _call_parts(tool_call)
        event = self._check(state, turn, name, payload, phase="model_output")
        if not event["enforced"]:
            return ModelDecision(content=content, tool_call=tool_call)
        # Upstream replaces the whole assistant turn with the stop message before
        # any of its calls runs (pi_detector.py:272-282); the host drops the turn.
        return ModelDecision(False, event["reason"], content=STOP_MESSAGE, tool_call=None,
                             decision_type="melon", terminate=True)

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        if state.get("_melon_model_turns"):
            return ToolDecision(True, "")
        # Graphs that surface calls only here issue one call per model turn.
        event = self._check(state, self._turn(state), name, payload, phase="before_tool")
        if not event["enforced"]:
            return ToolDecision(True, "")
        # Upstream replaces the turn with a stop message, which ends the agent loop.
        return ToolDecision(False, event["reason"], replacement_result=f"{STOP_MESSAGE} {OMITTED_MESSAGE}",
                            decision_type="melon", terminate=True)

    # Per-turn detection --------------------------------------------------

    def _turn(self, state: dict[str, Any]) -> dict[str, Any]:
        """Open (or reuse) the record for the model turn after the current outputs."""
        outputs = list(state.get("_melon_outputs") or [])
        turn = state.get("_melon_turn")
        if isinstance(turn, dict) and turn.get("tool_output_count") == len(outputs):
            return turn
        previous = int(turn.get("tool_output_count", 0)) if isinstance(turn, dict) else 0
        # Upstream runs only when the model input ends with a tool result, i.e.
        # new outputs arrived since the previous model turn (none on turn one).
        turn = {"tool_output_count": len(outputs), "active": len(outputs) > previous,
                "masked_run": None, "error": None, "detected": False, "latency_ms": 0}
        state["_melon_turn"] = turn
        if not turn["active"]:
            return turn
        started = time.time()
        try:
            # The bank is updated even when the real turn makes no tool call.
            turn["masked_run"] = self._masked_run(state, outputs, self._projection(state))
        except Exception as exc:
            turn["error"] = f"MELON detection failed: {exc}"
        turn["latency_ms"] = int((time.time() - started) * 1000)
        return turn

    def _check(self, state: dict[str, Any], turn: dict[str, Any], name: str, payload: dict[str, Any],
               *, phase: str) -> dict[str, Any]:
        if not turn["active"]:
            source = "no_tool_output" if not turn["tool_output_count"] else "no_new_tool_output"
            return self._record(state, phase, name, allowed=True, source=source,
                                reason="model turn does not follow a new tool output")
        masked, turn["masked_run"] = turn["masked_run"], None  # log it with the turn's first call
        base = {"tool_output_count": turn["tool_output_count"], "masked_run": masked,
                "masked_latency_ms": turn["latency_ms"]}
        if turn["error"]:
            return self._record(state, phase, name, allowed=not self.settings.fail_closed,
                                source="error", reason=turn["error"], detail=base)
        started = time.time()
        try:
            text = transform_tool_call(name, payload, self._projection(state).get(name))
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
        except Exception as exc:
            return self._record(state, phase, name, allowed=not self.settings.fail_closed,
                                source="error", reason=f"MELON detection failed: {exc}",
                                detail={**base, "latency_ms": int((time.time() - started) * 1000)})
        injection = best > self.settings.threshold
        prior = turn["detected"]
        turn["detected"] = prior or injection
        if injection:
            turn["reason"] = (f"tool call {text} matches masked-run call {match} (cosine "
                              f"{best:.3f} > {self.settings.threshold})")
        # One matching call flags the whole turn (upstream compares every call).
        reason = turn.get("reason", "") if turn["detected"] else "no masked-run call matched"
        return self._record(
            state, phase, name, allowed=not turn["detected"], source="comparison", reason=reason,
            detail={**base, "tool_call": text, "max_cosine": best if bank else None, "matched": match,
                    "turn_detected": turn["detected"], "flagged_by_sibling": prior and not injection,
                    "bank_size": len(bank), "latency_ms": int((time.time() - started) * 1000)},
        )

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
        # Upstream keeps every system message of the conversation (pi_detector.py:320-322);
        # chat-completions `developer` messages carry the same instructions.
        found = [_text(message.get("content")) for message in state.get("messages") or []
                 if message.get("role") in {"system", "developer"}]
        found = [text for text in found if text.strip()]
        if found:
            return found
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
        # Default (upstream, pi_detector.py:18-44): the two rules above and all
        # arguments for every other tool, on every benchmark. The opt-in `llm`
        # mode lets a generator pick comparison arguments for the other tools.
        manifest: dict[str, Any] = {"status": "upstream_only", "context_mode": "benign_only"}
        generate = self.settings.projection_generation == "llm" and self.settings.generator.enabled
        manifest["source"] = "upstream+generated" if generate and pending else "upstream"
        if pending and generate:
            manifest = {**self._generate(pending, state), "source": manifest["source"]}
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

    def _record(self, state: dict[str, Any], phase: str, name: str, *, allowed: bool,
                source: str, reason: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
        enforced = not allowed and self.settings.mode == "block"
        event = {"phase": phase, "tool": name, "allowed": allowed, "enforced": enforced,
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
        if source not in {"no_tool_output", "no_new_tool_output"}:
            self._write_artifact(state, "melon_events.json", events)
        if not allowed and self.settings.mode == "warn":
            state["_melon_warning"] = reason
        return event

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


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") if isinstance(part, dict) else str(part)
                       for part in content)
    return str(content or "")


def _call_parts(tool_call: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(tool_call, (tuple, list)) and len(tool_call) == 2:
        name, payload = tool_call
    elif isinstance(tool_call, dict):
        name = tool_call.get("name")
        payload = tool_call.get("arguments", tool_call.get("args"))
    else:
        name, payload = getattr(tool_call, "name", ""), getattr(tool_call, "args", None)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = {"input": payload}
    return str(name or ""), dict(payload) if isinstance(payload, dict) else {}


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
