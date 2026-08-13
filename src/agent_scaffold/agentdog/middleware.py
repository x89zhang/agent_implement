from __future__ import annotations

import json
import os
import time
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision
from .client import AgentDoGCompletion, OpenAICompatibleAgentDoGClient
from .model import AgentDoGResult
from .parser import parse_agentdog_response
from .prompt import build_agentdog_prompt
from .trajectory import build_agentdog_trajectory, is_tool_syntax


class AgentDoGMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        self.settings = cfg.agentdog
        api_key = self.settings.api_key
        if not api_key and self.settings.api_key_env:
            api_key = os.environ.get(self.settings.api_key_env, "")
        base_url = self.settings.base_url
        if not base_url and self.settings.base_url_env:
            base_url = os.environ.get(self.settings.base_url_env, "")
        self.client = client or OpenAICompatibleAgentDoGClient(
            base_url=base_url,
            api_key=api_key,
            model=self.settings.model,
            timeout_seconds=self.settings.timeout_seconds,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        if state.get("_agentdog_final_revision_active") and (
            tool_call is not None or is_tool_syntax(content)
        ):
            reason = (
                "AgentDoG final-response revision attempted a new tool call; "
                "tools are disabled during PRE_REPLY revision"
            )
            state.setdefault("trace", []).append(
                {
                    "step": "agentdog_revision_policy",
                    "timestamp": time.time(),
                    "latency_ms": 0,
                    "input": {"checkpoint": "pre_reply"},
                    "output": {"action": "gate", "reason": reason},
                    "usage": {},
                }
            )
            state.pop("_agentdog_final_revision_active", None)
            return ModelDecision(
                False,
                reason,
                content=self.settings.replacement_message,
                tool_call=None,
                decision_type="agentdog_gate",
                terminate=True,
            )
        if "pre_reply" not in self.settings.checkpoints or tool_call is not None:
            return ModelDecision(content=content, tool_call=tool_call)
        result = self._evaluate(
            state,
            checkpoint="pre_reply",
            candidate_content=content,
        )
        action = self._action(state, result, "pre_reply")
        self._record(state, result, action)

        if action in {"allow", "diagnose", "error_open"}:
            state.pop("_agentdog_final_revision_active", None)
            return ModelDecision(
                content=content,
                tool_call=None,
                decision_type=f"agentdog_{action}",
            )
        if action == "revise":
            state["_agentdog_final_revision_active"] = True
            return ModelDecision(
                False,
                _reason(result),
                content=content,
                tool_call=None,
                retry=True,
                feedback=_revision_feedback(result),
                decision_type="agentdog_revise",
            )
        state.pop("_agentdog_final_revision_active", None)
        return ModelDecision(
            False,
            _reason(result),
            content=self.settings.replacement_message,
            tool_call=None,
            decision_type="agentdog_gate",
            terminate=True,
        )

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        if "pre_action" not in self.settings.checkpoints:
            return ToolDecision()
        result = self._evaluate(
            state,
            checkpoint="pre_action",
            candidate_content=f"TOOL_CALL: {name} {json.dumps(payload, ensure_ascii=False)}",
            candidate_tool_call=(name, payload),
        )
        action = self._action(state, result, "pre_action")
        self._record(state, result, action)
        event = dict(state.get("_last_agentdog_decision") or {})
        state["_agentdog_pending_call"] = {
            "tool_name": name,
            "arguments": dict(payload),
            "decision": event,
        }

        if action in {"allow", "diagnose", "error_open"}:
            return ToolDecision(True, "", decision_type=f"agentdog_{action}")
        feedback = _revision_feedback(result)
        if action == "revise":
            return ToolDecision(
                False,
                _reason(result),
                replacement_result=(
                    "AgentDoG trajectory diagnosis rejected the proposed action. "
                    "The action was not executed. Replan from the original request.\n"
                    + feedback
                ),
                decision_type="agentdog_revise",
            )
        return ToolDecision(
            False,
            _reason(result),
            replacement_result=self.settings.replacement_message,
            decision_type="agentdog_gate",
            terminate=True,
        )

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        pending = state.pop("_agentdog_pending_call", None)
        if isinstance(pending, dict):
            pending["result"] = str(result)
            pending["failed"] = bool(failed)
            state.setdefault("_agentdog_action_history", []).append(pending)
        return ResultDecision(result=result)

    def _evaluate(
        self,
        state: dict[str, Any],
        *,
        checkpoint: str,
        candidate_content: str,
        candidate_tool_call: Any = None,
    ) -> AgentDoGResult:
        started = time.monotonic()
        trajectory = build_agentdog_trajectory(
            self.cfg,
            state,
            candidate_content=candidate_content,
            candidate_tool_call=candidate_tool_call,
            max_chars=self.settings.max_trajectory_chars,
        )
        prompt = build_agentdog_prompt(
            task=self.settings.task,
            formatted_trajectory=trajectory.formatted,
            tool_list_text=trajectory.tool_list_text,
        )
        raw_response = ""
        usage: dict[str, Any] = {}
        try:
            completion = self.client.complete(prompt)
            if isinstance(completion, AgentDoGCompletion):
                raw_response = completion.content
                usage = dict(completion.usage)
            elif hasattr(completion, "content"):
                raw_response = str(completion.content)
                candidate_usage = getattr(completion, "usage", {})
                usage = (
                    dict(candidate_usage) if isinstance(candidate_usage, dict) else {}
                )
            else:
                raw_response = str(completion)
            judgment, analysis, failure_modes, consequences, sources = (
                parse_agentdog_response(raw_response, self.settings.task)
            )
            return AgentDoGResult(
                judgment=judgment,
                analysis=analysis,
                failure_modes=failure_modes,
                risk_consequences=consequences,
                risk_sources=sources,
                checkpoint=checkpoint,
                task=self.settings.task,
                mode=self.settings.mode,
                model=self.settings.model,
                latency_ms=int((time.monotonic() - started) * 1000),
                trajectory_steps=trajectory.step_count,
                trajectory_chars=len(trajectory.formatted),
                truncated=trajectory.truncated,
                raw_response=raw_response if self.settings.include_raw_response else "",
                usage=usage,
            )
        except Exception as exc:  # noqa: BLE001 - endpoint/parser failures share policy
            return AgentDoGResult(
                judgment="error",
                analysis="",
                checkpoint=checkpoint,
                task=self.settings.task,
                mode=self.settings.mode,
                model=self.settings.model,
                latency_ms=int((time.monotonic() - started) * 1000),
                trajectory_steps=trajectory.step_count,
                trajectory_chars=len(trajectory.formatted),
                truncated=trajectory.truncated,
                raw_response=raw_response if self.settings.include_raw_response else "",
                error=str(exc),
                usage=usage,
            )

    def _action(
        self, state: dict[str, Any], result: AgentDoGResult, checkpoint: str
    ) -> str:
        if result.safe is True:
            return "allow"
        if result.safe is None:
            if self.settings.mode == "diagnose" or not self.settings.fail_closed:
                return "error_open"
            return "gate"
        if self.settings.mode == "diagnose":
            return "diagnose"
        if self.settings.mode == "gate":
            return "gate"
        counts = state.setdefault("_agentdog_revision_counts", {})
        count = int(counts.get(checkpoint, 0))
        if count < self.settings.max_revisions:
            counts[checkpoint] = count + 1
            return "revise"
        return "gate"

    def _record(
        self, state: dict[str, Any], result: AgentDoGResult, action: str
    ) -> None:
        data = result.to_dict()
        data["action"] = action
        data["reason"] = _reason(result)
        data["revision_count"] = int(
            (state.get("_agentdog_revision_counts") or {}).get(result.checkpoint, 0)
        )
        data["max_revisions"] = self.settings.max_revisions
        state["_last_agentdog_decision"] = data
        events = state.setdefault("agentdog_events", [])
        events.append(data)

        harness = state.setdefault("harness", {}).setdefault("agentdog", {})
        usage_totals = harness.setdefault(
            "usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_totals[key] = int(usage_totals.get(key, 0)) + int(
                result.usage.get(key, 0) or 0
            )
        harness.update(
            {
                "enabled": True,
                "status": "error" if result.safe is None else "completed",
                "mode": self.settings.mode,
                "task": self.settings.task,
                "model": self.settings.model,
                "checkpoints": list(self.settings.checkpoints),
                "event_count": len(events),
                "latest": data,
                "usage": usage_totals,
            }
        )
        state.setdefault("trace", []).append(
            {
                "step": "agentdog",
                "timestamp": time.time(),
                "latency_ms": result.latency_ms,
                "input": {
                    "checkpoint": result.checkpoint,
                    "trajectory_steps": result.trajectory_steps,
                    "trajectory_chars": result.trajectory_chars,
                    "truncated": result.truncated,
                },
                "output": data,
                "usage": dict(result.usage),
            }
        )


def _reason(result: AgentDoGResult) -> str:
    if result.safe is None:
        return f"AgentDoG evaluation failed: {result.error}"
    if result.safe:
        return ""
    labels = result.failure_modes + result.risk_consequences + result.risk_sources
    suffix = f" ({'; '.join(labels)})" if labels else ""
    return f"AgentDoG classified the accumulated trajectory as unsafe{suffix}"


def _revision_feedback(result: AgentDoGResult) -> str:
    return json.dumps(
        {
            "judgment": result.judgment,
            "analysis": result.analysis,
            "failure_modes": result.failure_modes,
            "risk_consequences": result.risk_consequences,
            "risk_sources": result.risk_sources,
        },
        ensure_ascii=False,
    )
