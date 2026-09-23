"""Adapt upstream AIRGuard's contextual action checks to project hooks."""

from __future__ import annotations

import importlib
import re
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision


_DENIED = frozenset({"block", "ask", "inspect", "sandbox", "quarantine"})
_READ = frozenset({"read", "get", "list", "search", "find", "query", "fetch", "browse", "view", "lookup", "inspect"})
_DELETE = frozenset({"delete", "remove", "purge", "drop", "truncate", "destroy", "rm"})
_EXEC = frozenset({"execute", "exec", "run", "shell", "bash", "terminal", "script"})
_NETWORK = frozenset({"http", "request", "download", "upload", "curl", "wget"})
_WRITE = frozenset({"write", "create", "add", "update", "set", "modify", "append", "save"})


class AIRGuardMiddleware(Middleware):
    def __init__(self, cfg: AppConfig, *, upstream: Any = None, llm_client: Any = None) -> None:
        self.cfg = cfg
        self.settings = cfg.airguard
        self._upstream = upstream
        self._llm_client = llm_client

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_airguard_warning", "")
        return [f"AIRGuard warning: {warning}"] if warning else []

    def guard_model_output(
        self, state: dict[str, Any], content: str, tool_call: Any
    ) -> ModelDecision:
        if tool_call is not None or not content.strip() or not self.settings.check_final_output:
            return ModelDecision(content=content, tool_call=tool_call)
        event = self._check(state, "output.respond", {"message": content}, "model_output")
        if event["blocked"]:
            return ModelDecision(
                allowed=False,
                reason=event["reason"],
                content="Final response blocked by AIRGuard.",
                tool_call=None,
                terminate=True,
                decision_type="airguard",
            )
        if self.settings.redact_credentials:
            try:
                clean = self._load_upstream().output_guard.redact_credentials(content)
            except Exception as exc:
                return self._output_error(state, content, tool_call, exc)
            if clean != content:
                event["redacted"] = True
                content = clean
        return ModelDecision(content=content, tool_call=tool_call)

    def before_tool(
        self, state: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> ToolDecision:
        state.pop("_airguard_pending_action", None)
        event = self._check(state, name, payload, "before_tool")
        state["_airguard_tool_blocked"] = event["blocked"]
        return ToolDecision(
            allowed=not event["blocked"],
            reason=event["reason"] if event["blocked"] else "",
            decision_type="airguard" if event["blocked"] else "",
        )

    def after_tool(
        self,
        state: dict[str, Any],
        name: str,
        payload: dict[str, Any],
        result: str,
        failed: bool,
    ) -> ResultDecision:
        pending = state.pop("_airguard_pending_action", None)
        if state.pop("_airguard_tool_blocked", False) or str(result).startswith(
            ("Tool execution blocked by middleware:", "Tool not found:")
        ):
            return ResultDecision(result=result)
        output = result
        if self.settings.redact_credentials:
            try:
                output = self._load_upstream().output_guard.redact_credentials(str(result))
            except Exception as exc:
                self._record(state, {
                    "phase": "after_tool", "tool": name, "outcome": "error",
                    "blocked": self.settings.mode == "block" and self.settings.fail_closed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "reason": f"AIRGuard output inspection failed: {exc}",
                    "mode": self.settings.mode,
                })
                if self.settings.fail_closed and self.settings.mode == "block":
                    return ResultDecision(
                        allowed=False,
                        reason=f"AIRGuard output inspection failed: {exc}",
                        result="Tool result withheld by AIRGuard.",
                        decision_type="airguard_error",
                    )
        if not failed:
            state["_airguard_resource"] = {
                "resource_id": f"tool:{name}:{uuid.uuid4().hex}",
                "publisher": self.settings.tool_publishers.get(
                    name, self.settings.default_tool_publisher
                ),
                "content_ref": str(output)[: self.settings.max_content_chars],
            }
        if pending is not None:
            try:
                self._post_audit(state, pending, str(output), failed)
            except Exception as exc:
                self._record(state, {
                    "phase": "after_tool", "tool": name, "outcome": "error",
                    "blocked": self.settings.mode == "block" and self.settings.fail_closed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "reason": f"AIRGuard post-action audit failed: {exc}",
                    "mode": self.settings.mode,
                })
                if self.settings.mode == "block" and self.settings.fail_closed:
                    return ResultDecision(
                        allowed=False, result="Tool result withheld by AIRGuard.",
                        reason="AIRGuard post-action audit failed",
                        decision_type="airguard_error",
                    )
        if output != result:
            event = {
                "phase": "after_tool", "tool": name, "outcome": "redact",
                "blocked": False, "redacted": True, "reason": "Credentials redacted from tool result",
            }
            self._record(state, event)
        return ResultDecision(result=output)

    def _check(
        self, state: dict[str, Any], name: str, payload: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        started = time.monotonic()
        event: dict[str, Any] = {
            "phase": phase, "tool": name, "outcome": "", "flagged": False,
            "blocked": False, "reason": "", "error": "", "risk_source": "",
            "risk_model": "", "target_trust_tier": "", "resource_publisher": "",
            "redacted": False,
        }
        try:
            upstream = self._load_upstream()
            action = self._action(upstream, name, payload, state)
            resource = self._resource(upstream, state)
            event["resource_publisher"] = str(resource.publisher)
            authority = upstream.types.AuthorityContext(
                issuer=upstream.types.Issuer.USER,
                subject=upstream.types.Subject.AGENT,
                scope=upstream.types.Scope.TASK,
                allow=list(self.settings.authority_allow),
                guard="ask",
                user_intent=str(state.get("_airguard_user_request") or self.cfg.agent.task),
            )
            ledger = upstream.audit_containment.LedgerStore()
            for row in state.get("_airguard_ledger", []):
                ledger.record(upstream.types.LedgerEntry(**row))
            decision = upstream.guard.check_action(
                resource=resource,
                authority_ctx=authority,
                action=action,
                history=ledger.all_entries(),
                ledger=ledger,
                llm_client=self._client(),
                risk_model=self.settings.model or self.cfg.llm.model,
            )
            state["_airguard_ledger"] = [asdict(row) for row in ledger.all_entries()]
            outcome = decision.outcome.value if hasattr(decision.outcome, "value") else str(decision.outcome)
            event.update({
                "outcome": outcome,
                "flagged": outcome in _DENIED,
                "reason": decision.reasoning,
                "risk_source": decision.risk_source,
                "risk_model": decision.risk_model,
                "risk_recommendation": decision.risk_recommendation,
                "risk_reason": decision.risk_reason,
                "risk_error": decision.risk_error,
                "target_trust_tier": decision.target_trust_tier,
                "ledger_entry_id": decision.ledger_entry_id,
            })
            event["blocked"] = event["flagged"] and self.settings.mode == "block"
            if phase == "before_tool" and not event["blocked"]:
                state["_airguard_pending_action"] = asdict(action)
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            event["reason"] = f"AIRGuard check failed: {event['error']}"
            event["outcome"] = "error"
            event["blocked"] = self.settings.mode == "block" and self.settings.fail_closed
        event["latency_ms"] = round((time.monotonic() - started) * 1000)
        event["mode"] = self.settings.mode
        event["authority_allow"] = list(self.settings.authority_allow)
        event["authority_source"] = self.settings.authority_source
        self._record(state, event)
        if event["flagged"] and self.settings.mode == "warn":
            state["_airguard_warning"] = event["reason"]
        return event

    def _post_audit(
        self, state: dict[str, Any], pending: dict[str, Any], output: str, failed: bool
    ) -> None:
        upstream = self._load_upstream()
        ledger = upstream.audit_containment.LedgerStore()
        for row in state.get("_airguard_ledger", []):
            ledger.record(upstream.types.LedgerEntry(**row))
        suspicions = upstream.guard.post_action_audit(
            upstream.types.Action(**pending),
            {"failed": failed, "result": output[: self.settings.max_content_chars]},
            ledger,
        )
        state["_airguard_ledger"] = [asdict(row) for row in ledger.all_entries()]
        self._record(state, {
            "phase": "after_tool", "tool": pending["name"],
            "outcome": "audit", "blocked": False,
            "suspicions": [asdict(item) for item in suspicions],
            "reason": f"Post-action audit found {len(suspicions)} suspicion(s)",
            "mode": self.settings.mode,
        })

    def _action(self, upstream: Any, name: str, payload: dict[str, Any], state: dict[str, Any]) -> Any:
        normalized = _normalized_action(name, payload)
        source = state.get("_airguard_resource") or {}
        return upstream.types.Action(
            action_id=uuid.uuid4().hex,
            name=name,
            args=dict(payload),
            source_resource_id=str(source.get("resource_id") or "trusted_user_task"),
            normalized_action=normalized,
        )

    def _resource(self, upstream: Any, state: dict[str, Any]) -> Any:
        source = state.get("_airguard_resource") or {
            "resource_id": "trusted_user_task", "publisher": "user",
            "trust_tier": "high", "content_ref": "",
        }
        return upstream.trust_labeling.label_resource(source)

    def _client(self) -> Any:
        if not self.settings.use_llm:
            return None
        if self._llm_client is not None:
            return self._llm_client
        provider = (self.settings.provider or self.cfg.llm.provider).lower()
        api_key = self.settings.api_key or self.cfg.llm.api_key
        base_url = self.settings.base_url or self.cfg.llm.base_url
        timeout = self.settings.timeout_seconds
        if provider == "openrouter":
            provider = "openai"
            base_url = base_url or "https://openrouter.ai/api/v1"
        if provider not in {"openai", "vllm_openai", "anthropic"}:
            return None  # AIRGuard uses its built-in heuristic risk model.
        if not api_key and not base_url:
            return None
        if provider in {"openai", "vllm_openai"}:
            from openai import OpenAI

            self._llm_client = OpenAI(
                api_key=api_key or ("local-airguard" if base_url else None),
                base_url=base_url or None,
                timeout=timeout,
            )
        elif provider == "anthropic":
            from anthropic import Anthropic

            self._llm_client = Anthropic(
                api_key=api_key or None,
                base_url=base_url or None,
                timeout=timeout,
            )
        return self._llm_client

    def _load_upstream(self) -> Any:
        if self._upstream is not None:
            return self._upstream
        source_root = self.settings.source_root.strip()
        if source_root:
            root = Path(source_root).expanduser()
            if not root.is_absolute():
                root = Path(self.cfg.config_dir) / root
            root = root.resolve()
            package_root = root if (root / "airguard" / "guard.py").is_file() else root / "src"
            if not (package_root / "airguard" / "guard.py").is_file():
                raise FileNotFoundError(f"AIRGuard source not found under {root}")
            if str(package_root) not in sys.path:
                sys.path.insert(0, str(package_root))
        try:
            package = importlib.import_module("airguard")
            package.guard = importlib.import_module("airguard.guard")
            package.types = importlib.import_module("airguard.types")
            package.trust_labeling = importlib.import_module("airguard.trust_labeling")
            package.audit_containment = importlib.import_module("airguard.audit_containment")
            package.output_guard = importlib.import_module("airguard.output_guard")
        except ImportError as exc:
            raise RuntimeError(
                "AIRGuard source is unavailable; set airguard.source_root to its checkout"
            ) from exc
        self._upstream = package
        return package

    def _record(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        state["_last_airguard_decision"] = event
        state.setdefault("airguard_events", []).append(event)
        state.setdefault("trace", []).append({
            "step": "airguard_decision", "timestamp": time.time(), "output": event,
        })
        state.setdefault("harness", {})["airguard"] = {
            "enabled": True,
            "mode": self.settings.mode,
            "authority_allow": list(self.settings.authority_allow),
            "authority_source": self.settings.authority_source,
            "status": "error" if event.get("error") else "blocked" if event.get("blocked") else "active",
            "event_count": len(state["airguard_events"]),
            "last_decision": event,
        }

    def _output_error(
        self, state: dict[str, Any], content: str, tool_call: Any, exc: Exception
    ) -> ModelDecision:
        reason = f"AIRGuard output inspection failed: {type(exc).__name__}: {exc}"
        blocked = self.settings.mode == "block" and self.settings.fail_closed
        self._record(state, {
            "phase": "model_output", "outcome": "error", "blocked": blocked,
            "reason": reason, "error": reason, "mode": self.settings.mode,
        })
        return ModelDecision(
            allowed=not blocked,
            reason=reason if blocked else "",
            content="Final response blocked by AIRGuard." if blocked else content,
            tool_call=None if blocked else tool_call,
            terminate=blocked,
            decision_type="airguard_error" if blocked else "",
        )


def _normalized_action(name: str, payload: dict[str, Any]) -> str:
    if name == "output.respond":
        return "output.respond"
    tokens = set(re.split(r"[_\-./\s]+", name.lower()))
    if "email" in tokens and "send" in tokens:
        return "email.send"
    if "mail" in tokens and "send" in tokens:
        return "email.send"
    if "browser" in tokens and tokens & {"navigate", "open", "go"}:
        return "browser.navigate"
    if "browser" in tokens and tokens & {"extract", "scrape", "read"}:
        return "browser.extract"
    if "memory" in tokens and tokens & _WRITE:
        return "memory.write"
    if "config" in tokens and tokens & (_WRITE | _DELETE):
        return "config.modify"
    if tokens & {"database", "db", "sql"} and tokens & (_READ | {"select"}):
        return "database.query"
    if tokens & {"package", "pip", "npm"} and tokens & {"install", "add"}:
        return "package.install"
    if tokens & _DELETE:
        return "file.delete"
    if tokens & _EXEC:
        return "process.exec"
    if tokens & _NETWORK:
        return "network.request"
    if tokens & _READ:
        return "file.read"
    if tokens & _WRITE:
        return "file.write"
    if "command" in payload or "cmd" in payload:
        return "process.exec"
    return "tool.call"
