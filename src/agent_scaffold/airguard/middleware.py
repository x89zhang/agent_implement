"""Adapt upstream AIRGuard's contextual action checks to project hooks."""

from __future__ import annotations

import importlib
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision


_DENIED = frozenset({"block", "ask", "inspect", "sandbox", "quarantine"})
# Upstream's reference integration (integrations/mcp_proxy.py:60-70) checks
# every tool call against one fixed low-trust resource and a constant
# authority; parsing user intent into allow/deny is an upstream TODO.
_PROXY_RESOURCE_ID = "mcp-proxy-tool"
_PROXY_CONSTRAINTS = ("no_secret", "no_network", "inspect_before_exec")
_PROXY_USER_INTENT_CHARS = 500  # benchmarks/dtap/agents/claudecli/agent.py:78


class _OpenAICompletionCompat:
    """Adapt upstream AIRGuard's Chat Completions call to GPT-5 parameters."""

    def __init__(self, client: Any) -> None:
        self.chat = SimpleNamespace(completions=self)
        self._create = client.chat.completions.create

    def create(self, **kwargs: Any) -> Any:
        if str(kwargs.get("model", "")).lower().startswith("gpt-5"):
            kwargs.pop("temperature", None)
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        return self._create(**kwargs)


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
        # Project extension (off by default): the upstream proxy only guards
        # tool calls, so final responses are not checked on the DTAP path.
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
        # Project extension (off by default): upstream's proxy runs no
        # post-action audit; it only redacts credentials in tool output.
        if pending is not None and self.settings.post_action_audit:
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
            "redacted": False, "sensitive_target": False, "normalized_action": "",
            "extension": phase == "model_output",
        }
        try:
            upstream = self._load_upstream()
            action = self._action(upstream, name, payload)
            event["normalized_action"] = _value(action.normalized_action)
            resource = self._resource(upstream)
            event["resource_publisher"] = _value(resource.publisher)
            intent = str(state.get("_airguard_user_request") or self.cfg.agent.task)
            authority = upstream.types.AuthorityContext(
                issuer=upstream.types.Issuer.USER,
                subject=upstream.types.Subject.AGENT,
                scope=upstream.types.Scope.TASK,
                allow=list(self.settings.authority_allow),
                guard="ask",
                user_intent=intent[:_PROXY_USER_INTENT_CHARS],
            )
            ledger = upstream.audit_containment.LedgerStore()
            for row in state.get("_airguard_ledger", []):
                ledger.record(upstream.types.LedgerEntry(**row))
            # Like mcp_proxy._guard_check, history is empty; the run-scoped
            # ledger still carries the sequence audit across calls.
            decision = upstream.guard.check_action(
                resource=resource,
                authority_ctx=authority,
                action=action,
                history=[],
                ledger=ledger,
                llm_client=self._client(),
                risk_model=self.settings.model or self.cfg.llm.model,
            )
            state["_airguard_ledger"] = [asdict(row) for row in ledger.all_entries()]
            outcome = decision.outcome.value if hasattr(decision.outcome, "value") else str(decision.outcome)
            flagged = outcome in _DENIED
            reason = decision.reasoning
            # mcp_proxy.py:158-161: exec/write on a sensitive target is blocked
            # regardless of the pipeline outcome.
            sensitive = _is_sensitive(upstream, action.args)
            event["sensitive_target"] = sensitive
            if sensitive and event["normalized_action"] in {"process.exec", "file.write"} and not flagged:
                outcome, flagged = "block", True
                reason = f"{reason} Proxy override: {event['normalized_action']} on a sensitive target."
            event.update({
                "outcome": outcome,
                "flagged": flagged,
                "reason": reason,
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
            "outcome": "audit", "blocked": False, "extension": True,
            "suspicions": [asdict(item) for item in suspicions],
            "reason": f"Post-action audit found {len(suspicions)} suspicion(s)",
            "mode": self.settings.mode,
        })

    def _action(self, upstream: Any, name: str, payload: dict[str, Any]) -> Any:
        # output.respond is only produced by the final-response extension.
        normalized = "output.respond" if name == "output.respond" else normalize_action(name, upstream)
        return upstream.types.Action(
            action_id=uuid.uuid4().hex,
            name=name,
            args=dict(payload),
            source_resource_id=_PROXY_RESOURCE_ID,
            required_capabilities=[_value(normalized).split(".")[0]],
            normalized_action=normalized,
        )

    def _resource(self, upstream: Any) -> Any:
        types = upstream.types
        return types.Resource(
            resource_id=_PROXY_RESOURCE_ID,
            publisher=types.Publisher.UNKNOWN_WEB,
            trust_tier=types.TrustTier.LOW,
            constraints=list(_PROXY_CONSTRAINTS),
        )

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

            client = OpenAI(
                api_key=api_key or ("local-airguard" if base_url else None),
                base_url=base_url or None,
                timeout=timeout,
            )
            self._llm_client = _OpenAICompletionCompat(client)
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
        try:
            # Needs the optional ``mcp`` package; the vendored copy is identical.
            package.mcp_proxy = importlib.import_module("airguard.integrations.mcp_proxy")
        except Exception:
            package.mcp_proxy = None
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


def _value(item: Any) -> str:
    return str(getattr(item, "value", item))


def normalize_action(name: str, upstream: Any = None) -> str:
    """Upstream DTAP proxy normalization; prefer the installed upstream copy."""
    proxy = getattr(upstream, "mcp_proxy", None)
    if proxy is not None:
        return _value(proxy._normalize(name))
    return _normalize(name)


def _is_sensitive(upstream: Any, args: dict[str, Any]) -> bool:
    proxy = getattr(upstream, "mcp_proxy", None)
    if proxy is not None:
        return bool(proxy._is_sensitive(args))
    return _proxy_is_sensitive(args)


# Vendored verbatim from AIRGuard 3def7bff3afbfb890e224131190a988bd3e4faba,
# src/airguard/integrations/mcp_proxy.py:84-127, for environments without the
# optional ``mcp`` dependency. Only the enum return values became strings.
def _normalize(name: str) -> str:
    """Map tool name to NormalizedAction.

    Order matters: destructive and exec checks run first so that names like
    ``delete_readme`` or ``run_query`` are not swallowed by the read branch.
    Matching uses ``_``/``-`` split tokens, not raw substrings, to avoid
    partial-word collisions.
    """
    import re as _re
    tokens = set(_re.split(r"[_\-./\s]+", name.lower()))

    # Destructive first — before read, so delete_readme → FILE_DELETE
    if tokens & {"delete", "remove", "rm", "drop", "purge", "truncate", "destroy"}:
        return "file.delete"
    # Exec before read — so run_query → PROCESS_EXEC
    if tokens & {"execute", "run", "exec", "bash", "terminal", "shell"}:
        return "process.exec"
    # Read
    if tokens & {"query", "search", "get", "list", "read", "find", "check", "view", "fetch",
                 "browse", "tree", "info", "show", "describe", "lookup", "inspect",
                 "login", "auth"}:
        return "file.read"
    # Write
    if tokens & {"write", "create", "add", "update", "set", "modify", "send", "put", "insert", "append"}:
        return "file.write"
    return "tool.call"


_SENSITIVE_TARGETS = [
    ".bashrc", ".zshrc", ".profile", ".bash_profile",
    ".ssh", ".env", ".aws", "id_rsa",
    "/etc/shadow", "/etc/passwd", "/etc/sudoers",
    "crontab", "systemd", "launchd",
]


def _proxy_is_sensitive(args: dict) -> bool:
    for v in args.values():
        if isinstance(v, str):
            vl = v.lower()
            for s in _SENSITIVE_TARGETS:
                if s in vl:
                    return True
    return False
