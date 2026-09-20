from __future__ import annotations

import copy
import importlib
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


_UPSTREAM_LOCK = threading.RLock()


@dataclass
class RuntimeResult:
    allowed: bool
    reason: str = ""
    policy: dict[str, Any] | None = None
    usage: dict[str, int] = field(default_factory=dict)


class ProgentRuntime:
    """Session facade over Progent's process-global ``secagent.tool`` API."""

    def __init__(
        self,
        *,
        tools: list[dict[str, Any]],
        query: str,
        policy: dict[str, Any] | None = None,
        completion: Callable[[str, str, float], tuple[str, dict[str, Any] | None]]
        | None = None,
    ) -> None:
        self.tools = copy.deepcopy(tools)
        self.query = query
        self.policy = copy.deepcopy(policy)
        self.completion = completion
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def available() -> bool:
        try:
            importlib.import_module("secagent.tool")
        except Exception:
            return False
        return True

    def generate(self) -> RuntimeResult:
        try:
            module = self._module()
            with self._activated(module):
                module.generate_security_policy(self.query, manual_check=False)
                self.policy = copy.deepcopy(module.get_current_config())
            return RuntimeResult(True, policy=copy.deepcopy(self.policy), usage=dict(self.usage))
        except Exception as exc:
            return RuntimeResult(
                False,
                f"Progent policy generation failed: {type(exc).__name__}: {exc}",
                policy=copy.deepcopy(self.policy),
                usage=dict(self.usage),
            )

    def check(self, name: str, arguments: dict[str, Any]) -> RuntimeResult:
        try:
            module = self._module()
            with self._activated(module):
                module.check_tool_call(name, copy.deepcopy(arguments))
                self.policy = copy.deepcopy(module.get_current_config())
            return RuntimeResult(True, policy=copy.deepcopy(self.policy))
        except Exception as exc:
            return RuntimeResult(
                False,
                str(exc),
                policy=copy.deepcopy(self.policy),
            )

    def update(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        *,
        only_allow_narrow: bool,
    ) -> RuntimeResult:
        previous = copy.deepcopy(self.policy)
        try:
            module = self._module()
            with self._activated(module):
                module.generate_update_security_policy(
                    [{"name": name, "args": copy.deepcopy(arguments)}],
                    str(result),
                    manual_check=False,
                )
                candidate = copy.deepcopy(module.get_current_config())
                if only_allow_narrow and not _is_policy_subset(module, previous, candidate):
                    self.policy = previous
                    return RuntimeResult(
                        False,
                        "Progent discarded a policy update that widened privileges",
                        policy=copy.deepcopy(previous),
                        usage=dict(self.usage),
                    )
                self.policy = candidate
            return RuntimeResult(True, policy=copy.deepcopy(self.policy), usage=dict(self.usage))
        except Exception as exc:
            self.policy = previous
            return RuntimeResult(
                False,
                f"Progent policy update failed: {type(exc).__name__}: {exc}",
                policy=copy.deepcopy(previous),
                usage=dict(self.usage),
            )

    def _module(self) -> Any:
        try:
            return importlib.import_module("secagent.tool")
        except Exception as exc:
            raise RuntimeError(
                "Progent is enabled but the optional 'secagent' package is unavailable; "
                "install requirements-progent.txt"
            ) from exc

    @contextmanager
    def _activated(self, module: Any) -> Iterator[None]:
        with _UPSTREAM_LOCK:
            names = (
                "available_tools",
                "security_policy",
                "init_user_query",
                "api_request",
                "generate_policy",
            )
            previous = {name: getattr(module, name, None) for name in names}
            module.available_tools = copy.deepcopy(self.tools)
            module.security_policy = copy.deepcopy(self.policy)
            module.init_user_query = self.query
            module.generate_policy = True
            if self.completion is not None:
                module.api_request = self._complete
            try:
                yield
            finally:
                for name, value in previous.items():
                    setattr(module, name, value)

    def _complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        if self.completion is None:  # pragma: no cover - guarded by _activated
            raise RuntimeError("No Progent policy model is configured")
        content, usage = self.completion(system, user, temperature)
        for key in self.usage:
            self.usage[key] += int((usage or {}).get(key) or 0)
        return content


def _is_policy_subset(module: Any, old: Any, new: Any) -> bool:
    if old is None:
        return True
    try:
        analysis = importlib.import_module("secagent.policy_analysis")
        return bool(analysis.security_policy_subset_check(old, new))
    except Exception:
        # A failed proof must never authorize a wider dynamic policy.
        return new == old
