from __future__ import annotations

import re
from typing import Callable

from jsonschema import validate


class ValidationError(Exception):
    pass


def check_arg(arg_name, value, restriction) -> None:
    """Raise if `value` violates `restriction`. Verbatim from secagent.tool.check_arg."""
    if isinstance(restriction, dict):
        validate(instance=value, schema=restriction)
    elif isinstance(restriction, str):
        if not re.match(restriction, value):
            raise ValidationError(
                f"Invalid value for argument '{arg_name}' value '{value}', the allowed value is '{restriction}'")
    elif isinstance(restriction, Callable):
        if not restriction(value):
            raise ValidationError(
                f"Invalid value for argument '{arg_name}': rejected by {type(restriction).__name__}.")
    else:
        raise NotImplementedError(f"Unsupported restriction type: {type(restriction)}")


class Enforcer:
    """Per-task policy store + guard. One instance per protected runtime; no global state."""

    def __init__(self) -> None:
        self.available_tools: list[dict] = []
        self.security_policy: dict | None = None
        self.user_query: str | None = None

    # ── configuration (mirrors secagent.update_* but on instance state) ───────────
    def set_user_query(self, query: str | None) -> None:
        self.user_query = query

    def update_available_tools(self, tools: list[dict]) -> None:
        self.available_tools = tools

    def _sort_policy(self) -> None:
        if self.security_policy is None:
            return
        for tool, policies in self.security_policy.items():
            self.security_policy[tool] = sorted(policies, key=lambda x: (x[0], -x[1]))

    def update_security_policy(self, policy: dict) -> None:
        self.security_policy = policy
        self._sort_policy()

    def update_always_allowed_tools(self, tools, allow_all_no_arg_tools: bool = False) -> None:
        if self.security_policy is None:
            self.security_policy = {}
        always_allowed = set(tools)
        if allow_all_no_arg_tools:
            always_allowed.update([t["name"] for t in self.available_tools if len(t["args"]) == 0])
        for tool in always_allowed:
            if tool not in self.security_policy:
                self.security_policy[tool] = [(1, 0, {}, 0)]
            else:
                self.security_policy[tool].insert(0, (1, 0, {}, 0))
        self._sort_policy()

    # ── enforcement (verbatim algorithm from secagent._check_tool_call) ───────────
    def _check_tool_call(self, tool_name, kwargs, policies):
        need_update_policies = None
        for policy in policies:
            if len(policy) == 4:
                priority, effect, policy, fallback = policy
            else:
                priority, effect, policy, fallback, need_update_policies = policy
            if effect == 0:
                flag = True
                try:
                    for arg_name, restriction in policy.items():
                        if arg_name in kwargs:
                            value = kwargs[arg_name]
                            check_arg(arg_name, value, restriction)
                except Exception as e:  # noqa: BLE001 -- a failed condition just means this rule doesn't apply
                    flag = False
                    if priority == 100:
                        if fallback == 0:
                            raise e
                if flag:
                    return need_update_policies
            elif effect == 1:
                try:
                    for arg_name, restriction in policy.items():
                        if arg_name in kwargs:
                            value = kwargs[arg_name]
                            check_arg(arg_name, value, restriction)
                except Exception:  # noqa: BLE001
                    continue
                if fallback == 0:
                    raise ValidationError(f"The tool '{tool_name}' is not allowed.")
                else:
                    raise ValidationError(f"The tool '{tool_name}' is not allowed.")
        raise ValidationError(f"The tool '{tool_name}' is not allowed.")

    def check_tool_call(self, tool_name, kwargs) -> None:
        """Raise ValidationError if the call is denied; return None if allowed."""
        if self.security_policy is None:
            return
        try:
            policies = self.security_policy.get(tool_name, None)
            if policies is None or len(policies) == 0:
                raise ValidationError(f"The tool '{tool_name}' is not allowed.")
            need_update_policies = self._check_tool_call(tool_name, kwargs, policies)
            if need_update_policies:
                for tool, ps in need_update_policies.items():
                    self.security_policy[tool] = ps
                self._sort_policy()
        except Exception as e:  # noqa: BLE001
            # THE FIX: interpolate the real user query (secagent used a global that we never set -> None).
            raise ValidationError(
                f"{e}. Please try other tools or arguments and continue to finish the user task: {self.user_query}.")

