from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ToolDecision
from .abstraction import ToolTraceAbstraction
from .model import JsonDTMC, Pro2GuardResult
from .prism import query_prism_probability


class Pro2GuardMiddleware(Middleware):
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.pg = cfg.pro2guard
        self.abstraction = _load_abstraction(self.pg.abstraction)
        self._json_model: JsonDTMC | None = None
        model_path = _resolve_config_path(cfg, self.pg.model_path or self.pg.dtmc_path)
        if model_path and Path(model_path).suffix.lower() == ".json":
            self._json_model = JsonDTMC(model_path)
        self._dtmc_path = _resolve_config_path(cfg, self.pg.dtmc_path)

    def before_model(self, state: dict[str, Any]) -> list[str]:
        warning = state.pop("_pro2guard_warning", "")
        return [warning] if warning else []

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        encoded = self.abstraction.encode_tool_call(state, name, payload)
        result = self._evaluate(encoded)
        state["_last_pro2guard_decision"] = result.to_dict()
        state.setdefault("pro2guard_events", []).append(result.to_dict())

        if result.allowed:
            return ToolDecision(True, "")
        if self.pg.mode == "warn":
            state["_pro2guard_warning"] = result.reason
            return ToolDecision(True, "")
        if self.pg.mode == "monitor":
            return ToolDecision(True, "")
        return ToolDecision(False, result.reason)

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> None:
        encoded = self.abstraction.encode_tool_result(name, payload, result, failed)
        state["_pro2guard_last_state"] = encoded
        state["_pro2guard_last_outcome"] = "failed" if failed else "ok"

    def _evaluate(self, encoded_state: str) -> Pro2GuardResult:
        try:
            probability, matched_state, source = self._probability(encoded_state)
        except Exception as exc:
            allowed = not self.pg.fail_closed
            return Pro2GuardResult(
                probability=None,
                state=encoded_state,
                matched_state=encoded_state,
                threshold=self.pg.threshold,
                allowed=allowed,
                reason=f"Pro2Guard failed {'closed' if self.pg.fail_closed else 'open'}: {exc}",
                mode=self.pg.mode,
                source="error",
            )

        allowed = probability <= self.pg.threshold
        reason = (
            f"Pro2Guard probability {probability:.4f} exceeds threshold {self.pg.threshold:.4f}"
            if not allowed
            else ""
        )
        return Pro2GuardResult(
            probability=probability,
            state=encoded_state,
            matched_state=matched_state,
            threshold=self.pg.threshold,
            allowed=allowed,
            reason=reason,
            mode=self.pg.mode,
            source=source,
        )

    def _probability(self, encoded_state: str) -> tuple[float, str, str]:
        if self._json_model is not None:
            probability, matched_state = self._json_model.probability_to_unsafe(
                encoded_state,
                self.pg.unsafe_states,
                self.pg.horizon,
            )
            return probability, matched_state, str(self._json_model.path)

        if not self._dtmc_path:
            return 0.0, encoded_state, "no_model"

        probability = query_prism_probability(
            prism_bin=self.pg.prism_bin,
            dtmc_path=self._dtmc_path,
            current_state=encoded_state,
            unsafe_states=self.pg.unsafe_states,
            timeout_seconds=self.pg.timeout_seconds,
        )
        return probability, encoded_state, self._dtmc_path


def _resolve_config_path(cfg: AppConfig, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(cfg.config_dir) / path).resolve())


def _load_abstraction(import_path: str) -> Any:
    if not import_path:
        return ToolTraceAbstraction()
    module_name, _, attr = import_path.partition(":")
    if not module_name or not attr:
        raise ValueError("Pro2Guard abstraction must use 'module:attribute' import syntax")
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    return obj() if isinstance(obj, type) else obj
