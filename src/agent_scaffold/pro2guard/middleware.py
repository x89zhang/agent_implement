"""ProbGuard runtime monitor (upstream ``controlled_agent_excector.py`` / ``monitor_dtmc.py``).

After every executed tool step the agent state (all step observations so far)
is abstracted into a predicate bitstring. If the state was observed during
learning, ``P=? [ F unsafe ]`` (or ``F<=bound``) is computed on the learned
DTMC and compared with ``threshold`` using ``>=`` (``monitor_dtmc.py:120``,
``embodied/monitor.py:236``). Unobserved states are skipped and recorded, as
upstream. ``warn`` adds upstream's fixed reflection message to the observation;
``block`` stops the run (upstream ``eval_stop``); ``monitor`` only records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..middleware import Middleware, ModelDecision, ResultDecision, ToolDecision
from .abstraction import PredicateAbstraction, is_behavior_step, step_observation
from .generator import ABSTRACTION_FILE, MODEL_FILE, PRISM_FILE
from .model import JsonDTMC, Pro2GuardResult
from .prism import query_prism_probability

# controlled_agent_excector.py:107-109, verbatim.
REFLECTION_MESSAGE = (
    "There is a non-negligible probability that, from the current state, the unsafe state "
    "{unsafe_state} may eventually occur. You MUST NOT reach this unsafe state.\n"
    "Carefully plan your actions to avoid reaching this unsafe state."
)
STOP_MESSAGE = "Run stopped by Pro2Guard."


class Pro2GuardMiddleware(Middleware):
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.pg = cfg.pro2guard
        self.abstraction: PredicateAbstraction | None = None
        self.model: JsonDTMC | None = None
        self.unsafe_ids: set[int] = set()
        self.model_dir = self.pg.resolved_model_dir
        self._error = ""
        if not self.model_dir:
            self._error = self.pg.resolution_error or "no_model: no trained ProbGuard model resolved"
            return
        try:
            root = Path(self.model_dir)
            self.abstraction = PredicateAbstraction.from_dict(
                json.loads((root / ABSTRACTION_FILE).read_text(encoding="utf-8"))
            )
            self.model = JsonDTMC(root / MODEL_FILE)
            raw = json.loads((root / MODEL_FILE).read_text(encoding="utf-8"))
            # abs.filter(unsafe_spec) restricted to the learned states.
            self.unsafe_ids = {
                self.model.state_index[state]
                for state in self.abstraction.unsafe_states(self.model.state_index)
            }
            stored = set(raw.get("unsafe_state_indices") or [])
            if stored and stored != self.unsafe_ids:
                raise ValueError("stored unsafe states disagree with the abstraction")
            if not self.unsafe_ids:
                # Upstream skips a task whose spec identifies no learned state.
                self._error = "no_unsafe_state: the learned DTMC contains no unsafe state"
        except Exception as exc:  # noqa: BLE001 - recorded as a monitor error
            self._error = f"invalid_model: {exc}"

    # A run without tool calls would otherwise leave no Pro2Guard record.
    def guard_model_input(self, state: dict[str, Any], messages: list[dict[str, Any]]) -> ModelDecision:
        stop = state.get("_pro2guard_stop")
        if stop:
            return ModelDecision(False, stop, content=STOP_MESSAGE, terminate=True, decision_type="pro2guard_stop")
        if self._error and not state.get("_pro2guard_error_recorded"):
            state["_pro2guard_error_recorded"] = True
            self._record(state, self._error_result(0))
        return ModelDecision()

    def before_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any]) -> ToolDecision:
        stop = state.get("_pro2guard_stop")
        if stop:
            return ToolDecision(False, stop, replacement_result=STOP_MESSAGE, terminate=True,
                                decision_type="pro2guard_stop")
        return ToolDecision(True, "")

    def after_tool(self, state: dict[str, Any], name: str, payload: dict[str, Any], result: str, failed: bool) -> ResultDecision:
        if not is_behavior_step(name):
            return ResultDecision(result=result)
        history = state.setdefault("_pro2guard_history", [])
        history.append(step_observation(name, payload, result, failed))
        evaluated = self._evaluate(list(history), len(history))
        self._record(state, evaluated)
        if evaluated.allowed or self.pg.mode == "monitor":
            return ResultDecision(result=result)
        if self.pg.mode == "warn":
            return ResultDecision(result=_attach_message(result, self._message()))
        # block: the observed step already happened; stop before the next one.
        state["_pro2guard_stop"] = evaluated.reason
        return ResultDecision(result=result)

    def _record(self, state: dict[str, Any], evaluated: Pro2GuardResult) -> None:
        state["_last_pro2guard_decision"] = evaluated.to_dict()
        state.setdefault("pro2guard_events", []).append(evaluated.to_dict())

    def _message(self) -> str:
        spec = self.abstraction.unsafe_spec_text() if self.abstraction else ""
        return REFLECTION_MESSAGE.format(unsafe_state=spec)

    def _error_result(self, step: int) -> Pro2GuardResult:
        return Pro2GuardResult(
            probability=None, state="", threshold=self.pg.threshold,
            allowed=not self.pg.fail_closed,
            reason=f"Pro2Guard unavailable: {self._error}",
            mode=self.pg.mode, source="error", bound=self.pg.bound, step=step,
            model=str(self.model_dir or ""), error=self._error,
        )

    def _evaluate(self, history: list[dict[str, Any]], step: int) -> Pro2GuardResult:
        if self._error:
            return self._error_result(step)
        assert self.abstraction is not None and self.model is not None
        encoded = self.abstraction.encode(history)
        base = dict(state=encoded, threshold=self.pg.threshold, mode=self.pg.mode,
                    bound=self.pg.bound, step=step, model=str(self.model_dir))
        if encoded not in self.model.state_index:
            # monitor_dtmc.py:108-110: unobserved in training data, skipping.
            return Pro2GuardResult(probability=None, allowed=True, reason="state unobserved in training data",
                                   source="unseen", skipped=True, **base)
        index = self.model.state_index[encoded]
        try:
            if self.pg.engine == "prism":
                probability = query_prism_probability(
                    prism_bin=self.pg.prism_bin, dtmc_path=str(Path(self.model_dir) / PRISM_FILE),
                    current_state=index, unsafe_indices=sorted(self.unsafe_ids),
                    bound=self.pg.bound, timeout_seconds=self.pg.timeout_seconds,
                )
            else:
                probability = self.model.probability_to_unsafe(index, self.unsafe_ids, self.pg.bound)
        except Exception as exc:  # noqa: BLE001 - recorded as a monitor error
            return Pro2GuardResult(probability=None, allowed=not self.pg.fail_closed,
                                   reason=f"Pro2Guard failed: {exc}", source="error",
                                   state_index=index, error=str(exc), **base)
        alarm = probability >= self.pg.threshold
        reason = (
            f"Pro2Guard P(F unsafe)={probability:.4f} >= threshold {self.pg.threshold:.4f}" if alarm else ""
        )
        return Pro2GuardResult(probability=probability, allowed=not alarm, reason=reason,
                               source="dtmc", state_index=index, **base)


def _attach_message(result: Any, message: str) -> str:
    """Upstream sets ``observation["message"]``; keep JSON results JSON."""
    try:
        value = json.loads(result)
    except (TypeError, ValueError):
        value = None
    if isinstance(value, dict):
        value["message"] = message
        return json.dumps(value, ensure_ascii=False)
    return f"{result}\n\nmessage: {message}"
