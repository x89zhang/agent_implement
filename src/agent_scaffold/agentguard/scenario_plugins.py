from __future__ import annotations

from typing import Any, ClassVar

from agentguard.plugins.base import BasePlugin, CheckResult
from agentguard.plugins.common.patterns import text_of
from agentguard.schemas.context import RuntimeContext
from agentguard.schemas.events import EventType, RuntimeEvent


class ScenarioSignalPlugin(BasePlugin):
    """Tag scenario-specific instruction-override phrases without making decisions."""

    name = "scenario_signals"
    description = "Detect prompt-injection phrases compiled for the current scenario."
    event_types: ClassVar[list[EventType]] = [
        EventType.LLM_INPUT,
        EventType.LLM_OUTPUT,
        EventType.TOOL_INVOKE,
        EventType.TOOL_RESULT,
    ]

    def __init__(
        self,
        *,
        phrases: list[str] | None = None,
        env: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, phrases=phrases or [], **kwargs)
        self.phrases = [
            str(phrase).strip().casefold()
            for phrase in phrases or []
            if str(phrase).strip()
        ]

    def check(self, event: RuntimeEvent, context: RuntimeContext) -> CheckResult:
        del context
        payload = event.payload.to_dict()
        text = text_of(payload).casefold()
        matches = [phrase for phrase in self.phrases if phrase in text]
        if not matches:
            return CheckResult.empty()
        signals = ["prompt_injection", "scenario_instruction_override"]
        if event.event_type == EventType.TOOL_RESULT:
            signals.append("tool_result_injection")
        return CheckResult(
            risk_signals=signals,
            metadata={"scenario_phrase_matches": matches[:10]},
        )


__all__ = ["ScenarioSignalPlugin"]
