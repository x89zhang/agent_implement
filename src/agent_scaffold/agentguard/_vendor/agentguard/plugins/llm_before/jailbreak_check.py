"""Plugin for user/LLM input events."""
from __future__ import annotations

import re

from agentguard.plugins.base import BasePlugin, CheckResult
from agentguard.plugins.common.patterns import text_of
from agentguard.plugins.llm_before.jailbreak_templates import SUSPICIOUS_PROMPT_TEMPLATES
from agentguard.plugins.registry import register
from agentguard.schemas.context import RuntimeContext
from agentguard.schemas.decisions import GuardDecision
from agentguard.schemas.events import EventType, RuntimeEvent


@register(
    name="jailbreak_check",
    description="Detect prompt-injection and system-prompt leak attempts in LLM input.",
)
class JailbreakCheckPlugin(BasePlugin):
    event_types = [EventType.LLM_INPUT]

    def check(self, event: RuntimeEvent, context: RuntimeContext) -> CheckResult:
        # System/developer instructions and prior assistant output are trusted
        # runtime context. Scanning them causes the guard's own tool-use and
        # security instructions to trigger broad jailbreak patterns. User and
        # tool messages are the untrusted inputs this plugin is intended to
        # inspect.
        text = _untrusted_input_text(event.payload.messages)
        signals: list[str] = []
        matched_templates: dict[str, list[str]] = {}

        for signal, templates in SUSPICIOUS_PROMPT_TEMPLATES.items():
            matches = [
                template
                for template in templates
                if re.search(template, text, flags=re.IGNORECASE)
            ]
            if not matches:
                continue
            signals.append(signal)
            matched_templates[signal] = matches

        if not signals:
            return CheckResult.empty()

        metadata = {"matched_prompt_templates": matched_templates} if matched_templates else {}
        return CheckResult(
            decision_candidate=GuardDecision.deny(
                "Prompt blocked by local jailbreak_check plugin.",
                policy_id="local:jailbreak_check:jailbreak_detected",
                risk_signals=list(signals),
                metadata=metadata,
            ),
            risk_signals=signals,
            is_final=True,
            metadata=metadata,
        )


def _untrusted_input_text(messages: list[dict]) -> str:
    return text_of(
        [
            message.get("content", "")
            for message in messages
            if str(message.get("role", "")).lower() in {"user", "tool"}
        ]
    )
