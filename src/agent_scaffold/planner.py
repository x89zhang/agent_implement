from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .config import AppConfig


@dataclass
class PlanStep:
    id: int
    description: str
    status: str = "pending"
    evidence: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def initialize_plan(cfg: AppConfig, task: str) -> list[dict[str, Any]]:
    if not cfg.planner.enabled:
        return []
    raw_steps = cfg.planner.steps[: cfg.planner.max_steps]
    if not raw_steps and task.strip():
        raw_steps = ["Understand the task and success criteria.", "Use available tools to gather required evidence.", "Produce the final answer or artifact and verify it against the task."]
    return [PlanStep(id=index + 1, description=step, evidence=[]).to_dict() for index, step in enumerate(raw_steps)]


def render_plan_context(state: dict[str, Any]) -> str:
    plan = state.get("plan") or []
    if not plan:
        return ""
    lines = ["# Current Plan", "Use this plan as the task checklist. Update your behavior based on completed and pending steps."]
    for step in plan:
        if not isinstance(step, dict):
            continue
        status = str(step.get("status", "pending"))
        desc = str(step.get("description", ""))
        lines.append(f"- [{status}] {step.get('id', '?')}. {desc}")
    return "\n".join(lines)


def mark_plan_progress(state: dict[str, Any], event: str, detail: str = "") -> None:
    plan = state.get("plan")
    if not isinstance(plan, list) or not plan:
        return
    for step in plan:
        if isinstance(step, dict) and step.get("status") in {"pending", "in_progress"}:
            if step.get("status") == "pending":
                step["status"] = "in_progress"
            evidence = step.setdefault("evidence", [])
            if isinstance(evidence, list) and detail:
                evidence.append(f"{event}: {detail[:200]}")
            break


def complete_plan_on_final(state: dict[str, Any]) -> None:
    plan = state.get("plan")
    if not isinstance(plan, list):
        return
    for step in plan:
        if isinstance(step, dict) and step.get("status") in {"pending", "in_progress"}:
            step["status"] = "done"
