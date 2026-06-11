from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

from .config import AppConfig


@dataclass
class SkillSpec:
    name: str
    description: str = ""
    triggers: list[str] | None = None
    requires_tools: list[str] | None = None
    inject_mode: str = "system_append"
    priority: int = 50
    path: str = ""
    instructions: str = ""

    def to_trace(self) -> dict[str, Any]:
        data = asdict(self)
        data["instructions"] = self.instructions[:500]
        return data


def load_enabled_skills(cfg: AppConfig) -> list[SkillSpec]:
    specs: list[SkillSpec] = []
    for item in cfg.skills.enabled:
        spec = _load_skill(cfg, item)
        if spec is not None:
            specs.append(spec)
    specs.sort(key=lambda spec: (spec.priority, spec.name))
    return specs


def render_skill_context(skills: list[SkillSpec]) -> str:
    if not skills:
        return ""
    sections = ["# Enabled Skills", "Use these reusable skill instructions when they apply to the task."]
    for skill in skills:
        sections.append(f"\n## {skill.name}")
        if skill.description:
            sections.append(skill.description.strip())
        if skill.instructions.strip():
            sections.append(skill.instructions.strip())
    return "\n".join(sections).strip()


def validate_skill_tools(skills: list[SkillSpec], tool_names: set[str]) -> list[str]:
    warnings: list[str] = []
    for skill in skills:
        missing = [name for name in (skill.requires_tools or []) if name not in tool_names]
        if missing:
            warnings.append(f"Skill {skill.name} requires missing tools: {', '.join(missing)}")
    return warnings


def _load_skill(cfg: AppConfig, item: Any) -> SkillSpec | None:
    if isinstance(item, str):
        name = item
        explicit_path = ""
        inline: dict[str, Any] = {}
    elif isinstance(item, dict):
        name = str(item.get("name") or item.get("path") or "").strip()
        explicit_path = str(item.get("path") or "").strip()
        inline = item
    else:
        return None
    if not name and not explicit_path:
        return None

    skill_dir = _resolve_skill_dir(cfg, name, explicit_path)
    meta: dict[str, Any] = {}
    instructions = ""
    if skill_dir is not None:
        meta_path = skill_dir / "skill.yaml"
        if meta_path.exists():
            loaded = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta.update(loaded)
        md_path = skill_dir / "SKILL.md"
        if md_path.exists():
            instructions = md_path.read_text(encoding="utf-8").strip()
    meta.update(inline)

    resolved_name = str(meta.get("name") or name or (skill_dir.name if skill_dir else "skill"))
    if not instructions:
        instructions = str(meta.get("instructions") or "").strip()
    return SkillSpec(
        name=resolved_name,
        description=str(meta.get("description") or ""),
        triggers=_string_list(meta.get("triggers")),
        requires_tools=_string_list(meta.get("requires_tools")),
        inject_mode=str(meta.get("inject_mode") or "system_append"),
        priority=int(meta.get("priority") or 50),
        path=str(skill_dir or explicit_path),
        instructions=instructions,
    )


def _resolve_skill_dir(cfg: AppConfig, name: str, explicit_path: str) -> Path | None:
    config_dir = Path(cfg.config_dir).resolve()
    cwd = Path.cwd().resolve()
    candidates: list[Path] = []
    if explicit_path:
        raw = Path(explicit_path)
        candidates.extend([raw if raw.is_absolute() else config_dir / raw, cwd / raw])
    if name:
        candidates.extend([
            config_dir / cfg.skills.base_dir / name,
            cwd / cfg.skills.base_dir / name,
            config_dir / name,
            cwd / name,
        ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
