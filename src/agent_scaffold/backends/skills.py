"""Materialize configured skills into an isolated native Hermes skill library."""

import hashlib
import re

import yaml

from ..skills import load_enabled_skills


def stage_skills(cfg, home):
    entries = []
    for skill in load_enabled_skills(cfg):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", skill.name):
            raise ValueError(
                f"Hermes skill name must be a single safe directory name: {skill.name!r}"
            )
        directory = home / "skills" / skill.name
        directory.mkdir(parents=True, exist_ok=False)
        content = (
            "---\n"
            + yaml.safe_dump(
                {"name": skill.name, "description": skill.description or skill.name}
            )
            + "---\n\n"
            + skill.instructions
            + "\n"
        )
        (directory / "SKILL.md").write_text(content, encoding="utf-8")
        entries.append(
            {
                "name": skill.name,
                "source": skill.path,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "delivery": "native_skill_view",
                "path": str(directory / "SKILL.md"),
            }
        )
    return entries
