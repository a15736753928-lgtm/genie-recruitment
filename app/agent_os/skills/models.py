"""
Skill data models — domain expertise packaged as markdown files.

Each skill is defined by a SKILL.md file with YAML frontmatter that
declares its name, description, triggers, and agent assignment.
The body contains domain-specific workflows, rules, and templates.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class Skill:
    """A loaded skill — domain expertise injected into the agent on demand."""

    name: str                                    # kebab-case identifier
    description: str                             # one-line summary for catalog
    content: str                                 # full markdown body
    triggers: list[str] = field(default_factory=list)  # keywords that activate this skill
    agent: str = "genie"                         # which agent this skill belongs to
    priority: int = 0                            # higher = matched first
    enabled: bool = True
    file_path: str = ""                          # source file location

    # ── serialization ──

    @classmethod
    def from_file(cls, path: str) -> Optional["Skill"]:
        """Parse a SKILL.md file into a Skill object."""
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls.from_text(text, path)

    @classmethod
    def from_text(cls, text: str, source_path: str = "") -> Optional["Skill"]:
        """Parse markdown with YAML frontmatter into a Skill."""
        if not text.startswith("---"):
            return None
        parts = re.split(r"^---\s*$", text, maxsplit=2, flags=re.MULTILINE)
        if len(parts) < 3:
            return None
        try:
            fm = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            return None
        if not isinstance(fm, dict):
            return None
        return cls(
            name=fm.get("name", ""),
            description=fm.get("description", ""),
            content=parts[2].strip(),
            triggers=fm.get("triggers", []),
            agent=fm.get("agent", "genie"),
            priority=fm.get("priority", 0),
            enabled=fm.get("enabled", True),
            file_path=source_path,
        )

    # ── prompt fragments ──

    def to_catalog_entry(self) -> str:
        """Compact entry for the skill catalog in system prompt."""
        return f"- **{self.name}**: {self.description}（触发词: {', '.join(self.triggers[:3])}）"

    def to_prompt(self) -> str:
        """Full skill content for injection into system prompt."""
        header = f"## 技能: {self.description}\n\n"
        if self.triggers:
            header += f"*适用场景: {', '.join(self.triggers)}*\n\n"
        return header + self.content

    def to_api_dict(self) -> dict:
        """API response format."""
        return {
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "agent": self.agent,
            "priority": self.priority,
            "enabled": self.enabled,
        }
