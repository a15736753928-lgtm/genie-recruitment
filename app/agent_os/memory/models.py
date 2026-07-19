"""
Memory data models — file-based persistent memory with frontmatter.

Each memory is stored as a Markdown file under ``memory/`` with YAML
frontmatter for structured metadata.  An index file (MEMORY.md) provides
a quick scan target for the LLM-based retriever.

Inspired by Claude Code's memory system: files not vectors, LLM scan not
embedding similarity.
"""

from __future__ import annotations

import os
import re
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ═══════════════════════════════════════════════════════════
#  Data Model
# ═══════════════════════════════════════════════════════════

@dataclass
class Memory:
    """A single persistent memory entry."""

    name: str                                    # kebab-case slug (filename stem)
    description: str                             # one-line summary for index scan
    content: str                                 # markdown body
    metadata: dict = field(default_factory=dict) # type, scope, agent, tags, …
    created: str = ""                            # ISO-format timestamp
    updated: str = ""                            # ISO-format timestamp

    # ── computed ──

    @property
    def filename(self) -> str:
        return f"{self.name}.md"

    @property
    def memory_type(self) -> str:
        return self.metadata.get("type", "reference")

    @property
    def scope(self) -> str:
        return self.metadata.get("scope", "general")

    @property
    def agent(self) -> str:
        return self.metadata.get("agent", "genie")

    # ── serialization ──

    def to_frontmatter(self) -> str:
        """Render the full markdown file (frontmatter + body)."""
        fm = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "created": self.created,
            "updated": self.updated,
        }
        yaml_str = yaml.dump(fm, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return f"---\n{yaml_str.strip()}\n---\n\n{self.content.strip()}\n"

    @classmethod
    def from_frontmatter(cls, text: str) -> Optional["Memory"]:
        """Parse a markdown file with YAML frontmatter into a Memory."""
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
            metadata=fm.get("metadata", {}),
            created=fm.get("created", ""),
            updated=fm.get("updated", ""),
        )

    def to_prompt_fragment(self) -> str:
        """Compact representation injected into the system prompt."""
        return (
            f"**{self.description}** (scope: {self.scope}, type: {self.memory_type})\n"
            f"{self.content[:500]}"
        )

    def to_index_line(self) -> str:
        """One-line entry for MEMORY.md."""
        return f"- [{self.description}]({self.filename}) — {self.memory_type}"


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
