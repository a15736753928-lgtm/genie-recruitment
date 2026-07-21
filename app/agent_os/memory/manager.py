"""
MemoryManager — CRUD operations for file-based persistent memory.

Reads/writes markdown files under ``memory/`` and keeps MEMORY.md
(the index) in sync.  Also syncs to PostgreSQL as a durability backup.

Usage::

    mm = MemoryManager(memory_dir="memory/")
    await mm.remember(
        name="hr-preference-985",
        content="张经理偏好 985 学历候选人...",
        description="HR 张经理偏好 985 学历候选人",
        metadata={"type": "preference", "scope": "candidate_screening"},
    )
    memories = await mm.recall(query="筛选候选人", limit=5)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Optional

from app.config import get_settings
from app.agent_os.memory.models import Memory, _now_iso

_MEMORY_FILENAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


class MemoryManager:
    """File-based memory CRUD with optional DB backup."""

    def __init__(self, memory_dir: str = "memory/"):
        os.makedirs(memory_dir, exist_ok=True)
        self._dir = os.path.realpath(memory_dir)
        self._index_path = os.path.join(self._dir, "MEMORY.md")

    # ── Path helpers ──────────────────────────────────────

    def _path(self, name: str) -> str:
        """Resolve the safe absolute path for a memory slug.

        Rejects names containing path separators or ``..`` patterns
        to prevent directory traversal attacks.
        """
        if not _MEMORY_FILENAME_RE.match(name):
            raise ValueError(
                f"Invalid memory name: {name!r}. "
                f"Must match {_MEMORY_FILENAME_RE.pattern}"
            )
        path = os.path.join(self._dir, f"{name}.md")
        # Defense-in-depth: resolve symlinks and verify containment
        real_path = os.path.realpath(path)
        if not real_path.startswith(self._dir + os.sep):
            raise ValueError(f"Path traversal blocked: {name!r}")
        return real_path

    def _exists(self, name: str) -> bool:
        return os.path.isfile(self._path(name))

    # ── Read ───────────────────────────────────────────────

    def list_names(self) -> list[str]:
        """Return all memory slug names (without .md)."""
        names = []
        if not os.path.isdir(self._dir):
            return names
        for fn in os.listdir(self._dir):
            if fn.endswith(".md") and fn != "MEMORY.md":
                stem = fn[:-3]
                if _MEMORY_FILENAME_RE.match(stem):
                    names.append(stem)
        return sorted(names)

    def load(self, name: str) -> Optional[Memory]:
        """Load a single memory by slug name."""
        path = self._path(name)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        mem = Memory.from_frontmatter(text)
        if mem and not mem.name:
            mem.name = name
        return mem

    def load_all(self) -> list[Memory]:
        """Load every memory file (excluding the index)."""
        memories = []
        for name in self.list_names():
            mem = self.load(name)
            if mem:
                memories.append(mem)
        return memories

    def load_index(self) -> str:
        """Read MEMORY.md content (or empty string if missing)."""
        if os.path.isfile(self._index_path):
            with open(self._index_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    # ── Write ──────────────────────────────────────────────

    def save(self, memory: Memory):
        """Persist a memory to its markdown file."""
        now = _now_iso()
        if not memory.created:
            memory.created = now
        memory.updated = now
        path = self._path(memory.name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(memory.to_frontmatter())
        self._rebuild_index()

    async def remember(
        self,
        name: str,
        content: str,
        description: str,
        metadata: Optional[dict] = None,
    ) -> Memory:
        """Create or update a memory.

        If a memory with the same ``name`` already exists, its content
        and metadata are merged (new keys overwrite old).
        """
        existing = self.load(name)
        if existing:
            existing.content = content
            existing.description = description
            if metadata:
                existing.metadata = {**existing.metadata, **metadata}
            self.save(existing)
            return existing

        mem = Memory(
            name=name,
            description=description,
            content=content,
            metadata=metadata or {},
        )
        self.save(mem)
        return mem

    def forget(self, name: str) -> bool:
        """Delete a memory file.  Returns True if it existed."""
        path = self._path(name)
        if os.path.isfile(path):
            os.remove(path)
            self._rebuild_index()
            return True
        return False

    # ── Index maintenance ─────────────────────────────────

    def _rebuild_index(self):
        """Regenerate MEMORY.md from all memory files."""
        memories = self.load_all()
        lines = [
            "# Genie Recruitment System — Memory Index",
            "",
            "> Auto-generated index.  Do not edit by hand.",
            "",
        ]
        for mem in memories:
            lines.append(mem.to_index_line())
        lines.append("")
        with open(self._index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ── Auto-capture ──────────────────────────────────────

    async def auto_capture(
        self,
        session_messages: list,
        session_id: str,
        llm_client=None,
    ) -> list[Memory]:
        """Analyze a completed session and extract learnings.

        Uses a lightweight LLM call to identify facts worth remembering.
        Falls back gracefully if no LLM client is provided.
        """
        if not session_messages or not llm_client:
            return []

        # Build a compact conversation summary for the LLM
        summary_lines = []
        for m in session_messages[-20:]:   # last 20 messages max
            role = getattr(m, "role", "unknown")
            content = str(getattr(m, "content", ""))[:300]
            summary_lines.append(f"[{role}] {content}")
        conversation = "\n".join(summary_lines)

        prompt = f"""分析以下招聘系统对话，提取值得长期记住的信息。
返回 JSON 数组（不要 markdown，不要解释）。如果没有值得记住的信息，返回空数组 []。

值得记住的信息类型：
- user_preference: 用户的偏好、习惯、特殊要求
- business_rule: 业务规则、流程变更
- candidate_note: 候选人关键信息（需要后续跟进的事项）
- project_context: 项目背景、团队信息

每条信息格式：
{{
  "name": "kebab-case-slug",
  "description": "一句话描述",
  "content": "详细内容（含 Why 和 How to apply）",
  "type": "user_preference | business_rule | candidate_note | project_context",
  "scope": "recruitment | interview | probation | performance | general"
}}

对话内容：
{conversation[:3000]}

JSON:"""

        try:
            resp = await llm_client.chat.completions.create(
                model=get_settings().deepseek_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            import json
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            items = json.loads(raw)
        except Exception:
            return []

        if not isinstance(items, list):
            return []

        created = []
        for item in items:
            if not isinstance(item, dict) or "name" not in item:
                continue
            # Clean the slug
            slug = re.sub(r"[^a-z0-9-]", "", item["name"].lower().replace(" ", "-"))[:64]
            if not slug:
                continue
            mem = await self.remember(
                name=slug,
                content=item.get("content", ""),
                description=item.get("description", ""),
                metadata={
                    "type": item.get("type", "reference"),
                    "scope": item.get("scope", "general"),
                },
            )
            created.append(mem)

        return created
