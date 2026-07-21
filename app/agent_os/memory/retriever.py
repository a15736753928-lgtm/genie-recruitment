"""
MemoryRetriever — LLM-based memory selection.

Given a user query, scans the memory index (MEMORY.md) and picks the
most relevant memories.  Uses the LLM (not embeddings) for selection,
matching Claude Code's approach: the model reads a short index of
titles + descriptions, then we load the full content of selected files.
"""

from __future__ import annotations

import os
import re
import json
from typing import Optional

from app.config import get_settings
from app.agent_os.memory.manager import MemoryManager
from app.agent_os.memory.models import Memory

# How many entries from the index to show the LLM per batch
_MAX_INDEX_ENTRIES = 50


class MemoryRetriever:
    """Select relevant memories for the current conversation turn."""

    def __init__(self, manager: MemoryManager):
        self._manager = manager

    # ── LLM-based retrieval ───────────────────────────────

    async def retrieve(
        self,
        query: str,
        llm_client=None,
        limit: int = 5,
        scope: Optional[str] = None,
    ) -> list[Memory]:
        """Return the top-K most relevant memories for a query.

        Strategy:
        1. If ≤ 15 total memories, just load them all (cheaper than an LLM call).
        2. Otherwise, send the index (titles + descriptions) to a fast LLM
           and ask it to pick the most relevant ones.
        3. Fall back to keyword matching if no LLM client is available.
        """
        all_names = self._manager.list_names()
        if not all_names:
            return []

        # Filter by scope if requested
        if scope:
            candidates = [
                m for m in (self._manager.load(n) for n in all_names)
                if m and m.scope == scope
            ]
        else:
            candidates = [self._manager.load(n) for n in all_names]
        candidates = [m for m in candidates if m is not None]

        if not candidates:
            return []

        # If there are very few memories, skip the LLM call
        if len(candidates) <= limit * 3 and len(candidates) <= 15:
            # Simple keyword relevance ranking
            ranked = self._keyword_rank(query, candidates)
            return ranked[:limit]

        # Use LLM to select
        if llm_client:
            try:
                return await self._llm_select(query, candidates, llm_client, limit)
            except Exception:
                pass

        # Fallback
        return self._keyword_rank(query, candidates)[:limit]

    # ── Quick path: load everything ───────────────────────

    async def retrieve_all(self, scope: Optional[str] = None) -> list[Memory]:
        """Load every memory, optionally filtered by scope."""
        all_names = self._manager.list_names()
        memories = [self._manager.load(n) for n in all_names]
        memories = [m for m in memories if m is not None]
        if scope:
            memories = [m for m in memories if m.scope == scope]
        return memories

    # ── LLM selection ─────────────────────────────────────

    async def _llm_select(
        self,
        query: str,
        candidates: list[Memory],
        llm_client,
        limit: int,
    ) -> list[Memory]:
        """Ask the LLM to pick the most relevant memories."""
        # Build compact index
        lines = ["以下是已存储的记忆索引，请选择与用户查询最相关的（最多 {} 条）：\n".format(limit)]
        for i, m in enumerate(candidates[:50]):
            lines.append(f"{i}: [{m.memory_type}] {m.description} (scope: {m.scope})")
        index_text = "\n".join(lines)

        prompt = f"""{index_text}

用户查询：{query[:300]}

返回一个 JSON 数组，包含最相关记忆的编号（如 [0, 3, 7]），最多 {limit} 个。
只返回 JSON 数组，不要其他文字。如果没有相关的，返回 []。"""

        resp = await llm_client.chat.completions.create(
            model=get_settings().deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        indices = json.loads(raw)
        if not isinstance(indices, list):
            return []
        return [candidates[i] for i in indices if 0 <= i < len(candidates)][:limit]

    # ── Keyword fallback ──────────────────────────────────

    def _keyword_rank(self, query: str, candidates: list[Memory]) -> list[Memory]:
        """Simple keyword overlap ranking (no LLM needed)."""
        query_lower = query.lower()
        scored = []
        for m in candidates:
            score = 0
            desc_lower = m.description.lower()
            # Title/description match
            for word in query_lower.split():
                if word in desc_lower:
                    score += 3
            # Content match
            content_lower = m.content.lower()
            for word in query_lower.split():
                if len(word) >= 2 and word in content_lower:
                    score += 1
            # Scope bonus
            if m.scope in query_lower:
                score += 2
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]
