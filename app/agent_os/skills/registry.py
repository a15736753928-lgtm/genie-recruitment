"""
SkillRegistry — progressive-disclosure skill loading.

Scans the ``skills/`` directory at startup, parses all SKILL.md files,
and provides on-demand loading and LLM-based matching.

The catalog (names + one-line descriptions only) is always available for
injection into the system prompt.  Full skill bodies are loaded only when
triggered by user intent or explicit invocation.
"""

from __future__ import annotations

import os
import re
import json
from typing import Optional

from app.agent_os.skills.models import Skill


class SkillRegistry:
    """Registry of available domain skills with progressive loading."""

    def __init__(self, skills_dir: str = "skills/"):
        self._dir = skills_dir
        self._skills: dict[str, Skill] = {}
        self._catalog: str = ""
        self.reload()

    # ── Loading ───────────────────────────────────────────

    def reload(self):
        """Rescan the skills directory and reload all SKILL.md files."""
        self._skills.clear()
        if not os.path.isdir(self._dir):
            return

        for root, dirs, files in os.walk(self._dir):
            for fn in files:
                if fn == "SKILL.md":
                    path = os.path.join(root, fn)
                    skill = Skill.from_file(path)
                    if skill and skill.enabled:
                        self._skills[skill.name] = skill

        # Rebuild catalog
        self._catalog = self._build_catalog()

    def _build_catalog(self) -> str:
        """Build the lightweight catalog string."""
        if not self._skills:
            return "（无可用技能）"
        lines = ["## 可用技能\n"]
        for skill in sorted(self._skills.values(), key=lambda s: -s.priority):
            lines.append(skill.to_catalog_entry())
        return "\n".join(lines)

    # ── Query ─────────────────────────────────────────────

    def list_all(self) -> list[Skill]:
        """Return all loaded skills."""
        return list(self._skills.values())

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_catalog(self) -> str:
        """Get the catalog string (for system prompt injection)."""
        return self._catalog

    # ── Matching ──────────────────────────────────────────

    def match_keyword(self, user_message: str, limit: int = 3) -> list[Skill]:
        """Fast keyword-based matching — no LLM call.

        Matches individual character bigrams from trigger words against
        the user message.  "筛选简历" will match "帮我筛选候选人" because
        "筛选" (2 chars) appears in both.
        """
        msg_lower = user_message.lower()
        scored = []
        import re as _re
        for skill in self._skills.values():
            score = 0
            for trigger in skill.triggers:
                trigger_lower = trigger.lower()
                # Full trigger match = strong signal
                if trigger_lower in msg_lower:
                    score += len(trigger) * 3
                else:
                    # Extract 2-char substrings and match individually
                    trig_chars = _re.sub(r'[，,、\s]+', '', trigger_lower)
                    for i in range(len(trig_chars) - 1):
                        bigram = trig_chars[i:i+2]
                        if bigram in msg_lower:
                            score += 1
            if score > 0:
                scored.append((score + skill.priority, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    async def match_llm(
        self,
        user_message: str,
        llm_client=None,
        limit: int = 3,
    ) -> list[Skill]:
        """LLM-based matching — more nuanced than keyword.

        Shows the model the skill catalog and asks which skills are
        most relevant to the user's request.
        """
        if not self._skills or not llm_client:
            return self.match_keyword(user_message, limit)

        # Build a compact option list
        options = [
            {"name": s.name, "desc": s.description, "triggers": s.triggers[:3]}
            for s in sorted(self._skills.values(), key=lambda s: -s.priority)
        ]

        prompt = f"""你是一个技能路由器。根据用户请求，选择最相关的技能。

可用技能列表：
{json.dumps(options, ensure_ascii=False, indent=2)}

用户请求：{user_message[:300]}

返回一个 JSON 数组，包含最相关技能的名称（如 ["resume-screening", "interview-question-gen"]），
最多 {limit} 个。只返回 JSON 数组，不要其他文字。如果没有相关的，返回 []。"""

        try:
            resp = await llm_client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            names = json.loads(raw)
            if not isinstance(names, list):
                return []
        except Exception:
            return self.match_keyword(user_message, limit)

        return [self._skills[n] for n in names if n in self._skills][:limit]

    async def match(
        self,
        user_message: str,
        llm_client=None,
        limit: int = 3,
    ) -> list[Skill]:
        """Match skills to a user message.

        Uses LLM if available, falls back to keyword matching.
        """
        if llm_client and len(self._skills) > 5:
            return await self.match_llm(user_message, llm_client, limit)
        return self.match_keyword(user_message, limit)

    # ── Prompt helpers ────────────────────────────────────

    def inject_skills(
        self,
        system_prompt: str,
        matched_skills: list[Skill],
    ) -> str:
        """Append matched skill content to the system prompt."""
        if not matched_skills:
            return system_prompt
        skill_text = "\n\n".join(s.to_prompt() for s in matched_skills)
        return f"{system_prompt}\n\n---\n\n## 当前激活技能\n\n{skill_text}"
