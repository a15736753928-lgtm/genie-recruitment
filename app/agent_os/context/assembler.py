"""
SystemPromptBuilder — layered system prompt assembly.

Builds the system prompt in layers so that the static (cacheable) portion
can be separated from the dynamic (session-specific) portion.  This
enables prompt caching on the LLM API side for significant cost savings.

Inspired by Claude Code's ``SYSTEM_PROMPT_DYNAMIC_BOUNDARY`` pattern.
"""

from __future__ import annotations

from typing import Optional

from app.agent_os.memory.models import Memory


class SystemPromptBuilder:
    """Assembles the full system prompt from cached and dynamic layers.

    Layers (top-to-bottom, static first):
      L1: Role definition + core rules  (from prompts/*.txt — static)
      L2: Tool catalog                  (names + one-liners — static)
      L3: Agent-specific instructions   (from AGENT_CONFIGS — static)
      --- DYNAMIC BOUNDARY (cache split) ---
      L4: Relevant memories             (MemoryManager.recall — dynamic)
      L5: Session context               (current position, candidate — dynamic)
      L6: Runtime directives            (intent gate, plan steps — dynamic)

    Layers 1-3 can be cached by the LLM API and reused across turns.
    Layers 4-6 change every turn and sit below the cache boundary.
    """

    def __init__(self):
        self._static_cache: Optional[str] = None
        self._static_cache_key: Optional[str] = None

    # ── Build ──────────────────────────────────────────────

    def build(
        self,
        base_prompt: str,
        agent_id: str,
        agent_config: dict,
        tool_catalog: str = "",
        memories: Optional[list[Memory]] = None,
        session_context: str = "",
        runtime_directives: str = "",
    ) -> tuple[str, str, str]:
        """Build the full system prompt and return (full, static, dynamic).

        Returns:
            full: The complete system prompt.
            static: The cacheable prefix (layers 1-3).
            dynamic: The per-turn suffix (layers 4-6).
        """
        cache_key = f"{agent_id}:{tool_catalog[:100]}"

        # ── Static portion ──
        if self._static_cache is None or self._static_cache_key != cache_key:
            parts = [base_prompt]
            if tool_catalog:
                parts.append(f"\n\n## 可用工具\n\n{tool_catalog}")
            if agent_config:
                parts.append(
                    f"\n\n你当前的角色是 **{agent_config.get('name', agent_id)}**，"
                    f"{agent_config.get('description', '')}"
                )
            self._static_cache = "\n".join(parts)
            self._static_cache_key = cache_key

        # ── Dynamic portion ──
        dynamic_parts = []

        if memories:
            mem_text = "\n".join(
                f"### {m.description}\n{m.content[:400]}"
                for m in memories
            )
            dynamic_parts.append(f"\n\n## 相关记忆\n\n{mem_text}")

        if session_context:
            dynamic_parts.append(f"\n\n## 当前上下文\n\n{session_context}")

        if runtime_directives:
            dynamic_parts.append(f"\n\n{runtime_directives}")

        dynamic = "\n".join(dynamic_parts)
        full = self._static_cache + dynamic

        return full, self._static_cache, dynamic

    # ── Tool catalog (deferred loading support) ────────────

    @staticmethod
    def build_tool_catalog(
        tool_defs: list[dict],
        deferred: bool = True,
    ) -> str:
        """Build a compact tool catalog string.

        Args:
            tool_defs: List of tool definition dicts (from TOOL_REGISTRY).
            deferred: If True, only emit name + one-line description.
                      Full schemas are loaded on demand when the LLM
                      requests a specific tool.

        Returns:
            Catalog text suitable for the system prompt.
        """
        if not tool_defs:
            return "（无可用工具）"

        lines = []
        # Group by category for readability
        categories: dict[str, list[dict]] = {}
        for td in tool_defs:
            cat = _tool_category(td["name"])
            categories.setdefault(cat, []).append(td)

        for cat, tools in sorted(categories.items()):
            lines.append(f"### {cat}")
            for t in tools:
                if deferred:
                    desc = t.get("description", "").split("。")[0]  # first sentence only
                    lines.append(f"- **{t['name']}**: {desc}")
                else:
                    lines.append(f"- **{t['name']}**: {t.get('description', '')}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def build_tool_detail(tool_def: dict) -> str:
        """Build the full detail for a single tool (on-demand loading)."""
        name = tool_def["name"]
        desc = tool_def.get("description", "")
        params = tool_def.get("parameters", {}).get("properties", {})
        required = tool_def.get("parameters", {}).get("required", [])

        lines = [f"### {name}", f"", desc, "", "**参数：**"]
        for pname, pschema in params.items():
            req_mark = " *(必填)*" if pname in required else ""
            pdesc = pschema.get("description", "")
            ptype = pschema.get("type", "string")
            lines.append(f"- `{pname}` ({ptype}){req_mark}: {pdesc}")

        return "\n".join(lines)

    # ── Cache management ──────────────────────────────────

    def invalidate_cache(self):
        """Clear the static cache (call when prompts/tools change)."""
        self._static_cache = None
        self._static_cache_key = None


# ── Helpers ────────────────────────────────────────────────

def _tool_category(name: str) -> str:
    """Map a tool name to its display category."""
    CATEGORY_MAP = {
        "resume": "📋 候选人管理",
        "position": "📌 岗位管理",
        "question": "📝 面试出题",
        "evaluation": "📊 面试评定",
        "leaderboard": "📊 面试评定",
        "ranking": "📊 面试评定",
        "score": "📊 面试评定",
        "probation": "🌱 试用期考核",
        "save_week": "🌱 试用期考核",
        "save_conversion": "🌱 试用期考核",
        "performance": "📈 绩效管理",
        "grade": "📈 绩效管理",
        "bonus": "📈 绩效管理",
        "quarter": "📈 绩效管理",
        "knowledge": "📚 知识库",
        "rag": "📚 知识库",
        "recall": "📚 知识库",
        "document": "📚 知识库",
        "dashboard": "📊 数据看板",
        "settings": "⚙️ 系统设置",
    }
    for prefix, cat in CATEGORY_MAP.items():
        if name.startswith(prefix) or prefix in name:
            return cat
    return "🔧 其他"
