"""
CacheAwarePromptBuilder — prompt assembly with explicit cache boundary.

Upgrades SystemPromptBuilder with:
- Explicit SYSTEM_PROMPT_DYNAMIC_BOUNDARY marker
- Static hash key computation for cache invalidation
- MCP instructions per-turn recompute (servers can connect/disconnect mid-session)
- CLAUDE.md per-turn re-injection (rules survive compaction)
- Deferred tool loading (only tool names in static, full schemas on demand)

Inspired by Claude Code's ``src/utils/systemPrompt.ts`` pattern.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional


# The cache boundary marker — identical to Claude Code's approach
CACHE_BOUNDARY = "\n\n--- SYSTEM_PROMPT_DYNAMIC_BOUNDARY ---\n\n"


class DeferredToolLoader:
    """Manages deferred (on-demand) tool schema loading.

    Only tool NAMES are included in the static system prompt.
    Full schemas are loaded when the model actually uses them,
    dramatically reducing static prompt size.
    """

    def __init__(self):
        self._schemas: dict[str, dict] = {}

    def register(self, name: str, description: str, parameters: dict):
        """Register a tool for deferred loading."""
        self._schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    def get_tool_names(self) -> list[str]:
        """Get all registered tool names (compact catalog)."""
        return sorted(self._schemas.keys())

    def get_detail(self, name: str) -> Optional[dict]:
        """Get full tool schema on demand."""
        return self._schemas.get(name)

    def build_compact_catalog(self) -> str:
        """Build a compact tool catalog (name + one-line description only)."""
        if not self._schemas:
            return "（无可用工具）"

        # Group by category
        categories: dict[str, list[dict]] = {}
        for name, schema in self._schemas.items():
            cat = _tool_category(name)
            categories.setdefault(cat, []).append(schema)

        lines = []
        for cat, tools in sorted(categories.items()):
            lines.append(f"### {cat}")
            for t in tools:
                desc = t["description"].split("。")[0][:120]  # First sentence only
                lines.append(f"- **{t['name']}**: {desc}")
            lines.append("")

        return "\n".join(lines)

    def build_full_tool_schemas(self) -> list[dict]:
        """Build full OpenAI-compatible tool schemas for an API call."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            }
            for name, schema in self._schemas.items()
        ]


class CacheAwarePromptBuilder:
    """Assembles system prompts with explicit cache boundary.

    Layers (static → dynamic):
        L1: Role definition + core identity (STATIC — cacheable)
        L2: Communication style & tone rules (STATIC — cacheable)
        L3: Deferred tool catalog (STATIC — cacheable, names only)
        L4: Agent-specific instructions (STATIC — cacheable)
        --- DYNAMIC BOUNDARY (API prompt cache split) ---
        L5: CWD/OS/shell environment (DYNAMIC — per-session)
        L6: Date/time (DYNAMIC — per-turn)
        L7: CLAUDE.md / project instructions (DYNAMIC — per-turn re-inject)
        L8: Relevant memories (DYNAMIC — per-turn query-dependent)
        L9: MCP server instructions (DYNAMIC — per-turn recompute)
        L10: Session context (DYNAMIC — per-turn)

    Layers 1-4 can be cached by the LLM API and reused across turns.
    Layers 5-10 change every turn and sit below the cache boundary.
    """

    def __init__(self):
        self._static_cache: dict[str, tuple[str, str]] = {}
        # (full_static, cache_hash)

    # ── Main build ──────────────────────────────────────────

    def build(
        self,
        base_prompt: str,
        agent_id: str,
        agent_config: dict,
        tool_loader: Optional[DeferredToolLoader] = None,
        memories: Optional[list] = None,
        session_context: str = "",
        runtime_directives: str = "",
        environment: Optional[dict] = None,
        claude_md_content: str = "",
        mcp_instructions: str = "",
    ) -> tuple[str, str, str]:
        """Build layered system prompt with cache boundary.

        Returns:
            (full_prompt, static_portion, dynamic_portion)

        The static_portion can be sent as the system message prefix for
        prompt caching.  The dynamic_portion should be appended each turn.
        """
        env = environment or {}

        # ── Static portion (cacheable) ──
        cache_key = self._compute_cache_key(
            agent_id, base_prompt, tool_loader
        )

        if cache_key not in self._static_cache:
            static_parts = [base_prompt]

            if agent_config:
                static_parts.append(
                    f"\n\n你当前的角色是 **{agent_config.get('name', agent_id)}**。"
                    f"{agent_config.get('description', '')}"
                )

            if tool_loader:
                catalog = tool_loader.build_compact_catalog()
                static_parts.append(f"\n\n## 可用工具\n\n{catalog}")

            cache_hash = hashlib.sha256(
                "\n".join(static_parts).encode()
            ).hexdigest()[:16]

            self._static_cache[cache_key] = (
                "\n".join(static_parts),
                cache_hash,
            )

        static_prompt, cache_hash = self._static_cache[cache_key]

        # ── Dynamic portion ──
        dynamic_parts = []

        # Environment
        cwd = env.get("cwd", env.get("working_dir", ""))
        os_name = env.get("os", env.get("platform", ""))
        if cwd or os_name:
            dynamic_parts.append(
                f"## 运行环境\n"
                f"- 工作目录: {cwd or '(未知)'}\n"
                f"- 操作系统: {os_name or '(未知)'}\n"
                f"- 当前时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )

        # CLAUDE.md content (per-turn re-injection)
        if claude_md_content:
            # Cap at 40K chars (Claude Code's limit)
            capped = claude_md_content[:40000]
            dynamic_parts.append(f"\n\n## 项目规则与偏好\n\n{capped}")

        # MCP server instructions (per-turn recompute)
        if mcp_instructions:
            dynamic_parts.append(f"\n\n## MCP 服务器指令\n\n{mcp_instructions}")

        # Relevant memories
        if memories:
            mem_lines = []
            for m in memories[:5]:
                desc = getattr(m, "description", str(m)[:100])
                content = getattr(m, "content", str(m))
                mem_lines.append(f"### {desc}\n{content[:400]}")
            dynamic_parts.append(f"\n\n## 相关记忆\n\n" + "\n".join(mem_lines))

        # Session context
        if session_context:
            dynamic_parts.append(f"\n\n## 当前会话上下文\n\n{session_context}")

        # Runtime directives
        if runtime_directives:
            dynamic_parts.append(f"\n\n{runtime_directives}")

        dynamic_prompt = "\n".join(dynamic_parts)

        # Full prompt with cache boundary
        full = static_prompt + CACHE_BOUNDARY + dynamic_prompt

        return full, static_prompt, dynamic_prompt

    # ── Cache management ────────────────────────────────────

    def _compute_cache_key(
        self,
        agent_id: str,
        base_prompt: str,
        tool_loader: Optional[DeferredToolLoader],
    ) -> str:
        """Compute a stable cache key for the static prompt portion."""
        parts = [agent_id, base_prompt[:200]]  # first 200 chars of base
        if tool_loader:
            parts.append(",".join(tool_loader.get_tool_names()))
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def invalidate_cache(self):
        """Clear the static cache (call when prompts/tools change)."""
        self._static_cache.clear()

    def get_cache_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        return {
            "cached_entries": len(self._static_cache),
            "cache_hashes": list(self._static_cache.values()),
        }


# ── Tool category mapping ────────────────────────────────────

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
        "delete": "⚠️ 危险操作",
    }
    for prefix, cat in CATEGORY_MAP.items():
        if name.startswith(prefix) or prefix in name:
            return cat
    return "🔧 其他"
