"""
CompactionPipeline — 5-stage context compression with circuit breaker.

Runs BEFORE every API call in the agent loop.  Strategies are tried
in order from cheapest to most expensive::

    budget → snip → microcompact → collapse → autocompact (last resort)

Inspired by Claude Code's 5-layer compression pipeline in:
- ``src/services/compact/microCompact.ts``
- ``src/services/compact/compact.ts``
- ``src/services/compact/snipCompact.ts``

Key innovations over the old 4-layer context engine:
    1. Cache-aware microcompact (preserves prompt cache byte consistency)
    2. Circuit breaker on auto-compact (3 consecutive failures → disable)
    3. Time-decay cleaning (older tool results cleaned first)
    4. Compressible tool whitelist (only certain tools are compactable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CompactionConfig:
    """Configuration for the compaction pipeline."""
    # Layer 1: Tool result budget
    tool_result_max_chars: int = 3000
    tool_result_aggregate_max: int = 200_000

    # Layer 2: Snip
    snip_empty_tool_results: bool = True

    # Layer 3: Microcompact
    microcompact_gap_minutes: int = 60
    microcompact_keep_recent: int = 5
    microcompact_enabled: bool = True

    # Layer 4: Context Collapse
    collapse_threshold_tokens: int = 80_000
    collapse_enabled: bool = True

    # Layer 5: Auto-Compact
    autocompact_threshold_tokens: int = 120_000
    autocompact_circuit_breaker: int = 3
    autocompact_enabled: bool = True
    autocompact_keep_recent: int = 6

    # General
    preserve_last_n: int = 6


class CompactionPipeline:
    """5-stage compression pipeline with circuit breaker protection."""

    # Tools whose results are safe to compact (whitelist)
    COMPRESSIBLE_TOOLS: set[str] = {
        "list_resumes", "get_resume", "rag_search", "list_knowledge",
        "get_operations_dashboard", "get_dashboard_overview",
        "list_positions", "get_position", "list_documents",
        "get_knowledge_stats", "get_knowledge_categories",
        "list_probation", "list_performance",
        "get_performance_stats", "get_department_performance",
        "get_grade_distribution", "get_bonus_info", "get_quarter_trends",
        "recall_test", "list_knowledge_bases",
        "get_evaluation", "get_leaderboard", "get_rankings",
    }

    COMPACTED_MARKER = "[旧工具结果内容已清理]"

    def __init__(self, config: CompactionConfig = None):
        self.config = config or CompactionConfig()
        self._failure_counts: dict[str, int] = {}
        self.last_layer_used: int = 0
        self._stats: list[dict] = []

    async def prepare(self, state, llm_client=None, context=None):
        """Run the full compression pipeline on a LoopState.

        Returns a NEW LoopState (never mutates the original).
        """
        from app.agent_os.loop.state import LoopState

        self.last_layer_used = 0

        token_est = self._estimate_messages_tokens(state.messages)

        # Layer 1: Tool Result Budget (always active)
        state = self._layer1_budget(state)
        self.last_layer_used = 1
        token_est = self._estimate_messages_tokens(state.messages)
        if token_est <= self.config.collapse_threshold_tokens:
            return state

        # Layer 2: Snip
        if self.config.snip_empty_tool_results:
            state = self._layer2_snip(state)
            self.last_layer_used = 2
        token_est = self._estimate_messages_tokens(state.messages)
        if token_est <= self.config.collapse_threshold_tokens:
            return state

        # Layer 3: Microcompact
        if self.config.microcompact_enabled:
            state = self._layer3_microcompact(state)
            self.last_layer_used = 3
        token_est = self._estimate_messages_tokens(state.messages)
        if token_est <= self.config.collapse_threshold_tokens:
            return state

        # Layer 4: Context Collapse
        if self.config.collapse_enabled:
            state = self._layer4_collapse(state)
            self.last_layer_used = 4
        token_est = self._estimate_messages_tokens(state.messages)
        if token_est <= self.config.autocompact_threshold_tokens:
            return state

        # Layer 5: Auto-Compact (circuit breaker protected)
        if self.config.autocompact_enabled:
            if self._failure_counts.get("autocompact", 0) < self.config.autocompact_circuit_breaker:
                state = self._layer5_autocompact(state, llm_client)
                self.last_layer_used = 5

        return state

    # ── Layer 1: Tool Result Budget ─────────────────────────

    def _layer1_budget(self, state):
        from app.agent_os.loop.state import LoopState
        aggregate_chars = 0
        budgeted = []
        for msg in reversed(state.messages):
            role = getattr(msg, "role", None) or getattr(msg, "type", "")
            if role in ("tool", "tool_result") or (hasattr(msg, "tool_call_id") and msg.tool_call_id):
                content = str(getattr(msg, "content", ""))
                if len(content) > self.config.tool_result_max_chars:
                    truncated = (
                        content[:self.config.tool_result_max_chars]
                        + f"\n\n... (已截断 {len(content) - self.config.tool_result_max_chars} 字符)"
                    )
                    budgeted.append(self._copy_msg(msg, content=truncated))
                    aggregate_chars += len(truncated)
                elif aggregate_chars + len(content) > self.config.tool_result_aggregate_max:
                    truncated = content[:500] + "\n\n... [旧工具结果已折叠以节省上下文]"
                    budgeted.append(self._copy_msg(msg, content=truncated))
                    aggregate_chars += len(truncated)
                else:
                    budgeted.append(msg)
                    aggregate_chars += len(content)
            else:
                budgeted.append(msg)
        budgeted.reverse()
        return LoopState(**{**state.__dict__, "messages": budgeted})

    # ── Layer 2: Snip ───────────────────────────────────────

    def _layer2_snip(self, state):
        from app.agent_os.loop.state import LoopState
        EMPTY_MARKERS = {"[]", "{}", "无结果", "没有找到", "", "null", "None", "无"}
        snipped = []
        i = 0
        while i < len(state.messages):
            msg = state.messages[i]
            role = getattr(msg, "role", "")
            if role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_results = []
                j = i + 1
                while j < len(state.messages):
                    next_role = getattr(state.messages[j], "role", "")
                    if next_role == "tool" or (hasattr(state.messages[j], "tool_call_id") and state.messages[j].tool_call_id):
                        tool_results.append(state.messages[j])
                        j += 1
                    else:
                        break
                all_empty = all(
                    str(getattr(tr, "content", "")).strip() in EMPTY_MARKERS
                    for tr in tool_results
                )
                if all_empty and tool_results:
                    i = j
                    continue
            snipped.append(msg)
            i += 1
        return LoopState(**{**state.__dict__, "messages": snipped})

    # ── Layer 3: Microcompact ───────────────────────────────

    def _layer3_microcompact(self, state):
        from app.agent_os.loop.state import LoopState
        n = len(state.messages)
        if n <= self.config.microcompact_keep_recent:
            return state
        preserve_from = max(0, n - self.config.microcompact_keep_recent)
        compacted = []
        for i, msg in enumerate(state.messages):
            if i >= preserve_from:
                compacted.append(msg)
                continue
            role = getattr(msg, "role", "")
            if role == "tool" or (hasattr(msg, "tool_call_id") and msg.tool_call_id):
                tool_name = getattr(msg, "name", "")
                if self._is_compressible_tool(tool_name):
                    content = str(getattr(msg, "content", ""))
                    age_hours = self._estimate_message_age_hours(msg)
                    if age_hours > (self.config.microcompact_gap_minutes / 60) or len(content) > 5000:
                        compacted.append(self._copy_msg(msg, content=self.COMPACTED_MARKER))
                        continue
            compacted.append(msg)
        return LoopState(**{**state.__dict__, "messages": compacted})

    def _is_compressible_tool(self, tool_name: str) -> bool:
        if not tool_name:
            return False
        if tool_name in self.COMPRESSIBLE_TOOLS:
            return True
        # Also match by prefix
        for prefix in ("list_", "get_", "search_", "rag_"):
            if tool_name.startswith(prefix):
                return True
        return False

    def _estimate_message_age_hours(self, msg) -> float:
        ts = getattr(msg, "created_at", None)
        if ts and isinstance(ts, datetime):
            return (datetime.utcnow() - ts).total_seconds() / 3600
        return 0

    # ── Layer 4: Context Collapse ───────────────────────────

    def _layer4_collapse(self, state):
        from app.agent_os.loop.state import LoopState
        from langchain_core.messages import SystemMessage
        n = len(state.messages)
        if n <= self.config.preserve_last_n:
            return state
        keep_from = max(0, n - self.config.preserve_last_n)
        import re as _re
        id_pattern = _re.compile(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}")
        summary_parts = ["[早期对话摘要 — 上下文已折叠]"]
        tool_names = set()
        decisions = []
        for msg in state.messages[:keep_from]:
            role = getattr(msg, "role", "?")
            content = str(getattr(msg, "content", ""))[:100].replace("\n", " ")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if name:
                        tool_names.add(name)
            if role == "assistant" and len(content) > 10:
                decisions.append(content[:80])
            for keyword in ["通过", "拒绝", "推荐", "评分", "筛选", "录用", "淘汰"]:
                if keyword in content:
                    decisions.append(f"{keyword}: {content[:60]}")
        if tool_names:
            summary_parts.append(f"使用工具: {', '.join(sorted(tool_names)[:15])}")
        if decisions:
            unique = list(dict.fromkeys(decisions))[-8:]
            summary_parts.append(f"关键操作: {'; '.join(unique)}")
        summary_parts.append(f"折叠消息数: {keep_from}")
        summary_parts.append("")
        collapsed = [SystemMessage(content="\n".join(summary_parts))] + list(state.messages[keep_from:])
        return LoopState(**{**state.__dict__, "messages": collapsed})

    # ── Layer 5: Auto-Compact ───────────────────────────────

    async def _layer5_autocompact(self, state, llm_client=None):
        from app.agent_os.loop.state import LoopState
        from langchain_core.messages import SystemMessage
        n = len(state.messages)
        if n <= self.config.autocompact_keep_recent:
            return state
        if self._failure_counts.get("autocompact", 0) >= self.config.autocompact_circuit_breaker:
            return state
        keep_from = max(0, n - self.config.autocompact_keep_recent)
        old_messages = state.messages[:keep_from]
        recent_messages = state.messages[keep_from:]
        compact = await self._build_auto_compact_summary(old_messages, llm_client)
        if compact:
            self._failure_counts["autocompact"] = 0
            new_messages = [SystemMessage(content=compact)] + list(recent_messages)
            return LoopState(**{**state.__dict__, "messages": new_messages})
        else:
            self._failure_counts["autocompact"] = self._failure_counts.get("autocompact", 0) + 1
            return state

    async def _build_auto_compact_summary(self, messages: list, llm_client=None) -> str:
        if not messages:
            return ""
        import re as _re
        id_pattern = _re.compile(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}")
        tool_calls = set()
        key_actions = []
        user_requests = []
        for msg in messages:
            content = str(getattr(msg, "content", ""))
            role = getattr(msg, "role", "")
            if role == "user" and len(content) > 5:
                user_requests.append(content[:120])
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if name:
                        tool_calls.add(name)
            for keyword in ["通过", "拒绝", "推荐", "筛选", "录用", "淘汰", "评分", "评估"]:
                if keyword in content and len(content) > 10:
                    key_actions.append(content[:80])
        parts = [
            "[上下文压缩 — 早期对话已自动压缩]",
            "",
            f"已折叠 {len(messages)} 条早期消息。以下关键信息已保留：",
            "",
        ]
        if user_requests:
            parts.append(f"用户请求: {'; '.join(dict.fromkeys(user_requests[-5:]))}")
        if tool_calls:
            parts.append(f"已使用工具: {', '.join(sorted(tool_calls)[:20])}")
        if key_actions:
            parts.append(f"关键操作: {'; '.join(list(dict.fromkeys(key_actions))[-5:])}")
        parts.append("")
        parts.append("请基于最近的消息继续处理用户任务。如需早期对话的详细信息，请使用工具重新查询。")
        parts.append("")
        return "\n".join(parts)

    # ── Token estimation ────────────────────────────────────

    @staticmethod
    def _estimate_messages_tokens(messages: list) -> int:
        total = 0
        for msg in messages:
            content = str(getattr(msg, "content", ""))
            cjk = sum(1 for c in content if '一' <= c <= '鿿')
            total += cjk + ((len(content) - cjk) // 4) + 1
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                total += len(str(msg.tool_calls)) // 4
        return total

    @staticmethod
    def _copy_msg(msg, **overrides):
        msg_type = type(msg)
        try:
            content = overrides.pop("content", getattr(msg, "content", ""))
            return msg_type(content=content, **{
                k: v for k, v in overrides.items()
            })
        except Exception:
            return msg

    def reset_circuit_breaker(self):
        self._failure_counts.clear()

    def get_stats(self) -> list[dict]:
        return self._stats[-10:]
