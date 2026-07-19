"""
ContextEngine — intelligent context window management.

Treats the context window as a scarce managed resource, not an infinite
buffer.  Four compression layers run sequentially before every LLM call,
cheapest first:

  Layer 1 — Truncation: single-message size caps + overflow hints
  Layer 2 — Deduplication: merge repeated tool calls
  Layer 3 — Summarization: LLM-generated summaries of old messages
  Layer 4 — Compaction: full sub-agent conversation summary (last resort)

Inspired by Claude Code's 5-layer compression pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreparedContext:
    """The result of context preparation — ready to send to the LLM."""
    messages: list
    system_prompt: str
    system_prompt_static: str        # cacheable portion
    system_prompt_dynamic: str       # session-specific portion
    token_estimate: int = 0
    compression_applied: bool = False
    compression_layer: int = 0


@dataclass
class ContextStats:
    """Usage statistics for monitoring."""
    total_messages: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    savings_percent: float = 0.0
    layer_used: int = 0


class ContextEngine:
    """Manages context window allocation and compression.

    Args:
        max_tokens: Target context window size (input tokens).
        reserve_tokens: Tokens reserved for the model's output.
        preserve_last_n: Number of recent messages to keep verbatim.
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        reserve_tokens: int = 2000,
        preserve_last_n: int = 6,
    ):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.preserve_last_n = preserve_last_n
        self._stats: list[ContextStats] = []

    @property
    def effective_limit(self) -> int:
        """The actual token budget for the input context."""
        return self.max_tokens - self.reserve_tokens

    # ── Main entry point ──────────────────────────────────

    async def prepare(
        self,
        messages: list,
        system_prompt: str,
        system_prompt_dynamic: str = "",
    ) -> PreparedContext:
        """Prepare context for an LLM call.

        Applies compression layers as needed, cheapest first, until the
        context fits within the token budget.

        IMPORTANT: Compression is NON-DESTRUCTIVE.  The original messages
        list is never modified — we return a (possibly compressed) copy.
        """
        original_count = len(messages)
        token_est = self._estimate_total(messages, system_prompt, system_prompt_dynamic)

        ctx = PreparedContext(
            messages=list(messages),  # shallow copy
            system_prompt=system_prompt,
            system_prompt_static=system_prompt,
            system_prompt_dynamic=system_prompt_dynamic,
            token_estimate=token_est,
        )

        # If we're within budget, return immediately
        if token_est <= self.effective_limit:
            return ctx

        # ── Layer 1: Truncation ──
        ctx = await self._layer1_truncation(ctx)
        if ctx.token_estimate <= self.effective_limit:
            return ctx

        # ── Layer 2: Deduplication ──
        ctx = await self._layer2_deduplicate(ctx)
        if ctx.token_estimate <= self.effective_limit:
            return ctx

        # ── Layer 3: Summarization ──
        ctx = await self._layer3_summarize(ctx)
        if ctx.token_estimate <= self.effective_limit:
            return ctx

        # ── Layer 4: Compaction ──
        ctx = await self._layer4_compact(ctx)

        self._stats.append(ContextStats(
            total_messages=original_count,
            original_tokens=token_est,
            compressed_tokens=ctx.token_estimate,
            savings_percent=(1 - ctx.token_estimate / max(token_est, 1)) * 100,
            layer_used=ctx.compression_layer,
        ))

        return ctx

    # ── Layer 1: Per-message truncation ───────────────────

    async def _layer1_truncation(self, ctx: PreparedContext) -> PreparedContext:
        """Cap individual message content lengths.

        For tool results exceeding 2000 chars, truncate with a hint
        so the model knows there is more data available.
        """
        MAX_PER_MESSAGE = 2000
        truncated = []
        for msg in ctx.messages:
            content = getattr(msg, "content", None)
            if content and isinstance(content, str) and len(content) > MAX_PER_MESSAGE:
                # Create a truncated copy
                new_content = (
                    content[:MAX_PER_MESSAGE]
                    + f"\n\n... (truncated {len(content) - MAX_PER_MESSAGE} chars, "
                    + "use offset/limit on the original tool to read more)"
                )
                # For LangChain messages, we need to preserve the type
                msg_type = type(msg)
                try:
                    truncated.append(msg_type(content=new_content))
                except Exception:
                    truncated.append(msg)  # fallback: keep original
            else:
                truncated.append(msg)
        ctx.messages = truncated
        ctx.token_estimate = self._estimate_total(
            ctx.messages, ctx.system_prompt, ctx.system_prompt_dynamic
        )
        ctx.compression_layer = 1
        return ctx

    # ── Layer 2: Deduplication ────────────────────────────

    async def _layer2_deduplicate(self, ctx: PreparedContext) -> PreparedContext:
        """Merge consecutive identical tool calls.

        When the agent calls the same tool multiple times with the same
        params (e.g. list_resumes with identical filters), keep only the
        most recent result and note the merge.
        """
        if len(ctx.messages) < 3:
            return ctx

        deduped = []
        i = 0
        while i < len(ctx.messages):
            msg = ctx.messages[i]
            # Only deduplicate tool result messages (role == "tool")
            role = getattr(msg, "role", None)
            if role == "tool" and i > 0:
                # Check if the previous tool call was the same
                prev = ctx.messages[i - 1]
                prev_name = getattr(prev, "name", None)
                curr_name = getattr(msg, "name", None)
                if prev_name and prev_name == curr_name:
                    # Skip older duplicate, keep latest
                    if i + 1 < len(ctx.messages) and getattr(ctx.messages[i + 1], "role", None) == "tool":
                        # There's a newer tool result — skip this one
                        i += 1
                        continue
            deduped.append(msg)
            i += 1

        ctx.messages = deduped
        ctx.token_estimate = self._estimate_total(
            ctx.messages, ctx.system_prompt, ctx.system_prompt_dynamic
        )
        ctx.compression_layer = 2
        return ctx

    # ── Layer 3: Old-message summarization ────────────────

    async def _layer3_summarize(self, ctx: PreparedContext) -> PreparedContext:
        """Replace old messages with brief structural summaries.

        Messages before the preserve_last_n window are replaced with
        compact notation: ``[role]: one-line content summary``.
        This is rule-based (no LLM call) — fast and cheap.
        """
        n = len(ctx.messages)
        if n <= self.preserve_last_n:
            return ctx

        keep_from = n - self.preserve_last_n
        summarized = []
        for i, msg in enumerate(ctx.messages[:keep_from]):
            role = getattr(msg, "role", "?")
            content = str(getattr(msg, "content", ""))[:80].replace("\n", " ")
            is_tool = ""
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tools = [tc.get("name", "?") for tc in msg.tool_calls]
                is_tool = f" [tools: {', '.join(tools)}]"
            summarized.append(f"[{role}{is_tool}] {content}")

        # Create a summary marker
        summary = "[对话历史摘要]\n" + "\n".join(summarized) + "\n\n[最近对话]\n"

        # Prepend summary to the preserved messages
        # We create a synthetic system-level message
        from langchain_core.messages import SystemMessage
        ctx.messages = [SystemMessage(content=summary)] + list(ctx.messages[keep_from:])
        ctx.token_estimate = self._estimate_total(
            ctx.messages, ctx.system_prompt, ctx.system_prompt_dynamic
        )
        ctx.compression_layer = 3
        return ctx

    # ── Layer 4: Full compaction ──────────────────────────

    async def _layer4_compact(self, ctx: PreparedContext) -> PreparedContext:
        """Full conversation compaction using a sub-agent LLM call.

        Generates a comprehensive summary of the entire conversation,
        preserving key decisions, data, and state.  This is the most
        expensive layer and should only fire as a last resort.

        For now this is a structural compaction (metadata extraction)
        without an LLM call, keeping it fast.  The full LLM-based
        compaction can be enabled via the sub-agent system in Phase 4.
        """
        if len(ctx.messages) <= self.preserve_last_n:
            return ctx

        # Extract key metadata from messages
        decisions = []
        tool_calls = []
        data_refs = []

        for msg in ctx.messages:
            content = str(getattr(msg, "content", ""))
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc.get("name", "?"))
            # Identify candidate IDs mentioned
            import re as _re
            ids = _re.findall(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", content)
            if ids:
                data_refs.extend(ids)

        # Build a compaction summary
        from langchain_core.messages import SystemMessage

        compact = (
            f"[上下文压缩] 早期对话已压缩。关键信息保留如下：\n"
            f"  涉及工具: {', '.join(set(tool_calls)) if tool_calls else '无'}\n"
            f"  引用对象: {len(set(data_refs))} 个\n"
            f"  请基于最近的消息继续处理用户任务。\n\n"
        )

        keep_from = max(0, len(ctx.messages) - self.preserve_last_n)
        ctx.messages = [SystemMessage(content=compact)] + list(ctx.messages[keep_from:])
        ctx.token_estimate = self._estimate_total(
            ctx.messages, ctx.system_prompt, ctx.system_prompt_dynamic
        )
        ctx.compression_applied = True
        ctx.compression_layer = 4
        return ctx

    # ── Token estimation ──────────────────────────────────

    def _estimate_total(
        self,
        messages: list,
        system_prompt: str,
        system_prompt_dynamic: str,
    ) -> int:
        """Estimate total token count (char / 2 for CJK, char / 4 for ASCII)."""
        total = self.estimate_tokens(system_prompt)
        total += self.estimate_tokens(system_prompt_dynamic)
        for msg in messages:
            content = str(getattr(msg, "content", ""))
            total += self.estimate_tokens(content)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    total += self.estimate_tokens(str(tc.get("arguments", "")))
        return total

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Conservative token count estimate.

        CJK characters ≈ 1 token each.  ASCII text ≈ 0.25 tokens per char.
        This is a heuristic — actual tokenization depends on the model.
        """
        if not text:
            return 0
        cjk = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿')
        ascii_chars = len(text) - cjk
        return cjk + (ascii_chars // 4) + 1
