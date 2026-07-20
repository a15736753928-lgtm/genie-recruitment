"""
StreamingToolExecutor — execute tools DURING model response streaming.

Claude Code's key innovation: tools start executing as soon as a complete
``tool_use`` JSON block is parsed from the streaming response, rather than
waiting for the entire model response to complete.

This eliminates the serial bottleneck where all tools wait for the full
API response.  In a typical 5-30 second streaming window, multiple tools
are dispatched and completed; by the time the stream finishes, results
are already available.

Architecture::

    Model streaming ──┬── content delta → yield to frontend
                      ├── tool_use block parsed → addTool() → execute NOW
                      └── stream ends → getRemainingResults() → yield to frontend

Tool lifecycle (4 states):
    queued → executing → completed → yielded
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Awaitable

logger = logging.getLogger("genie.streaming_executor")


class ToolExecutionState(str, Enum):
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    YIELDED = "yielded"
    FAILED = "failed"


@dataclass
class ToolBlock:
    """A parsed tool_use block from the streaming response."""
    id: str
    name: str
    arguments: dict = field(default_factory=dict)
    raw_arguments: str = ""  # raw JSON before parsing


@dataclass
class ToolExecutionResult:
    """Result of executing a single tool."""
    tool_id: str
    tool_name: str
    success: bool
    result_text: str = ""
    error: Optional[str] = None
    duration_ms: float = 0
    display_hint: str = "text"
    structured_data: Optional[dict] = None


# ── Concurrency safety markers ──────────────────────────────

# Tools that ONLY read data → safe to run in parallel
READ_ONLY_TOOLS: set[str] = {
    "list_resumes", "get_resume", "list_positions", "get_position",
    "get_questions", "get_evaluation", "get_leaderboard", "get_rankings",
    "list_probation", "get_probation_stats", "get_probation_employee",
    "list_performance", "get_performance_stats", "get_department_performance",
    "get_grade_distribution", "get_bonus_info", "get_quarter_trends",
    "rag_search", "list_knowledge", "get_knowledge_stats", "get_knowledge_categories",
    "recall_test", "list_knowledge_bases", "list_documents",
    "get_operations_dashboard", "get_dashboard_overview",
    "get_settings",
    # Position read tools
    "get_position_questions",
}

# Tools that MODIFY data → must run sequentially
WRITE_TOOLS: set[str] = {
    "update_resume", "delete_resume", "upload_resume", "batch_parse_resumes",
    "reanalyze_resume",
    "create_position", "update_position", "delete_position",
    "save_position_questions",
    "generate_questions", "save_questions", "replace_question",
    "save_evaluation", "submit_evaluation", "ai_score_question",
    "create_probation_employee", "save_week1_assessment", "save_conversion",
    "create_probation_task", "update_probation_task", "ai_evaluate_probation",
    "update_probation_status", "manual_review_probation",
    "initiate_appraisal", "update_bonus",
    "upload_knowledge_file", "create_knowledge_item", "update_knowledge_item",
    "delete_knowledge_item", "create_knowledge_base", "update_knowledge_base",
    "delete_knowledge_base", "upload_document", "delete_document",
    "update_settings",
}


def is_concurrency_safe(tool_name: str) -> bool:
    """Check if a tool can run concurrently with other tools.

    Read-only tools are concurrency-safe.  Write tools are NOT.
    Unknown tools default to NOT safe (fail-closed).
    """
    if tool_name in READ_ONLY_TOOLS:
        return True
    if tool_name in WRITE_TOOLS:
        return False
    # Unknown tools: assume unsafe (fail-closed for safety)
    return False


# ── Streaming Executor ──────────────────────────────────────

class StreamingToolExecutor:
    """Executes tools in parallel during model streaming.

    Usage::

        executor = StreamingToolExecutor(max_concurrent=10)
        async for chunk in stream_model(messages):
            if chunk.type == "tool_use":
                executor.add_tool(chunk.tool_block)
                yield sse_event("tool_call", chunk.tool_block)
            elif chunk.type == "content":
                yield sse_event("content", chunk)
            # Check for completed tools during streaming
            for result in executor.get_completed_results():
                yield sse_event("tool_result", result)

        # After stream ends, wait for any stragglers
        for result in await executor.get_remaining_results():
            yield sse_event("tool_result", result)
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        tool_executor: Optional[Callable[..., Awaitable[str]]] = None,
        db_factory=None,
    ):
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tool_executor = tool_executor
        self._db_factory = db_factory

        # Tool state tracking
        self._queue: list[ToolBlock] = []
        self._executing: dict[str, asyncio.Task] = {}
        self._completed: dict[str, ToolExecutionResult] = {}
        self._yielded: set[str] = set()
        self._sibling_abort = asyncio.Event()  # set when a write tool fails

        # Scheduling state
        self._has_unsafe_running = False
        self._scheduler_task: Optional[asyncio.Task] = None

    # ── Public API ──────────────────────────────────────────

    def add_tool(self, block: ToolBlock):
        """Called when a complete tool_use JSON block is parsed from the stream.

        Does NOT wait — the tool is queued and an internal scheduler
        tries to dispatch it immediately.
        """
        self._queue.append(block)
        # Kick the scheduler (non-blocking)
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    def get_completed_results(self) -> list[ToolExecutionResult]:
        """Non-blocking harvest of tools that finished since last call.

        Call this from the streaming loop to yield results as they become
        available, without blocking the stream.
        """
        results = []
        for tid, task in list(self._executing.items()):
            if task.done() and tid not in self._yielded:
                self._yielded.add(tid)
                try:
                    result = task.result()
                except Exception as e:
                    result = ToolExecutionResult(
                        tool_id=tid,
                        tool_name=self._completed.get(tid, ToolExecutionResult(tool_id=tid, tool_name="?")).tool_name,
                        success=False,
                        error=str(e),
                    )
                self._completed[tid] = result
                results.append(result)
        return results

    async def get_remaining_results(self) -> list[ToolExecutionResult]:
        """Wait for all still-executing tools to complete.

        Call this AFTER the model stream ends to collect stragglers.
        """
        if self._executing:
            # Wait for all outstanding tasks
            await asyncio.gather(*self._executing.values(), return_exceptions=True)

        results = []
        for tid in self._completed:
            if tid not in self._yielded:
                self._yielded.add(tid)
                results.append(self._completed[tid])

        # Also handle completed tasks not yet in self._completed
        for tid, task in self._executing.items():
            if tid not in self._yielded:
                self._yielded.add(tid)
                try:
                    result = task.result()
                except Exception as e:
                    result = ToolExecutionResult(
                        tool_id=tid, tool_name="?",
                        success=False, error=str(e),
                    )
                results.append(result)

        return results

    def cancel_all(self):
        """Cancel all pending and executing tools (on user abort)."""
        for task in self._executing.values():
            if not task.done():
                task.cancel()
        self._queue.clear()
        self._sibling_abort.set()

    @property
    def pending_count(self) -> int:
        return len(self._queue) + len(self._executing)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    # ── Internal scheduler ──────────────────────────────────

    async def _scheduler_loop(self):
        """Background coroutine that continuously dispatches queued tools.

        Rules:
        - Read-only tools: dispatch immediately (up to max_concurrent)
        - Write tools: wait for ALL executing tools to finish first
        - If a write tool fails, abort sibling tools (cascading failure)
        """
        try:
            while self._queue:
                # Check abort signal
                if self._sibling_abort.is_set():
                    self._queue.clear()
                    break

                # Determine what can be dispatched
                has_write_running = any(
                    not is_concurrency_safe(
                        self._completed.get(tid, ToolExecutionResult(
                            tool_id=tid, tool_name="?"
                        )).tool_name
                    )
                    for tid in self._executing
                    if not self._executing[tid].done()
                )

                if has_write_running:
                    # Wait for all executing tools to finish
                    if self._executing:
                        await asyncio.gather(
                            *self._executing.values(), return_exceptions=True
                        )
                    continue

                # Dispatch next batch
                batch = self._pop_safe_batch()
                if not batch:
                    # No safe tools to dispatch right now,
                    # but queue isn't empty → wait briefly
                    await asyncio.sleep(0.05)
                    continue

                for block in batch:
                    if len(self._executing) >= self._max_concurrent:
                        # Wait for some to finish
                        done, _ = await asyncio.wait(
                            self._executing.values(),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    task = asyncio.create_task(self._execute_one(block))
                    self._executing[block.id] = task

        except asyncio.CancelledError:
            pass
        finally:
            self._scheduler_task = None

    def _pop_safe_batch(self) -> list[ToolBlock]:
        """Pop a batch of tools that can run concurrently.

        Returns contiguous concurrency-safe tools from the front of the queue.
        Stops at the first write tool (which must run alone).
        """
        if not self._queue:
            return []

        batch = []
        while self._queue:
            block = self._queue[0]
            if is_concurrency_safe(block.name):
                batch.append(self._queue.pop(0))
            else:
                # Write tool: if batch is empty, we can dispatch it alone
                if not batch:
                    batch.append(self._queue.pop(0))
                break

        return batch

    # ── Single tool execution ───────────────────────────────

    async def _execute_one(self, block: ToolBlock) -> ToolExecutionResult:
        """Execute a single tool and record the result."""
        start = time.time()

        async with self._semaphore:
            try:
                if self._tool_executor and self._db_factory:
                    async with self._db_factory() as db:
                        try:
                            result_text = await self._tool_executor(
                                block.name, block.arguments, db
                            )
                            await db.commit()
                        except Exception as e:
                            result_text = f"工具执行错误: {e}"
                elif self._tool_executor:
                    result_text = await self._tool_executor(
                        block.name, block.arguments
                    )
                else:
                    result_text = f"[Tool '{block.name}' would execute with params: {json.dumps(block.arguments, ensure_ascii=False)[:200]}]"

                duration = (time.time() - start) * 1000

                result = ToolExecutionResult(
                    tool_id=block.id,
                    tool_name=block.name,
                    success=True,
                    result_text=str(result_text)[:4000],  # cap individual results
                    duration_ms=duration,
                    display_hint=_infer_display_hint(block.name),
                )

            except asyncio.CancelledError:
                duration = (time.time() - start) * 1000
                result = ToolExecutionResult(
                    tool_id=block.id,
                    tool_name=block.name,
                    success=False,
                    error="任务已被取消",
                    duration_ms=duration,
                )

            except Exception as e:
                duration = (time.time() - start) * 1000
                logger.warning("Tool '%s' failed: %s", block.name, e)
                result = ToolExecutionResult(
                    tool_id=block.id,
                    tool_name=block.name,
                    success=False,
                    error=str(e),
                    duration_ms=duration,
                )

                # Cascading failure: if a write tool fails, abort siblings
                if not is_concurrency_safe(block.name):
                    self._sibling_abort.set()

            self._completed[block.id] = result
            return result


# ── Helpers ──────────────────────────────────────────────────

def _infer_display_hint(tool_name: str) -> str:
    """Infer the display hint for a tool based on its name."""
    HINTS = {
        "list": ["list_resumes", "list_positions", "list_probation",
                  "list_performance", "list_knowledge", "rag_search",
                  "list_knowledge_bases", "list_documents", "recall_test"],
        "card": ["get_resume", "get_position", "get_probation_employee",
                  "upload_resume", "reanalyze_resume", "create_position",
                  "update_position", "create_knowledge_item", "update_knowledge_item",
                  "create_knowledge_base", "update_knowledge_base"],
        "stats": ["get_probation_stats", "get_performance_stats",
                   "get_department_performance", "get_grade_distribution",
                   "get_bonus_info", "get_quarter_trends", "get_knowledge_stats",
                   "get_operations_dashboard", "get_dashboard_overview"],
        "score": ["get_evaluation", "ai_score_question", "save_week1_assessment",
                   "save_conversion", "ai_evaluate_probation", "manual_review_probation"],
        "table": ["get_leaderboard", "get_rankings"],
        "questions": ["get_questions", "generate_questions", "get_position_questions"],
    }
    for hint, names in HINTS.items():
        if tool_name in names:
            return hint
    return "text"
