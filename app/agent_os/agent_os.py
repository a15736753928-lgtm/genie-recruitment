"""
AgentOS v3 — facade that wires all Agent OS modules together.

Upgraded from v2 with Claude Code architecture:
- Native async generator agent loop (replaces LangGraph ReAct)
- StreamingToolExecutor (tools execute DURING model streaming)
- 5-stage compaction pipeline with circuit breaker
- Deny-First permission engine
- 27 lifecycle hooks across 6 execution types
- Cache-aware prompt building with explicit boundary
- Smart retry with model fallback

This is the single entry point used by the chat endpoint.
"""

from __future__ import annotations

from typing import Optional

from app.agent_os.memory.manager import MemoryManager
from app.agent_os.memory.retriever import MemoryRetriever
from app.agent_os.memory.models import Memory
from app.agent_os.context.engine import ContextEngine
from app.agent_os.cache.prompt_cache import CacheAwarePromptBuilder, DeferredToolLoader
from app.agent_os.compact.pipeline import CompactionPipeline, CompactionConfig
from app.agent_os.quality.guard import QualityGuard, GuardResult
from app.agent_os.skills.registry import SkillRegistry
from app.agent_os.skills.models import Skill
from app.agent_os.hooks.manager import ExtendedHookManager
from app.agent_os.hooks.events import HookEvent
from app.agent_os.orchestration.plan_coordinator import PlanCoordinator, PendingPlan
from app.agent_os.orchestration.dispatcher import SubAgentDispatcher, SubAgentTask, SubAgentResult
from app.agent_os.orchestration.workflow import WorkflowEngine, WorkflowDefinition
from app.agent_os.loop.state import LoopState, ContinueReason, ExitReason, QueryContext, LoopResult
from app.agent_os.loop.streaming_executor import StreamingToolExecutor
from app.agent_os.loop.query_loop import query_loop, query, QueryEngine
from app.agent_os.permissions.engine import (
    PermissionEngine, PermissionDecision, PermissionMode, PermissionResult,
)
from app.agent_os.retry.smart_retry import SmartRetry, RetryConfig
from app.config import get_settings

_settings = get_settings()


class AgentOS:
    """Top-level Agent OS coordinator (v3).

    Upgraded with Claude Code architecture patterns:
    - Agent loop driven by async generator state machine
    - Stream tool execution with parallel/concurrent safety
    - Deny-first permission engine for tool authorization
    - Cache-aware layered prompt building

    Thread-safe (no mutable shared state).  Create one instance per
    application process.

    Usage::

        agent_os = AgentOS()

        # Build prompts
        full, static, dynamic = agent_os.build_prompt(
            base_prompt, "genie", agent_config, tool_defs, memories,
        )

        # Run agent loop
        async for event in agent_os.run_loop(
            messages, static, dynamic, tools, context,
            llm_client=client, tool_executor=executor, db_factory=factory,
        ):
            yield event
    """

    def __init__(
        self,
        memory_dir: str = "memory/",
        skills_dir: str = "skills/",
        max_context_tokens: int = 8000,
    ):
        # ── Memory ──
        self.memory_manager = MemoryManager(memory_dir=memory_dir)
        self.memory_retriever = MemoryRetriever(self.memory_manager)

        # ── Context ──
        self.context_engine = ContextEngine(max_tokens=max_context_tokens)
        self.compaction_pipeline = CompactionPipeline(CompactionConfig())

        # ── Prompt ──
        self.prompt_builder = CacheAwarePromptBuilder()
        self.tool_loader = DeferredToolLoader()

        # ── Quality ──
        self.quality_guard = QualityGuard()

        # ── Skills ──
        self.skill_registry = SkillRegistry(skills_dir=skills_dir)

        # ── Hooks (v3: extended) ──
        self.hook_manager = ExtendedHookManager()

        # ── Permissions (v3: new) ──
        self.permission_engine = PermissionEngine()

        # ── Execution engine (v3: new) ──
        self.query_engine = QueryEngine()

        # ── Plan coordination ──
        self.plan_coordinator = PlanCoordinator()

        # ── Sub-agent dispatch ──
        self.dispatcher = SubAgentDispatcher(
            max_concurrent=getattr(_settings, "max_subagents", 10)
        )

        # ── Workflow ──
        self.workflow_engine = WorkflowEngine(self.dispatcher)

        # ── Retry ──
        self.retry = SmartRetry(RetryConfig())

    # ── Memory ──────────────────────────────────────────────

    async def load_memories(
        self,
        query: str,
        llm_client=None,
        limit: int = 5,
    ) -> list[Memory]:
        """Load relevant memories for a user query."""
        return await self.memory_retriever.retrieve(
            query=query,
            llm_client=llm_client,
            limit=limit,
        )

    async def capture_session_learnings(
        self,
        messages: list,
        session_id: str,
        llm_client=None,
    ) -> list[Memory]:
        """Auto-capture learnings after a session completes."""
        return await self.memory_manager.auto_capture(
            session_messages=messages,
            session_id=session_id,
            llm_client=llm_client,
        )

    # ── Prompt building (v3: cache-aware) ───────────────────

    def register_tools(self, tools: list[dict]):
        """Register tools for deferred loading.

        Args:
            tools: List of tool definitions with name, description, parameters.
        """
        for tool in tools:
            name = tool.get("name", "")
            if isinstance(tool, dict):
                self.tool_loader.register(
                    name=name,
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {}),
                )
            else:
                # LangChain tool object
                self.tool_loader.register(
                    name=getattr(tool, "name", ""),
                    description=getattr(tool, "description", ""),
                    parameters=getattr(tool, "args_schema", None),
                )

    def build_prompt(
        self,
        base_prompt: str,
        agent_id: str,
        agent_config: dict,
        memories: Optional[list[Memory]] = None,
        session_context: str = "",
        runtime_directives: str = "",
        claude_md_content: str = "",
        mcp_instructions: str = "",
        environment: Optional[dict] = None,
    ) -> tuple[str, str, str]:
        """Build layered system prompt with cache boundary.

        Returns (full_prompt, static_portion, dynamic_portion).
        """
        return self.prompt_builder.build(
            base_prompt=base_prompt,
            agent_id=agent_id,
            agent_config=agent_config,
            tool_loader=self.tool_loader,
            memories=memories,
            session_context=session_context,
            runtime_directives=runtime_directives,
            environment=environment or {},
            claude_md_content=claude_md_content,
            mcp_instructions=mcp_instructions,
        )

    def get_tool_schemas(self) -> list[dict]:
        """Get full tool schemas for API calls."""
        return self.tool_loader.build_full_tool_schemas()

    # ── Quality ─────────────────────────────────────────────

    async def verify_quality(
        self,
        user_message: str,
        agent_response: str,
        tool_calls: list[dict],
        tool_results: list[str],
        tool_executor,
    ) -> GuardResult:
        """Run post-execution quality checks."""
        return await self.quality_guard.guard(
            user_message=user_message,
            agent_response=agent_response,
            tool_calls=tool_calls,
            tool_results=tool_results,
            tool_executor=tool_executor,
        )

    # ── Skills ──────────────────────────────────────────────

    async def match_skills(
        self,
        user_message: str,
        llm_client=None,
        limit: int = 3,
    ) -> list[Skill]:
        """Match relevant skills for a user message."""
        return await self.skill_registry.match(user_message, llm_client, limit)

    def inject_skills_into_prompt(self, prompt: str, skills: list[Skill]) -> str:
        """Append matched skill content to a system prompt."""
        return self.skill_registry.inject_skills(prompt, skills)

    # ── Hooks (v3: extended) ────────────────────────────────

    async def fire_hooks(
        self,
        event: HookEvent | str,
        context: dict | None = None,
        tool_name: str = "",
    ) -> list:
        """Fire hooks for a lifecycle event."""
        return await self.hook_manager.fire(event, context, tool_name)

    # ── Permissions (v3: new) ───────────────────────────────

    def check_permission(
        self,
        tool_name: str,
        tool_params: dict,
        mode: str = "default",
        context: dict | None = None,
    ) -> PermissionResult:
        """Check if a tool call is permitted."""
        try:
            pmode = PermissionMode(mode)
        except ValueError:
            pmode = PermissionMode.DEFAULT
        return self.permission_engine.evaluate(
            tool_name, tool_params, pmode, context
        )

    # ── Agent loop (v3: new architecture) ───────────────────

    async def run_loop(
        self,
        messages: list,
        system_prompt_static: str,
        system_prompt_dynamic: str,
        tools: list[dict],
        context: QueryContext,
        *,
        llm_client=None,
        tool_executor=None,
        db_factory=None,
    ):
        """Run the main agent loop (async generator).

        Yields SSE event dicts for the frontend.
        """
        async for event in query_loop(
            initial_messages=messages,
            system_prompt_static=system_prompt_static,
            system_prompt_dynamic=system_prompt_dynamic,
            tools=tools,
            context=context,
            llm_client=llm_client,
            tool_executor=tool_executor,
            db_factory=db_factory,
            hook_manager=self.hook_manager,
            permission_engine=self.permission_engine,
        ):
            yield event

    # ── Cache management ────────────────────────────────────

    def invalidate_caches(self):
        """Clear all internal caches (call when prompts/tools change)."""
        self.prompt_builder.invalidate_cache()
        self.skill_registry.reload()
        self.compaction_pipeline.reset_circuit_breaker()
