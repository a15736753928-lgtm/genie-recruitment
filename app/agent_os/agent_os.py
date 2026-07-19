"""
AgentOS — facade that wires all Agent OS modules together.

This is the single entry point used by the chat endpoint.  It owns the
lifecycle of MemoryManager, ContextEngine, SystemPromptBuilder, and
QualityGuard, and provides a simple async API for the HTTP layer.
"""

from __future__ import annotations

from typing import Optional

from app.agent_os.memory.manager import MemoryManager
from app.agent_os.memory.retriever import MemoryRetriever
from app.agent_os.memory.models import Memory
from app.agent_os.context.engine import ContextEngine
from app.agent_os.context.assembler import SystemPromptBuilder
from app.agent_os.quality.guard import QualityGuard, GuardResult
from app.agent_os.skills.registry import SkillRegistry
from app.agent_os.skills.models import Skill
from app.agent_os.hooks.manager import HookManager
from app.agent_os.hooks.events import HookEvent
from app.agent_os.orchestration.plan_coordinator import PlanCoordinator, PendingPlan
from app.agent_os.orchestration.dispatcher import SubAgentDispatcher, SubAgentTask, SubAgentResult
from app.agent_os.orchestration.workflow import WorkflowEngine, WorkflowDefinition
from app.config import get_settings

_settings = get_settings()


class AgentOS:
    """Top-level Agent OS coordinator.

    Thread-safe (no mutable shared state).  Create one instance per
    application process — it's cheap and stateless aside from caches.

    Usage::

        agent_os = AgentOS()
        memories = await agent_os.load_memories(query="筛选候选人", llm=client)
        prompt, static, dynamic = agent_os.build_prompt(
            base_prompt, "genie", agent_config, tool_defs, memories,
            runtime_directives="[Intent Gate] ...",
        )
    """

    def __init__(
        self,
        memory_dir: str = "memory/",
        skills_dir: str = "skills/",
        max_context_tokens: int = 8000,
    ):
        self.memory_manager = MemoryManager(memory_dir=memory_dir)
        self.memory_retriever = MemoryRetriever(self.memory_manager)
        self.context_engine = ContextEngine(max_tokens=max_context_tokens)
        self.prompt_builder = SystemPromptBuilder()
        self.quality_guard = QualityGuard()
        self.skill_registry = SkillRegistry(skills_dir=skills_dir)
        self.hook_manager = HookManager()
        self.plan_coordinator = PlanCoordinator()
        self.dispatcher = SubAgentDispatcher(
            max_concurrent=getattr(_settings, "max_subagents", 10)
        )
        self.workflow_engine = WorkflowEngine(self.dispatcher)

    # ── Memory ────────────────────────────────────────────

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

    # ── Prompt building ───────────────────────────────────

    def build_prompt(
        self,
        base_prompt: str,
        agent_id: str,
        agent_config: dict,
        tool_defs: list[dict],
        memories: Optional[list[Memory]] = None,
        session_context: str = "",
        runtime_directives: str = "",
    ) -> tuple[str, str, str]:
        """Build the layered system prompt.

        Returns (full_prompt, static_portion, dynamic_portion).
        """
        tool_catalog = self.prompt_builder.build_tool_catalog(
            tool_defs, deferred=True
        )
        return self.prompt_builder.build(
            base_prompt=base_prompt,
            agent_id=agent_id,
            agent_config=agent_config,
            tool_catalog=tool_catalog,
            memories=memories,
            session_context=session_context,
            runtime_directives=runtime_directives,
        )

    # ── Context management ────────────────────────────────

    async def prepare_context(
        self,
        messages: list,
        system_prompt: str,
        system_prompt_dynamic: str = "",
    ):
        """Run the compression pipeline and return prepared context."""
        return await self.context_engine.prepare(
            messages=messages,
            system_prompt=system_prompt,
            system_prompt_dynamic=system_prompt_dynamic,
        )

    # ── Quality ───────────────────────────────────────────

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

    # ── Skills ─────────────────────────────────────────────

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

    # ── Hooks ──────────────────────────────────────────────

    async def fire_hooks(
        self,
        event: HookEvent,
        context: dict | None = None,
        tool_name: str = "",
    ) -> list[dict]:
        """Fire hooks for a lifecycle event."""
        return await self.hook_manager.fire(event, context, tool_name)

    # ── Plan coordination ──────────────────────────────────

    # (plan_coordinator is used directly via agent_os.plan_coordinator.confirm())

    # ── Cache management ──────────────────────────────────

    def invalidate_caches(self):
        """Clear all internal caches (call when prompts/tools change)."""
        self.prompt_builder.invalidate_cache()
        self.skill_registry.reload()
