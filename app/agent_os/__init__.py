"""
Agent OS — the intelligence layer merged into the single LangGraph agent.

The genuinely useful pieces of the former standalone "Agent OS" stack, now
imported directly by the one and only agent endpoint (app/api/ai/agent_chat.py):
- SkillRegistry: on-demand domain expertise loading
- QualityGuard: post-execution write verification + self-reflection
- PermissionEngine: deny-first tool authorization
- ToolResultAdapter (output/): display-hint mapping for the frontend

The memory subsystem (MemoryManager / MemoryRetriever) was removed because
its file-based memories degraded into noise — per-candidate notes became
stale after interviews, and stale business-rule fragments confused the agent.
The dead subsystems (query_loop, streaming_executor, context engine, cache
prompt builder, orchestration, hooks, retry, compaction pipeline, the AgentOS
facade) were also removed during the v1/v3 consolidation.
"""
