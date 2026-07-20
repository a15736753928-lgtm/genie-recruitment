"""
Agent OS — the intelligence layer merged into the single LangGraph agent.

The genuinely useful pieces of the former standalone "Agent OS" stack, now
imported directly by the one and only agent endpoint (app/api/ai/agent_chat.py):
- MemoryManager / MemoryRetriever: persistent file-based memory across sessions
- SkillRegistry: on-demand domain expertise loading
- QualityGuard: post-execution write verification + self-reflection
- PermissionEngine: deny-first tool authorization
- ToolResultAdapter (output/): display-hint mapping for the frontend

The dead subsystems (query_loop, streaming_executor, context engine, cache
prompt builder, orchestration, hooks, retry, compaction pipeline, the AgentOS
facade) were removed during the v1/v3 consolidation — token-budget truncation
now lives inline in agent_chat.py. There is now a single agent path.
"""
