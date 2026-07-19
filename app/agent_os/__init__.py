"""
Agent OS — the intelligence layer above LangGraph.

Inspired by Claude Code's architecture, this package provides:
- MemoryManager: persistent file-based memory across sessions
- ContextEngine: intelligent context window management
- QualityGuard: post-execution verification + self-reflection
- SkillRegistry: on-demand domain expertise loading
- HookSystem: event-driven automation
- SubAgentDispatcher: multi-agent orchestration with context isolation
- WorkflowEngine: declarative multi-step workflows
- StructuredOutput: typed tool results

All modules are independent and can be enabled/disabled via config flags.
"""
