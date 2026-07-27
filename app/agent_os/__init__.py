"""
Agent OS — the two remaining support pieces for the single LangGraph agent.

What's left after the over-engineering cleanup, imported directly by the one
and only agent endpoint (app/api/ai/agent_chat.py):
- QualityGuard (quality/): post-execution read-after-write verification
- DISPLAY_HINTS (output/): display-hint mapping for the frontend

Removed during the cleanup: SkillRegistry (skills/), the deny-first
PermissionEngine (permissions/), the keyword Reflector, the intent
classifier + tool gating, the supervisor graph, the plan generator, and
the never-called ToolResultAdapter + its duplicate ToolResult schemas
(the live one is app/agent/tool_result.py).
The ReAct loop now sees the full tool set and self-corrects instead.
"""
