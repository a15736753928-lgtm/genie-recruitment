"""
Reflector — self-reflection prompts for the agent.

After the agent produces a response, the reflector decides whether the
response should be re-examined.  If so, it injects a reflection prompt
that asks the model to check its own work.

The reflection loop:
  1. Agent produces response + tool calls
  2. Reflector checks: should we reflect?
  3. If yes → inject reflection prompt → Agent re-evaluates → return (possibly corrected) response
  4. Max 2 reflection iterations to prevent infinite loops
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Patterns that suggest the agent might need to re-examine its work
_UNCERTAINTY_MARKERS = [
    "可能", "大概", "也许", "不确定", "不太清楚",
    "maybe", "perhaps", "uncertain", "not sure",
]

_ERROR_MARKERS = [
    "工具执行错误", "查询失败", "不存在", "无权限",
    "error", "failed", "not found",
]

_SUBJECTIVE_TASKS = [
    "推荐", "筛选", "评估", "建议", "分析", "对比",
    "recommend", "evaluate", "assess", "analyze", "compare",
]


@dataclass
class ReflectionOutput:
    should_reflect: bool
    reason: str = ""
    reflection_prompt: str = ""


class Reflector:
    """Decides when the agent should reflect on its own output."""

    def __init__(self, max_iterations: int = 2):
        self.max_iterations = max_iterations

    def should_reflect(
        self,
        user_message: str,
        agent_response: str,
        tool_results_summary: str,
        iteration: int = 0,
    ) -> ReflectionOutput:
        """Check whether a reflection cycle is warranted.

        Triggers when:
        1. The agent used uncertain language in a recommendation task
        2. Any tool returned an error
        3. The user asked a subjective/evaluative question
        4. The response seems too short for the query complexity
        """
        if iteration >= self.max_iterations:
            return ReflectionOutput(should_reflect=False)

        triggers = []

        # Check 1: Uncertainty in subjective task
        user_lower = user_message.lower()
        is_subjective = any(t in user_lower for t in _SUBJECTIVE_TASKS)
        has_uncertainty = any(m in agent_response for m in _UNCERTAINTY_MARKERS)
        if is_subjective and has_uncertainty:
            triggers.append("响应中包含不确定表达，但用户要求的是分析/推荐类任务")

        # Check 2: Tool errors
        has_errors = any(m in tool_results_summary for m in _ERROR_MARKERS)
        if has_errors:
            triggers.append("工具执行中出现了错误")

        # Check 3: Response too short
        if is_subjective and len(agent_response) < 100:
            triggers.append("用户要求分析/推荐，但响应过短（<100 字）")

        if not triggers:
            return ReflectionOutput(should_reflect=False)

        return ReflectionOutput(
            should_reflect=True,
            reason="; ".join(triggers),
            reflection_prompt=self._build_reflection_prompt(
                user_message, agent_response, triggers
            ),
        )

    def _build_reflection_prompt(
        self,
        user_message: str,
        agent_response: str,
        triggers: list[str],
    ) -> str:
        """Build the reflection prompt injected into the conversation."""
        return f"""请回顾你刚才的回答，并检查以下问题：

{chr(10).join(f'- {t}' for t in triggers)}

请回答：
1. 你的回答是否基于工具返回的实际数据？（不要编造数据）
2. 你的建议是否有明确的依据？（每个结论都要能追溯到数据）
3. 你是否有遗漏的重要信息？（如果遗漏了，请补充查询）
4. 你的回答是否过于笼统？（如果过于笼统，请具体化）

基于以上反思，修正你的回答（如有必要），然后重新输出完整回复。"""
