"""
QualityGuard — execution quality gatekeeper.

Runs after the agent produces a response but before it's delivered to the
user.  Orchestrates verification (read-after-write) and reflection
(self-review) checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from app.agent_os.quality.verifier import WriteVerifier, VerificationResult
from app.agent_os.quality.reflector import Reflector, ReflectionOutput


@dataclass
class GuardResult:
    """Output from QualityGuard.guard()."""
    passed: bool = True
    verification_results: list[VerificationResult] = field(default_factory=list)
    reflection_output: Optional[ReflectionOutput] = None
    issues: list[str] = field(default_factory=list)


class QualityGuard:
    """Runs post-execution quality checks on agent output.

    Usage::

        guard = QualityGuard()
        result = await guard.guard(
            user_message="推荐候选人",
            agent_response="...",
            tool_calls=[...],
            tool_results=[...],
            tool_executor=execute_tool_call,
        )
        if not result.passed:
            # Inject issues into reflection loop
            ...
    """

    def __init__(self):
        self.verifier = WriteVerifier()
        self.reflector = Reflector()

    async def guard(
        self,
        user_message: str,
        agent_response: str,
        tool_calls: list[dict],
        tool_results: list[str],
        tool_executor: Callable,          # async (tool_name, params) -> str
    ) -> GuardResult:
        """Run all quality checks.

        Returns:
            GuardResult with passed=False if any issues were found
            that should block or modify the response.
        """
        issues: list[str] = []
        verification_results: list[VerificationResult] = []

        # ── Step 1: Verification for write operations ──
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            if not self.verifier.should_verify(tool_name):
                continue
            params = tc.get("params", {}) or tc.get("arguments", {}) or {}
            vresult = await self.verifier.verify(
                tool_name=tool_name,
                params=params,
                executor=tool_executor,
            )
            verification_results.append(vresult)
            if not vresult.verified:
                issues.append(
                    f"验证失败 [{tool_name}]: {vresult.discrepancy}"
                )

        # ── Step 2: Reflection check ──
        tool_results_summary = "\n".join(
            str(r)[:200] for r in tool_results[-10:]
        )
        reflection = self.reflector.should_reflect(
            user_message=user_message,
            agent_response=agent_response,
            tool_results_summary=tool_results_summary,
        )

        if reflection.should_reflect:
            issues.append(f"建议反思: {reflection.reason}")

        return GuardResult(
            passed=len(issues) == 0,
            verification_results=verification_results,
            reflection_output=reflection if reflection.should_reflect else None,
            issues=issues,
        )
