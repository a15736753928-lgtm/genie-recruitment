"""
QualityGuard — write-verification gatekeeper.

Runs after the agent produces a response.  For write operations it re-reads
the affected record to confirm the change actually took effect (read-after-write).
The keyword-based "reflection" heuristic was removed — it was decorative and
never fed back into the loop.  Verification issues are now returned to the
caller so they CAN be injected back into the agent for a correction pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.agent_os.quality.verifier import WriteVerifier, VerificationResult


@dataclass
class GuardResult:
    """Output from QualityGuard.guard()."""
    passed: bool = True
    verification_results: list[VerificationResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


class QualityGuard:
    """Runs post-execution read-after-write verification on agent output."""

    def __init__(self):
        self.verifier = WriteVerifier()

    async def guard(
        self,
        tool_calls: list[dict],
        tool_executor: Callable,          # async (tool_name, params) -> str
    ) -> GuardResult:
        """Verify that write operations actually took effect.

        Returns:
            GuardResult with passed=False if any write could not be confirmed.
        """
        issues: list[str] = []
        verification_results: list[VerificationResult] = []

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
                    f"写操作 [{tool_name}] 未能确认生效: {vresult.discrepancy}"
                )

        return GuardResult(
            passed=len(issues) == 0,
            verification_results=verification_results,
            issues=issues,
        )
