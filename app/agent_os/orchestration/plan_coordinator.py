"""
PlanCoordinator — generate plan, wait for user confirmation, execute, verify.

Upgrades the existing supervisor_graph's auto-execute behavior with a user
approval gate.  Complex plans are presented to the user as a ``plan_proposal``
SSE event; the stream PAUSES until the user calls the confirm-plan API.

Usage (in chat endpoint)::

    coordinator = PlanCoordinator()
    async for event in coordinator.coordinate(message, session_id, ...):
        yield event
"""

from __future__ import annotations

import json
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional
from datetime import datetime


@dataclass
class PendingPlan:
    """A plan awaiting user confirmation."""
    plan_id: str
    session_id: str
    title: str
    steps: list[dict]
    created_at: datetime = field(default_factory=datetime.utcnow)
    confirmed: bool = False
    action: str = ""          # "approve" | "reject" | "modify"
    modifications: str = ""
    event: asyncio.Event = field(default_factory=asyncio.Event)


class PlanCoordinator:
    """Coordinates the plan → confirm → execute → verify cycle.

    Stores pending plans in memory (keyed by plan_id).  The confirm-plan
    API endpoint signals the event to resume the paused stream.
    """

    def __init__(self, timeout: int = 300):
        self._pending: dict[str, PendingPlan] = {}
        self._timeout = timeout

    # ── Plan lifecycle ────────────────────────────────────

    async def propose_plan(
        self,
        plan_id: str,
        session_id: str,
        title: str,
        steps: list[dict],
    ) -> PendingPlan:
        """Register a plan and return it for SSE emission."""
        plan = PendingPlan(
            plan_id=plan_id,
            session_id=session_id,
            title=title,
            steps=steps,
        )
        self._pending[plan_id] = plan
        return plan

    async def wait_for_confirmation(self, plan_id: str) -> dict:
        """Block until the user confirms/rejects the plan (or timeout).

        Returns:
            {"action": "approve" | "reject" | "timeout", "modifications": "..."}
        """
        plan = self._pending.get(plan_id)
        if not plan:
            return {"action": "reject", "modifications": "plan not found"}

        try:
            await asyncio.wait_for(plan.event.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            plan.action = "timeout"
            return {"action": "timeout", "modifications": "user did not respond"}

        return {
            "action": plan.action,
            "modifications": plan.modifications,
        }

    def confirm(self, plan_id: str, action: str, modifications: str = "") -> bool:
        """Called by the confirm-plan API endpoint.

        Returns True if the plan was found and signaled.
        """
        plan = self._pending.get(plan_id)
        if not plan:
            return False
        plan.action = action
        plan.modifications = modifications
        plan.confirmed = (action == "approve")
        plan.event.set()
        return True

    # ── SSE helpers ───────────────────────────────────────

    @staticmethod
    def _sse(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def plan_to_sse_dict(self, plan: PendingPlan) -> dict:
        return {
            "plan_id": plan.plan_id,
            "title": plan.title,
            "steps": plan.steps,
            "total_steps": len(plan.steps),
        }

    # ── Cleanup ───────────────────────────────────────────

    def cleanup(self, plan_id: str):
        """Remove a plan from pending storage."""
        self._pending.pop(plan_id, None)
