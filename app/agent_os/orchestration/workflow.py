"""
WorkflowEngine — declarative multi-stage workflow execution.

Allows the agent to decompose complex tasks into stages that can run
in parallel, sequentially, or as a dependency graph.

Inspired by Claude Code's pipeline() / parallel() patterns::

    pipeline(items, stage1, stage2)  → items flow through stages independently
    parallel([task1, task2])          → all tasks run concurrently

A workflow definition looks like::

    {
      "stages": [
        {"name": "screening", "parallel": true, "tasks": [...]},
        {"name": "merge", "depends_on": ["screening"], "tasks": [...]},
        {"name": "verify", "depends_on": ["merge"], "tasks": [...]}
      ]
    }
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.agent_os.orchestration.dispatcher import (
    SubAgentDispatcher, SubAgentTask, SubAgentResult,
)


@dataclass
class WorkflowStage:
    """A single stage in a workflow (one or more parallel tasks)."""
    name: str
    tasks: list[SubAgentTask] = field(default_factory=list)
    parallel: bool = True
    depends_on: list[str] = field(default_factory=list)


@dataclass
class WorkflowDefinition:
    """A complete workflow definition."""
    workflow_id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:8]}")
    name: str = ""
    stages: list[WorkflowStage] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """Result of executing a workflow."""
    workflow_id: str
    success: bool
    stage_results: dict[str, list[SubAgentResult]] = field(default_factory=dict)
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    error: Optional[str] = None


class WorkflowEngine:
    """Executes multi-stage workflows with dependency management.

    Usage::

        engine = WorkflowEngine(dispatcher)
        wf = WorkflowDefinition(
            name="批量筛选+出题",
            stages=[
                WorkflowStage(name="filter", tasks=[...], parallel=True),
                WorkflowStage(name="questions", tasks=[...], depends_on=["filter"]),
            ],
        )
        result = await engine.execute(wf, llm_client, tool_executor, db_factory)
    """

    def __init__(self, dispatcher: SubAgentDispatcher):
        self.dispatcher = dispatcher

    # ── Execution ──────────────────────────────────────────

    async def execute(
        self,
        workflow: WorkflowDefinition,
        llm_client,
        tool_executor,
        db_factory,
    ) -> WorkflowResult:
        """Execute a workflow, respecting dependencies between stages."""
        completed_stages: dict[str, list[SubAgentResult]] = {}
        total_tasks = sum(len(s.tasks) for s in workflow.stages)
        completed = 0
        failed = 0

        # Topological sort: stages without unmet dependencies go first
        remaining = list(workflow.stages)
        while remaining:
            # Find stages whose dependencies are all satisfied
            ready = [
                s for s in remaining
                if all(d in completed_stages for d in s.depends_on)
            ]
            if not ready:
                return WorkflowResult(
                    workflow_id=workflow.workflow_id,
                    success=False,
                    stage_results=completed_stages,
                    total_tasks=total_tasks,
                    completed_tasks=completed,
                    failed_tasks=failed,
                    error=f"循环依赖或无法满足的依赖: {[s.name for s in remaining]}",
                )

            # Execute all ready stages in parallel (each stage's tasks
            # may themselves be parallel or sequential)
            stage_coros = [
                self._execute_stage(stage, llm_client, tool_executor, db_factory)
                for stage in ready
            ]
            stage_result_lists = await asyncio.gather(*stage_coros)

            for stage, results in zip(ready, stage_result_lists):
                completed_stages[stage.name] = results
                completed += len(results)
                failed += sum(1 for r in results if not r.success)

            remaining = [s for s in remaining if s not in ready]

        return WorkflowResult(
            workflow_id=workflow.workflow_id,
            success=failed == 0,
            stage_results=completed_stages,
            total_tasks=total_tasks,
            completed_tasks=completed,
            failed_tasks=failed,
        )

    async def _execute_stage(
        self,
        stage: WorkflowStage,
        llm_client,
        tool_executor,
        db_factory,
    ) -> list[SubAgentResult]:
        """Execute all tasks in a stage."""
        if not stage.tasks:
            return []
        return await self.dispatcher.dispatch(
            tasks=stage.tasks,
            llm_client=llm_client,
            tool_executor=tool_executor,
            db_factory=db_factory,
            parallel=stage.parallel,
        )

    # ── Workflow builders ──────────────────────────────────

    @staticmethod
    def from_json(data: dict) -> WorkflowDefinition:
        """Parse a workflow from a JSON dict (e.g., agent-generated plan)."""
        stages = []
        for s in data.get("stages", []):
            tasks = [
                SubAgentTask(
                    agent_type=t.get("agent", "genie"),
                    task_description=t.get("task", ""),
                    tools=t.get("tools"),
                    context_data=t.get("context", {}),
                )
                for t in s.get("tasks", [])
            ]
            stages.append(WorkflowStage(
                name=s.get("name", f"stage_{len(stages)}"),
                tasks=tasks,
                parallel=s.get("parallel", True),
                depends_on=s.get("depends_on", []),
            ))
        return WorkflowDefinition(
            name=data.get("name", "未命名工作流"),
            stages=stages,
        )

    # ── SSE helpers ────────────────────────────────────────

    @staticmethod
    def _sse(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def stage_to_sse(self, stage_name: str, results: list[SubAgentResult]) -> str:
        """Emit a stage completion SSE event."""
        return self._sse("plan_step_done", {
            "stage": stage_name,
            "tasks": len(results),
            "success": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
        })
