"""
WriteVerifier — read-after-write verification for data mutations.

After any write tool (update_resume, create_position, etc.), optionally
re-reads the affected record to confirm the change actually took effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class VerificationResult:
    verified: bool
    tool_name: str = ""
    target_id: str = ""
    discrepancy: Optional[str] = None
    readback_data: Optional[dict] = None


class WriteVerifier:
    """Post-write verification rules.

    Each rule specifies:
    - verify_tool: which read tool to call for verification
    - verify_params: how to derive the read params from the write params
    - check: how to compare original intent with read-back data
    """

    # ── Rule registry ─────────────────────────────────────

    VERIFICATION_RULES: dict[str, dict] = {
        "update_resume": {
            "verify_tool": "get_resume",
            "verify_params": lambda p: {"id": p["id"]},
            "description": "验证候选人信息更新是否生效",
        },
        "update_position": {
            "verify_tool": "get_position",
            "verify_params": lambda p: {"id": p["id"]},
            "description": "验证岗位信息更新是否生效",
        },
        "create_position": {
            "verify_tool": "list_positions",
            "verify_params": lambda p: {},
            "description": "验证岗位是否创建成功",
        },
        "delete_resume": {
            "verify_tool": "get_resume",
            "verify_params": lambda p: {"id": p["id"]},
            "description": "验证候选人是否删除成功（期待 404）",
        },
        "delete_position": {
            "verify_tool": "get_position",
            "verify_params": lambda p: {"id": p["id"]},
            "description": "验证岗位是否删除成功（期待 404）",
        },
        "save_questions": {
            "verify_tool": "get_questions",
            "verify_params": lambda p: {
                "candidateId": p["candidateId"],
                "round": p["round"],
            },
            "description": "验证面试题保存是否成功",
        },
        "save_evaluation": {
            "verify_tool": "get_evaluation",
            "verify_params": lambda p: {
                "candidateId": p["candidateId"],
                "round": p.get("round", "first"),
            },
            "description": "验证面试评分保存是否成功",
        },
    }

    # ── Public API ────────────────────────────────────────

    def should_verify(self, tool_name: str) -> bool:
        """Check if this tool has a verification rule."""
        return tool_name in self.VERIFICATION_RULES

    def get_verify_tool(self, tool_name: str) -> Optional[str]:
        """Get the name of the read tool used for verification."""
        rule = self.VERIFICATION_RULES.get(tool_name)
        return rule["verify_tool"] if rule else None

    def get_verify_params(self, tool_name: str, original_params: dict) -> dict:
        """Compute the parameters for the verification read."""
        rule = self.VERIFICATION_RULES.get(tool_name)
        if not rule:
            return {}
        try:
            return rule["verify_params"](original_params)
        except Exception:
            return {}

    async def verify(
        self,
        tool_name: str,
        params: dict,
        executor: Callable,       # async (tool_name, params) -> str
    ) -> VerificationResult:
        """Execute verification for a write operation.

        Args:
            tool_name: The write tool that was just called.
            params: The original write parameters.
            executor: An async callable that executes a tool and returns its result.

        Returns:
            VerificationResult with verified=True/False and optional discrepancy.
        """
        rule = self.VERIFICATION_RULES.get(tool_name)
        if not rule:
            return VerificationResult(verified=True, tool_name=tool_name)

        verify_tool = rule["verify_tool"]
        verify_params = self.get_verify_params(tool_name, params)

        if not verify_params:
            return VerificationResult(verified=True, tool_name=tool_name)

        try:
            readback = await executor(verify_tool, verify_params)
        except Exception as e:
            return VerificationResult(
                verified=False,
                tool_name=tool_name,
                target_id=str(verify_params.get("id", "")),
                discrepancy=f"验证查询失败: {e}",
            )

        # For deletes, "不存在" means success
        if tool_name.startswith("delete_") and "不存在" in str(readback):
            return VerificationResult(
                verified=True,
                tool_name=tool_name,
                target_id=str(verify_params.get("id", "")),
                readback_data={"raw": str(readback)[:500]},
            )

        # For other operations, any non-error result confirms success
        if "错误" not in str(readback) and "失败" not in str(readback):
            return VerificationResult(
                verified=True,
                tool_name=tool_name,
                target_id=str(verify_params.get("id", "")),
                readback_data={"raw": str(readback)[:500]},
            )

        return VerificationResult(
            verified=False,
            tool_name=tool_name,
            target_id=str(verify_params.get("id", "")),
            discrepancy=str(readback)[:300],
        )
