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
    # 结论强度。只有做过「期望值 vs 回读值」比对的才叫「已校验」；
    # 其余只能说明回读请求没报错，不能证明写入内容正确 → 「回读可达」。
    conclusion: str = ""


def _expected_update_resume(params: dict) -> dict[str, str]:
    """update_resume 的明确目标值：status。

    handler 会把中文口语（「淘汰」「一面中」）映射成英文 key 后才落库，
    这里复用同一张表，否则会把成功的写入误判成失败。
    """
    from app.api.ai.tool_handlers.recruitment import STATUS_ALIASES

    raw = params.get("status") or (params.get("fields") or {}).get("status")
    if not raw:
        return {}
    raw = str(raw).strip()
    return {"status": STATUS_ALIASES.get(raw, raw)}


class WriteVerifier:
    """Post-write verification rules.

    Each rule specifies:
    - verify_tool: which read tool to call for verification
    - verify_params: how to derive the read params from the write params
    - expect (可选): 从写参数里提取「明确的目标值」，回读文本里必须能找到它们。
      没有 expect 的规则只能确认「回读可达」，不能确认写入内容正确。
    """

    # ── Rule registry ─────────────────────────────────────

    VERIFICATION_RULES: dict[str, dict] = {
        "update_resume": {
            "verify_tool": "get_resume",
            # detail 视图一定包含 status，回读文本里能直接比对目标状态
            "verify_params": lambda p: {"id": p["id"], "view": "detail"},
            "expect": _expected_update_resume,
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
            return VerificationResult(
                verified=True, tool_name=tool_name, conclusion="未配置校验规则，未做回读"
            )

        verify_tool = rule["verify_tool"]
        verify_params = self.get_verify_params(tool_name, params)

        if not verify_params:
            return VerificationResult(
                verified=True, tool_name=tool_name, conclusion="缺少回读参数，未做回读"
            )

        try:
            readback = await executor(verify_tool, verify_params)
        except Exception as e:
            return VerificationResult(
                verified=False,
                tool_name=tool_name,
                target_id=str(verify_params.get("id", "")),
                discrepancy=f"验证查询失败: {e}",
                conclusion="回读失败",
            )

        text = str(readback)
        target_id = str(verify_params.get("id", ""))

        # For deletes, "不存在" means success
        if tool_name.startswith("delete_") and "不存在" in text:
            return VerificationResult(
                verified=True,
                tool_name=tool_name,
                target_id=target_id,
                readback_data={"raw": text[:500]},
                conclusion="已校验：记录确实已不存在",
            )

        # 回读本身失败 → 直接判未通过
        if "错误" in text or "失败" in text:
            return VerificationResult(
                verified=False,
                tool_name=tool_name,
                target_id=target_id,
                discrepancy=text[:300],
                conclusion="回读失败",
            )

        # 有明确目标值的写操作 → 做实际值比对
        expect_fn = rule.get("expect")
        expected = {}
        if expect_fn:
            try:
                expected = expect_fn(params) or {}
            except Exception:
                expected = {}
        if expected:
            missing = [f"{k}={v}" for k, v in expected.items() if str(v) not in text]
            if missing:
                return VerificationResult(
                    verified=False,
                    tool_name=tool_name,
                    target_id=target_id,
                    readback_data={"raw": text[:500]},
                    discrepancy=(
                        f"期望写入 {', '.join(missing)}，但回读内容里找不到该值；"
                        f"回读片段：{text[:200]}"
                    ),
                    conclusion="已校验：实际值与目标值不一致（写入很可能没生效）",
                )
            return VerificationResult(
                verified=True,
                tool_name=tool_name,
                target_id=target_id,
                readback_data={"raw": text[:500]},
                conclusion=f"已校验：回读值与目标值一致（{', '.join(f'{k}={v}' for k, v in expected.items())}）",
            )

        # 没有可比对的目标值：只能说明回读请求没报错，不能证明写入内容正确。
        return VerificationResult(
            verified=True,
            tool_name=tool_name,
            target_id=target_id,
            readback_data={"raw": text[:500]},
            conclusion="回读可达（未做实际值比对，不代表写入内容正确）",
        )
