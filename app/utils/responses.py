"""Standardized API response factories.

Replaces manual ``{"code": 0, "message": "ok", "data": ...}`` with::

    from app.utils.responses import ok, fail, not_found, conflict
    return ok(data=result)
    return not_found("候选人不存在")
"""

from __future__ import annotations


def ok(data=None, message: str = "ok") -> dict:
    return {"code": 0, "message": message, "data": data}


def fail(code: int = 400, message: str = "error", data=None) -> dict:
    return {"code": code, "message": message, "data": data}


def not_found(message: str = "资源不存在") -> dict:
    return fail(404, message)


def conflict(message: str = "资源冲突") -> dict:
    return fail(409, message)
