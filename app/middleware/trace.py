"""Request-level trace ID via contextvars.

Every request gets a unique ``trace_id`` that flows through the entire
pipeline — HTTP middleware → SSE streaming → LLM calls → tool execution →
log output.  Searching for one ``trace_id`` in the logs gives the full
trace for a single user interaction.

Usage (anywhere in the codebase)::

    from app.middleware.trace import get_trace_id
    trace_id = get_trace_id()
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

__all__ = ["TraceMiddleware", "get_trace_id", "trace_id_var"]

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """Return the current request's trace ID, or an empty string."""
    return trace_id_var.get()


def _generate_trace_id() -> str:
    """Generate a short, sortable trace ID.

    Uses uuid4 (random) truncated to 12 hex chars — short enough to paste
    into a chat or log grep, unique enough for a single deployment.
    """
    return uuid.uuid4().hex[:12]


class TraceMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that sets up a per-request trace ID.

    Ordering: register this BEFORE CORS middleware so that the
    ``X-Trace-ID`` response header is visible to browser clients.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Read from upstream (e.g. nginx / frontend) or generate locally
        trace_id = request.headers.get("X-Trace-ID") or request.headers.get(
            "X-Request-ID"
        ) or _generate_trace_id()

        token = trace_id_var.set(trace_id)
        try:
            response = await call_next(request)
        finally:
            trace_id_var.reset(token)

        # Echo back so the caller (browser / frontend) can correlate
        response.headers["X-Trace-ID"] = trace_id
        return response
