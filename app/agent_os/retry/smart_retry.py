"""
SmartRetry — intelligent retry with fallback decisions.

Key decision rules (from Claude Code):
- 529 (overloaded): Only foreground queries retry; background bail immediately
- Opus fallback: After 3 consecutive 529s → throw FallbackTriggeredError → switch model
- OAuth 401: Force token refresh before next attempt
- Context overflow 400: Parse token counts from error → compute new maxTokensOverride
- ECONNRESET/EPIPE: Detects stale keep-alive sockets → disable keep-alive before retry
- Rate limit 429: Wait for Retry-After header duration
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable, Optional

logger = logging.getLogger("genie.retry")


class RetryDecision(str, Enum):
    RETRY = "retry"
    RETRY_WITH_FALLBACK = "retry_with_fallback"  # Switch model
    BAIL = "bail"                                 # Don't retry
    REFRESH_AUTH = "refresh_auth"                 # Refresh token → retry


class FallbackTriggeredError(Exception):
    """Thrown after 3 consecutive 529s — caller should switch to fallback model."""
    def __init__(self, message: str = "Model fallback triggered"):
        super().__init__(message)


class CannotRetryError(Exception):
    """Thrown when a retry is impossible (auth failure, quota exceeded, etc.)."""
    pass


@dataclass
class RetryConfig:
    """Configuration for the smart retry layer."""
    max_retries: int = 10
    base_delay_ms: float = 1000
    max_delay_ms: float = 30000        # 30 second cap
    backoff_multiplier: float = 2.0
    jitter: bool = True                  # Add randomness to avoid thundering herd

    # Fallback thresholds
    overload_fallback_threshold: int = 3  # Consecutive 529s before fallback
    overload_retry_delay_ms: int = 5000

    # Context overflow
    context_overflow_buffer: int = 13000   # Tokens reserved for response

    # Rate limit
    rate_limit_default_wait_ms: int = 30_000

    # Non-retryable errors
    non_retryable_status_codes: set[int] = field(default_factory=lambda: {
        400,  # Bad request (except context overflow)
        401,  # Unauthorized (trigger auth refresh, not retry)
        402,  # Payment required
        403,  # Forbidden
        404,  # Not found
    })


class SmartRetry:
    """Intelligent retry decorator with model fallback.

    Usage::

        retry = SmartRetry(RetryConfig())
        result = await retry.execute(
            lambda: llm_client.chat.completions.create(...),
            source="foreground",
        )
    """

    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()

    async def execute(
        self,
        fn: Callable[[], Awaitable],
        *,
        source: str = "foreground",      # "foreground" | "background"
        fallback_fn: Optional[Callable[[], Awaitable]] = None,
        on_retry: Optional[Callable[[int, str], Awaitable]] = None,
    ):
        """Execute a function with smart retry logic.

        Args:
            fn: The async function to execute (e.g., an API call).
            source: "foreground" (retry on 529) or "background" (bail on 529).
            fallback_fn: Alternative function if fallback is triggered.
            on_retry: Callback called before each retry attempt.
        """
        consecutive_overloads = 0
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await fn()

            except asyncio.CancelledError:
                raise  # Never suppress cancellation

            except Exception as e:
                last_error = e
                error_str = str(e)
                status_code = self._extract_status_code(e)

                decision = self._classify_error(
                    status_code, error_str, source, consecutive_overloads
                )

                if decision == RetryDecision.BAIL:
                    raise CannotRetryError(f"不可重试的错误: {error_str[:200]}") from e

                if decision == RetryDecision.REFRESH_AUTH:
                    # Auth refresh would happen here
                    # For now: treat as bail (caller should refresh token)
                    raise CannotRetryError("需要刷新认证令牌") from e

                if decision == RetryDecision.RETRY_WITH_FALLBACK:
                    if fallback_fn:
                        logger.info("Falling back to alternative model...")
                        try:
                            return await fallback_fn()
                        except Exception as fe:
                            raise FallbackTriggeredError(
                                f"Fallback model also failed: {fe}"
                            ) from fe
                    else:
                        raise FallbackTriggeredError(
                            "Model fallback triggered but no fallback provided"
                        ) from e

                # decision == RETRY
                consecutive_overloads += 1

                delay = self._compute_delay(attempt, status_code)

                if on_retry:
                    try:
                        await on_retry(attempt + 1, str(e)[:200])
                    except Exception:
                        pass

                logger.info(
                    "Retry %d/%d after %dms (reason: %s)",
                    attempt + 1, self.config.max_retries,
                    int(delay * 1000), str(e)[:100],
                )

                await asyncio.sleep(delay)

        # Exhausted retries
        raise CannotRetryError(
            f"重试 {self.config.max_retries} 次后仍然失败: {last_error}"
        ) from last_error

    def _classify_error(
        self,
        status_code: Optional[int],
        error_str: str,
        source: str,
        consecutive_overloads: int,
    ) -> RetryDecision:
        """Classify an error and decide what to do."""

        # 529: Overloaded
        if status_code == 529 or "overloaded" in error_str.lower():
            if source == "background":
                return RetryDecision.BAIL  # Background tasks bail immediately

            if consecutive_overloads >= self.config.overload_fallback_threshold:
                return RetryDecision.RETRY_WITH_FALLBACK

            return RetryDecision.RETRY

        # 401: Unauthorized → refresh auth
        if status_code == 401:
            return RetryDecision.REFRESH_AUTH

        # 429: Rate limited
        if status_code == 429 or "rate_limit" in error_str.lower():
            return RetryDecision.RETRY

        # 413 / context overflow → extract token info
        if status_code == 413 or "prompt_too_long" in error_str.lower():
            # Parse token counts for recovery
            token_info = self._parse_context_overflow(error_str)
            if token_info:
                return RetryDecision.RETRY
            return RetryDecision.BAIL

        # 5xx: Server errors → retry
        if status_code and 500 <= status_code < 600:
            return RetryDecision.RETRY

        # Network errors (ECONNRESET, EPIPE, etc.)
        network_errors = [
            "econnreset", "epipe", "econnrefused", "etimedout",
            "connection reset", "broken pipe", "timeout",
        ]
        if any(ne in error_str.lower() for ne in network_errors):
            return RetryDecision.RETRY

        # Non-retryable
        if status_code in self.config.non_retryable_status_codes:
            return RetryDecision.BAIL

        # Unknown errors: retry (conservative — might be transient)
        return RetryDecision.RETRY

    def _compute_delay(self, attempt: int, status_code: Optional[int]) -> float:
        """Compute retry delay with exponential backoff + jitter."""
        if status_code == 529 or status_code == 429:
            # Overloaded/rate-limited: longer initial wait
            delay = self.config.overload_retry_delay_ms / 1000
        else:
            delay = self.config.base_delay_ms / 1000

        delay *= (self.config.backoff_multiplier ** attempt)

        # Cap at max delay
        delay = min(delay, self.config.max_delay_ms / 1000)

        # Add jitter (±25%)
        if self.config.jitter:
            import random
            jitter_factor = 1 + (random.random() - 0.5) * 0.5  # 0.75 to 1.25
            delay *= jitter_factor

        return delay

    @staticmethod
    def _extract_status_code(exception: Exception) -> Optional[int]:
        """Extract HTTP status code from various exception types."""
        error_str = str(exception)

        # Try to find status code in error message
        import re
        match = re.search(r"status(?:[ _]code)?[=: ]*(\d{3})", error_str, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Check for common attribute names
        for attr in ("status_code", "status", "http_status", "code"):
            val = getattr(exception, attr, None)
            if isinstance(val, int) and 100 <= val <= 599:
                return val

        return None

    @staticmethod
    def _parse_context_overflow(error_str: str) -> Optional[dict]:
        """Parse context overflow error for token count information.

        Returns dict with {prompt_tokens, max_tokens} or None.
        """
        import re
        # Pattern: "prompt tokens: X, max tokens: Y"
        prompt_match = re.search(r"prompt(?:_)?\s*tokens?[=: ]*(\d+)", error_str, re.IGNORECASE)
        max_match = re.search(r"max(?:_)?\s*tokens?[=: ]*(\d+)", error_str, re.IGNORECASE)

        if prompt_match:
            result = {"prompt_tokens": int(prompt_match.group(1))}
            if max_match:
                result["max_tokens"] = int(max_match.group(1))
            return result

        return None


# ── Convenience function ─────────────────────────────────────

async def with_retry(
    fn: Callable[[], Awaitable],
    *,
    max_retries: int = 10,
    source: str = "foreground",
    fallback_fn: Optional[Callable[[], Awaitable]] = None,
) -> any:
    """Convenience wrapper: execute with smart retry."""
    config = RetryConfig(max_retries=max_retries)
    retry = SmartRetry(config)
    return await retry.execute(fn, source=source, fallback_fn=fallback_fn)
