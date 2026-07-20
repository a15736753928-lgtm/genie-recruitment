"""Injectable clock — replaces direct `date.today()` / `datetime.utcnow()` calls.

Usage::

    from app.utils.clock import clock
    today = clock.today()
    now = clock.utcnow()

Tests can monkeypatch ``clock.today`` / ``clock.utcnow`` to freeze time.
"""

from __future__ import annotations

from datetime import date, datetime


class Clock:
    """Default clock that delegates to the real system clock."""

    def today(self) -> date:
        return date.today()

    def utcnow(self) -> datetime:
        return datetime.utcnow()


clock = Clock()
