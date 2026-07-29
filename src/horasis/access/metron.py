from __future__ import annotations

import threading
import time
from typing import Protocol

from horasis.foundation.kanon import QuotaRule
from horasis.foundation.sphalma import QuotaExceededError


class QuotaStore(Protocol):
    """Pluggable counter backend for :class:`Metron` (e.g. Redis in production)."""

    def increment_and_get(self, key: str, *, window_seconds: int) -> int:
        """Increment the counter for `key` in the current window and return the new count."""
        ...

    def current(self, key: str, *, window_seconds: int) -> int:
        """Return the current count for `key` without incrementing it."""
        ...


class InMemoryQuotaStore:
    """Thread-safe, fixed-window counter held in process memory.

    Not suitable for multi-process deployments; swap in a Redis-backed
    :class:`QuotaStore` for production.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> (window_start_epoch, count)
        self._buckets: dict[str, tuple[int, int]] = {}

    def _window_start(self, window_seconds: int) -> int:
        now = int(time.time())
        return now - (now % window_seconds)

    def increment_and_get(self, key: str, *, window_seconds: int) -> int:
        window_start = self._window_start(window_seconds)
        with self._lock:
            stored_start, count = self._buckets.get(key, (window_start, 0))
            if stored_start != window_start:
                count = 0
                stored_start = window_start
            count += 1
            self._buckets[key] = (stored_start, count)
            return count

    def current(self, key: str, *, window_seconds: int) -> int:
        window_start = self._window_start(window_seconds)
        with self._lock:
            stored_start, count = self._buckets.get(key, (window_start, 0))
            if stored_start != window_start:
                return 0
            return count


class Metron:
    """Enforces :class:`QuotaRule` limits per principal."""

    def __init__(self, store: QuotaStore | None = None) -> None:
        self._store = store or InMemoryQuotaStore()

    def check_and_consume(self, principal_id: str, rule: QuotaRule) -> int:
        """Consume one unit of quota for `principal_id` under `rule`.

        Returns:
            The caller's usage count within the current window, including
            this call.

        Raises:
            QuotaExceededError: if this call would exceed `rule.limit`.
        """
        key = f"{principal_id}:{rule.window_seconds}"
        count = self._store.increment_and_get(key, window_seconds=rule.window_seconds)
        if count > rule.limit:
            raise QuotaExceededError(
                f"Quota exceeded for principal {principal_id!r}: "
                f"{rule.limit} calls / {rule.window_seconds}s",
                limit=rule.limit,
                used=count,
                window_seconds=rule.window_seconds,
                context={"principal_id": principal_id},
            )
        return count

    def remaining(self, principal_id: str, rule: QuotaRule) -> int:
        key = f"{principal_id}:{rule.window_seconds}"
        used = self._store.current(key, window_seconds=rule.window_seconds)
        return max(rule.limit - used, 0)
