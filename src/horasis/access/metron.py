from __future__ import annotations

import threading
import time
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from horasis.foundation.kanon import QuotaRule
from horasis.foundation.sphalma import QuotaExceededError


def window_start(window_seconds: int, now: float | None = None) -> int:
    """Start (epoch seconds) of the fixed window containing `now`."""
    t = int(time.time() if now is None else now)
    return t - (t % window_seconds)


def seconds_until_reset(window_seconds: int, now: float | None = None) -> float:
    """Seconds left in the current fixed window."""
    t = time.time() if now is None else now
    return window_seconds - (t - window_start(window_seconds, t))


class QuotaStore(Protocol):
    """Pluggable counter backend for :class:`Metron`.

    Implementations must make ``increment_and_get`` atomic: concurrent
    callers (threads or processes) each see a distinct, consecutive count.
    Counters are fixed-window and reset when the window rolls over.
    """

    def increment_and_get(self, key: str, *, window_seconds: int, amount: int = 1) -> int:
        """Atomically add `amount` (negative to refund) and return the new count."""
        ...

    def current(self, key: str, *, window_seconds: int) -> int:
        """Return the current count for `key` without changing it."""
        ...


class InMemoryQuotaStore:
    """Thread-safe, fixed-window counter held in process memory.

    Not suitable for multi-process deployments; use ``SQLiteQuotaStore``
    (one host) or ``RedisQuotaStore`` (many hosts).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> (window_start_epoch, count)
        self._buckets: dict[str, tuple[int, int]] = {}

    def increment_and_get(self, key: str, *, window_seconds: int, amount: int = 1) -> int:
        start = window_start(window_seconds)
        with self._lock:
            stored_start, count = self._buckets.get(key, (start, 0))
            if stored_start != start:
                count = 0
            count = max(0, count + amount)
            self._buckets[key] = (start, count)
            return count

    def current(self, key: str, *, window_seconds: int) -> int:
        start = window_start(window_seconds)
        with self._lock:
            stored_start, count = self._buckets.get(key, (start, 0))
            return 0 if stored_start != start else count


class WindowUsage(BaseModel):
    """Usage within one quota window."""

    model_config = ConfigDict(frozen=True)

    scope: str = Field(description='"quota" for the overall limit or "praxis:<name>".')
    window_seconds: int
    limit: int
    used: int
    remaining: int
    resets_in_seconds: float


class UsageReport(BaseModel):
    """What a caller has used and has left, in quota units (see ``Kanon.praxis_cost``).

    Attributes:
        principal_id: Who the report is for.
        tier: Their tier name.
        overall: Usage against the tier's main quota window.
        per_praxis: Usage per praxis (each praxis's own quota window if the
            tier defines one, otherwise the tier window with the overall limit).
    """

    model_config = ConfigDict(frozen=True)

    principal_id: str
    tier: str
    overall: WindowUsage
    per_praxis: dict[str, WindowUsage]


class Metron:
    """Enforces :class:`QuotaRule` limits per principal, weighted by call cost."""

    def __init__(self, store: QuotaStore | None = None) -> None:
        self._store = store or InMemoryQuotaStore()

    @staticmethod
    def _key(principal_id: str, rule: QuotaRule, scope: str) -> str:
        return f"{principal_id}:{scope}:{rule.window_seconds}"

    def check_and_consume(
        self, principal_id: str, rule: QuotaRule, *, cost: int = 1, scope: str = "quota"
    ) -> int:
        """Consume `cost` units of quota for `principal_id` under `rule`.

        A rejected call is refunded, so denials never eat into the quota.

        Returns:
            Usage within the current window, including this call.

        Raises:
            QuotaExceededError: if this call would exceed `rule.limit`; it
                carries ``limit``, ``used``, ``scope`` and ``retry_after``.
        """
        key = self._key(principal_id, rule, scope)
        count = self._store.increment_and_get(key, window_seconds=rule.window_seconds, amount=cost)
        if count > rule.limit:
            self._store.increment_and_get(key, window_seconds=rule.window_seconds, amount=-cost)
            raise QuotaExceededError(
                f"Quota exceeded for principal {principal_id!r} ({scope}): "
                f"{rule.limit} units / {rule.window_seconds}s",
                limit=rule.limit,
                used=count,
                window_seconds=rule.window_seconds,
                retry_after=seconds_until_reset(rule.window_seconds),
                scope=scope,
                context={"principal_id": principal_id},
            )
        return count

    def refund(
        self, principal_id: str, rule: QuotaRule, *, cost: int = 1, scope: str = "quota"
    ) -> None:
        """Give back `cost` units (e.g. when a later check rejected the call)."""
        self._store.increment_and_get(
            self._key(principal_id, rule, scope), window_seconds=rule.window_seconds, amount=-cost
        )

    def used(self, principal_id: str, rule: QuotaRule, *, scope: str = "quota") -> int:
        return self._store.current(
            self._key(principal_id, rule, scope), window_seconds=rule.window_seconds
        )

    def remaining(self, principal_id: str, rule: QuotaRule, *, scope: str = "quota") -> int:
        return max(rule.limit - self.used(principal_id, rule, scope=scope), 0)

    def window_usage(
        self, principal_id: str, rule: QuotaRule, *, scope: str = "quota"
    ) -> WindowUsage:
        used = self.used(principal_id, rule, scope=scope)
        return WindowUsage(
            scope=scope,
            window_seconds=rule.window_seconds,
            limit=rule.limit,
            used=used,
            remaining=max(rule.limit - used, 0),
            resets_in_seconds=seconds_until_reset(rule.window_seconds),
        )
