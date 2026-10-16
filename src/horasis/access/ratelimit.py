"""Per-minute token-bucket rate limiting (in-process)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from horasis.foundation.kanon import RateRule, TierRule
from horasis.foundation.sphalma import RateLimitedError


class TokenBucketLimiter:
    """Thread-safe token buckets keyed by principal (and praxis).

    Each :class:`RateRule` is a bucket holding ``capacity`` tokens that
    refills at ``per_minute / 60`` tokens per second. A call costs ``cost``
    tokens. State lives in this process, so with several workers each one
    enforces its own bucket.

    Args:
        clock: Monotonic seconds source (injectable for tests).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_seen)

    def _peek(self, key: str, rule: RateRule, now: float) -> float:
        tokens, last = self._buckets.get(key, (float(rule.capacity), now))
        return min(float(rule.capacity), tokens + (now - last) * rule.per_minute / 60.0)

    @staticmethod
    def _rules(
        principal_id: str, praxis: str, tier: TierRule
    ) -> list[tuple[str, str, RateRule]]:
        rules: list[tuple[str, str, RateRule]] = []
        if tier.rate is not None:
            rules.append(("rate", f"{principal_id}:*", tier.rate))
        if praxis in tier.praxis_rates:
            rules.append((f"rate:{praxis}", f"{principal_id}:{praxis}", tier.praxis_rates[praxis]))
        return rules

    def acquire(self, principal_id: str, praxis: str, tier: TierRule, cost: int = 1) -> None:
        """Take `cost` tokens from every applicable bucket, or take none.

        Raises:
            RateLimitedError: with ``retry_after`` (seconds) and the ``limit``
                of the bucket that rejected the call.
        """
        rules = self._rules(principal_id, praxis, tier)
        if not rules:
            return
        with self._lock:
            now = self._clock()
            levels = [(scope, key, rule, self._peek(key, rule, now)) for scope, key, rule in rules]
            for scope, _key, rule, tokens in levels:
                if tokens < cost:
                    never = cost > rule.capacity
                    wait = None if never else (cost - tokens) * 60.0 / rule.per_minute
                    raise RateLimitedError(
                        f"Rate limit exceeded ({scope}): {rule.per_minute} units/minute"
                        + (" (call costs more than the burst size)" if never else ""),
                        limit=rule.per_minute,
                        retry_after=wait,
                        scope=scope,
                        context={"principal_id": principal_id},
                    )
            for _scope, key, _rule, tokens in levels:
                self._buckets[key] = (tokens - cost, now)

    def refund(self, principal_id: str, praxis: str, tier: TierRule, cost: int = 1) -> None:
        """Return tokens taken by `acquire` (when a later check rejected the call)."""
        with self._lock:
            now = self._clock()
            for _scope, key, rule in self._rules(principal_id, praxis, tier):
                tokens = min(float(rule.capacity), self._peek(key, rule, now) + cost)
                self._buckets[key] = (tokens, now)
