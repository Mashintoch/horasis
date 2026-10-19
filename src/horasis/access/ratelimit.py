"""Per-minute token-bucket rate limiting: in-process, or shared through Redis."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from horasis.foundation.config import RateRule, TierRule
from horasis.foundation.errors import BackendUnavailableError, RateLimitedError


@runtime_checkable
class RateLimiter(Protocol):
    """What `AccessContext` needs from a rate limiter.

    ``shared`` is True when the state is visible to every worker process.
    """

    shared: bool

    def acquire(self, principal_id: str, task: str, tier: TierRule, cost: int = 1) -> None:
        """Take `cost` tokens from every applicable bucket, or none (raises ``RateLimitedError``)."""

    def refund(self, principal_id: str, task: str, tier: TierRule, cost: int = 1) -> None:
        """Return tokens taken by `acquire`."""


def _rules_for(
    principal_id: str, task: str, tier: TierRule
) -> list[tuple[str, str, RateRule]]:
    """``(scope, bucket name, rule)`` for every rate rule that applies to this call."""
    rules: list[tuple[str, str, RateRule]] = []
    if tier.rate is not None:
        rules.append(("rate", f"{principal_id}:*", tier.rate))
    if task in tier.task_rates:
        rules.append((f"rate:{task}", f"{principal_id}:{task}", tier.task_rates[task]))
    return rules


def _rate_error(
    scope: str, rule: RateRule, principal_id: str, wait: float | None
) -> RateLimitedError:
    return RateLimitedError(
        f"Rate limit exceeded ({scope}): {rule.per_minute} units/minute"
        + (" (call costs more than the burst size)" if wait is None else ""),
        limit=rule.per_minute,
        retry_after=wait,
        scope=scope,
        context={"principal_id": principal_id},
    )


class TokenBucketLimiter:
    """Thread-safe token buckets keyed by principal (and task).

    Each :class:`RateRule` is a bucket holding ``capacity`` tokens that
    refills at ``per_minute / 60`` tokens per second. A call costs ``cost``
    tokens. State lives in this process, so with several workers each one
    enforces its own bucket -- use :class:`RedisRateLimiter` to share them.

    Args:
        clock: Monotonic seconds source (injectable for tests).
    """

    shared = False

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_seen)

    def _peek(self, key: str, rule: RateRule, now: float) -> float:
        tokens, last = self._buckets.get(key, (float(rule.capacity), now))
        return min(float(rule.capacity), tokens + (now - last) * rule.per_minute / 60.0)

    @staticmethod
    def _rules(
        principal_id: str, task: str, tier: TierRule
    ) -> list[tuple[str, str, RateRule]]:
        return _rules_for(principal_id, task, tier)

    def acquire(self, principal_id: str, task: str, tier: TierRule, cost: int = 1) -> None:
        """Take `cost` tokens from every applicable bucket, or take none.

        Raises:
            RateLimitedError: with ``retry_after`` (seconds) and the ``limit``
                of the bucket that rejected the call.
        """
        rules = self._rules(principal_id, task, tier)
        if not rules:
            return
        with self._lock:
            now = self._clock()
            levels = [(scope, key, rule, self._peek(key, rule, now)) for scope, key, rule in rules]
            for scope, _key, rule, tokens in levels:
                if tokens < cost:
                    wait = None if cost > rule.capacity else (cost - tokens) * 60.0 / rule.per_minute
                    raise _rate_error(scope, rule, principal_id, wait)
            for _scope, key, _rule, tokens in levels:
                self._buckets[key] = (tokens - cost, now)

    def refund(self, principal_id: str, task: str, tier: TierRule, cost: int = 1) -> None:
        """Return tokens taken by `acquire` (when a later check rejected the call)."""
        with self._lock:
            now = self._clock()
            for _scope, key, rule in self._rules(principal_id, task, tier):
                tokens = min(float(rule.capacity), self._peek(key, rule, now) + cost)
                self._buckets[key] = (tokens, now)


# Both scripts read Redis server time (TIME), never a client clock, and run atomically
# across all of a call's buckets. Bucket state: hash {t: tokens, ts: last update in ms}.
_ACQUIRE_LUA = """
local time = redis.call('TIME')
local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)
local cost = tonumber(ARGV[1])
local levels = {}
for i = 1, #KEYS do
  local cap = tonumber(ARGV[2 * i])
  local per_minute = tonumber(ARGV[2 * i + 1])
  local data = redis.call('HMGET', KEYS[i], 't', 'ts')
  local tokens = tonumber(data[1])
  local ts = tonumber(data[2])
  if tokens == nil then
    tokens = cap
    ts = now
  end
  tokens = math.min(cap, tokens + math.max(0, now - ts) * per_minute / 60000)
  levels[i] = tokens
  if tokens < cost then
    if cost > cap then
      return {0, -1, i}
    end
    return {0, math.ceil((cost - tokens) * 60000 / per_minute), i}
  end
end
for i = 1, #KEYS do
  local cap = tonumber(ARGV[2 * i])
  local per_minute = tonumber(ARGV[2 * i + 1])
  redis.call('HSET', KEYS[i], 't', tostring(levels[i] - cost), 'ts', now)
  redis.call('PEXPIRE', KEYS[i], math.ceil(cap * 60000 / per_minute) * 2 + 60000)
end
return {1, 0, 0}
"""

_REFUND_LUA = """
local time = redis.call('TIME')
local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)
local cost = tonumber(ARGV[1])
for i = 1, #KEYS do
  local cap = tonumber(ARGV[2 * i])
  local per_minute = tonumber(ARGV[2 * i + 1])
  local data = redis.call('HMGET', KEYS[i], 't', 'ts')
  local tokens = tonumber(data[1])
  local ts = tonumber(data[2])
  if tokens == nil then
    tokens = cap
    ts = now
  end
  tokens = math.min(cap, tokens + math.max(0, now - ts) * per_minute / 60000)
  redis.call('HSET', KEYS[i], 't', tostring(math.min(cap, tokens + cost)), 'ts', now)
  redis.call('PEXPIRE', KEYS[i], math.ceil(cap * 60000 / per_minute) * 2 + 60000)
end
return 1
"""


class RedisRateLimiter:
    """Token buckets shared by every worker through Redis (extra: ``horasis[redis]``).

    The same bucket semantics as :class:`TokenBucketLimiter`, but the whole
    check-refill-take step is one Lua script, so it is atomic across workers
    and hosts, and it uses the Redis server's clock (``TIME``) so clock skew
    between workers cannot matter. Buckets are keyed per principal + scope.

    Args:
        client: A ``redis.Redis``-compatible client (it may be the very same
            one given to :class:`~horasis.access.stores.RedisQuotaStore`). If
            omitted, one is built from `url`.
        url: Redis URL used when no client is given.
        prefix: Key prefix. The principal is wrapped in ``{}`` so all of a
            principal's buckets share a Redis Cluster hash slot.
    """

    shared = True

    def __init__(
        self,
        client: Any = None,
        *,
        url: str = "redis://localhost:6379/0",
        prefix: str = "horasis:r:",
    ) -> None:
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise BackendUnavailableError(
                    "The 'redis' package is required for RedisRateLimiter. "
                    "Install it with: pip install 'horasis[redis]'"
                ) from exc
            client = redis.Redis.from_url(url)
        self._client = client
        self._prefix = prefix
        self._acquire = client.register_script(_ACQUIRE_LUA)
        self._refund = client.register_script(_REFUND_LUA)

    @property
    def client(self) -> Any:
        return self._client

    def _call(
        self, rules: list[tuple[str, str, RateRule]], principal_id: str, cost: int
    ) -> tuple[list[str], list[Any]]:
        keys = [
            f"{self._prefix}{{{principal_id}}}:{name[len(principal_id) + 1 :]}" for _, name, _ in rules
        ]
        args: list[Any] = [cost]
        for _, _, rule in rules:
            args += [rule.capacity, rule.per_minute]
        return keys, args

    def acquire(self, principal_id: str, task: str, tier: TierRule, cost: int = 1) -> None:
        rules = _rules_for(principal_id, task, tier)
        if not rules:
            return
        keys, args = self._call(rules, principal_id, cost)
        allowed, wait_ms, index = (int(v) for v in self._acquire(keys=keys, args=args))
        if allowed:
            return
        scope, _name, rule = rules[index - 1]
        wait = None if wait_ms < 0 else math.ceil(wait_ms) / 1000.0
        raise _rate_error(scope, rule, principal_id, wait)

    def refund(self, principal_id: str, task: str, tier: TierRule, cost: int = 1) -> None:
        rules = _rules_for(principal_id, task, tier)
        if rules:
            keys, args = self._call(rules, principal_id, cost)
            self._refund(keys=keys, args=args)
