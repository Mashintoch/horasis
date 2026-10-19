from __future__ import annotations

import logging

import pytest

from horasis import Client
from horasis.access import RedisQuotaStore, RedisRateLimiter, TokenBucketLimiter, redis_backends
from horasis.access.ratelimit import RateLimiter
from horasis.cli import warn_if_limiter_not_shared
from horasis.foundation.config import Config, QuotaRule, RateRule, TierRule
from horasis.foundation.errors import ConfigurationError, RateLimitedError

fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("lupa")  # fakeredis needs it to run Lua scripts

KEY = "horasis:r:{u}:*"


def _tier(per_minute: int = 60, burst: int = 2, **task: RateRule) -> TierRule:
    return TierRule(
        name="t",
        quota=QuotaRule(limit=1000, window_seconds=60),
        rate=RateRule(per_minute=per_minute, burst=burst),
        task_rates=task,
    )


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


def _limiter(server: fakeredis.FakeServer) -> RedisRateLimiter:
    return RedisRateLimiter(fakeredis.FakeRedis(server=server))


def test_both_limiters_satisfy_the_protocol(server: fakeredis.FakeServer) -> None:
    assert isinstance(TokenBucketLimiter(), RateLimiter)
    assert isinstance(_limiter(server), RateLimiter)
    assert TokenBucketLimiter.shared is False
    assert RedisRateLimiter.shared is True


def test_bucket_is_shared_between_workers_and_reports_retry_after(
    server: fakeredis.FakeServer,
) -> None:
    worker_a, worker_b = _limiter(server), _limiter(server)
    tier = _tier()
    worker_a.acquire("u", "detect", tier)
    worker_b.acquire("u", "detect", tier)  # burst of 2 is spent across both workers
    with pytest.raises(RateLimitedError) as info:
        worker_a.acquire("u", "detect", tier)
    err = info.value
    assert err.scope == "rate" and err.limit == 60
    assert err.retry_after is not None and 0 < err.retry_after <= 1.0
    worker_a.acquire("other", "detect", tier)  # other principals are independent


def test_refill_uses_elapsed_time_from_redis(server: fakeredis.FakeServer) -> None:
    limiter, raw = _limiter(server), fakeredis.FakeRedis(server=server, decode_responses=True)
    tier = _tier()
    limiter.acquire("u", "detect", tier, cost=2)
    with pytest.raises(RateLimitedError):
        limiter.acquire("u", "detect", tier)
    raw.hset(KEY, "ts", int(raw.hget(KEY, "ts")) - 2000)  # pretend 2s passed on the server
    limiter.acquire("u", "detect", tier, cost=2)


def test_refund_returns_tokens_but_not_past_capacity(server: fakeredis.FakeServer) -> None:
    limiter = _limiter(server)
    tier = _tier()
    limiter.acquire("u", "detect", tier, cost=2)
    limiter.refund("u", "detect", tier, cost=2)
    limiter.refund("u", "detect", tier, cost=5)
    limiter.acquire("u", "detect", tier, cost=2)
    with pytest.raises(RateLimitedError):
        limiter.acquire("u", "detect", tier)


def test_all_or_nothing_across_buckets(server: fakeredis.FakeServer) -> None:
    limiter = _limiter(server)
    tier = _tier(burst=5, detect=RateRule(per_minute=60, burst=1))
    limiter.acquire("u", "detect", tier)
    with pytest.raises(RateLimitedError) as info:
        limiter.acquire("u", "detect", tier)
    assert info.value.scope == "rate:detect"
    for _ in range(4):  # the general bucket only paid for the one successful call
        limiter.acquire("u", "ocr", tier)


def test_cost_larger_than_burst_never_succeeds(server: fakeredis.FakeServer) -> None:
    with pytest.raises(RateLimitedError) as info:
        _limiter(server).acquire("u", "detect", _tier(), cost=3)
    assert info.value.retry_after is None


def test_no_rate_rule_means_no_redis_traffic(server: fakeredis.FakeServer) -> None:
    _limiter(server).acquire("u", "detect", TierRule(name="t", quota=QuotaRule(limit=1, window_seconds=60)))
    assert fakeredis.FakeRedis(server=server).keys("*") == []


def test_redis_backends_share_one_client() -> None:
    client = fakeredis.FakeRedis()
    store, limiter = redis_backends(client)
    assert store.client is client and limiter.client is client


def test_build_wires_redis_limiter_from_config() -> None:
    store = RedisQuotaStore(fakeredis.FakeRedis())
    config = Config.default().model_copy(update={"rate_limit_store": "redis"})
    client = Client.build(config=config, quota_store=store)
    assert isinstance(client.rate_limiter, RedisRateLimiter)
    assert client.rate_limiter.client is store.client
    with pytest.raises(ConfigurationError):
        Client.build(config=config)  # redis requested, nothing to connect to


def test_build_accepts_an_explicit_limiter() -> None:
    limiter = TokenBucketLimiter()
    assert Client.build(rate_limiter=limiter).rate_limiter is limiter


def test_serve_warns_for_multiple_workers_with_in_process_limiter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    local = Client.build()
    shared = Client.build(rate_limiter=_limiter(fakeredis.FakeServer()))
    with caplog.at_level(logging.WARNING, logger="horasis"):
        assert warn_if_limiter_not_shared(1, local) is False
        assert warn_if_limiter_not_shared(4, shared) is False
        assert caplog.records == []
        assert warn_if_limiter_not_shared(4, local) is True
    assert "in-process" in caplog.text and "--redis-url" in caplog.text
