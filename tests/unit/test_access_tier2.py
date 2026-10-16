from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from horasis import Skopos
from horasis.access.keys import HashedApiKeyValidator, InMemoryApiKeyValidator
from horasis.access.klimax import Klimax
from horasis.access.metron import Metron
from horasis.access.ratelimit import TokenBucketLimiter
from horasis.access.stores import RedisQuotaStore, SQLiteQuotaStore
from horasis.backend.taxis import Taxis
from horasis.foundation.kanon import Kanon, QuotaRule, RateRule, TierRule
from horasis.foundation.sphalma import (
    AccessDeniedError,
    BackendUnavailableError,
    ExpiredApiKeyError,
    InvalidApiKeyError,
    QuotaExceededError,
    RateLimitedError,
    RevokedApiKeyError,
)
from tests.conftest import FakeMechane, png_bytes


class FakeRedis:
    """Just enough of redis-py for RedisQuotaStore (single-threaded)."""

    def __init__(self) -> None:
        self.data: dict[str, int] = {}
        self.ttl: dict[str, int] = {}

    def pipeline(self, transaction: bool = True) -> Any:
        outer = self
        ops: list[tuple[str, tuple[Any, ...]]] = []

        class Pipe:
            def incrby(self, k: str, n: int) -> None:
                ops.append(("incrby", (k, n)))

            def expire(self, k: str, s: int) -> None:
                ops.append(("expire", (k, s)))

            def execute(self) -> list[Any]:
                return [getattr(outer, op)(*a) for op, a in ops]

        return Pipe()

    def incrby(self, k: str, n: int) -> int:
        self.data[k] = self.data.get(k, 0) + n
        return self.data[k]

    def expire(self, k: str, s: int) -> bool:
        self.ttl[k] = s
        return True

    def get(self, k: str) -> bytes | None:
        return str(self.data[k]).encode() if k in self.data else None


@pytest.fixture(params=["memory", "sqlite", "redis"])
def metron(request: pytest.FixtureRequest, tmp_path: Path) -> Metron:
    if request.param == "memory":
        return Metron()
    if request.param == "sqlite":
        return Metron(SQLiteQuotaStore(tmp_path / "q.db"))
    return Metron(RedisQuotaStore(FakeRedis()))


# ---- stores / Metron ---------------------------------------------------


def test_weighted_consume_and_denial_refund(metron: Metron) -> None:
    rule = QuotaRule(limit=5, window_seconds=3600)
    assert metron.check_and_consume("a", rule, cost=2) == 2
    assert metron.check_and_consume("a", rule, cost=3) == 5
    with pytest.raises(QuotaExceededError) as info:
        metron.check_and_consume("a", rule, cost=1)
    err = info.value
    assert (err.limit, err.used, err.scope) == (5, 6, "quota")
    assert err.retry_after is not None and 0 < err.retry_after <= 3600
    assert metron.used("a", rule) == 5  # the denied call was refunded
    assert metron.remaining("a", rule) == 0
    metron.refund("a", rule, cost=2)
    assert metron.remaining("a", rule) == 2


def test_scopes_are_independent_and_refunds_never_go_negative(metron: Metron) -> None:
    rule = QuotaRule(limit=5, window_seconds=3600)
    metron.check_and_consume("a", rule, scope="praxis:ocr")
    assert metron.used("a", rule) == 0
    metron.refund("a", rule, cost=10)
    assert metron.used("a", rule) == 0


def test_window_usage_report(metron: Metron) -> None:
    rule = QuotaRule(limit=10, window_seconds=60)
    metron.check_and_consume("a", rule, cost=4)
    usage = metron.window_usage("a", rule)
    assert (usage.limit, usage.used, usage.remaining) == (10, 4, 6)
    assert 0 < usage.resets_in_seconds <= 60


def test_sqlite_is_atomic_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "shared.db"
    metrons = [Metron(SQLiteQuotaStore(path)) for _ in range(4)]  # like 4 worker processes
    rule = QuotaRule(limit=40, window_seconds=3600)

    def hit(i: int) -> bool:
        try:
            metrons[i % 4].check_and_consume("p", rule)
            return True
        except QuotaExceededError:
            return False

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(hit, range(120)))
    assert results.count(True) == 40


def test_sqlite_persists_and_resets_per_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "p.db"
    SQLiteQuotaStore(path).increment_and_get("k", window_seconds=60, amount=3)
    store = SQLiteQuotaStore(path)
    assert store.current("k", window_seconds=60) == 3
    real = time.time()
    monkeypatch.setattr(time, "time", lambda: real + 120)
    assert store.current("k", window_seconds=60) == 0
    assert store.increment_and_get("k", window_seconds=60) == 1


def test_redis_store_sets_expiry_and_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRedis()
    RedisQuotaStore(fake).increment_and_get("k", window_seconds=60)
    assert list(fake.ttl.values()) == [120]
    monkeypatch.setitem(__import__("sys").modules, "redis", None)
    with pytest.raises(BackendUnavailableError, match="horasis\\[redis\\]"):
        RedisQuotaStore()


# ---- token bucket ------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_token_bucket_burst_refill_and_retry_after() -> None:
    clock = Clock()
    limiter = TokenBucketLimiter(clock)
    tier = TierRule(
        name="t",
        quota=QuotaRule(limit=9, window_seconds=60),
        rate=RateRule(per_minute=60, burst=3),
    )
    for _ in range(3):
        limiter.acquire("u", "detect", tier)
    with pytest.raises(RateLimitedError) as info:
        limiter.acquire("u", "detect", tier)
    assert info.value.limit == 60 and info.value.retry_after == pytest.approx(1.0)
    clock.t += 1.0  # one token per second
    limiter.acquire("u", "detect", tier)
    with pytest.raises(RateLimitedError):
        limiter.acquire("u", "detect", tier)
    limiter.acquire("other", "detect", tier)  # independent principal


def test_token_bucket_weighted_cost_per_praxis_and_never_case() -> None:
    limiter = TokenBucketLimiter(Clock())
    tier = TierRule(
        name="t",
        quota=QuotaRule(limit=9, window_seconds=60),
        rate=RateRule(per_minute=60, burst=4),
        praxis_rates={"ocr": RateRule(per_minute=6, burst=2)},
    )
    limiter.acquire("u", "ocr", tier, cost=2)  # drains ocr bucket; 2 of 4 overall
    with pytest.raises(RateLimitedError) as info:
        limiter.acquire("u", "ocr", tier, cost=1)
    assert info.value.scope == "rate:ocr" and info.value.retry_after == pytest.approx(10.0)
    limiter.acquire("u", "detect", tier, cost=2)  # overall bucket not charged by the failure
    with pytest.raises(RateLimitedError) as never:
        limiter.acquire("u", "detect", tier, cost=5)
    assert never.value.retry_after is None


# ---- API keys ----------------------------------------------------------


def test_hashed_keys_store_no_plaintext_and_validate() -> None:
    v = HashedApiKeyValidator()
    key = v.issue(principal_id="alice", tier=Klimax.PRO, scopes=["detect"])
    secret = key.split(".", 1)[1]
    assert secret not in repr(v.export_records())
    principal = v.validate(key)
    assert principal.principal_id == "alice" and principal.scopes == ["detect"]
    assert v.issue(principal_id="a", tier=Klimax.FREE) != v.issue(principal_id="a", tier=Klimax.FREE)


def test_hashed_keys_reject_tampering_unknown_and_malformed() -> None:
    v = HashedApiKeyValidator()
    key = v.issue(principal_id="alice", tier=Klimax.PRO)
    tampered = key[:-1] + ("A" if key[-1] != "A" else "B")
    for bad in (tampered, "hrs_deadbeef.zzz", "nope", "", "hrs_."):
        with pytest.raises(InvalidApiKeyError):
            v.validate(bad)


def test_hashed_keys_expiry_and_revocation() -> None:
    v = HashedApiKeyValidator()
    expired = v.issue(principal_id="a", tier=Klimax.PRO, expires_at=time.time() - 1)
    with pytest.raises(ExpiredApiKeyError):
        v.validate(expired)
    key = v.issue(principal_id="b", tier=Klimax.PRO)
    v.validate(key)
    v.revoke(key)
    with pytest.raises(RevokedApiKeyError):
        v.validate(key)
    by_id = v.issue(principal_id="c", tier=Klimax.PRO)
    v.revoke(by_id.removeprefix("hrs_").split(".")[0])
    with pytest.raises(RevokedApiKeyError):
        v.validate(by_id)


def test_hashed_keys_survive_export_and_load() -> None:
    v = HashedApiKeyValidator()
    key = v.issue(principal_id="alice", tier=Klimax.PRO, scopes=["ocr"])
    restored = HashedApiKeyValidator()
    restored.load_records(v.export_records())
    assert restored.validate(key).scopes == ["ocr"]


# ---- end to end through Skopos ----------------------------------------


def _client(validator: Any, **tier_kwargs: Any) -> Skopos:
    kanon = Kanon(
        default_backend="fake",
        tiers={
            "pro": TierRule(
                name="pro", quota=QuotaRule(limit=10, window_seconds=3600), **tier_kwargs
            )
        },
    )
    taxis = Taxis()
    taxis.register(FakeMechane())
    return Skopos(kanon=kanon, taxis=taxis, validator=validator)


def _pro_validator(**kwargs: Any) -> InMemoryApiKeyValidator:
    v = InMemoryApiKeyValidator()
    v.register_simple("k", principal_id="u", tier=Klimax.PRO, **kwargs)
    return v


def test_scopes_restrict_calls_within_the_tier() -> None:
    client = _client(_pro_validator(scopes=["detect"]))
    client.detection.detect(png_bytes(), api_key="k")
    with pytest.raises(AccessDeniedError, match="not scoped"):
        client.ocr.extract_text(png_bytes(), api_key="k")


def test_costs_usage_report_and_retry_after_end_to_end() -> None:
    client = _client(_pro_validator())
    client.detection.detect(png_bytes(), api_key="k")  # cost 1
    client.ocr.extract_text(png_bytes(), api_key="k")  # cost 2
    report = client.usage("k")
    assert (report.overall.used, report.overall.remaining, report.overall.limit) == (3, 7, 10)
    assert report.per_praxis["ocr"].used == 2 and report.per_praxis["detect"].used == 1
    assert report.per_praxis["face"].used == 0
    assert client.usage("k").overall.used == 3  # reading usage consumes nothing
    for _ in range(3):
        client.moderation.moderate(png_bytes(), api_key="k")  # cost 2 each -> 9
    with pytest.raises(QuotaExceededError) as info:
        client.ocr.extract_text(png_bytes(), api_key="k")
    assert info.value.limit == 10 and info.value.retry_after is not None
    with pytest.raises(InvalidApiKeyError):
        client.usage("nope")


def test_usage_hides_praxeis_outside_key_scopes() -> None:
    report = _client(_pro_validator(scopes=["detect"])).usage("k")
    assert set(report.per_praxis) == {"detect"}


def test_per_praxis_quota_and_rollback_of_overall() -> None:
    client = _client(
        _pro_validator(), praxis_quotas={"detect": QuotaRule(limit=2, window_seconds=3600)}
    )
    client.detection.detect(png_bytes(), api_key="k")
    client.detection.detect(png_bytes(), api_key="k")
    with pytest.raises(QuotaExceededError) as info:
        client.detection.detect(png_bytes(), api_key="k")
    assert info.value.scope == "praxis:detect" and info.value.limit == 2
    report = client.usage("k")
    assert report.overall.used == 2  # denied call refunded from the overall quota
    assert report.per_praxis["detect"].limit == 2


def test_rate_limit_end_to_end_and_refund_when_quota_rejects() -> None:
    client = _client(_pro_validator(), rate=RateRule(per_minute=60, burst=3))
    client.detection.detect(png_bytes(), api_key="k")  # 3 -> 2 tokens
    client.ocr.extract_text(png_bytes(), api_key="k")  # cost 2 -> 0 tokens
    with pytest.raises(RateLimitedError) as info:
        client.detection.detect(png_bytes(), api_key="k")
    assert info.value.retry_after is not None and info.value.retry_after > 0

    # Quota rejection must hand the rate tokens back.
    quota_client = _client(
        _pro_validator(),
        rate=RateRule(per_minute=60, burst=5),
        praxis_quotas={"detect": QuotaRule(limit=1, window_seconds=3600)},
    )
    quota_client.detection.detect(png_bytes(), api_key="k")  # 5 -> 4 tokens
    with pytest.raises(QuotaExceededError):
        quota_client.detection.detect(png_bytes(), api_key="k")
    for _ in range(4):  # all 4 remaining tokens are still there
        quota_client.classification.classify(png_bytes(), api_key="k")
