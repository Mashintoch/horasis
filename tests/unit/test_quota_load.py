"""Quota/load coverage: concurrent access to Metron must not over-admit callers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from horasis.access.metron import Metron
from horasis.foundation.kanon import QuotaRule
from horasis.foundation.sphalma import QuotaExceededError


def test_metron_is_thread_safe_under_concurrent_load() -> None:
    metron = Metron()
    rule = QuotaRule(limit=50, window_seconds=3600)
    successes = 0
    failures = 0

    def attempt() -> str:
        try:
            metron.check_and_consume("load-test-principal", rule)
            return "ok"
        except QuotaExceededError:
            return "denied"

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: attempt(), range(200)))

    successes = results.count("ok")
    failures = results.count("denied")

    # Exactly `limit` calls should be admitted; no more, no fewer.
    assert successes == 50
    assert failures == 150
    assert successes + failures == 200


def test_metron_load_across_many_principals_stays_isolated() -> None:
    metron = Metron()
    rule = QuotaRule(limit=5, window_seconds=3600)

    def hammer(principal_id: str) -> int:
        admitted = 0
        for _ in range(10):
            try:
                metron.check_and_consume(principal_id, rule)
                admitted += 1
            except QuotaExceededError:
                pass
        return admitted

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(hammer, [f"principal-{i}" for i in range(20)]))

    assert all(count == 5 for count in results)
