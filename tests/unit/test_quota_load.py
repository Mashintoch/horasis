from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from horasis.access.quota import QuotaManager
from horasis.foundation.config import QuotaRule
from horasis.foundation.errors import QuotaExceededError


def test_quota_is_thread_safe_under_concurrent_load() -> None:
    quota_manager = QuotaManager()
    rule = QuotaRule(limit=50, window_seconds=3600)
    successes = 0
    failures = 0

    def attempt() -> str:
        try:
            quota_manager.check_and_consume("load-test-principal", rule)
            return "ok"
        except QuotaExceededError:
            return "denied"

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: attempt(), range(200)))

    successes = results.count("ok")
    failures = results.count("denied")

    assert successes == 50
    assert failures == 150
    assert successes + failures == 200


def test_quota_load_across_many_principals_stays_isolated() -> None:
    quota_manager = QuotaManager()
    rule = QuotaRule(limit=5, window_seconds=3600)

    def hammer(principal_id: str) -> int:
        admitted = 0
        for _ in range(10):
            try:
                quota_manager.check_and_consume(principal_id, rule)
                admitted += 1
            except QuotaExceededError:
                pass
        return admitted

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(hammer, [f"principal-{i}" for i in range(20)]))

    assert all(count == 5 for count in results)
