from __future__ import annotations

import pytest

from horasis.access.metron import InMemoryQuotaStore, Metron
from horasis.foundation.kanon import QuotaRule
from horasis.foundation.sphalma import QuotaExceededError


def test_metron_allows_up_to_limit() -> None:
    metron = Metron()
    rule = QuotaRule(limit=3, window_seconds=3600)
    for expected_count in (1, 2, 3):
        assert metron.check_and_consume("alice", rule) == expected_count


def test_metron_raises_once_limit_exceeded() -> None:
    metron = Metron()
    rule = QuotaRule(limit=1, window_seconds=3600)
    metron.check_and_consume("alice", rule)
    with pytest.raises(QuotaExceededError) as exc_info:
        metron.check_and_consume("alice", rule)
    assert exc_info.value.limit == 1
    assert exc_info.value.used == 2


def test_metron_tracks_principals_independently() -> None:
    metron = Metron()
    rule = QuotaRule(limit=1, window_seconds=3600)
    metron.check_and_consume("alice", rule)
    # bob has his own independent bucket
    assert metron.check_and_consume("bob", rule) == 1


def test_remaining_reflects_usage() -> None:
    metron = Metron()
    rule = QuotaRule(limit=5, window_seconds=3600)
    assert metron.remaining("alice", rule) == 5
    metron.check_and_consume("alice", rule)
    metron.check_and_consume("alice", rule)
    assert metron.remaining("alice", rule) == 3


def test_in_memory_quota_store_resets_after_window(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryQuotaStore()
    times = iter([1000, 1000, 1065])  # third call is past a 60s window
    monkeypatch.setattr("time.time", lambda: next(times))

    assert store.increment_and_get("k", window_seconds=60) == 1
    assert store.increment_and_get("k", window_seconds=60) == 2  # same window as call 1
    assert store.increment_and_get("k", window_seconds=60) == 1  # new window -> reset
