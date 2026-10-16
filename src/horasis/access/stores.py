"""Durable `QuotaStore` implementations: SQLite (one host) and Redis (many)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from horasis.access.metron import window_start
from horasis.foundation.sphalma import BackendUnavailableError


class SQLiteQuotaStore:
    """Fixed-window counters in a SQLite file; safe across threads *and* processes.

    Each increment runs inside a ``BEGIN IMMEDIATE`` transaction, so SQLite's
    write lock serializes concurrent writers and counts never collide.
    Old windows are pruned opportunistically.

    Args:
        path: Database file (``":memory:"`` works for single-connection tests).
        timeout: Seconds to wait for another writer's lock.
    """

    def __init__(self, path: str | Path = "horasis_quota.db", *, timeout: float = 30.0) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._path, timeout=timeout, isolation_level=None, check_same_thread=False
        )
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS quota ("
            " key TEXT NOT NULL, window_start INTEGER NOT NULL, count INTEGER NOT NULL,"
            " PRIMARY KEY (key, window_start))"
        )

    def increment_and_get(self, key: str, *, window_seconds: int, amount: int = 1) -> int:
        start = window_start(window_seconds)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO quota (key, window_start, count) VALUES (?, ?, MAX(?, 0)) "
                    "ON CONFLICT(key, window_start) DO UPDATE SET count = MAX(count + ?, 0)",
                    (key, start, amount, amount),
                )
                row = self._conn.execute(
                    "SELECT count FROM quota WHERE key = ? AND window_start = ?", (key, start)
                ).fetchone()
                self._conn.execute(
                    "DELETE FROM quota WHERE key = ? AND window_start < ?", (key, start)
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return int(row[0])

    def current(self, key: str, *, window_seconds: int) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT count FROM quota WHERE key = ? AND window_start = ?",
                (key, window_start(window_seconds)),
            ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()


class RedisQuotaStore:
    """Fixed-window counters in Redis (extra: ``horasis[redis]``).

    Uses ``INCRBY`` (atomic) on a key per window, expired automatically.

    Args:
        client: A ``redis.Redis``-compatible client. If omitted, one is built
            from `url`.
        url: Redis URL used when no client is given.
        prefix: Key prefix.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        url: str = "redis://localhost:6379/0",
        prefix: str = "horasis:q:",
    ) -> None:
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise BackendUnavailableError(
                    "The 'redis' package is required for RedisQuotaStore. "
                    "Install it with: pip install 'horasis[redis]'"
                ) from exc
            client = redis.Redis.from_url(url)
        self._client = client
        self._prefix = prefix

    def _key(self, key: str, window_seconds: int) -> str:
        return f"{self._prefix}{key}:{window_start(window_seconds)}"

    def increment_and_get(self, key: str, *, window_seconds: int, amount: int = 1) -> int:
        full = self._key(key, window_seconds)
        pipe = self._client.pipeline(transaction=True)
        pipe.incrby(full, amount)
        pipe.expire(full, window_seconds * 2)
        count = int(pipe.execute()[0])
        if count < 0:  # refunds can't push a window below zero
            self._client.incrby(full, -count)
            return 0
        return count

    def current(self, key: str, *, window_seconds: int) -> int:
        raw = self._client.get(self._key(key, window_seconds))
        return max(0, int(raw)) if raw is not None else 0
