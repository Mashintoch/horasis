"""Layer 4 -- Access control.

API-key based, tiered, quota-aware authorization. Depends only on the
foundation layer (Layer 1). Higher layers (orchestration, public API)
depend on this layer, never the reverse.
"""

from horasis.access.keys import ApiKeyValidator, HashedApiKeyValidator, InMemoryApiKeyValidator
from horasis.access.klimax import Klimax
from horasis.access.limen import AccessContext, limen
from horasis.access.metron import (
    InMemoryQuotaStore,
    Metron,
    QuotaStore,
    UsageReport,
    WindowUsage,
)
from horasis.access.nomos import Nomos
from horasis.access.prosopon import AnonymousProsopon, Prosopon
from horasis.access.ratelimit import RateLimiter, RedisRateLimiter, TokenBucketLimiter
from horasis.access.stores import RedisQuotaStore, SQLiteQuotaStore, redis_backends

__all__ = [
    "AccessContext",
    "AnonymousProsopon",
    "ApiKeyValidator",
    "HashedApiKeyValidator",
    "InMemoryApiKeyValidator",
    "InMemoryQuotaStore",
    "Klimax",
    "Metron",
    "Nomos",
    "Prosopon",
    "QuotaStore",
    "RateLimiter",
    "RedisQuotaStore",
    "RedisRateLimiter",
    "SQLiteQuotaStore",
    "TokenBucketLimiter",
    "UsageReport",
    "WindowUsage",
    "limen",
    "redis_backends",
]
