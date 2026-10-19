"""Layer 4 -- Access control.

API-key based, tiered, quota-aware authorization. Depends only on the
foundation layer (Layer 1). Higher layers (orchestration, public API)
depend on this layer, never the reverse.
"""

from horasis.access.guard import AccessContext, requires_access
from horasis.access.keys import ApiKeyValidator, HashedApiKeyValidator, InMemoryApiKeyValidator
from horasis.access.policy import Policy
from horasis.access.principal import AnonymousPrincipal, Principal
from horasis.access.quota import (
    InMemoryQuotaStore,
    QuotaManager,
    QuotaStore,
    UsageReport,
    WindowUsage,
)
from horasis.access.ratelimit import RateLimiter, RedisRateLimiter, TokenBucketLimiter
from horasis.access.stores import RedisQuotaStore, SQLiteQuotaStore, redis_backends
from horasis.access.tiers import Tier

__all__ = [
    "AccessContext",
    "AnonymousPrincipal",
    "ApiKeyValidator",
    "HashedApiKeyValidator",
    "InMemoryApiKeyValidator",
    "InMemoryQuotaStore",
    "Policy",
    "Principal",
    "QuotaManager",
    "QuotaStore",
    "RateLimiter",
    "RedisQuotaStore",
    "RedisRateLimiter",
    "SQLiteQuotaStore",
    "Tier",
    "TokenBucketLimiter",
    "UsageReport",
    "WindowUsage",
    "redis_backends",
    "requires_access",
]
