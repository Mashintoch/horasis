"""Layer 4: API keys, tiers, quotas and rate limits.

Imports only from the foundation layer.
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
