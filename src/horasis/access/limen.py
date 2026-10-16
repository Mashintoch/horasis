"""limen -- the access-control decorator.

``limen`` is Latin for "threshold." Every public-API call crosses this
threshold before it reaches orchestration: the caller's API key is
resolved to a :class:`~horasis.access.prosopon.Prosopon`, :class:`~horasis.access.nomos.Nomos`
authorizes the requested :class:`~horasis.foundation.schemas.Praxis`,
:class:`~horasis.access.metron.Metron` consumes one unit of quota, and
:class:`~horasis.foundation.aisthesis.Aisthesis` hooks fire around the
wrapped call.

`limen` is deliberately generic: it decorates any callable whose owning
object exposes an :class:`AccessContext` via a `_access` attribute, and
that accepts `api_key` as a keyword argument.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from horasis.access.keys import ApiKeyValidator
from horasis.access.metron import Metron, UsageReport
from horasis.access.nomos import Nomos
from horasis.access.prosopon import Prosopon
from horasis.access.ratelimit import TokenBucketLimiter
from horasis.foundation.aisthesis import Aisthesis, NullAisthesis
from horasis.foundation.kanon import Kanon, QuotaRule
from horasis.foundation.schemas import Praxis
from horasis.foundation.sphalma import QuotaExceededError, Sphalma, ValidationError

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class AccessContext:
    """Bundles everything `limen` needs to guard a call.

    An object exposing a `_access: AccessContext` attribute (typically a
    Layer 5 public-API class) can have its methods decorated with `@limen`.
    """

    kanon: Kanon
    validator: ApiKeyValidator
    nomos: Nomos
    metron: Metron
    rate_limiter: TokenBucketLimiter
    aisthesis: Aisthesis = field(default_factory=NullAisthesis)

    @classmethod
    def build(
        cls,
        *,
        kanon: Kanon,
        validator: ApiKeyValidator,
        metron: Metron | None = None,
        rate_limiter: TokenBucketLimiter | None = None,
        aisthesis: Aisthesis | None = None,
    ) -> AccessContext:
        return cls(
            kanon=kanon,
            validator=validator,
            nomos=Nomos(kanon),
            metron=metron or Metron(),
            rate_limiter=rate_limiter or TokenBucketLimiter(),
            aisthesis=aisthesis or NullAisthesis(),
        )


    def charge(self, principal: Prosopon, praxis: Praxis, *, units: int = 1) -> None:
        """Rate-limit and meter `units` calls of `praxis` for `principal`.

        Each unit costs ``Kanon.cost_for(praxis)``. All-or-nothing: if any
        limit rejects, everything taken so far is given back.

        Raises:
            RateLimitedError: per-minute bucket empty (has ``retry_after``).
            QuotaExceededError: window quota exhausted (has ``retry_after``,
                ``limit`` and ``scope``).
        """
        rule = self.kanon.tier(principal.tier.value)
        pid, name = principal.principal_id, praxis.value
        cost = self.kanon.cost_for(name) * units
        self.rate_limiter.acquire(pid, name, rule, cost)
        try:
            self.metron.check_and_consume(pid, rule.quota, cost=cost)
            praxis_rule = self._praxis_rule(rule.praxis_quotas.get(name), rule.quota)
            try:
                self.metron.check_and_consume(pid, praxis_rule, cost=cost, scope=f"praxis:{name}")
            except QuotaExceededError:
                self.metron.refund(pid, rule.quota, cost=cost)
                raise
        except QuotaExceededError:
            self.rate_limiter.refund(pid, name, rule, cost)
            raise

    @staticmethod
    def _praxis_rule(specific: QuotaRule | None, overall: QuotaRule) -> QuotaRule:
        return specific or overall

    def usage(self, principal: Prosopon) -> UsageReport:
        """Used/remaining units for `principal`, overall and per allowed praxis."""
        rule = self.kanon.tier(principal.tier.value)
        pid = principal.principal_id
        per_praxis = {
            name: self.metron.window_usage(
                pid, self._praxis_rule(rule.praxis_quotas.get(name), rule.quota), scope=f"praxis:{name}"
            )
            for name in rule.allowed_praxeis
            if principal.scopes is None or name in principal.scopes
        }
        return UsageReport(
            principal_id=pid,
            tier=principal.tier.value,
            overall=self.metron.window_usage(pid, rule.quota),
            per_praxis=per_praxis,
        )


def limen(praxis: Praxis) -> Callable[[F], F]:
    """Decorator that enforces authentication, authorization, and quota.

    Args:
        praxis: The task type this method performs; used both for
            authorization against :class:`~horasis.access.nomos.Nomos` and
            for telemetry.

    The wrapped method must be called with an `api_key: str` keyword
    argument, and the wrapped instance must expose `self._access: AccessContext`.
    On success, the wrapped method additionally receives `principal:
    Prosopon` as a keyword argument.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            access: AccessContext | None = getattr(self, "_access", None)
            if access is None:
                raise ValidationError(
                    f"{type(self).__name__} has no AccessContext; cannot enforce @limen"
                )

            api_key = kwargs.pop("api_key", None)
            if not api_key:
                raise ValidationError("api_key is required")

            principal: Prosopon = access.validator.validate(api_key)
            access.nomos.authorize(principal, praxis)
            access.charge(principal, praxis)

            access.aisthesis.on_call_start(praxis=praxis.value, principal_id=principal.principal_id)
            start = time.perf_counter()
            try:
                result = func(self, *args, principal=principal, **kwargs)
            except Sphalma as exc:
                access.aisthesis.on_error(
                    praxis=praxis.value, principal_id=principal.principal_id, error=exc
                )
                raise
            except Exception as exc:  # pragma: no cover - defensive
                access.aisthesis.on_error(
                    praxis=praxis.value, principal_id=principal.principal_id, error=exc
                )
                raise
            else:
                latency_ms = (time.perf_counter() - start) * 1000
                backend = getattr(result, "backend", "unknown")
                access.aisthesis.on_call_end(
                    praxis=praxis.value,
                    principal_id=principal.principal_id,
                    backend=backend,
                    latency_ms=latency_ms,
                    success=True,
                )
                return result

        return wrapper  # type: ignore[return-value]

    return decorator
