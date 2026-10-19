"""The access-control decorator.

Every public-API call passes through :func:`requires_access` before it
reaches orchestration. The API key is resolved to a
:class:`~horasis.access.principal.Principal`,
:class:`~horasis.access.policy.Policy` authorizes the task, rate limits and
quota are charged, and :class:`~horasis.foundation.telemetry.Telemetry`
hooks fire around the call.

It decorates methods of any object that keeps an :class:`AccessContext` in
``_access`` and takes ``api_key`` as a keyword argument.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from horasis.access.keys import ApiKeyValidator
from horasis.access.policy import Policy
from horasis.access.principal import Principal
from horasis.access.quota import QuotaManager, UsageReport
from horasis.access.ratelimit import RateLimiter, TokenBucketLimiter
from horasis.foundation.config import Config, QuotaRule
from horasis.foundation.errors import HorasisError, QuotaExceededError, ValidationError
from horasis.foundation.imaging import to_image
from horasis.foundation.schemas import Task
from horasis.foundation.telemetry import NullTelemetry, Telemetry

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class AccessContext:
    """Everything `requires_access` needs to guard a call."""

    config: Config
    validator: ApiKeyValidator
    policy: Policy
    quota_manager: QuotaManager
    rate_limiter: RateLimiter
    telemetry: Telemetry = field(default_factory=NullTelemetry)

    @classmethod
    def build(
        cls,
        *,
        config: Config,
        validator: ApiKeyValidator,
        quota_manager: QuotaManager | None = None,
        rate_limiter: RateLimiter | None = None,
        telemetry: Telemetry | None = None,
    ) -> AccessContext:
        return cls(
            config=config,
            validator=validator,
            policy=Policy(config),
            quota_manager=quota_manager or QuotaManager(),
            rate_limiter=rate_limiter or TokenBucketLimiter(),
            telemetry=telemetry or NullTelemetry(),
        )


    def charge(self, principal: Principal, task: Task, *, units: int = 1) -> None:
        """Rate-limit and meter `units` calls of `task` for `principal`.

        Each unit costs ``Config.cost_for(task)``. All-or-nothing: if any
        limit rejects, everything taken so far is given back.

        Raises:
            RateLimitedError: per-minute bucket empty (has ``retry_after``).
            QuotaExceededError: window quota exhausted (has ``retry_after``,
                ``limit`` and ``scope``).
        """
        rule = self.config.tier(principal.tier.value)
        pid, name = principal.principal_id, task.value
        cost = self.config.cost_for(name) * units
        self.rate_limiter.acquire(pid, name, rule, cost)
        try:
            self.quota_manager.check_and_consume(pid, rule.quota, cost=cost)
            task_rule = self._task_rule(rule.task_quotas.get(name), rule.quota)
            try:
                self.quota_manager.check_and_consume(pid, task_rule, cost=cost, scope=f"task:{name}")
            except QuotaExceededError:
                self.quota_manager.refund(pid, rule.quota, cost=cost)
                raise
        except QuotaExceededError:
            self.rate_limiter.refund(pid, name, rule, cost)
            raise

    @staticmethod
    def _task_rule(specific: QuotaRule | None, overall: QuotaRule) -> QuotaRule:
        return specific or overall

    def usage(self, principal: Principal) -> UsageReport:
        """Return used and remaining units for `principal`, overall and per task."""
        rule = self.config.tier(principal.tier.value)
        pid = principal.principal_id
        per_task = {
            name: self.quota_manager.window_usage(
                pid, self._task_rule(rule.task_quotas.get(name), rule.quota), scope=f"task:{name}"
            )
            for name in rule.allowed_tasks
            if principal.scopes is None or name in principal.scopes
        }
        return UsageReport(
            principal_id=pid,
            tier=principal.tier.value,
            overall=self.quota_manager.window_usage(pid, rule.quota),
            per_task=per_task,
        )


def _validated_inputs(
    access: AccessContext, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Decode and validate the image argument (first positional, or ``image=``).

    The decoded `Image` replaces the raw input, so the wrapped method does not
    decode twice and malformed input is rejected before anything is charged.
    """
    policy = access.config.inputs
    if "image" in kwargs:
        return args, {**kwargs, "image": to_image(kwargs["image"], policy)}
    if args:
        return (to_image(args[0], policy), *args[1:]), kwargs
    return args, kwargs


def requires_access(task: Task, *, validate_image: bool = True) -> Callable[[F], F]:
    """Authenticate, authorize and charge quota before running the method.

    The method must be called with an ``api_key`` keyword and its instance
    must keep an :class:`AccessContext` in ``_access``. It receives the
    resolved ``principal`` as a keyword argument.

    Args:
        task: Task the method performs, for authorization and telemetry.
        validate_image: Decode and validate the image argument before
            charging, so invalid input costs nothing. Set False for video.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            access: AccessContext | None = getattr(self, "_access", None)
            if access is None:
                raise ValidationError(
                    f"{type(self).__name__} has no AccessContext; cannot enforce @requires_access"
                )

            api_key = kwargs.pop("api_key", None)
            if not api_key:
                raise ValidationError("api_key is required")

            principal: Principal = access.validator.validate(api_key)
            access.policy.authorize(principal, task)
            if validate_image:
                args, kwargs = _validated_inputs(access, args, kwargs)
            access.charge(principal, task)

            access.telemetry.on_call_start(task=task.value, principal_id=principal.principal_id)
            start = time.perf_counter()
            try:
                result = func(self, *args, principal=principal, **kwargs)
            except HorasisError as exc:
                access.telemetry.on_error(
                    task=task.value, principal_id=principal.principal_id, error=exc
                )
                raise
            except Exception as exc:  # pragma: no cover - defensive
                access.telemetry.on_error(
                    task=task.value, principal_id=principal.principal_id, error=exc
                )
                raise
            latency_ms = (time.perf_counter() - start) * 1000
            backend = getattr(result, "backend", "unknown")
            access.telemetry.on_call_end(
                task=task.value,
                principal_id=principal.principal_id,
                backend=backend,
                latency_ms=latency_ms,
                success=True,
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
