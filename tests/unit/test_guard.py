from __future__ import annotations

import pytest

from horasis.access.guard import AccessContext, requires_access
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.tiers import Tier
from horasis.foundation.config import Config, QuotaRule, TierRule
from horasis.foundation.errors import (
    AccessDeniedError,
    InvalidApiKeyError,
    QuotaExceededError,
    ValidationError,
)
from horasis.foundation.schemas import Task


class Widget:
    """Minimal object guarded by `@requires_access`, without orchestration."""

    def __init__(self, access: AccessContext) -> None:
        self._access = access
        self.calls_with_principal: list[str] = []

    @requires_access(Task.DETECT)
    def do_detect(self, *, principal=None):
        self.calls_with_principal.append(principal.principal_id)
        return "ok"


@pytest.fixture
def guard_config() -> Config:
    return Config(
        tiers={
            "free": TierRule(
                name="free",
                quota=QuotaRule(limit=1, window_seconds=3600),
                allowed_tasks=["detect"],
            ),
            "pro": TierRule(
                name="pro",
                quota=QuotaRule(limit=1000, window_seconds=3600),
                allowed_tasks=["classify"],  # deliberately does NOT allow detect
            ),
        }
    )


@pytest.fixture
def guard_validator() -> InMemoryApiKeyValidator:
    v = InMemoryApiKeyValidator()
    v.register_simple("free-key", principal_id="alice", tier=Tier.FREE)
    v.register_simple("pro-key", principal_id="bob", tier=Tier.PRO)
    return v


def test_guard_allows_authorized_call(guard_config: Config, guard_validator: InMemoryApiKeyValidator) -> None:
    widget = Widget(AccessContext.build(config=guard_config, validator=guard_validator))
    assert widget.do_detect(api_key="free-key") == "ok"
    assert widget.calls_with_principal == ["alice"]


def test_guard_rejects_missing_api_key(guard_config: Config, guard_validator: InMemoryApiKeyValidator) -> None:
    widget = Widget(AccessContext.build(config=guard_config, validator=guard_validator))
    with pytest.raises(ValidationError):
        widget.do_detect()


def test_guard_rejects_unknown_api_key(guard_config: Config, guard_validator: InMemoryApiKeyValidator) -> None:
    widget = Widget(AccessContext.build(config=guard_config, validator=guard_validator))
    with pytest.raises(InvalidApiKeyError):
        widget.do_detect(api_key="not-a-real-key")


def test_guard_rejects_task_not_allowed_for_tier(
    guard_config: Config, guard_validator: InMemoryApiKeyValidator
) -> None:
    widget = Widget(AccessContext.build(config=guard_config, validator=guard_validator))
    # bob is "pro" tier, but the pro tier here is not allowed to `detect`
    with pytest.raises(AccessDeniedError):
        widget.do_detect(api_key="pro-key")


def test_guard_enforces_quota(guard_config: Config, guard_validator: InMemoryApiKeyValidator) -> None:
    widget = Widget(AccessContext.build(config=guard_config, validator=guard_validator))
    widget.do_detect(api_key="free-key")  # consumes alice's only unit of quota
    with pytest.raises(QuotaExceededError):
        widget.do_detect(api_key="free-key")


def test_guard_requires_access_context() -> None:
    class NoAccess:
        @requires_access(Task.DETECT)
        def do_detect(self, *, principal=None):
            return "ok"

    with pytest.raises(ValidationError):
        NoAccess().do_detect(api_key="whatever")
