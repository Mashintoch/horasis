from __future__ import annotations

import pytest

from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.klimax import Klimax
from horasis.access.limen import AccessContext, limen
from horasis.foundation.kanon import Kanon, QuotaRule, TierRule
from horasis.foundation.schemas import Praxis
from horasis.foundation.sphalma import (
    AccessDeniedError,
    InvalidApiKeyError,
    QuotaExceededError,
    ValidationError,
)


class Widget:
    """A minimal object exercising `@limen` without any orchestration."""

    def __init__(self, access: AccessContext) -> None:
        self._access = access
        self.calls_with_principal: list[str] = []

    @limen(Praxis.DETECT)
    def do_detect(self, *, principal=None):
        self.calls_with_principal.append(principal.principal_id)
        return "ok"


@pytest.fixture
def limen_kanon() -> Kanon:
    return Kanon(
        tiers={
            "free": TierRule(
                name="free",
                quota=QuotaRule(limit=1, window_seconds=3600),
                allowed_praxeis=["detect"],
            ),
            "pro": TierRule(
                name="pro",
                quota=QuotaRule(limit=1000, window_seconds=3600),
                allowed_praxeis=["classify"],  # deliberately does NOT allow detect
            ),
        }
    )


@pytest.fixture
def limen_validator() -> InMemoryApiKeyValidator:
    v = InMemoryApiKeyValidator()
    v.register_simple("free-key", principal_id="alice", tier=Klimax.FREE)
    v.register_simple("pro-key", principal_id="bob", tier=Klimax.PRO)
    return v


def test_limen_allows_authorized_call(limen_kanon: Kanon, limen_validator: InMemoryApiKeyValidator) -> None:
    widget = Widget(AccessContext.build(kanon=limen_kanon, validator=limen_validator))
    assert widget.do_detect(api_key="free-key") == "ok"
    assert widget.calls_with_principal == ["alice"]


def test_limen_rejects_missing_api_key(limen_kanon: Kanon, limen_validator: InMemoryApiKeyValidator) -> None:
    widget = Widget(AccessContext.build(kanon=limen_kanon, validator=limen_validator))
    with pytest.raises(ValidationError):
        widget.do_detect()


def test_limen_rejects_unknown_api_key(limen_kanon: Kanon, limen_validator: InMemoryApiKeyValidator) -> None:
    widget = Widget(AccessContext.build(kanon=limen_kanon, validator=limen_validator))
    with pytest.raises(InvalidApiKeyError):
        widget.do_detect(api_key="not-a-real-key")


def test_limen_rejects_task_not_allowed_for_tier(
    limen_kanon: Kanon, limen_validator: InMemoryApiKeyValidator
) -> None:
    widget = Widget(AccessContext.build(kanon=limen_kanon, validator=limen_validator))
    # bob is "pro" tier, but the pro tier here is not allowed to `detect`
    with pytest.raises(AccessDeniedError):
        widget.do_detect(api_key="pro-key")


def test_limen_enforces_quota(limen_kanon: Kanon, limen_validator: InMemoryApiKeyValidator) -> None:
    widget = Widget(AccessContext.build(kanon=limen_kanon, validator=limen_validator))
    widget.do_detect(api_key="free-key")  # consumes alice's only unit of quota
    with pytest.raises(QuotaExceededError):
        widget.do_detect(api_key="free-key")


def test_limen_requires_access_context() -> None:
    class NoAccess:
        @limen(Praxis.DETECT)
        def do_detect(self, *, principal=None):
            return "ok"

    with pytest.raises(ValidationError):
        NoAccess().do_detect(api_key="whatever")
