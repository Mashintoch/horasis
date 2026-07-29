from __future__ import annotations

from horasis.access.prosopon import Prosopon
from horasis.foundation.kanon import Kanon
from horasis.foundation.schemas import Praxis
from horasis.foundation.sphalma import AccessDeniedError, ConfigurationError


class Nomos:
    """Evaluates whether a principal's tier permits a given task."""

    def __init__(self, kanon: Kanon) -> None:
        self._kanon = kanon

    def authorize(self, principal: Prosopon, praxis: Praxis) -> None:
        """Raise :class:`AccessDeniedError` if `principal` may not invoke `praxis`.

        Raises:
            AccessDeniedError: if the principal's tier is unknown or does
                not include `praxis` in its allow-list.
        """
        try:
            rule = self._kanon.tier(principal.tier.value)
        except ConfigurationError as exc:
            raise AccessDeniedError(
                f"Tier {principal.tier.value!r} is not configured",
                context={"principal_id": principal.principal_id},
            ) from exc

        if praxis.value not in rule.allowed_praxeis:
            raise AccessDeniedError(
                f"Tier {principal.tier.value!r} may not invoke {praxis.value!r}",
                context={
                    "principal_id": principal.principal_id,
                    "tier": principal.tier.value,
                    "praxis": praxis.value,
                },
            )
