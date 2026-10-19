from __future__ import annotations

from horasis.access.principal import Principal
from horasis.foundation.config import Config
from horasis.foundation.errors import AccessDeniedError, ConfigurationError
from horasis.foundation.schemas import Task


class Policy:
    """Evaluates whether a principal's tier permits a given task."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def authorize(self, principal: Principal, task: Task) -> None:
        """Raise :class:`AccessDeniedError` if `principal` may not invoke `task`.

        Raises:
            AccessDeniedError: if the principal's tier is unknown or does
                not include `task` in its allow-list.
        """
        try:
            rule = self._config.tier(principal.tier.value)
        except ConfigurationError as exc:
            raise AccessDeniedError(
                f"Tier {principal.tier.value!r} is not configured",
                context={"principal_id": principal.principal_id},
            ) from exc

        if task.value not in rule.allowed_tasks:
            raise AccessDeniedError(
                f"Tier {principal.tier.value!r} may not invoke {task.value!r}",
                context={
                    "principal_id": principal.principal_id,
                    "tier": principal.tier.value,
                    "task": task.value,
                },
            )

        if principal.scopes is not None and task.value not in principal.scopes:
            raise AccessDeniedError(
                f"API key is not scoped for {task.value!r}",
                context={
                    "principal_id": principal.principal_id,
                    "task": task.value,
                    "scopes": list(principal.scopes),
                },
            )
