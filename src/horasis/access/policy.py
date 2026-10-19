from __future__ import annotations

from horasis.access.principal import Principal
from horasis.foundation.config import Config
from horasis.foundation.errors import AccessDeniedError, ConfigurationError
from horasis.foundation.schemas import Task


class Policy:
    """Checks whether a principal's tier and key scopes allow a task."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def authorize(self, principal: Principal, task: Task) -> None:
        """Raise :class:`AccessDeniedError` unless `principal` may call `task`.

        The tier must be configured and allow `task`, and the key's scopes,
        if set, must include it.
        """
        try:
            rule = self._config.tier(principal.tier.value)
        except ConfigurationError as exc:
            raise AccessDeniedError(
                f"tier {principal.tier.value!r} is not configured",
                context={"principal_id": principal.principal_id},
            ) from exc

        if task.value not in rule.allowed_tasks:
            raise AccessDeniedError(
                f"tier {principal.tier.value!r} may not call {task.value!r}",
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
