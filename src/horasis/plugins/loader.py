from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points

from horasis.backend.base import Backend
from horasis.backend.registry import BackendRegistry
from horasis.foundation.errors import PluginError

ENTRY_POINT_GROUP = "horasis.backends"


class PluginLoader:
    """Discovers and registers third-party backends."""

    def __init__(self, group: str = ENTRY_POINT_GROUP) -> None:
        self._group = group

    def discover_entry_points(self) -> list[EntryPoint]:
        return list(entry_points(group=self._group))

    def load(self, entry_point: EntryPoint, **construction_kwargs: object) -> Backend:
        try:
            factory = entry_point.load()
        except Exception as exc:
            raise PluginError(
                f"cannot load plugin entry point {entry_point.name!r}: {exc}"
            ) from exc

        try:
            instance = factory(**construction_kwargs)
        except Exception as exc:
            raise PluginError(
                f"cannot construct backend from plugin {entry_point.name!r}: {exc}"
            ) from exc

        if not isinstance(instance, Backend):
            raise PluginError(
                f"plugin {entry_point.name!r} did not produce a Backend instance "
                f"(got {type(instance).__name__})"
            )
        return instance

    def discover(self, registry: BackendRegistry, *, strict: bool = False) -> list[str]:
        """Register every plugin in the entry-point group on `registry`.

        Args:
            strict: Raise `PluginError` on the first failing plugin instead
                of skipping it.

        Returns:
            Names of the backends registered.
        """
        registered: list[str] = []
        for ep in self.discover_entry_points():
            try:
                backend = self.load(ep)
            except PluginError:
                if strict:
                    raise
                continue
            registry.register(backend)
            registered.append(backend.name)
        return registered
