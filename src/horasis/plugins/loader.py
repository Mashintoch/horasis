from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points

from horasis.backend.base import Backend
from horasis.backend.registry import BackendRegistry
from horasis.foundation.errors import PluginError

ENTRY_POINT_GROUP = "horasis.backends"


class PluginLoader:
    """Discovers and registers third-party `Backend` backends."""

    def __init__(self, group: str = ENTRY_POINT_GROUP) -> None:
        self._group = group

    def discover_entry_points(self) -> list[EntryPoint]:
        return list(entry_points(group=self._group))

    def load(self, entry_point: EntryPoint, **construction_kwargs: object) -> Backend:
        try:
            factory = entry_point.load()
        except Exception as exc:
            raise PluginError(
                f"Failed to load plugin entry point {entry_point.name!r}: {exc}"
            ) from exc

        try:
            instance = factory(**construction_kwargs)
        except Exception as exc:
            raise PluginError(
                f"Failed to construct backend from plugin {entry_point.name!r}: {exc}"
            ) from exc

        if not isinstance(instance, Backend):
            raise PluginError(
                f"Plugin {entry_point.name!r} did not produce a Backend instance "
                f"(got {type(instance).__name__})"
            )
        return instance

    def discover(self, registry: BackendRegistry, *, strict: bool = False) -> list[str]:
        """Discover and register all plugins in the entry-point group onto `registry`.

        Args:
            registry: Registry to register discovered backends onto.
            strict: If True, a single failing plugin raises `PluginError`
                immediately. If False (default), failing plugins are
                skipped and discovery continues.

        Returns:
            Names of the backends successfully registered.
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
