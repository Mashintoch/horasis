"""Entry-point plugin loading; sits alongside the backend layer."""

from horasis.plugins.loader import ENTRY_POINT_GROUP, PluginLoader

__all__ = ["ENTRY_POINT_GROUP", "PluginLoader"]
