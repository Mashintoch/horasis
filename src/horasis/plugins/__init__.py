"""Plugin mechanism -- cross-cutting, sits alongside the backend layer."""

from horasis.plugins.prosthesis import ENTRY_POINT_GROUP, Prosthesis

__all__ = ["ENTRY_POINT_GROUP", "Prosthesis"]
