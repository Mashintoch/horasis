"""Shared helpers for the examples (not an example itself)."""

from __future__ import annotations

import sys
from typing import Any

from horasis import HashedApiKeyValidator, Klimax, Skopos


def demo_client(**build_kwargs: Any) -> tuple[Skopos, str]:
    """A client with one freshly issued PRO-tier key. Returns ``(client, api_key)``."""
    validator = HashedApiKeyValidator()
    key = validator.issue(principal_id="demo", tier=Klimax.PRO)
    return Skopos.build(validator=validator, **build_kwargs), key


def image_argument() -> str:
    """The image path from the command line, or exit with a usage message."""
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: python {sys.argv[0]} IMAGE")
    return sys.argv[1]
