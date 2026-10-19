"""Shared helpers for the examples (not an example itself)."""

from __future__ import annotations

import sys
from typing import Any

from horasis import HashedApiKeyValidator, Tier, Client


def demo_client(**build_kwargs: Any) -> tuple[Client, str]:
    """Return a client and a freshly issued pro-tier API key."""
    validator = HashedApiKeyValidator()
    key = validator.issue(principal_id="demo", tier=Tier.PRO)
    return Client.build(validator=validator, **build_kwargs), key


def image_argument() -> str:
    """Return the image path from the command line, or exit with usage."""
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: python {sys.argv[0]} IMAGE")
    return sys.argv[1]
