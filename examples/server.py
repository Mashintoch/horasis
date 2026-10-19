"""Serve horasis over HTTP.

    pip install 'horasis[server]'
    HORASIS_API_KEY=change-me python examples/server.py
    curl -H "X-API-Key: change-me" -F image=@photo.jpg http://127.0.0.1:8000/v1/detect

(`horasis serve` does the same from the command line.)
"""

from __future__ import annotations

import os

from horasis import InMemoryApiKeyValidator, Tier, Client


def build_app(api_key):
    from horasis.server import create_app

    validator = InMemoryApiKeyValidator()
    validator.register_simple(api_key, principal_id="http-client", tier=Tier.PRO)
    return create_app(Client.build(validator=validator))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(os.environ["HORASIS_API_KEY"]), host="127.0.0.1", port=8000)
