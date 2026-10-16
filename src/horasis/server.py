"""HTTP server exposing a :class:`Skopos` client (extra: ``horasis[server]``).

This is what :class:`~horasis.backend.remote_adapter.RemoteAdapter` talks to,
but any HTTP client works::

    curl -H "X-API-Key: $KEY" -F image=@photo.jpg -F options='{"conf": 0.4}' \\
         http://localhost:8000/v1/detect

Endpoints:

* ``GET  /health`` -- liveness and the praxeis each backend serves (no auth).
* ``GET  /usage``  -- used/remaining quota for the caller (``X-API-Key``).
* ``POST /v1/{praxis}`` -- run one of ``detect classify ocr face moderate
  segment pose`` on an uploaded ``image`` (multipart). Optional form fields:
  ``options`` (JSON object, whitelisted per praxis) and ``backend``.

Errors are ``{"error": {"type": ..., "message": ...}}`` with the status codes
documented on :func:`create_app`. Build the app with your own ``Skopos`` so you
control the API-key validator, quota store, rate limits and backends.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from horasis import __version__
from horasis.api.skopos import Skopos
from horasis.foundation.schemas import Praxis, Theoria
from horasis.foundation.sphalma import (
    AccessDeniedError,
    AdapterError,
    BackendUnavailableError,
    ExpiredApiKeyError,
    InvalidApiKeyError,
    QuotaExceededError,
    RateLimitedError,
    RevokedApiKeyError,
    Sphalma,
    ValidationError,
)

try:
    from fastapi import FastAPI, File, Form, Header, Request, UploadFile
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "horasis.server needs FastAPI. Install it with: pip install 'horasis[server]'"
    ) from exc


class UploadTooLargeError(Sphalma):
    """The uploaded image exceeded ``Kanon.inputs.max_bytes``."""


_Num = (int, float)

#: Options a caller may pass per praxis, with their types. Anything else is
#: rejected: options are forwarded to model code, so they must be whitelisted.
ALLOWED_OPTIONS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "detect": {
        "conf": _Num,
        "iou": _Num,
        "max_det": int,
        "conf_threshold": _Num,
        "iou_threshold": _Num,
        "estimate_demographics": bool,
    },
    "classify": {"top_k": int},
    "ocr": {"min_confidence": _Num},
    "face": {"landmarks": bool},
    "moderate": {},
    "segment": {"conf": _Num, "iou": _Num, "max_det": int},
    "pose": {"conf": _Num, "iou": _Num, "max_det": int},
}


def parse_options(praxis: str, raw: str | None) -> dict[str, Any]:
    """Parse and validate the JSON ``options`` form field for `praxis`."""
    if not raw:
        return {}
    try:
        options = json.loads(raw)
    except ValueError as exc:
        raise ValidationError(f"options is not valid JSON: {exc}") from exc
    if not isinstance(options, dict):
        raise ValidationError("options must be a JSON object")
    allowed = ALLOWED_OPTIONS[praxis]
    for key, value in options.items():
        if key not in allowed:
            raise ValidationError(
                f"Option {key!r} is not allowed for {praxis!r}; allowed: {sorted(allowed)}"
            )
        wants_bool = allowed[key] is bool
        if isinstance(value, bool) != wants_bool or not isinstance(value, allowed[key]):
            raise ValidationError(f"Option {key!r} has the wrong type")
    return options


def _error(status: int, kind: str, message: str, **extra: Any) -> JSONResponse:
    headers = {}
    retry_after = extra.get("retry_after")
    if retry_after is not None:
        headers["Retry-After"] = str(max(1, math.ceil(retry_after)))
    return JSONResponse(
        {"error": {"type": kind, "message": message, **extra}}, status_code=status, headers=headers
    )


def error_response(exc: Exception) -> JSONResponse:
    """Map an exception to the wire error format (also used by the tests)."""
    message = exc.message if isinstance(exc, Sphalma) else "Internal server error"
    if isinstance(exc, ExpiredApiKeyError):
        return _error(401, "expired_api_key", message)
    if isinstance(exc, RevokedApiKeyError):
        return _error(401, "revoked_api_key", message)
    if isinstance(exc, InvalidApiKeyError):
        return _error(401, "invalid_api_key", message)
    if isinstance(exc, QuotaExceededError):
        return _error(
            429,
            "quota_exceeded",
            message,
            limit=exc.limit,
            used=exc.used,
            window_seconds=exc.window_seconds,
            scope=exc.scope,
            retry_after=exc.retry_after,
        )
    if isinstance(exc, RateLimitedError):
        return _error(
            429,
            "rate_limited",
            message,
            limit=exc.limit,
            scope=exc.scope,
            retry_after=exc.retry_after,
        )
    if isinstance(exc, AccessDeniedError):
        return _error(403, "forbidden", message)
    if isinstance(exc, UploadTooLargeError):
        return _error(413, "too_large", message)
    if isinstance(exc, ValidationError):
        return _error(422, "invalid_request", message)
    if isinstance(exc, BackendUnavailableError):
        return _error(501, "backend_unavailable", message)
    if isinstance(exc, AdapterError):
        return _error(500, "adapter_error", message)
    return _error(500, "internal_error", message)


def _dispatch(client: Skopos, praxis: str) -> Callable[..., Theoria]:
    table: dict[str, Callable[..., Theoria]] = {
        "detect": client.detection.detect,
        "classify": client.classification.classify,
        "ocr": client.ocr.extract_text,
        "face": client.face.detect_faces,
        "moderate": client.moderation.moderate,
        "segment": client.segmentation.segment,
        "pose": client.pose.estimate,
    }
    return table[praxis]


def create_app(client: Skopos) -> FastAPI:
    """Build the FastAPI app serving `client`.

    Status codes: 401 bad/expired/revoked key, 403 not allowed (tier/scope),
    404 unknown praxis, 413 upload too large, 422 invalid input or options,
    429 quota or rate limit (with ``Retry-After``), 501 no backend for the
    praxis, 500 backend failure.
    """
    app = FastAPI(
        title="horasis", version=__version__, docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.exception_handler(Sphalma)
    async def _handle_sphalma(_: Request, exc: Sphalma) -> JSONResponse:
        return error_response(exc)

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        return error_response(exc)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "backends": {
                name: sorted(p.value for p in client.taxis.get(name).supported_praxeis)
                for name in client.taxis.names()
            },
        }

    @app.get("/usage")
    def usage(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        if not x_api_key:
            raise InvalidApiKeyError("Missing X-API-Key header")
        return client.usage(x_api_key).model_dump(mode="json")

    @app.post("/v1/{praxis}")
    def infer(
        praxis: str,
        image: UploadFile = File(...),  # noqa: B008
        options: str | None = Form(default=None),
        backend: str | None = Form(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if praxis not in {p.value for p in Praxis}:
            return _error(404, "unknown_praxis", f"Unknown praxis {praxis!r}")
        if not x_api_key:
            raise InvalidApiKeyError("Missing X-API-Key header")
        parsed = parse_options(praxis, options)
        limit = client.kanon.inputs.max_bytes
        data = image.file.read(limit + 1)
        if len(data) > limit:
            raise UploadTooLargeError(f"Upload exceeds the {limit}-byte limit")
        call = _dispatch(client, praxis)
        kwargs: dict[str, Any] = {"api_key": x_api_key, "backend": backend}
        if praxis == "detect":
            kwargs["estimate_demographics"] = bool(parsed.pop("estimate_demographics", False))
        if parsed:
            kwargs["options"] = parsed
        return call(data, **kwargs).to_dict()

    return app
