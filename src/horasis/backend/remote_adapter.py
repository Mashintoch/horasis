"""Backend that calls a remote horasis server over HTTP (extra: ``horasis[remote]``).

Wire protocol (served by ``horasis.server``):

* ``POST {endpoint}/v1/{task}`` -- multipart form with an ``image`` file and
  an optional JSON ``options`` field; credential in the ``X-API-Key`` header.
* ``200`` -> JSON body in the shape of the matching ``Result`` result.
* Errors -> ``{"error": {"type": "...", "message": "..."}}`` with a status code.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterable
from typing import Any

from horasis.backend.base import Backend
from horasis.foundation.errors import (
    AccessDeniedError,
    BackendUnavailableError,
    ExpiredApiKeyError,
    HorasisError,
    InvalidApiKeyError,
    QuotaExceededError,
    RateLimitedError,
    RemoteError,
    RevokedApiKeyError,
    ValidationError,
)
from horasis.foundation.imaging import encode_image
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    FaceResult,
    Image,
    ModerationResult,
    OcrResult,
    PoseResult,
    Result,
    SegmentationResult,
    Task,
)

_RESULT_TYPES: dict[Task, type[Result]] = {
    Task.DETECT: DetectionResult,
    Task.CLASSIFY: ClassificationResult,
    Task.OCR: OcrResult,
    Task.FACE: FaceResult,
    Task.MODERATE: ModerationResult,
    Task.SEGMENT: SegmentationResult,
    Task.POSE: PoseResult,
}
_RETRYABLE_STATUS = frozenset({502, 503, 504})


def map_error_response(status: int, body: Any, headers: Any = None) -> HorasisError:
    """Translate a failed HTTP response into the matching `HorasisError` exception."""
    error = body.get("error") if isinstance(body, dict) else None
    error = error if isinstance(error, dict) else {}
    kind = str(error.get("type", ""))
    message = str(error.get("message") or f"Remote server returned HTTP {status}")
    ctx: dict[str, object] = {"status_code": status, "error_type": kind}

    if status == 401 or kind in {"invalid_api_key", "expired_api_key", "revoked_api_key"}:
        if kind == "expired_api_key":
            return ExpiredApiKeyError(message, context=ctx)
        if kind == "revoked_api_key":
            return RevokedApiKeyError(message, context=ctx)
        return InvalidApiKeyError(message, context=ctx)
    if status == 403:
        return AccessDeniedError(message, context=ctx)
    if status == 429:
        retry_after = _retry_after(headers)
        if retry_after is None and isinstance(error.get("retry_after"), int | float):
            retry_after = float(error["retry_after"])
        if retry_after is not None:
            ctx["retry_after"] = retry_after
        if kind == "rate_limited":
            return RateLimitedError(
                message,
                limit=int(error.get("limit", 0)),
                retry_after=retry_after,
                scope=str(error.get("scope", "rate")),
                context=ctx,
            )
        return QuotaExceededError(
            message,
            limit=int(error.get("limit", 0)),
            used=int(error.get("used", 0)),
            window_seconds=int(error.get("window_seconds", 0)),
            retry_after=retry_after,
            scope=str(error.get("scope", "quota")),
            context=ctx,
        )
    if status in (400, 413, 415, 422):
        return ValidationError(message, context=ctx)
    if status in (404, 501):
        return BackendUnavailableError(message, context=ctx)
    return RemoteError(message, status_code=status, retryable=status in _RETRYABLE_STATUS, context=ctx)


def _retry_after(headers: Any) -> float | None:
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except (TypeError, ValueError):
        return None


class RemoteAdapter(Backend):
    """`Backend` that delegates inference to a remote horasis server.

    Transient failures (timeouts, connection errors, HTTP 502/503/504, and
    429 with a short ``Retry-After``) are retried with exponential backoff
    and jitter. Everything else fails fast, mapped onto `HorasisError` types:
    401 -> ``InvalidApiKeyError`` (or Expired/Revoked), 403 ->
    ``AccessDeniedError``, 429 -> ``QuotaExceededError``, 400/413/415/422 ->
    ``ValidationError``, 404/501 -> ``BackendUnavailableError``, and
    anything else -> ``RemoteError``.

    Args:
        endpoint_url: Base URL of the remote server.
        api_key: Credential for the remote server (not a local horasis key).
        timeout: Total per-request timeout, seconds.
        max_retries: Extra attempts after the first.
        backoff_base: First backoff delay in seconds (doubles each retry).
        backoff_max: Upper bound for any single delay, and for honouring
            ``Retry-After``.
        tasks: Tasks to expose; defaults to all of them.
        image_format: ``"JPEG"`` (smaller) or ``"PNG"`` (lossless) upload encoding.
        client: A pre-built ``httpx.Client`` (e.g. with a mock transport).
        sleep: Delay function; injectable so tests need not wait.
        name: Registry name; defaults to ``"remote"``.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        backoff_max: float = 8.0,
        tasks: Iterable[Task] | None = None,
        image_format: str = "JPEG",
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        name: str = "remote",
    ) -> None:
        self.name = name
        self.supported_tasks = frozenset(tasks) if tasks is not None else frozenset(Task)
        self._endpoint_url = endpoint_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._image_format = image_format
        self._client: Any = client
        self._sleep = sleep

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'httpx' package is required for RemoteAdapter. "
                "Install it with: pip install 'horasis[remote]'"
            ) from exc
        self._client = httpx.Client(
            timeout=httpx.Timeout(self._timeout, connect=min(10.0, self._timeout))
        )
        return self._client

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self._backoff_max)
        return random.uniform(0, min(self._backoff_max, self._backoff_base * 2**attempt))

    def infer(self, task: Task, image: Image, **options: Any) -> Result:
        if not self.supports(task):
            raise BackendUnavailableError(f"RemoteAdapter does not support task {task.value!r}")
        client = self._load_client()
        upload = encode_image(image, fmt=self._image_format)
        mime = "image/png" if self._image_format.upper() == "PNG" else "image/jpeg"
        url = f"{self._endpoint_url}/v1/{task.value}"
        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        form = {"options": json.dumps(_json_safe(options))}

        start = time.perf_counter()
        last_error: HorasisError | None = None
        for attempt in range(self._max_retries + 1):
            retry_after: float | None = None
            try:
                response = client.post(
                    url,
                    headers=headers,
                    data=form,
                    files={"image": ("image", upload, mime)},
                )
            except Exception as exc:
                if not _is_transport_error(exc):
                    raise RemoteError(f"Remote request failed: {exc}") from exc
                last_error = RemoteError(
                    f"Remote request to {url} failed: {type(exc).__name__}: {exc}", retryable=True
                )
            else:
                if response.status_code == 200:
                    return self._parse(task, response, (time.perf_counter() - start) * 1000)
                body = _safe_json(response)
                last_error = map_error_response(response.status_code, body, response.headers)
                retry_after = _retry_after(response.headers)
                retryable = response.status_code in _RETRYABLE_STATUS or (
                    response.status_code == 429
                    and retry_after is not None
                    and retry_after <= self._backoff_max
                )
                if not retryable:
                    raise last_error
            if attempt < self._max_retries:
                self._sleep(self._delay(attempt, retry_after))
        assert last_error is not None
        raise last_error

    def _parse(self, task: Task, response: Any, latency_ms: float) -> Result:
        payload = _safe_json(response)
        if not isinstance(payload, dict):
            raise RemoteError("Remote server returned a non-object JSON body", status_code=200)
        try:
            result = _RESULT_TYPES[task].model_validate(
                {**payload, "task": task.value, "backend": self.name, "latency_ms": latency_ms}
            )
        except Exception as exc:
            raise RemoteError(
                f"Remote server returned an invalid {task.value} result: {exc}", status_code=200
            ) from exc
        return result


def _is_transport_error(exc: Exception) -> bool:
    try:
        import httpx
    except ImportError:  # pragma: no cover - client exists only if httpx does
        return False
    return isinstance(exc, httpx.TransportError)


def _safe_json(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _json_safe(options: dict[str, Any]) -> dict[str, Any]:
    """Keep only options that survive JSON encoding."""
    kept: dict[str, Any] = {}
    for key, value in options.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        kept[key] = value
    return kept
