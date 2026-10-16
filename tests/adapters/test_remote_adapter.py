from __future__ import annotations

from typing import Any

import pytest

httpx = pytest.importorskip("httpx")

from horasis.backend.remote_adapter import RemoteAdapter, map_error_response
from horasis.foundation.schemas import DetectionResult, Physis, Praxis
from horasis.foundation.sphalma import (
    AccessDeniedError,
    BackendUnavailableError,
    ExpiredApiKeyError,
    InvalidApiKeyError,
    QuotaExceededError,
    RemoteError,
    RevokedApiKeyError,
    ValidationError,
)
from tests.conftest import png_bytes

IMAGE = Physis.from_bytes(png_bytes())
OK_BODY = {
    "praxis": "detect",
    "backend": "server-side",
    "latency_ms": 1.0,
    "detections": [
        {
            "label": "cat",
            "confidence": {"value": 0.9},
            "box": {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4},
        }
    ],
}


def _adapter(handler: Any, **kwargs: Any) -> tuple[RemoteAdapter, list[float]]:
    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = RemoteAdapter(
        endpoint_url="https://vision.example.com/",
        api_key="secret",
        client=client,
        sleep=sleeps.append,
        **kwargs,
    )
    return adapter, sleeps


def test_success_sends_expected_request_and_parses_result() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK_BODY)

    adapter, _ = _adapter(handler)
    result = adapter.infer(Praxis.DETECT, IMAGE, conf_threshold=0.3, weird=object())
    assert isinstance(result, DetectionResult)
    assert result.backend == "remote" and result.detections[0].label == "cat"
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "https://vision.example.com/v1/detect"
    assert request.headers["x-api-key"] == "secret"
    body = request.read()
    assert b'name="image"' in body and b"image/jpeg" in body
    assert b"conf_threshold" in body and b"weird" not in body  # non-JSON options dropped


def test_retries_503_with_backoff_then_succeeds() -> None:
    attempts = iter([503, 502, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(attempts)
        return httpx.Response(status, json=OK_BODY if status == 200 else {})

    adapter, sleeps = _adapter(handler, max_retries=2, backoff_base=1.0)
    assert adapter.infer(Praxis.DETECT, IMAGE).backend == "remote"
    assert len(sleeps) == 2
    assert 0 <= sleeps[0] <= 1.0 and 0 <= sleeps[1] <= 2.0  # full-jitter, doubling cap


def test_gives_up_after_max_retries_with_remote_error() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": {"type": "overloaded", "message": "busy"}})

    adapter, sleeps = _adapter(handler, max_retries=2)
    with pytest.raises(RemoteError, match="busy") as info:
        adapter.infer(Praxis.DETECT, IMAGE)
    assert len(calls) == 3 and len(sleeps) == 2
    assert info.value.status_code == 503 and info.value.retryable


def test_timeouts_and_connection_errors_are_retried() -> None:
    attempts: list[Exception | None] = [httpx.ReadTimeout("slow"), httpx.ConnectError("down"), None]

    def handler(request: httpx.Request) -> httpx.Response:
        err = attempts.pop(0)
        if err:
            raise err
        return httpx.Response(200, json=OK_BODY)

    adapter, sleeps = _adapter(handler, max_retries=2)
    assert adapter.infer(Praxis.DETECT, IMAGE).praxis is Praxis.DETECT
    assert len(sleeps) == 2


def test_persistent_timeout_becomes_retryable_remote_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("nope")

    adapter, _ = _adapter(handler, max_retries=1)
    with pytest.raises(RemoteError) as info:
        adapter.infer(Praxis.DETECT, IMAGE)
    assert info.value.status_code is None and info.value.retryable


@pytest.mark.parametrize(
    ("status", "error_type", "expected"),
    [
        (401, "invalid_api_key", InvalidApiKeyError),
        (401, "expired_api_key", ExpiredApiKeyError),
        (401, "revoked_api_key", RevokedApiKeyError),
        (403, "forbidden", AccessDeniedError),
        (422, "invalid_image", ValidationError),
        (413, "too_large", ValidationError),
        (404, "no_backend", BackendUnavailableError),
        (500, "internal", RemoteError),
    ],
)
def test_error_status_mapping_does_not_retry(status: int, error_type: str, expected: type) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, json={"error": {"type": error_type, "message": "nope"}})

    adapter, sleeps = _adapter(handler, max_retries=3)
    with pytest.raises(expected, match="nope"):
        adapter.infer(Praxis.DETECT, IMAGE)
    assert len(calls) == 1 and sleeps == []


def test_429_quota_is_mapped_and_not_retried_without_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"type": "quota_exceeded", "message": "daily quota", "limit": 100,
                            "used": 100, "window_seconds": 86400}},
        )

    adapter, sleeps = _adapter(handler)
    with pytest.raises(QuotaExceededError) as info:
        adapter.infer(Praxis.DETECT, IMAGE)
    assert (info.value.limit, info.value.used, info.value.window_seconds) == (100, 100, 86400)
    assert sleeps == []


def test_429_with_short_retry_after_is_retried_and_honoured() -> None:
    responses = iter([httpx.Response(429, headers={"Retry-After": "2"}, json={}), None])

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses) or httpx.Response(200, json=OK_BODY)

    adapter, sleeps = _adapter(handler)
    adapter.infer(Praxis.DETECT, IMAGE)
    assert sleeps == [2.0]


def test_429_with_huge_retry_after_fails_fast_with_hint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "3600"}, json={})

    adapter, sleeps = _adapter(handler)
    with pytest.raises(QuotaExceededError) as info:
        adapter.infer(Praxis.DETECT, IMAGE)
    assert info.value.context["retry_after"] == 3600.0 and sleeps == []


def test_bad_success_bodies_are_remote_errors() -> None:
    adapter, _ = _adapter(lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(RemoteError, match="non-object"):
        adapter.infer(Praxis.DETECT, IMAGE)
    adapter, _ = _adapter(lambda r: httpx.Response(200, json={"detections": "not-a-list"}))
    with pytest.raises(RemoteError, match="invalid detect result"):
        adapter.infer(Praxis.DETECT, IMAGE)


def test_unsupported_praxis_and_defaults() -> None:
    adapter, _ = _adapter(lambda r: httpx.Response(200), praxeis=[Praxis.OCR])
    assert adapter.supports(Praxis.OCR) and not adapter.supports(Praxis.DETECT)
    with pytest.raises(BackendUnavailableError):
        adapter.infer(Praxis.DETECT, IMAGE)
    assert RemoteAdapter(endpoint_url="http://x").supported_praxeis == frozenset(Praxis)


def test_map_error_response_handles_unparseable_body() -> None:
    err = map_error_response(500, None, None)
    assert isinstance(err, RemoteError) and "500" in err.message
