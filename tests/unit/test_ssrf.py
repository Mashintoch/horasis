from __future__ import annotations

import socket
from typing import Any

import pytest

from horasis.foundation import imaging
from horasis.foundation.config import InputPolicy
from horasis.foundation.errors import ValidationError
from tests.conftest import png_bytes

PUBLIC = "93.184.216.34"


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status = status
        self._headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._body = body
        self._pos = 0

    def getheader(self, name: str) -> str | None:
        return self._headers.get(name.lower())

    def read(self, n: int) -> bytes:
        chunk = self._body[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


class FakeConn:
    def __init__(self, response: FakeResponse, log: list[tuple[Any, ...]]) -> None:
        self._response = response
        self._log = log

    def request(self, method: str, target: str, headers: dict[str, str]) -> None:
        self._log.append(("request", method, target))

    def getresponse(self) -> FakeResponse:
        return self._response

    def close(self) -> None:
        pass


def _dns(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, list[str]]) -> list[str]:
    looked_up: list[str] = []

    def fake_getaddrinfo(host: str, port: int, **_: Any) -> list[Any]:
        looked_up.append(host)
        addrs = mapping[host]
        return [
            (socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port))
            for a in addrs
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    return looked_up


def _transport(
    monkeypatch: pytest.MonkeyPatch, responses: list[FakeResponse]
) -> list[tuple[Any, ...]]:
    log: list[tuple[Any, ...]] = []
    queue = list(responses)

    def fake_open(scheme: str, host: str, port: int, addresses: list[str], timeout: float) -> Any:
        log.append(("open", scheme, host, port, tuple(addresses)))
        return FakeConn(queue.pop(0), log)

    monkeypatch.setattr(imaging, "_open_connection", fake_open)
    return log


def _fetch(url: str, policy: InputPolicy | None = None, max_bytes: int = 1_000_000) -> bytes:
    return imaging.fetch_url(url, timeout=1, max_bytes=max_bytes, policy=policy)


def _image(body: bytes | None = None) -> FakeResponse:
    return FakeResponse(200, {"Content-Type": "image/png"}, body or png_bytes())


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.5",
        "192.168.1.1",
        "169.254.169.254",
        "224.0.0.1",
        "240.0.0.1",
        "0.0.0.0",
        "100.64.0.1",
        "::1",
        "::",
        "fc00::1",
        "fd12:3456::1",
        "fe80::1",
        "ff02::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
    ],
)
def test_blocked_ranges(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    _dns(monkeypatch, {"evil.test": [address]})
    log = _transport(monkeypatch, [_image()])
    with pytest.raises(ValidationError, match="non-public") as info:
        _fetch("http://evil.test/a.png")
    assert address not in str(info.value)  # never leak the internal address
    assert log == []  # no connection was attempted


def test_mixed_public_and_private_addresses_are_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"mixed.test": [PUBLIC, "10.0.0.8"]})
    log = _transport(monkeypatch, [_image()])
    with pytest.raises(ValidationError, match="non-public"):
        _fetch("https://mixed.test/a.png")
    assert log == []


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x.test/a.png", "gopher://x.test/"])
def test_only_http_schemes(url: str) -> None:
    with pytest.raises(ValidationError, match="http"):
        _fetch(url)


def test_credentials_in_url_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": [PUBLIC]})
    with pytest.raises(ValidationError, match="credentials"):
        _fetch("http://user:pw@ok.test/a.png")


def test_allowed_public_fetch_connects_to_the_validated_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    looked_up = _dns(monkeypatch, {"ok.test": [PUBLIC]})
    log = _transport(monkeypatch, [_image()])
    assert _fetch("https://ok.test/img/a.png?x=1") == png_bytes()
    assert log[0] == ("open", "https", "ok.test", 443, (PUBLIC,))
    assert log[1] == ("request", "GET", "/img/a.png?x=1")
    assert looked_up == ["ok.test"]  # resolved once; the connection reuses that result


def test_redirect_to_private_ip_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": [PUBLIC], "internal.test": ["10.0.0.1"]})
    log = _transport(
        monkeypatch, [FakeResponse(302, {"Location": "http://internal.test/secret.png"}), _image()]
    )
    with pytest.raises(ValidationError, match="non-public"):
        _fetch("http://ok.test/a.png")
    assert sum(1 for entry in log if entry[0] == "open") == 1


def test_redirect_to_literal_metadata_ip_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": [PUBLIC], "169.254.169.254": ["169.254.169.254"]})
    _transport(monkeypatch, [FakeResponse(301, {"Location": "http://169.254.169.254/latest"})])
    with pytest.raises(ValidationError, match="non-public"):
        _fetch("http://ok.test/a.png")


def test_public_redirect_is_followed_and_relative_locations_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dns(monkeypatch, {"ok.test": [PUBLIC], "cdn.test": ["93.184.216.35"]})
    log = _transport(
        monkeypatch,
        [
            FakeResponse(302, {"Location": "/moved.png"}),
            FakeResponse(307, {"Location": "https://cdn.test/final.png"}),
            _image(),
        ],
    )
    assert _fetch("http://ok.test/a.png") == png_bytes()
    assert [e[2] for e in log if e[0] == "open"] == ["ok.test", "ok.test", "cdn.test"]


def test_too_many_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": [PUBLIC]})
    _transport(monkeypatch, [FakeResponse(302, {"Location": "/again"}) for _ in range(5)])
    with pytest.raises(ValidationError, match="redirects"):
        _fetch("http://ok.test/a.png")


def test_oversize_body_aborts_while_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": [PUBLIC]})
    _transport(monkeypatch, [_image(b"x" * 5000)])
    with pytest.raises(ValidationError, match="exceeds"):
        _fetch("http://ok.test/a.png", max_bytes=1000)


def test_oversize_content_length_is_rejected_up_front(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": [PUBLIC]})
    _transport(
        monkeypatch,
        [FakeResponse(200, {"Content-Type": "image/png", "Content-Length": "999999"}, b"x")],
    )
    with pytest.raises(ValidationError, match="exceeds"):
        _fetch("http://ok.test/a.png", max_bytes=1000)


def test_wrong_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": [PUBLIC]})
    _transport(monkeypatch, [FakeResponse(200, {"Content-Type": "text/html"}, b"<html>")])
    with pytest.raises(ValidationError, match="image"):
        _fetch("http://ok.test/a.png")


def test_http_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": [PUBLIC]})
    _transport(monkeypatch, [FakeResponse(404)])
    with pytest.raises(ValidationError, match="404"):
        _fetch("http://ok.test/a.png")


def test_allow_and_deny_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": [PUBLIC], "other.test": [PUBLIC]})
    _transport(monkeypatch, [_image(), _image()])
    policy = InputPolicy(allow_urls=True, url_allow_hosts=["OK.test"])
    assert _fetch("http://ok.test/a.png", policy)
    with pytest.raises(ValidationError, match="allow list"):
        _fetch("http://other.test/a.png", policy)
    deny = InputPolicy(allow_urls=True, url_deny_hosts=["other.test"])
    with pytest.raises(ValidationError, match="deny list"):
        _fetch("http://other.test/a.png", deny)


def test_allow_list_does_not_bypass_address_check(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": ["127.0.0.1"]})
    _transport(monkeypatch, [_image()])
    with pytest.raises(ValidationError, match="non-public"):
        _fetch("http://ok.test/a.png", InputPolicy(url_allow_hosts=["ok.test"]))


def test_dns_failure_is_a_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> Any:
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(ValidationError, match="resolve"):
        _fetch("http://nowhere.test/a.png")


def test_to_image_uses_ssrf_checked_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    _dns(monkeypatch, {"ok.test": ["10.0.0.9"]})
    _transport(monkeypatch, [_image()])
    with pytest.raises(ValidationError, match="non-public"):
        imaging.to_image("http://ok.test/a.png", InputPolicy(allow_urls=True))
