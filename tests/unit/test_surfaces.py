"""CLI and HTTP server tests (all with fake backends)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from horasis import cli
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.klimax import Klimax
from horasis.api.skopos import Skopos
from horasis.backend.remote_adapter import RemoteAdapter
from horasis.backend.taxis import Taxis
from horasis.foundation.kanon import InputPolicy, Kanon, QuotaRule, RateRule, TierRule
from horasis.foundation.schemas import (
    BoundingBox,
    Detection,
    DetectionResult,
    Doxa,
    Glyph,
    OcrResult,
    Physis,
    PoseResult,
    Praxis,
    SegmentationResult,
    Theoria,
)
from horasis.foundation.sphalma import (
    AccessDeniedError,
    InvalidApiKeyError,
    RateLimitedError,
    ValidationError,
)
from horasis.server import create_app, parse_options
from tests.conftest import FakeMechane, png_bytes


class Scripted(FakeMechane):
    """Returns recognisable results and records the options it received."""

    def __init__(self, name: str = "fake", praxeis: frozenset[Praxis] | None = None) -> None:
        super().__init__(name, praxeis or frozenset(Praxis))
        self.options: list[dict[str, Any]] = []

    def infer(self, praxis: Praxis, image: Physis, **options: object) -> Theoria:
        self.options.append(dict(options))
        if praxis is Praxis.DETECT:
            det = Detection(
                label="cat",
                confidence=Doxa(value=0.9),
                box=BoundingBox(x_min=1, y_min=2, x_max=30, y_max=40),
            )
            return DetectionResult(backend=self.name, latency_ms=1.0, detections=[det])
        if praxis is Praxis.OCR:
            glyph = Glyph(text="HELLO", confidence=Doxa(value=0.8), line_index=0)
            return OcrResult(backend=self.name, latency_ms=1.0, glyphs=[glyph])
        if praxis is Praxis.SEGMENT:
            return SegmentationResult(backend=self.name, latency_ms=1.0)
        if praxis is Praxis.POSE:
            return PoseResult(backend=self.name, latency_ms=1.0)
        return super().infer(praxis, image, **options)


def make_client(**tier_kw: Any) -> tuple[Skopos, Scripted]:
    backend = Scripted()
    kanon = Kanon(
        default_backend="fake",
        tiers={
            "pro": TierRule(name="pro", quota=QuotaRule(limit=100, window_seconds=3600), **tier_kw)
        },
        inputs=InputPolicy(max_bytes=5000),
    )
    validator = InMemoryApiKeyValidator()
    validator.register_simple("good", principal_id="u", tier=Klimax.PRO)
    validator.register_simple("scoped", principal_id="s", tier=Klimax.PRO, scopes=["ocr"])
    validator.register_simple("old", principal_id="o", tier=Klimax.PRO, expires_at=1.0)
    taxis = Taxis()
    taxis.register(backend)
    return Skopos(kanon=kanon, taxis=taxis, validator=validator), backend


@pytest.fixture
def served() -> tuple[TestClient, Scripted, Skopos]:
    client, backend = make_client()
    return TestClient(create_app(client)), backend, client


def post(http: TestClient, praxis: str, key: str | None = "good", **form: str) -> Any:
    headers = {"X-API-Key": key} if key else {}
    return http.post(
        f"/v1/{praxis}", files={"image": ("i.png", png_bytes(), "image/png")}, data=form, headers=headers
    )


# ---- server --------------------------------------------------------------------


def test_health_lists_backends_without_auth(served: Any) -> None:
    http, _, _ = served
    body = http.get("/health").json()
    assert body["status"] == "ok" and "detect" in body["backends"]["fake"]


def test_detect_roundtrip_and_options_forwarding(served: Any) -> None:
    http, backend, _ = served
    res = post(http, "detect", options=json.dumps({"conf": 0.4}))
    assert res.status_code == 200
    body = res.json()
    assert body["praxis"] == "detect" and body["detections"][0]["label"] == "cat"
    assert backend.options[-1] == {"estimate_demographics": False, "conf": 0.4}


def test_usage_endpoint(served: Any) -> None:
    http, _, _ = served
    post(http, "ocr")
    usage = http.get("/usage", headers={"X-API-Key": "good"}).json()
    assert usage["principal_id"] == "u" and usage["overall"]["used"] == 2  # ocr costs 2
    assert http.get("/usage").status_code == 401


@pytest.mark.parametrize(
    ("kwargs", "status", "kind"),
    [
        ({"key": None}, 401, "invalid_api_key"),
        ({"key": "nope"}, 401, "invalid_api_key"),
        ({"key": "old"}, 401, "expired_api_key"),
        ({"key": "scoped", "praxis": "detect"}, 403, "forbidden"),
        ({"praxis": "teleport"}, 404, "unknown_praxis"),
        ({"options": "{bad json"}, 422, "invalid_request"),
        ({"options": json.dumps({"save": True})}, 422, "invalid_request"),
        ({"options": json.dumps({"conf": "high"})}, 422, "invalid_request"),
        ({"options": json.dumps([1])}, 422, "invalid_request"),
        ({"backend": "ghost"}, 501, "backend_unavailable"),
    ],
)
def test_error_mapping(served: Any, kwargs: dict[str, Any], status: int, kind: str) -> None:
    http, _, _ = served
    kw = dict(kwargs)
    praxis = kw.pop("praxis", "detect")
    key = kw.pop("key", "good")
    res = post(http, praxis, key, **kw)
    assert res.status_code == status
    assert res.json()["error"]["type"] == kind


def test_garbage_and_oversized_uploads(served: Any) -> None:
    http, _, _ = served
    bad = http.post(
        "/v1/detect", files={"image": ("i", b"junk", "image/png")}, headers={"X-API-Key": "good"}
    )
    assert bad.status_code == 422
    big = http.post(
        "/v1/detect", files={"image": ("i", b"x" * 6000, "image/png")}, headers={"X-API-Key": "good"}
    )
    assert big.status_code == 413 and big.json()["error"]["type"] == "too_large"


def test_quota_and_rate_errors_carry_retry_after() -> None:
    client, _ = make_client(rate=RateRule(per_minute=60, burst=1))
    http = TestClient(create_app(client))
    assert post(http, "detect").status_code == 200
    limited = post(http, "detect")
    assert limited.status_code == 429
    err = limited.json()["error"]
    assert err["type"] == "rate_limited" and err["limit"] == 60 and err["retry_after"] > 0
    assert int(limited.headers["Retry-After"]) >= 1

    tiny = make_client()[0]
    tiny_kanon = tiny.kanon.model_copy(
        update={"tiers": {"pro": TierRule(name="pro", quota=QuotaRule(limit=1, window_seconds=60))}}
    )
    validator = InMemoryApiKeyValidator()
    validator.register_simple("good", principal_id="u", tier=Klimax.PRO)
    taxis = Taxis()
    taxis.register(Scripted())
    http2 = TestClient(create_app(Skopos(kanon=tiny_kanon, taxis=taxis, validator=validator)))
    post(http2, "detect")
    over = post(http2, "detect")
    body = over.json()["error"]
    assert over.status_code == 429 and body["type"] == "quota_exceeded"
    assert (body["limit"], body["scope"]) == (1, "quota") and "Retry-After" in over.headers


def test_backend_failure_does_not_leak_internals() -> None:
    class Crash(Scripted):
        def infer(self, praxis: Praxis, image: Physis, **options: object) -> Theoria:
            raise RuntimeError("secret path /srv/models")

    client, _ = make_client()
    client.taxis.register(Crash("crash"))
    http = TestClient(create_app(client), raise_server_exceptions=False)
    res = post(http, "detect", backend="crash")
    assert res.status_code == 500
    assert "secret" not in res.text


def test_parse_options_accepts_empty_and_valid() -> None:
    assert parse_options("detect", None) == {} and parse_options("detect", "") == {}
    assert parse_options("ocr", '{"min_confidence": 1}') == {"min_confidence": 1}
    with pytest.raises(ValidationError):
        parse_options("detect", '{"max_det": true}')  # bool is not an int here


# ---- remote adapter <-> server ------------------------------------------------------------


def test_remote_adapter_talks_to_the_server() -> None:
    server_client, backend = make_client()
    http = TestClient(create_app(server_client), base_url="http://horasis.test")
    remote = RemoteAdapter(endpoint_url="http://horasis.test", api_key="good", client=http)
    result = remote.infer(Praxis.OCR, Physis.from_bytes(png_bytes()))
    assert isinstance(result, OcrResult) and result.text == "HELLO" and result.backend == "remote"

    local_client = Skopos.build(
        kanon=server_client.kanon.model_copy(update={"default_backend": "remote"}),
        validator=_validator("local"),
        backends=[remote],
    )
    detected = local_client.detection.detect(png_bytes(), api_key="local", options={"conf": 0.5})
    assert detected.detections[0].label == "cat" and detected.backend == "remote"
    assert backend.options[-1]["conf"] == 0.5


def _validator(key: str) -> InMemoryApiKeyValidator:
    v = InMemoryApiKeyValidator()
    v.register_simple(key, principal_id="l", tier=Klimax.PRO)
    return v


def test_remote_adapter_maps_server_errors_back_to_exceptions() -> None:
    server_client, _ = make_client(rate=RateRule(per_minute=60, burst=1))
    http = TestClient(create_app(server_client), base_url="http://horasis.test")
    img = Physis.from_bytes(png_bytes())

    def adapter(key: str) -> RemoteAdapter:
        return RemoteAdapter(
            endpoint_url="http://horasis.test", api_key=key, client=http, max_retries=0
        )

    with pytest.raises(InvalidApiKeyError):
        adapter("wrong").infer(Praxis.DETECT, img)
    with pytest.raises(AccessDeniedError):
        adapter("scoped").infer(Praxis.DETECT, img)
    adapter("good").infer(Praxis.DETECT, img)
    with pytest.raises(RateLimitedError) as info:
        adapter("good").infer(Praxis.DETECT, img)
    assert info.value.limit == 60 and info.value.retry_after is not None


# ---- CLI -----------------------------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(cli, "_default_backends", lambda args: [Scripted("fake")])
    monkeypatch.setattr(cli.Prosthesis, "discover_entry_points", lambda self: [])
    image = tmp_path / "pic.png"
    image.write_bytes(png_bytes(64, 48))
    return image


def test_cli_text_and_json_output(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["detect", str(cli_env), "--backend", "fake"]) == 0
    text = capsys.readouterr().out
    assert "cat" in text and "0.90" in text and "[1, 2, 30, 40]" in text
    assert cli.main(["detect", str(cli_env), "--backend", "fake", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["detections"][0]["label"] == "cat"


def test_cli_text_format_for_each_command(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for command in ("classify", "ocr", "faces", "moderate", "segment", "pose"):
        assert cli.main([command, str(cli_env), "--backend", "fake"]) == 0
    capsys.readouterr()
    assert cli.main(["ocr", str(cli_env), "--backend", "fake"]) == 0
    assert capsys.readouterr().out.strip() == "HELLO"
    assert cli.main(["faces", str(cli_env), "--backend", "fake"]) == 0
    assert "(nothing found)" in capsys.readouterr().out


def test_cli_min_conf_draw_and_errors(
    cli_env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_png = tmp_path / "drawn.png"
    code = cli.main(
        ["detect", str(cli_env), "--backend", "fake", "--min-conf", "0.95", "--draw", str(out_png)]
    )
    assert code == 0 and out_png.is_file()
    assert "(nothing found)" in capsys.readouterr().out
    assert cli.main(["detect", str(tmp_path / "missing.png"), "--backend", "fake"]) == 1
    assert "error:" in capsys.readouterr().err
    assert cli.main(["detect", str(cli_env), "--backend", "ghost"]) == 1


def test_cli_urls_blocked_by_default(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["detect", "https://example.com/a.png", "--backend", "fake"]) == 1
    assert "disabled" in capsys.readouterr().err


def test_cli_backends_command(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["backends"]) == 0
    text = capsys.readouterr().out
    assert "ultralytics" in text and "fake" in text and "builtin" in text
    assert cli.main(["backends", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    names = {b["name"]: b for b in data["backends"]}
    assert {"ultralytics", "fake"} <= set(names) and "detect" in names["ultralytics"]["praxeis"]


def test_cli_lists_plugins_including_broken_ones(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from importlib.metadata import EntryPoint

    good = EntryPoint("goodplug", "tests.unit.test_surfaces:PluginBackend", "horasis.backends")
    broken = EntryPoint("brokenplug", "no_such_module:Thing", "horasis.backends")
    monkeypatch.setattr(cli.Prosthesis, "discover_entry_points", lambda self: [good, broken])
    assert cli.main(["backends", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    plug = next(b for b in data["backends"] if b["name"] == "plugged")
    assert plug["origin"] == "plugin"
    assert data["failed_plugins"][0]["name"] == "brokenplug"


class PluginBackend(FakeMechane):
    def __init__(self) -> None:
        super().__init__("plugged")


def test_cli_serve_requires_keys_and_version_flag(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HORASIS_API_KEY", raising=False)
    assert cli.main(["serve"]) == 1
    assert "API keys" in capsys.readouterr().err
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    with pytest.raises(SystemExit) as bad:
        cli.main(["detect"])  # missing image
    assert bad.value.code == 2
