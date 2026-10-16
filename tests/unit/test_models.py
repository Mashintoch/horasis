from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import IO

import pytest

from horasis import cli, models
from horasis.backend.face_adapter import MODEL_ENV_VAR, YuNetFaceAdapter
from horasis.foundation.schemas import Praxis
from horasis.foundation.sphalma import (
    BackendUnavailableError,
    ConfigurationError,
    ModelDownloadError,
)

PAYLOAD = b"fake-onnx-bytes" * 1000


@pytest.fixture
def fake_spec(monkeypatch: pytest.MonkeyPatch) -> models.ModelSpec:
    spec = models.ModelSpec(
        name="yunet",
        url="https://example.test/m.onnx",
        sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        filename="m.onnx",
        license="MIT",
    )
    monkeypatch.setitem(models.REGISTRY, "yunet", spec)
    return spec


class Opener:
    def __init__(self, payload: bytes = PAYLOAD) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> IO[bytes]:
        self.urls.append(url)
        return io.BytesIO(self.payload)


def test_registry_entries_are_complete_and_pinned() -> None:
    spec = models.get_spec("yunet")
    assert len(spec.sha256) == 64 and spec.filename.endswith(".onnx")
    assert "/raw/main/" not in spec.url  # pinned to a commit, not a moving branch
    with pytest.raises(ConfigurationError, match="Unknown model"):
        models.get_spec("nope")


def test_cache_dir_env_override_and_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(models.CACHE_ENV_VAR, str(tmp_path / "c"))
    assert models.cache_dir() == tmp_path / "c"
    assert models.model_path("yunet").parent == tmp_path / "c"
    monkeypatch.delenv(models.CACHE_ENV_VAR)
    assert "horasis" in str(models.cache_dir())


def test_pull_downloads_verifies_and_caches(fake_spec: models.ModelSpec) -> None:
    opener = Opener()
    path = models.pull("yunet", opener=opener)
    assert path.read_bytes() == PAYLOAD and path == models.model_path("yunet")
    assert opener.urls == [fake_spec.url]
    assert [p.name for p in path.parent.iterdir()] == ["m.onnx"]  # no temp files left
    assert models.verify("yunet")


def test_cache_hit_does_not_download(fake_spec: models.ModelSpec) -> None:
    models.pull("yunet", opener=Opener())
    again = Opener(b"different")
    models.pull("yunet", opener=again)
    assert again.urls == []
    models.pull("yunet", opener=Opener(), force=True)


def test_checksum_mismatch_deletes_and_raises(fake_spec: models.ModelSpec) -> None:
    with pytest.raises(ModelDownloadError, match="Checksum mismatch"):
        models.pull("yunet", opener=Opener(b"tampered"))
    assert list(models.cache_dir().iterdir()) == []
    assert not models.is_cached("yunet")


def test_network_failure_is_a_model_error(fake_spec: models.ModelSpec) -> None:
    def broken(url: str, timeout: float) -> IO[bytes]:
        raise OSError("offline")

    with pytest.raises(ModelDownloadError, match="offline"):
        models.pull("yunet", opener=broken)
    assert list(models.cache_dir().iterdir()) == []


def test_verify_detects_corruption(fake_spec: models.ModelSpec) -> None:
    assert not models.verify("yunet")  # not cached
    path = models.pull("yunet", opener=Opener())
    path.write_bytes(b"corrupt")
    assert not models.verify("yunet")


def test_adapter_without_model_tells_you_to_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    with pytest.raises(BackendUnavailableError, match="horasis models pull yunet"):
        YuNetFaceAdapter().warm_up()


def test_adapter_resolves_cached_model_and_env_override(
    fake_spec: models.ModelSpec, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    cached = models.pull("yunet", opener=Opener())
    adapter = YuNetFaceAdapter()
    assert adapter._resolve_model_path() == cached  # pylint: disable=protected-access
    other = tmp_path / "mine.onnx"
    other.write_bytes(b"x")
    monkeypatch.setenv(MODEL_ENV_VAR, str(other))
    assert adapter._resolve_model_path() == other  # pylint: disable=protected-access
    explicit = tmp_path / "explicit.onnx"
    explicit.write_bytes(b"x")
    assert YuNetFaceAdapter(model_path=explicit)._resolve_model_path() == explicit  # pylint: disable=protected-access


def test_adapter_auto_download_only_when_enabled(
    fake_spec: models.ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    opener = Opener()
    monkeypatch.setattr(models, "_urlopen", opener)
    off = YuNetFaceAdapter()
    with pytest.raises(BackendUnavailableError):
        off._resolve_model_path()  # pylint: disable=protected-access
    assert opener.urls == []
    on = YuNetFaceAdapter()
    on.default_auto_download = True
    assert on._resolve_model_path() == models.model_path("yunet")  # pylint: disable=protected-access
    assert opener.urls == [fake_spec.url]


def test_kanon_flag_reaches_backends() -> None:
    from horasis import Skopos
    from horasis.foundation.kanon import Kanon

    adapter = YuNetFaceAdapter()
    client = Skopos.build(kanon=Kanon.default().model_copy(update={"auto_download_models": True}))
    client.taxis.register(adapter)
    assert adapter.default_auto_download is True
    assert Praxis.FACE in adapter.supported_praxeis


def test_models_cli(
    fake_spec: models.ModelSpec, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(models, "_urlopen", Opener())
    assert cli.main(["models", "list"]) == 0
    assert "not cached" in capsys.readouterr().out
    assert cli.main(["models", "verify"]) == 0
    assert cli.main(["models", "pull", "yunet"]) == 0
    pulled = capsys.readouterr().out.strip().splitlines()[-1]
    assert cli.main(["models", "path", "yunet"]) == 0
    assert capsys.readouterr().out.strip() == pulled
    assert cli.main(["models", "verify"]) == 0
    Path(pulled).write_bytes(b"bad")
    assert cli.main(["models", "verify"]) == 1
    assert "MISMATCH" in capsys.readouterr().out
    assert cli.main(["models", "pull", "nope"]) == 1


