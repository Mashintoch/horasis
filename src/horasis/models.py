"""Model registry and download/cache manager.

Most backends fetch their own weights (Ultralytics, EasyOCR, transformers).
This module covers the files that nothing else downloads for you -- today the
YuNet face detector -- with pinned URLs and SHA-256 checksums, a per-user
cache directory, and atomic, verified downloads.

Cache directory: ``$HORASIS_CACHE_DIR``, else the platform user cache dir
(via ``platformdirs`` when installed), else ``~/.cache/horasis``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from horasis.foundation.sphalma import ConfigurationError, ModelDownloadError

CACHE_ENV_VAR = "HORASIS_CACHE_DIR"
_CHUNK = 256 * 1024
_TIMEOUT = 30.0


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable model file.

    Attributes:
        name: Registry name (``horasis models pull <name>``).
        url: Exact download URL (pinned to a commit, not a moving branch).
        sha256: Expected SHA-256 of the file, lowercase hex.
        filename: File name inside the cache directory.
        license: Licence note for the weights.
        description: One-line description.
    """

    name: str
    url: str
    sha256: str
    filename: str
    license: str
    description: str = ""


_YUNET_COMMIT = "f12e12798e8314f7c074a6656816c048dcc95b7a"

REGISTRY: dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        name="yunet",
        url=(
            f"https://github.com/opencv/opencv_zoo/raw/{_YUNET_COMMIT}"
            "/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        filename="face_detection_yunet_2023mar.onnx",
        license="MIT (OpenCV Zoo, (c) 2020 Shiqi Yu)",
        description="YuNet face detector (OpenCV Zoo, 2023 Mar), ~230 KB ONNX",
    ),
}

#: Opens a URL and returns a readable binary stream (replaced in tests).
Opener = Callable[[str, float], IO[bytes]]


def _urlopen(url: str, timeout: float) -> IO[bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": "horasis"})
    return urllib.request.urlopen(request, timeout=timeout)  # type: ignore[no-any-return]


def cache_dir() -> Path:
    """Directory models are cached in (not created)."""
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser()
    try:
        from platformdirs import user_cache_dir
    except ImportError:
        return Path.home() / ".cache" / "horasis"
    return Path(user_cache_dir("horasis"))


def get_spec(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ConfigurationError(
            f"Unknown model {name!r}; available: {', '.join(sorted(REGISTRY))}"
        ) from None


def model_path(name: str) -> Path:
    """Where `name` lives (or will live) in the cache; the file may not exist yet."""
    return cache_dir() / get_spec(name).filename


def is_cached(name: str) -> bool:
    return model_path(name).is_file()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(name: str) -> bool:
    """True if `name` is cached and matches its registered checksum."""
    path = model_path(name)
    return path.is_file() and file_sha256(path) == get_spec(name).sha256


def pull(name: str, *, force: bool = False, opener: Opener | None = None) -> Path:
    """Make sure `name` is in the cache and return its path.

    A cache hit returns immediately (use `verify` to re-check the checksum, or
    ``force=True`` to download again). Otherwise the file is streamed to a
    temporary file next to the destination, hashed, and atomically renamed into
    place only if the checksum matches.

    Raises:
        ModelDownloadError: on a network failure or a checksum mismatch (the
            partial file is deleted).
    """
    spec = get_spec(name)
    dest = model_path(name)
    if dest.is_file() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{spec.filename}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(handle, "wb") as out, (opener or _urlopen)(spec.url, _TIMEOUT) as source:
            while chunk := source.read(_CHUNK):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != spec.sha256:
            raise ModelDownloadError(
                f"Checksum mismatch for model {name!r} downloaded from {spec.url}: "
                f"expected {spec.sha256}, got {digest.hexdigest()}. The file was discarded.",
                context={"model": name},
            )
        os.replace(tmp, dest)
    except OSError as exc:
        raise ModelDownloadError(
            f"Could not download model {name!r} from {spec.url}: {exc}", context={"model": name}
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)
    return dest
