"""Image input handling for `Physis`.

Turns whatever a caller hands us (path, bytes, URL, PIL image, numpy array)
into a validated, EXIF-corrected RGB image carried on a :class:`Physis`.
Decoding happens once, at the API boundary; backends then read
``physis.pixels`` via :func:`load_rgb` and never touch raw caller input.

Security notes:
    * URLs are fetched only when ``InputPolicy.allow_urls`` is True. Only
      ``http``/``https`` are accepted. If you enable URLs for untrusted
      callers, also firewall the host against requests to internal networks
      (SSRF); this module does not resolve or filter addresses.
    * Encoded size and decoded pixel count are both capped to blunt
      decompression bombs.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias
from urllib.parse import urlparse

from PIL import Image, ImageOps, UnidentifiedImageError

from horasis.foundation.kanon import InputPolicy
from horasis.foundation.schemas import Physis
from horasis.foundation.sphalma import ValidationError

if TYPE_CHECKING:
    import numpy.typing as npt

    ImageInput: TypeAlias = str | Path | bytes | Physis | Image.Image | npt.NDArray[Any]
else:
    ImageInput = Any

_DECODE_ERRORS = (
    UnidentifiedImageError,
    OSError,
    SyntaxError,
    ValueError,
    Image.DecompressionBombError,
)


def is_url(text: str) -> bool:
    """True if `text` looks like an http(s) URL."""
    return urlparse(text).scheme in ("http", "https")


def decode_image(data: bytes, policy: InputPolicy | None = None) -> Image.Image:
    """Decode encoded image `data` to an upright RGB image, enforcing `policy`.

    Raises:
        ValidationError: if the data is empty, too large, not an allowed
            image format, has too many pixels, or is corrupt.
    """
    policy = policy or InputPolicy()
    if not data:
        raise ValidationError("Image data is empty")
    if len(data) > policy.max_bytes:
        raise ValidationError(
            f"Image is {len(data)} bytes; the limit is {policy.max_bytes} bytes",
            context={"size": len(data), "max_bytes": policy.max_bytes},
        )
    try:
        opened = Image.open(io.BytesIO(data))
        fmt = (opened.format or "").upper()
        allowed = {f.upper() for f in policy.allowed_formats}
        if fmt not in allowed:
            raise ValidationError(
                f"Image format {fmt or 'unknown'!r} is not allowed; allowed: {sorted(allowed)}"
            )
        if opened.width * opened.height > policy.max_pixels:
            raise ValidationError(
                f"Image is {opened.width}x{opened.height} pixels; the limit is {policy.max_pixels}"
            )
        opened.load()
        img: Image.Image = opened
        if policy.apply_exif_orientation:
            img = ImageOps.exif_transpose(img) or img
        return img.convert("RGB")
    except _DECODE_ERRORS as exc:
        raise ValidationError(f"Could not decode image: {exc}") from exc


def array_to_image(array: Any, policy: InputPolicy | None = None) -> Image.Image:
    """Convert a numpy array to an RGB image.

    Accepts ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)`` arrays
    in **RGB(A)** channel order (convert OpenCV's BGR first). ``uint8`` is
    used as-is; floats are read as ``0..1`` when their max is <= 1, else as
    ``0..255``; other integer types are clipped to ``0..255``.
    """
    import numpy as np

    policy = policy or InputPolicy()
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (3, 4)):
        raise ValidationError(f"Unsupported array shape {arr.shape}; expected HxW, HxWx3 or HxWx4")
    if arr.size == 0:
        raise ValidationError("Image array is empty")
    if arr.shape[0] * arr.shape[1] > policy.max_pixels:
        raise ValidationError(
            f"Image is {arr.shape[1]}x{arr.shape[0]} pixels; the limit is {policy.max_pixels}"
        )
    if arr.dtype == np.uint8:
        pass
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.integer):
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise ValidationError(f"Unsupported array dtype {arr.dtype}")
    return Image.fromarray(arr).convert("RGB")


def fetch_url(url: str, *, timeout: float, max_bytes: int) -> bytes:
    """Download `url` (http/https only), refusing bodies over `max_bytes`."""
    if not is_url(url):
        raise ValidationError(f"Only http(s) URLs can be fetched: {url!r}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body: bytes = response.read(max_bytes + 1)
    except Exception as exc:
        raise ValidationError(f"Failed to fetch image from URL: {url}") from exc
    if len(body) > max_bytes:
        raise ValidationError(f"Image at {url} exceeds the {max_bytes}-byte limit")
    return body


def _physis_from_image(img: Image.Image, *, source: str, path: str | Path | None = None) -> Physis:
    return Physis(source=source, path=path, pixels=img, width=img.width, height=img.height)


def to_physis(image: Any, policy: InputPolicy | None = None) -> Physis:
    """Normalize any supported image input into a decoded `Physis`.

    Supported: `Physis`, ``str``/``Path`` file paths, ``bytes``, PIL images,
    numpy arrays, and http(s) URL strings (only if ``policy.allow_urls``).

    Raises:
        ValidationError: for unsupported types, disabled URLs, missing
            files, or any decode/size/format failure.
    """
    policy = policy or InputPolicy()

    if isinstance(image, Physis):
        if image.pixels is not None:
            return image
        if image.data is not None:
            img = decode_image(image.data, policy)
        elif image.path is not None:
            return to_physis(str(image.path), policy)
        else:
            raise ValidationError("Physis has neither `pixels`, `data`, nor `path` set")
        return image.model_copy(update={"pixels": img, "width": img.width, "height": img.height})

    if isinstance(image, bytes | bytearray | memoryview):
        return _physis_from_image(decode_image(bytes(image), policy), source="<bytes>")

    if isinstance(image, str | Path):
        text = str(image)
        if isinstance(image, str) and is_url(text):
            if not policy.allow_urls:
                raise ValidationError(
                    "URL inputs are disabled. Set Kanon.inputs.allow_urls=True to opt in, "
                    "or download the image yourself and pass bytes."
                )
            data = fetch_url(text, timeout=policy.url_timeout_seconds, max_bytes=policy.max_bytes)
            return _physis_from_image(decode_image(data, policy), source=text)
        path = Path(image)
        if not path.is_file():
            raise ValidationError(f"Image file not found: {text}")
        if path.stat().st_size > policy.max_bytes:
            raise ValidationError(
                f"Image file is {path.stat().st_size} bytes; the limit is {policy.max_bytes} bytes"
            )
        return _physis_from_image(decode_image(path.read_bytes(), policy), source=text, path=path)

    if isinstance(image, Image.Image):
        return _physis_from_image(_normalize_pil(image, policy), source="<pil>")

    if type(image).__module__.split(".")[0] == "numpy":
        return _physis_from_image(array_to_image(image, policy), source="<ndarray>")

    raise ValidationError(f"Unsupported image input type: {type(image).__name__}")


def _normalize_pil(img: Image.Image, policy: InputPolicy) -> Image.Image:
    if img.width * img.height > policy.max_pixels:
        raise ValidationError(
            f"Image is {img.width}x{img.height} pixels; the limit is {policy.max_pixels}"
        )
    if policy.apply_exif_orientation:
        img = ImageOps.exif_transpose(img) or img
    return img.convert("RGB")


def load_rgb(image: Physis, policy: InputPolicy | None = None) -> Image.Image:
    """Return the RGB pixels of `image`, decoding them if not yet loaded.

    Backends call this instead of reading ``path``/``data`` themselves, so
    every backend sees identical, validated, upright pixels.
    """
    if image.pixels is not None:
        pixels: Image.Image = image.pixels
        return pixels
    decoded: Image.Image = to_physis(image, policy).pixels
    return decoded


def encode_image(image: Physis, *, fmt: str = "JPEG", quality: int = 95) -> bytes:
    """Encode `image` for transport (e.g. to a remote server). `fmt` is JPEG or PNG."""
    buf = io.BytesIO()
    rgb = load_rgb(image)
    if fmt.upper() == "PNG":
        rgb.save(buf, format="PNG")
    else:
        rgb.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
