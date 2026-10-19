"""Load and validate caller-supplied images.

Paths, bytes, URLs, PIL images and numpy arrays become an EXIF-corrected RGB
:class:`Image`. Decoding happens once, at the API boundary; backends read
pixels through :func:`load_rgb` and never touch raw caller input.

Security notes:
    * URLs are fetched only when ``InputPolicy.allow_urls`` is set, and only
      over http(s). Every resolved address must be globally routable. The
      connection goes to the validated IP (no second lookup, so DNS rebinding
      cannot swap the target), with SNI and certificate checks against the
      original hostname. Redirects are followed manually, at most 3, and each
      hop is validated again. Run an egress firewall as well.
    * Encoded size and decoded pixel count are both capped to blunt
      decompression bombs.
"""

from __future__ import annotations

import http.client
import io
import ipaddress
import socket
import ssl
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias
from urllib.parse import urljoin, urlparse

from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError

from horasis.foundation.config import InputPolicy
from horasis.foundation.errors import ValidationError
from horasis.foundation.schemas import Image

if TYPE_CHECKING:
    import numpy.typing as npt

    ImageInput: TypeAlias = str | Path | bytes | Image | PILImage.Image | npt.NDArray[Any]
else:
    ImageInput = Any

_DECODE_ERRORS = (
    UnidentifiedImageError,
    OSError,
    SyntaxError,
    ValueError,
    PILImage.DecompressionBombError,
)


def is_url(text: str) -> bool:
    """Return True if `text` is an http(s) URL."""
    return urlparse(text).scheme in ("http", "https")


def decode_image(data: bytes, policy: InputPolicy | None = None) -> PILImage.Image:
    """Decode encoded image `data` to an upright RGB image, enforcing `policy`.

    Raises:
        ValidationError: if the data is empty, too large, not an allowed
            image format, has too many pixels, or is corrupt.
    """
    policy = policy or InputPolicy()
    if not data:
        raise ValidationError("image data is empty")
    if len(data) > policy.max_bytes:
        raise ValidationError(
            f"image is {len(data)} bytes, limit is {policy.max_bytes}",
            context={"size": len(data), "max_bytes": policy.max_bytes},
        )
    try:
        opened = PILImage.open(io.BytesIO(data))
        fmt = (opened.format or "").upper()
        allowed = {f.upper() for f in policy.allowed_formats}
        if fmt not in allowed:
            raise ValidationError(
                f"image format {fmt or 'unknown'!r} not allowed; allowed: {sorted(allowed)}"
            )
        if opened.width * opened.height > policy.max_pixels:
            raise ValidationError(
                f"image is {opened.width}x{opened.height} pixels, limit is {policy.max_pixels}"
            )
        opened.load()
        img: PILImage.Image = opened
        if policy.apply_exif_orientation:
            img = ImageOps.exif_transpose(img) or img
        return img.convert("RGB")
    except _DECODE_ERRORS as exc:
        raise ValidationError(f"cannot decode image: {exc}") from exc


def array_to_image(array: Any, policy: InputPolicy | None = None) -> PILImage.Image:
    """Convert a numpy array to an RGB image.

    Accepts ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or ``(H, W, 4)`` arrays
    in RGB(A) channel order (convert OpenCV's BGR first). ``uint8`` is
    used as-is; floats are read as ``0..1`` when their max is <= 1, else as
    ``0..255``; other integer types are clipped to ``0..255``.
    """
    import numpy as np

    policy = policy or InputPolicy()
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (3, 4)):
        raise ValidationError(f"unsupported array shape {arr.shape}; expected HxW, HxWx3 or HxWx4")
    if arr.size == 0:
        raise ValidationError("image array is empty")
    if arr.shape[0] * arr.shape[1] > policy.max_pixels:
        raise ValidationError(
            f"image is {arr.shape[1]}x{arr.shape[0]} pixels, limit is {policy.max_pixels}"
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
        raise ValidationError(f"unsupported array dtype {arr.dtype}")
    return PILImage.fromarray(arr).convert("RGB")


MAX_REDIRECTS = 3
_CHUNK = 64 * 1024


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True only for globally routable unicast addresses (embedded IPv4 is unwrapped)."""
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip.teredo is not None:
            embedded = ip.teredo[1]
        if embedded is not None:
            return _is_public(embedded)
    return ip.is_global and not (
        ip.is_multicast or ip.is_reserved or ip.is_loopback or ip.is_link_local or ip.is_unspecified
    )


def _check_host_policy(host: str, policy: InputPolicy) -> None:
    name = host.lower().rstrip(".")
    deny = {h.lower().rstrip(".") for h in policy.url_deny_hosts}
    allow = {h.lower().rstrip(".") for h in policy.url_allow_hosts}
    if name in deny:
        raise ValidationError(f"URL host {host!r} is on the deny list")
    if allow and name not in allow:
        raise ValidationError(f"URL host {host!r} is not on the allow list")


def _resolve_public(host: str, port: int, policy: InputPolicy) -> list[str]:
    """Resolve `host`; return its addresses only if all of them are public.

    The offending address is deliberately not included in the error, so an
    internal IP is never echoed back to a caller.
    """
    _check_host_policy(host, policy)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValidationError(f"cannot resolve URL host {host!r}") from exc
    addresses: list[str] = []
    for info in infos:
        text = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(text)
        except ValueError as exc:
            raise ValidationError(f"URL host {host!r} resolved to an invalid address") from exc
        if not _is_public(ip):
            raise ValidationError(
                f"URL host {host!r} resolves to a non-public address; request blocked"
            )
        if text not in addresses:
            addresses.append(text)
    if not addresses:
        raise ValidationError(f"cannot resolve URL host {host!r}")
    return addresses


def _dial(addresses: list[str], port: int, timeout: float) -> socket.socket:
    last: OSError | None = None
    for addr in addresses:
        try:
            return socket.create_connection((addr, port), timeout=timeout)
        except OSError as exc:
            last = exc
    raise last or OSError("no address to connect to")


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that dials pre-validated IPs instead of resolving the hostname."""

    def __init__(self, host: str, port: int, addresses: list[str], timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._addresses = addresses
        self._timeout = timeout

    def connect(self) -> None:
        self.sock = _dial(self._addresses, self.port, self._timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS equivalent; TLS SNI and certificate checks still use the original hostname."""

    def __init__(self, host: str, port: int, addresses: list[str], timeout: float) -> None:
        self._tls = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=self._tls)
        self._addresses = addresses
        self._timeout = timeout

    def connect(self) -> None:
        raw = _dial(self._addresses, self.port, self._timeout)
        self.sock = self._tls.wrap_socket(raw, server_hostname=self.host)


def _open_connection(
    scheme: str, host: str, port: int, addresses: list[str], timeout: float
) -> http.client.HTTPConnection:
    """Return a connection pinned to `addresses`. Tests replace this."""
    if scheme == "https":
        return _PinnedHTTPSConnection(host, port, addresses, timeout)
    return _PinnedHTTPConnection(host, port, addresses, timeout)


def fetch_url(
    url: str, *, timeout: float, max_bytes: int, policy: InputPolicy | None = None
) -> bytes:
    """Download an image from `url` (http/https only) with SSRF protections.

    Every resolved address must be globally routable and the connection goes
    to the validated IP. Redirects are followed manually, at most
    ``MAX_REDIRECTS``, and each hop is validated again. The body is streamed
    and aborted past `max_bytes`. The response must be ``image/*``.

    Raises:
        ValidationError: if the request is blocked, fails, redirects too
            often, is not an image, or is larger than `max_bytes`.
    """
    policy = policy or InputPolicy()
    deadline = time.monotonic() + timeout * 4
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        parts = urlparse(current)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValidationError(f"only http(s) URLs can be fetched: {current!r}")
        if parts.username or parts.password:
            raise ValidationError("URLs with embedded credentials are not allowed")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError as exc:
            raise ValidationError(f"invalid URL port: {current!r}") from exc
        addresses = _resolve_public(parts.hostname, port, policy)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        conn = _open_connection(parts.scheme, parts.hostname, port, addresses, timeout)
        try:
            conn.request(
                "GET",
                target,
                headers={
                    "Accept": "image/*",
                    "Accept-Encoding": "identity",
                    "User-Agent": "horasis",
                },
            )
            response = conn.getresponse()
            location = response.getheader("Location")
            if response.status in (301, 302, 303, 307, 308) and location:
                current = urljoin(current, location)
                continue
            return _read_image_body(response, current, max_bytes, deadline)
        except (OSError, http.client.HTTPException) as exc:
            raise ValidationError(f"failed to fetch image from {url}") from exc
        finally:
            conn.close()
    raise ValidationError(f"too many redirects (max {MAX_REDIRECTS}) fetching {url}")


def _read_image_body(
    response: http.client.HTTPResponse, url: str, max_bytes: int, deadline: float
) -> bytes:
    if response.status != 200:
        raise ValidationError(f"fetching {url} failed with HTTP status {response.status}")
    ctype = (response.getheader("Content-Type") or "").split(";")[0].strip().lower()
    if not ctype.startswith("image/"):
        raise ValidationError(f"URL did not return an image (content type {ctype or 'missing'!r})")
    declared = response.getheader("Content-Length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise ValidationError(f"image at {url} exceeds the {max_bytes}-byte limit")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValidationError(f"image at {url} exceeds the {max_bytes}-byte limit")
        if time.monotonic() > deadline:
            raise ValidationError(f"timed out downloading image from {url}")
        chunks.append(chunk)
    return b"".join(chunks)


def _image_from_pil(img: PILImage.Image, *, source: str, path: str | Path | None = None) -> Image:
    return Image(source=source, path=path, pixels=img, width=img.width, height=img.height)


def to_image(image: Any, policy: InputPolicy | None = None) -> Image:
    """Convert any supported image input into a decoded `Image`.

    Supported: `Image`, ``str``/``Path`` file paths, ``bytes``, PIL images,
    numpy arrays, and http(s) URL strings (only if ``policy.allow_urls``).

    Raises:
        ValidationError: for unsupported types, disabled URLs, missing
            files, or any decode/size/format failure.
    """
    policy = policy or InputPolicy()

    if isinstance(image, Image):
        if image.pixels is not None:
            return image
        if image.data is not None:
            img = decode_image(image.data, policy)
        elif image.path is not None:
            return to_image(str(image.path), policy)
        else:
            raise ValidationError("image has neither `pixels`, `data` nor `path` set")
        return image.model_copy(update={"pixels": img, "width": img.width, "height": img.height})

    if isinstance(image, bytes | bytearray | memoryview):
        return _image_from_pil(decode_image(bytes(image), policy), source="<bytes>")

    if isinstance(image, str | Path):
        text = str(image)
        if isinstance(image, str) and is_url(text):
            if not policy.allow_urls:
                raise ValidationError(
                    "URL inputs are disabled; set Config.inputs.allow_urls=True "
                    "or pass the image as bytes"
                )
            data = fetch_url(
                text, timeout=policy.url_timeout_seconds, max_bytes=policy.max_bytes, policy=policy
            )
            return _image_from_pil(decode_image(data, policy), source=text)
        path = Path(image)
        if not path.is_file():
            raise ValidationError(f"image file not found: {text}")
        if path.stat().st_size > policy.max_bytes:
            raise ValidationError(
                f"image file is {path.stat().st_size} bytes, limit is {policy.max_bytes}"
            )
        return _image_from_pil(decode_image(path.read_bytes(), policy), source=text, path=path)

    if isinstance(image, PILImage.Image):
        return _image_from_pil(_normalize_pil(image, policy), source="<pil>")

    if type(image).__module__.split(".")[0] == "numpy":
        return _image_from_pil(array_to_image(image, policy), source="<ndarray>")

    raise ValidationError(f"unsupported image input type: {type(image).__name__}")


def _normalize_pil(img: PILImage.Image, policy: InputPolicy) -> PILImage.Image:
    if img.width * img.height > policy.max_pixels:
        raise ValidationError(
            f"image is {img.width}x{img.height} pixels, limit is {policy.max_pixels}"
        )
    if policy.apply_exif_orientation:
        img = ImageOps.exif_transpose(img) or img
    return img.convert("RGB")


def load_rgb(image: Image, policy: InputPolicy | None = None) -> PILImage.Image:
    """Return the RGB pixels of `image`, decoding them if not yet loaded.

    Backends call this instead of reading ``path`` or ``data``, so they all
    get the same validated pixels.
    """
    if image.pixels is not None:
        pixels: PILImage.Image = image.pixels
        return pixels
    decoded: PILImage.Image = to_image(image, policy).pixels
    return decoded


def encode_image(image: Image, *, fmt: str = "JPEG", quality: int = 95) -> bytes:
    """Encode `image` as JPEG or PNG bytes for transport."""
    buf = io.BytesIO()
    rgb = load_rgb(image)
    if fmt.upper() == "PNG":
        rgb.save(buf, format="PNG")
    else:
        rgb.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
