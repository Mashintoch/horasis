"""Command-line interface: ``horasis detect photo.jpg --json``.

Commands::

    horasis detect|classify|ocr|faces|moderate|segment|pose IMAGE [options]
    horasis backends [--json]
    horasis models list|pull NAME|path NAME|verify
    horasis serve [--host H] [--port P] [--keys-file FILE]   (needs horasis[server])

The image commands run locally as an internal enterprise-tier principal, so
they need no API key.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from horasis import __version__, models
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.stores import redis_backends
from horasis.access.tiers import Tier
from horasis.api.client import Client
from horasis.backend.base import Backend
from horasis.backend.easyocr_adapter import EasyOCRAdapter
from horasis.backend.face_adapter import YuNetFaceAdapter
from horasis.backend.moderation_adapter import TransformersModerationAdapter
from horasis.backend.tesseract_adapter import TesseractAdapter
from horasis.foundation.config import Config, InputPolicy
from horasis.foundation.errors import HorasisError, PluginError
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    FaceResult,
    ModerationResult,
    OcrResult,
    PoseResult,
    Result,
    SegmentationResult,
)
from horasis.plugins.loader import PluginLoader

CLI_KEY = "cli-local"

#: command name -> (capability attribute, method name)
COMMANDS: dict[str, tuple[str, str]] = {
    "detect": ("detection", "detect"),
    "classify": ("classification", "classify"),
    "ocr": ("ocr", "extract_text"),
    "faces": ("face", "detect_faces"),
    "moderate": ("moderation", "moderate"),
    "segment": ("segmentation", "segment"),
    "pose": ("pose", "estimate"),
}


def _default_backends(args: argparse.Namespace) -> list[Backend]:
    """Return the optional backends registered next to Ultralytics; all load lazily."""
    return [
        EasyOCRAdapter(),
        TesseractAdapter(),
        YuNetFaceAdapter(model_path=getattr(args, "face_model", None)),
        TransformersModerationAdapter(),
    ]


def _build_client(args: argparse.Namespace) -> tuple[Client, list[tuple[str, str]]]:
    """Build the client and return it with ``(plugin_name, status)`` for each plugin found."""
    config = Config.from_yaml(args.config) if args.config else Config.default()
    update: dict[str, Any] = {"device": args.device}
    if args.allow_urls:
        update["inputs"] = InputPolicy(allow_urls=True)
    if not config.backend_order:
        update["backend_order"] = {"ocr": ["easyocr", "tesseract"]}
    config = config.model_copy(update=update)

    validator = InMemoryApiKeyValidator()
    validator.register_simple(CLI_KEY, principal_id="cli", tier=Tier.ENTERPRISE)
    client = Client.build(config=config, validator=validator, backends=_default_backends(args))

    plugins: list[tuple[str, str]] = []
    loader = PluginLoader()
    for ep in loader.discover_entry_points():
        try:
            backend = loader.load(ep)
        except PluginError as exc:
            plugins.append((ep.name, f"failed to load: {exc.message}"))
            continue
        client.registry.register(backend)
        plugins.append((backend.name, "plugin"))
    return client, plugins


def format_result(result: Result) -> str:
    """Format any result as plain text for the terminal."""

    def box(b: Any) -> str:
        return f"[{b.x_min:.0f}, {b.y_min:.0f}, {b.x_max:.0f}, {b.y_max:.0f}]"

    lines: list[str] = []
    if isinstance(result, DetectionResult):
        lines = [f"{d.label:<16} {d.confidence.value:.2f}  {box(d.box)}" for d in result.detections]
    elif isinstance(result, FaceResult):
        lines = [f"face             {f.confidence.value:.2f}  {box(f.box)}" for f in result.faces]
    elif isinstance(result, ClassificationResult):
        lines = [f"{e.label:<24} {e.confidence.value:.3f}" for e in result.entries]
    elif isinstance(result, OcrResult):
        lines = result.text.splitlines()
    elif isinstance(result, ModerationResult):
        lines = [
            f"{lb.category:<20} {lb.confidence.value:.3f}{'  FLAGGED' if lb.flagged else ''}"
            for lb in result.labels
        ]
    elif isinstance(result, SegmentationResult):
        lines = [
            f"{s.label:<16} {s.confidence.value:.2f}  {box(s.box)}  {len(s.polygon)} points"
            for s in result.segments
        ]
    elif isinstance(result, PoseResult):
        lines = [
            f"{p.label:<16} {p.confidence.value:.2f}  {box(p.box)}  {len(p.keypoints)} keypoints"
            for p in result.poses
        ]
    if not lines:
        lines = ["(nothing found)"]
    return "\n".join(lines)


def _add_common(p: argparse.ArgumentParser, *, image: bool) -> None:
    if image:
        p.add_argument("image", help="path to an image file (URLs need --allow-urls)")
        p.add_argument("--backend", help="force a backend by name (see `horasis backends`)")
        p.add_argument("--json", action="store_true", help="print the result as JSON")
        p.add_argument("--min-conf", type=float, help="drop results below this confidence")
        p.add_argument("--draw", metavar="OUT", help="also save an annotated copy to OUT")
        p.add_argument("--face-model", help="path to the YuNet .onnx (or set HORASIS_YUNET_MODEL)")
    else:
        p.add_argument("--json", action="store_true", help="print as JSON")
        p.add_argument("--face-model", help=argparse.SUPPRESS)
    p.add_argument("--config", help="YAML config file")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--allow-urls", action="store_true", help="allow http(s) image URLs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horasis", description="computer vision from the shell")
    parser.add_argument("--version", action="version", version=f"horasis {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        _add_common(sub.add_parser(name, help=f"run {name} on an image"), image=True)
    _add_common(sub.add_parser("backends", help="list registered and plugin backends"), image=False)
    models_parser = sub.add_parser("models", help="manage downloadable model files")
    msub = models_parser.add_subparsers(dest="models_command", required=True)
    msub.add_parser("list", help="show known models and whether they are cached")
    for cmd, text in (("pull", "download a model into the cache"), ("path", "print a model's cache path")):
        msub.add_parser(cmd, help=text).add_argument("name")
    msub.add_parser("verify", help="check cached models against their checksums")
    serve = sub.add_parser("serve", help="serve the HTTP API (needs horasis[server])")
    _add_common(serve, image=False)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WEB_CONCURRENCY", "1")),
        help="worker processes (default: $WEB_CONCURRENCY or 1)",
    )
    serve.add_argument(
        "--redis-url",
        default=os.environ.get("HORASIS_REDIS_URL"),
        help="share quotas and rate limits across workers through Redis",
    )
    serve.add_argument(
        "--keys-file",
        required=False,
        help='JSON {"<api key>": {"principal_id": "...", "tier": "pro", "scopes": [...]}}',
    )
    return parser


def _cmd_image(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    client, _ = _build_client(args)
    attr, method = COMMANDS[args.command]
    call = getattr(getattr(client, attr), method)
    result: Result = call(args.image, api_key=CLI_KEY, backend=args.backend)
    if args.min_conf is not None:
        result = result.filter(min_conf=args.min_conf)
    if args.draw:
        result.draw(args.image).save(args.draw)
    out(result.to_json(indent=2) if args.json else format_result(result))
    return 0


def _cmd_models(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    if args.models_command == "list":
        for spec in models.REGISTRY.values():
            state = "cached" if models.is_cached(spec.name) else "not cached"
            out(f"{spec.name:<10} {state:<11} {spec.license}  {spec.description}")
        out(f"cache dir: {models.cache_dir()}")
    elif args.models_command == "pull":
        out(str(models.pull(args.name)))
    elif args.models_command == "path":
        out(str(models.model_path(args.name)))
    else:
        ok = True
        for spec in models.REGISTRY.values():
            if not models.is_cached(spec.name):
                out(f"{spec.name}: not cached")
                continue
            good = models.verify(spec.name)
            ok = ok and good
            out(f"{spec.name}: {'ok' if good else 'CHECKSUM MISMATCH'}")
        return 0 if ok else 1
    return 0


def _cmd_backends(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    client, plugins = _build_client(args)
    plugin_status = dict(plugins)
    rows: list[dict[str, Any]] = []
    for name in client.registry.names():
        backend = client.registry.get(name)
        rows.append(
            {
                "name": name,
                "tasks": sorted(p.value for p in backend.supported_tasks),
                "origin": "plugin" if plugin_status.get(name) == "plugin" else "builtin",
                "loaded": backend.is_loaded(),
            }
        )
    failed = [{"name": n, "error": s} for n, s in plugins if s != "plugin"]
    if args.json:
        out(json.dumps({"backends": rows, "failed_plugins": failed}, indent=2))
        return 0
    for row in rows:
        out(f"{row['name']:<16} {row['origin']:<8} {', '.join(row['tasks'])}")
    for f in failed:
        out(f"{f['name']:<16} plugin   {f['error']}")
    return 0


def warn_if_limiter_not_shared(workers: int, client: Client) -> bool:
    """Log a warning when several workers each keep their own rate-limit buckets."""
    if workers > 1 and not getattr(client.rate_limiter, "shared", False):
        logging.getLogger("horasis").warning(
            "serving with %d workers but the rate limiter is in-process: each worker enforces "
            "its own per-minute buckets, so the effective limit is up to %dx the configured one; "
            "pass --redis-url or set Config.rate_limit_store: redis to share them",
            workers,
            workers,
        )
        return True
    return False


def _serve_client(args: argparse.Namespace) -> Client:
    config = Config.from_yaml(args.config) if args.config else Config.default()
    validator = InMemoryApiKeyValidator()
    if args.keys_file:
        for key, spec in json.loads(Path(args.keys_file).read_text(encoding="utf-8")).items():
            validator.register_simple(
                key,
                principal_id=spec["principal_id"],
                tier=Tier.from_value(spec["tier"]),
                scopes=spec.get("scopes"),
            )
    elif env_key := os.environ.get("HORASIS_API_KEY"):
        validator.register_simple(env_key, principal_id="env", tier=Tier.ENTERPRISE)
    else:
        raise HorasisError("no API keys configured: pass --keys-file or set HORASIS_API_KEY")
    update: dict[str, Any] = {"device": args.device}
    extra: dict[str, Any] = {}
    if args.redis_url:
        extra["quota_store"], extra["rate_limiter"] = redis_backends(url=args.redis_url)
    client = Client.build(
        config=config.model_copy(update=update),
        validator=validator,
        backends=_default_backends(args),
        **extra,
    )
    warn_if_limiter_not_shared(args.workers, client)
    return client


SERVE_ARGV_ENV = "HORASIS_SERVE_ARGV"


def worker_app() -> Any:
    """Build the app in a uvicorn worker from the serve arguments."""
    from horasis.server import create_app

    argv = json.loads(os.environ[SERVE_ARGV_ENV])
    return create_app(_serve_client(build_parser().parse_args(argv)))


def _cmd_serve(args: argparse.Namespace, out: Callable[[str], None], argv: Sequence[str]) -> int:
    try:
        import uvicorn

        from horasis.server import create_app
    except ImportError:
        raise HorasisError("`horasis serve` needs the server extra: pip install 'horasis[server]'") from None
    client = _serve_client(args)
    out(f"serving on http://{args.host}:{args.port}")
    if args.workers > 1:
        os.environ[SERVE_ARGV_ENV] = json.dumps(list(argv))
        uvicorn.run(
            "horasis.cli:worker_app",
            factory=True,
            host=args.host,
            port=args.port,
            workers=args.workers,
        )
    else:
        uvicorn.run(create_app(client), host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the exit code: 0 ok, 1 error, 2 bad usage."""
    args = build_parser().parse_args(argv)
    try:
        if args.command in COMMANDS:
            return _cmd_image(args, print)
        if args.command == "models":
            return _cmd_models(args, print)
        if args.command == "backends":
            return _cmd_backends(args, print)
        return _cmd_serve(args, print, sys.argv[1:] if argv is None else argv)
    except HorasisError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
