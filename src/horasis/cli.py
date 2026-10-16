"""Command-line interface: ``horasis detect photo.jpg --json``.

Commands::

    horasis detect|classify|ocr|faces|moderate|segment|pose IMAGE [options]
    horasis backends [--json]
    horasis serve [--host H] [--port P] [--keys-file FILE]   (needs horasis[server])

The CLI is a local, single-user tool: it runs as an internal enterprise-tier
principal, so no API key is needed for the image commands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from horasis import __version__
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.klimax import Klimax
from horasis.api.skopos import Skopos
from horasis.backend.easyocr_adapter import EasyOCRAdapter
from horasis.backend.face_adapter import YuNetFaceAdapter
from horasis.backend.mechane import Mechane
from horasis.backend.moderation_adapter import TransformersModerationAdapter
from horasis.backend.tesseract_adapter import TesseractAdapter
from horasis.foundation.kanon import InputPolicy, Kanon
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    FaceResult,
    ModerationResult,
    OcrResult,
    PoseResult,
    SegmentationResult,
    Theoria,
)
from horasis.foundation.sphalma import PluginError, Sphalma
from horasis.plugins.prosthesis import Prosthesis

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


def _default_backends(args: argparse.Namespace) -> list[Mechane]:
    """Optional backends registered next to Ultralytics (all load lazily)."""
    return [
        EasyOCRAdapter(),
        TesseractAdapter(),
        YuNetFaceAdapter(model_path=getattr(args, "face_model", None)),
        TransformersModerationAdapter(),
    ]


def _build_client(args: argparse.Namespace) -> tuple[Skopos, list[tuple[str, str]]]:
    """Create the client; also returns ``(plugin_name, status)`` for each plugin found."""
    kanon = Kanon.from_yaml(args.config) if args.config else Kanon.default()
    update: dict[str, Any] = {"device": args.device}
    if args.allow_urls:
        update["inputs"] = InputPolicy(allow_urls=True)
    if not kanon.backend_order:
        update["backend_order"] = {"ocr": ["easyocr", "tesseract"]}
    kanon = kanon.model_copy(update=update)

    validator = InMemoryApiKeyValidator()
    validator.register_simple(CLI_KEY, principal_id="cli", tier=Klimax.ENTERPRISE)
    client = Skopos.build(kanon=kanon, validator=validator, backends=_default_backends(args))

    plugins: list[tuple[str, str]] = []
    prosthesis = Prosthesis()
    for ep in prosthesis.discover_entry_points():
        try:
            backend = prosthesis.load(ep)
        except PluginError as exc:
            plugins.append((ep.name, f"failed to load: {exc.message}"))
            continue
        client.taxis.register(backend)
        plugins.append((backend.name, "plugin"))
    return client, plugins


def format_result(result: Theoria) -> str:
    """Human-readable rendering of any result type."""

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
    p.add_argument("--config", help="Kanon YAML config file")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--allow-urls", action="store_true", help="allow http(s) image URLs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horasis", description="Computer vision from the shell.")
    parser.add_argument("--version", action="version", version=f"horasis {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        _add_common(sub.add_parser(name, help=f"run {name} on an image"), image=True)
    _add_common(sub.add_parser("backends", help="list registered and plugin backends"), image=False)
    serve = sub.add_parser("serve", help="serve the HTTP API (needs horasis[server])")
    _add_common(serve, image=False)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
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
    result: Theoria = call(args.image, api_key=CLI_KEY, backend=args.backend)
    if args.min_conf is not None:
        result = result.filter(min_conf=args.min_conf)
    if args.draw:
        result.draw(args.image).save(args.draw)
    out(result.to_json(indent=2) if args.json else format_result(result))
    return 0


def _cmd_backends(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    client, plugins = _build_client(args)
    plugin_status = dict(plugins)
    rows: list[dict[str, Any]] = []
    for name in client.taxis.names():
        backend = client.taxis.get(name)
        rows.append(
            {
                "name": name,
                "praxeis": sorted(p.value for p in backend.supported_praxeis),
                "origin": "plugin" if plugin_status.get(name) == "plugin" else "builtin",
                "loaded": backend.is_loaded(),
            }
        )
    failed = [{"name": n, "error": s} for n, s in plugins if s != "plugin"]
    if args.json:
        out(json.dumps({"backends": rows, "failed_plugins": failed}, indent=2))
        return 0
    for row in rows:
        out(f"{row['name']:<16} {row['origin']:<8} {', '.join(row['praxeis'])}")
    for f in failed:
        out(f"{f['name']:<16} plugin   {f['error']}")
    return 0


def _cmd_serve(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    try:
        import uvicorn

        from horasis.server import create_app
    except ImportError:
        raise Sphalma("`horasis serve` needs: pip install 'horasis[server]'") from None
    kanon = Kanon.from_yaml(args.config) if args.config else Kanon.default()
    validator = InMemoryApiKeyValidator()
    if args.keys_file:
        for key, spec in json.loads(Path(args.keys_file).read_text(encoding="utf-8")).items():
            validator.register_simple(
                key,
                principal_id=spec["principal_id"],
                tier=Klimax.from_value(spec["tier"]),
                scopes=spec.get("scopes"),
            )
    elif env_key := os.environ.get("HORASIS_API_KEY"):
        validator.register_simple(env_key, principal_id="env", tier=Klimax.ENTERPRISE)
    else:
        raise Sphalma("No API keys configured: pass --keys-file or set HORASIS_API_KEY")
    client = Skopos.build(
        kanon=kanon.model_copy(update={"device": args.device}),
        validator=validator,
        backends=_default_backends(args),
    )
    out(f"serving on http://{args.host}:{args.port}")
    uvicorn.run(create_app(client), host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code (0 ok, 1 error, 2 bad usage)."""
    args = build_parser().parse_args(argv)
    try:
        if args.command in COMMANDS:
            return _cmd_image(args, print)
        if args.command == "backends":
            return _cmd_backends(args, print)
        return _cmd_serve(args, print)
    except Sphalma as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
