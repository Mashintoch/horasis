# horasis

A layered, plugin-friendly computer vision SDK with API-key based, tiered,
quota-aware access control.

`horasis` (Greek "vision, seeing") wraps object detection, image
classification, OCR, face detection, and content moderation behind one
small, stable public API — while keeping backend, orchestration, and
access-control concerns cleanly separated so the SDK can grow new
backends and capabilities without breaking callers.

## Architecture

Five layers, each depending only on the ones below it:

| Layer | Name | Responsibility |
|---|---|---|
| 5 | Public API | `Skopos` client + capability classes (`ObjectDetector`, `ImageClassifier`, `TextExtractor`, `FaceDetector`, `ContentModerator`) |
| 4 | Access control | `Klimax` tiers, `Prosopon` identity, `Nomos` policy, `Metron` quotas, `@limen` decorator |
| 3 | Orchestration | `Organon` pipeline, `Kairos` hooks |
| 2 | Backend abstraction | `Mechane` interface, `Taxis` registry, adapters (Ultralytics, ONNX Runtime, EasyOCR, Tesseract, YuNet faces, transformers moderation, remote HTTP) |
| 1 | Foundation | `Kanon` config, `Sphalma` exceptions, result schemas (`Theoria`, `Doxa`, `Praxis`, `Glyph`, `Physis`), `Aisthesis` telemetry |

Plugins (`Prosthesis`) register additional `Mechane` backends via the
`horasis.backends` Python entry-point group, independent of the core layers.

## Install

```bash
pip install horasis[ultralytics]   # detection + classification
pip install horasis[ocr]           # EasyOCR + pytesseract
pip install horasis[face]          # OpenCV (YuNet face detection)
pip install horasis[moderation]    # transformers image classifier
pip install horasis[onnx]          # onnxruntime
pip install horasis[remote]        # httpx client for a remote horasis server
pip install horasis[all]
```

## Quick start

```python
from horasis import Skopos, InMemoryApiKeyValidator, Klimax

validator = InMemoryApiKeyValidator()
validator.register_simple("demo-key", principal_id="alice", tier=Klimax.FREE)

client = Skopos.build(validator=validator)

result = client.detection.detect("photo.jpg", api_key="demo-key")
for d in result.detections:
    print(d.label, float(d.confidence), d.box)
```

Each capability always returns a `Theoria` subclass -- never a raw backend object.

## Inputs

Every capability accepts a file path (`str`/`Path`), raw `bytes`, a `PIL.Image`,
a numpy array (RGB order; `uint8`, or floats in `0..1`), or a pre-built `Physis`.
Images are decoded **once**, at the API boundary:

- size and format are validated (`Kanon.inputs`: `max_bytes`, `max_pixels`,
  `allowed_formats`) and bad input raises `ValidationError`;
- EXIF orientation is applied, so every backend sees upright pixels;
- **URLs are off by default.** Opt in with `Kanon(inputs=InputPolicy(allow_urls=True))`.
  If you do this for untrusted callers, also block requests to internal networks.

## Backends and routing

```python
from horasis import Kanon, Skopos
from horasis.backend import EasyOCRAdapter, TesseractAdapter, YuNetFaceAdapter

kanon = Kanon(
    default_backend="ultralytics",
    backend_order={"ocr": ["easyocr", "tesseract"]},  # first registered match wins
    fallback_order=["remote"],                         # tried after the default
)
client = Skopos.build(
    kanon=kanon,
    backends=[EasyOCRAdapter(), TesseractAdapter(), YuNetFaceAdapter(model_path="yunet.onnx")],
)
client.ocr.extract_text("scan.png", api_key="...")            # easyocr, else tesseract
client.ocr.extract_text("scan.png", api_key="...", backend="tesseract")  # forced
```

Resolution: explicit `backend=` -> `backend_order[praxis]` -> `default_backend` ->
`fallback_order` -> any registered backend. If nothing supports the praxis,
`NoBackendError` lists what is registered.

| Backend | Praxeis | Notes |
|---|---|---|
| `UltralyticsAdapter` | detect, classify (+ video detect) | default; weights download on first use |
| `OnnxAdapter` | detect *or* classify | letterbox + NMS, CPU/CUDA, labels from a `<model>.labels.txt` / `.json` sidecar |
| `EasyOCRAdapter` / `TesseractAdapter` | ocr | `Glyph`s with polygon and `line_index`; `result.lines` merges words into lines |
| `YuNetFaceAdapter` | face | boxes, confidence, 5-point landmarks; **detection only, no recognition**; you supply the YuNet `.onnx` |
| `TransformersModerationAdapter` | moderate | per-category scores; `flagged` uses `Kanon.moderation_threshold(s)` |
| `RemoteAdapter` | all | httpx; timeouts, retried with backoff, errors mapped to `Sphalma` types |

## Configuration

Tiers and quotas live in a `Kanon`, loadable from YAML:

```python
from horasis import Kanon

kanon = Kanon.from_yaml("kanon.yaml")
client = Skopos.build(kanon=kanon)
```

See `src/horasis/config/default.yaml` for the shipped defaults.

## Writing a backend plugin

```toml
# your package's pyproject.toml
[project.entry-points."horasis.backends"]
my_backend = "my_package.adapter:MyAdapter"
```

```python
from horasis.backend.mechane import Mechane

class MyAdapter(Mechane):
    name = "my_backend"
    supported_praxeis = frozenset({...})

    def infer(self, praxis, image, **options):
        ...
```

Then discover and register it:

```python
from horasis.plugins import Prosthesis

Prosthesis().discover(client.taxis)
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
mypy
```

## Status

Version 1.0.0. Implemented and unit-tested with fake backends (no model
downloads or network in the test suite): detection, classification, OCR, face
detection, content moderation, ONNX and remote adapters, input validation,
config-driven backend routing. Real-model tests are marked `integration` and
skipped by default (`pytest -m integration`).

Not yet done: persistent/shared quota stores, per-minute rate limits, key
hashing/expiry/scopes, async API, batching, streaming, a CLI and an HTTP
server (the remote adapter speaks the protocol documented in
`horasis/backend/remote_adapter.py`; no server ships yet).

## License

Apache-2.0



