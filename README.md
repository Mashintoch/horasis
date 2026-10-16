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

## Access control

```python
from horasis import HashedApiKeyValidator, Klimax, Skopos
from horasis.access import SQLiteQuotaStore   # or RedisQuotaStore (pip install horasis[redis])

validator = HashedApiKeyValidator()           # stores only salted hashes, compares in constant time
key = validator.issue(principal_id="alice", tier=Klimax.PRO,
                      expires_at=None, scopes=["detect", "ocr"])   # shown once; keep it
client = Skopos.build(validator=validator, quota_store=SQLiteQuotaStore("quota.db"))

client.usage(key)   # UsageReport: used/remaining overall and per praxis, plus seconds until reset
validator.revoke(key)
```

- **Cost-weighted quotas:** each call consumes `Kanon.praxis_cost[praxis]` units
  (default: OCR and moderation 2, the rest 1). Counters are atomic in every store.
- **Rate limits:** per-minute token buckets per tier (`TierRule.rate`) and per praxis
  (`praxis_rates`); in-process, so each worker enforces its own bucket. Extra
  windowed quotas per praxis via `praxis_quotas`.
- **Errors:** `QuotaExceededError` and `RateLimitedError` carry `limit`, `scope`
  and `retry_after` (seconds). Denied calls are refunded.
- **Scopes** narrow what a key may call within its tier. `InMemoryApiKeyValidator`
  keeps plaintext keys and is for tests and demos only.

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

## Async, batching, streaming

```python
from horasis import AsyncSkopos

async with AsyncSkopos.build(validator=validator) as client:      # blocking backends run on a thread pool
    result = await client.detection.detect("photo.jpg", api_key=key)
    items = await client.detection.detect_batch(paths, api_key=key)   # concurrent
    async for frame in client.detection.stream(0, api_key=key, every_n_frames=5):  # webcam 0
        print(frame.frame_index, len(frame.result.detections))
```

- **Batches** (`detect_batch`, `classify_batch`, `extract_text_batch`, `detect_faces_batch`,
  `moderate_batch`) return one `BatchItem` per input: `.result` or `.error` (a `Sphalma`),
  never both. Quota and rate limits are charged **per item** -- including items that then
  fail validation. An invalid API key aborts the whole batch.
- **Streams** (`client.detection.stream(source, every_n_frames=N, max_frames=...)`) yield
  a `FrameResult` per processed frame from a video file or camera index (needs
  `horasis[video]`). Each processed frame is charged as one detection call.
- **Lifecycle:** models load lazily. `client.warmup()` loads them up front;
  `Kanon(device="auto"|"cpu"|"cuda"|"mps")` picks the device for backends that can choose
  (an adapter's own `device=` wins); `Kanon(max_loaded_models=N)` unloads the least
  recently used backend beyond N (it reloads on next use; granularity is per backend).

## Results

Every result has `.to_dict()`, `.to_json()`, `.filter(min_conf=..., labels=[...])` and
`.draw(image)` (returns an annotated PIL image; saves nothing). Detections also export COCO:
`result.to_coco(image_id=1, file_name="a.jpg", width=W, height=H)`.

## Hooks and telemetry

```python
from horasis.orchestration import LoggingHook, Moment, TimingHook

LoggingHook().attach(client.kairos)
timing = TimingHook().attach(client.kairos)      # timing.summary() -> per-praxis count/mean/max
client.kairos.on(Moment.ON_ERROR, lambda **p: print("failed:", p["error"]))
```

Hook moments and their keyword payloads are documented in `horasis.orchestration.kairos`.
Hooks observe only; an exception in a before/after hook propagates, one in an `ON_ERROR`
hook is logged and swallowed.

`OpenTelemetryAisthesis` (extra `horasis[otel]`) emits a span per call plus `horasis.calls`,
`horasis.errors` and `horasis.latency_ms`; pass it as `Skopos.build(aisthesis=...)`. It is a
no-op if OpenTelemetry is not installed.

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

Not yet done: shared (cross-process) rate limits, async API, batching, streaming, a CLI and an HTTP
server (the remote adapter speaks the protocol documented in
`horasis/backend/remote_adapter.py`; no server ships yet).

## License

Apache-2.0



