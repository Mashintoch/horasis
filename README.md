# horasis

A layered, plugin-friendly computer vision SDK with API-key based, tiered,
quota-aware access control.

`horasis` (Greek "vision, seeing") puts object detection, image
classification, segmentation, pose estimation, OCR, face detection and
content moderation behind one small, stable public API. Backend,
orchestration and access-control concerns are kept cleanly separated, so the
SDK can gain new backends and capabilities without breaking callers. You can
use it as a library in a script, as a command-line tool, or as a metered,
multi-tenant HTTP vision service.

## Contents

- [What horasis can do](#what-horasis-can-do)
- [Install](#install)
- [Quick start](#quick-start)
- [Capabilities](#capabilities)
- [Inputs](#inputs)
- [Results](#results)
- [Access control](#access-control)
- [Backends and routing](#backends-and-routing)
- [Async, batching, streaming](#async-batching-streaming)
- [Hooks and telemetry](#hooks-and-telemetry)
- [Command line and HTTP server](#command-line-and-http-server)
- [Configuration](#configuration)
- [Errors](#errors)
- [Writing a backend plugin](#writing-a-backend-plugin)
- [Architecture](#architecture)
- [Development](#development)
- [Status and known limits](#status-and-known-limits)

## What horasis can do

**See**
- Detect objects in an image or video, with labels, confidence and boxes
- Classify images, returning top-k labels
- Segment instances, with a mask per object
- Estimate human pose, with keypoints per person
- Read text (OCR), returning words with polygons and confidence, merged into lines
- Detect faces, with boxes, confidence and 5-point landmarks. Detection only, never identification.
- Moderate content, returning per-category scores and a `flagged` decision from configurable thresholds

**Accept any reasonable input**
- File paths, raw bytes, PIL images, numpy arrays or pre-built `Physis` objects
- Image URLs, if you opt in
- Video files and webcams
- Inputs are validated for size, pixel count and format, and corrected for EXIF orientation, once at the API boundary

**Return typed, useful results**
- Every call returns a `Theoria` result, never a raw backend object
- Filter by confidence or label, draw annotations, and serialise to dict or JSON
- Export detections as COCO
- Annotate video output (`viz` extra)

**Control and meter access**
- API keys stored as salted hashes, compared in constant time
- Key expiry, revocation and scopes
- Tiers with quotas, plus per-praxis cost weights
- Per-minute rate limits per tier and per praxis
- Quotas in memory, SQLite or Redis, with atomic counters
- Usage reports per key, and `retry_after` on every limit error

**Run fast and flexibly**
- Sync and async clients
- Concurrent batches with per-item errors
- Frame-skipping video streams
- Lazy model loading, warmup, CPU/CUDA/MPS selection and an LRU cap on loaded backends
- Config-driven backend routing with fallbacks, or a backend forced per call

**Plug in anywhere**
- A CLI for every capability
- A FastAPI server with API-key auth
- A remote backend, so one horasis can call another
- Lifecycle hooks, and OpenTelemetry traces and metrics
- Third-party backends via Python entry points

## Install

Requires Python 3.10–3.12. Install only what you need:

```bash
pip install horasis[ultralytics]   # detection, classification, segmentation, pose (YOLO)
pip install horasis[ocr]           # EasyOCR + pytesseract
pip install horasis[face]          # OpenCV (YuNet face detection)
pip install horasis[moderation]    # transformers image classifier
pip install horasis[onnx]          # onnxruntime
pip install horasis[remote]        # httpx client for a remote horasis server
pip install horasis[redis]         # Redis-backed quota store
pip install horasis[video]         # OpenCV, for detection.stream()
pip install horasis[viz]           # video annotation output (imageio)
pip install horasis[otel]          # OpenTelemetry spans and metrics
pip install horasis[server]        # FastAPI server (`horasis serve`)
pip install horasis[all]           # everything
```

Tesseract also needs the `tesseract` binary installed on the system. YuNet
needs its `.onnx` model file, which isn't downloaded for you.

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

Each capability always returns a `Theoria` subclass, never a raw backend object.

## Capabilities

| Capability | Class | Batch method | Result | CLI | Backends |
|---|---|---|---|---|---|
| Object detection | `ObjectDetector` | `detect_batch` | `DetectionResult` | `horasis detect` | Ultralytics, ONNX, remote |
| Video detection | `ObjectDetector.stream` | n/a | `FrameResult` / `VideoDetectionResult` | n/a | Ultralytics, any detect backend |
| Classification | `ImageClassifier` | `classify_batch` | `ClassificationResult` | `horasis classify` | Ultralytics, ONNX, remote |
| Segmentation | `ObjectSegmenter` | n/a | `SegmentationResult` | `horasis segment` | Ultralytics, remote |
| Pose estimation | `PoseEstimator` | n/a | `PoseResult` | `horasis pose` | Ultralytics, remote |
| OCR | `TextExtractor` | `extract_text_batch` | `OcrResult` | `horasis ocr` | EasyOCR, Tesseract, remote |
| Face detection | `FaceDetector` | `detect_faces_batch` | `FaceResult` | `horasis faces` | YuNet, remote |
| Moderation | `ContentModerator` | `moderate_batch` | `ModerationResult` | `horasis moderate` | transformers, remote |

Each capability is reached through the `Skopos` client (for example
`client.detection` and `client.ocr`). The `examples/` folder has a runnable,
tested script for every capability: `detect`, `classify`, `segment`, `pose`,
`ocr`, `faces`, `moderate`, `async_batch`, `stream_video`, `server` and
`remote_client`.

```python
# OCR, with routing and a forced backend
client.ocr.extract_text("scan.png", api_key=key)
client.ocr.extract_text("scan.png", api_key=key, backend="tesseract")
```

## Inputs

Every capability accepts a file path (`str`/`Path`), raw `bytes`, a `PIL.Image`,
a numpy array (RGB order; `uint8`, or floats in `0..1`), or a pre-built `Physis`.
Images are decoded **once**, at the API boundary:

- Size and format are validated against `Kanon.inputs` (`max_bytes`, `max_pixels`,
  `allowed_formats`). Bad input raises `ValidationError`.
- EXIF orientation is applied, so every backend sees upright pixels.
- **URLs are off by default.** Opt in with `Kanon(inputs=InputPolicy(allow_urls=True))`.
  If you do this for untrusted callers, also block requests to internal networks.

## Results

Every result has:

- `.to_dict()` and `.to_json()` for serialisation
- `.filter(min_conf=..., labels=[...])`, which returns a narrowed result
- `.draw(image)`, which returns an annotated PIL image and saves nothing

Detections also export COCO:

```python
result.to_coco(image_id=1, file_name="a.jpg", width=W, height=H)
```

OCR results expose `result.lines`, which merges words into lines using each
`Glyph`'s `line_index`.

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

- **Tiers (`Klimax`)** set each caller's quotas and rate limits.
- **Cost-weighted quotas:** each call consumes `Kanon.praxis_cost[praxis]` units
  (by default, 2 for OCR and moderation and 1 for everything else). Counters are
  atomic in every store.
- **Rate limits:** there are per-minute token buckets per tier (`TierRule.rate`)
  and per praxis (`praxis_rates`). By default they live in-process, so each
  worker enforces its own bucket. To share them across workers, use the Redis
  limiter (atomic Lua script on Redis server time, keyed per principal + scope):
  `store, limiter = redis_backends(url=...)` then
  `Skopos.build(quota_store=store, rate_limiter=limiter)`, or set
  `Kanon.rate_limit_store: redis` with `redis_url`. `horasis serve --workers N`
  warns when the limiter is in-process; `--redis-url` shares quotas and limits.
  You can add windowed quotas per praxis with `praxis_quotas`.
- **Errors:** `QuotaExceededError` and `RateLimitedError` carry `limit`, `scope`
  and `retry_after` (seconds). Denied calls are refunded.
- **Scopes** narrow what a key may call within its tier.
- **Persistence:** `HashedApiKeyValidator` keeps its records in memory. Persist them
  with `export_records()`. `InMemoryApiKeyValidator` keeps plaintext keys and is
  for tests and demos only.

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
```

The backend for each call is resolved in this order: an explicit `backend=`,
then `backend_order[praxis]`, then `default_backend`, then `fallback_order`,
then any registered backend. If nothing supports the praxis, `NoBackendError`
lists what is registered.

| Backend | Praxeis | Notes |
|---|---|---|
| `UltralyticsAdapter` | detect, classify, segment, pose (+ video detect) | Default; weights download on first use |
| `OnnxAdapter` | detect *or* classify | Letterbox + NMS, CPU/CUDA, labels from a `<model>.labels.txt` / `.json` sidecar |
| `EasyOCRAdapter` / `TesseractAdapter` | ocr | `Glyph`s with polygon and `line_index` |
| `YuNetFaceAdapter` | face | Boxes, confidence, 5-point landmarks; **detection only, no recognition**; you supply the YuNet `.onnx` |
| `TransformersModerationAdapter` | moderate | Per-category scores; `flagged` uses `Kanon.moderation_threshold(s)` |
| `RemoteAdapter` | all | httpx; timeouts, retries with backoff, errors mapped to `Sphalma` types |

## Async, batching, streaming

```python
from horasis import AsyncSkopos

async with AsyncSkopos.build(validator=validator) as client:      # blocking backends run on a thread pool
    result = await client.detection.detect("photo.jpg", api_key=key)
    items = await client.detection.detect_batch(paths, api_key=key)   # concurrent
    async for frame in client.detection.stream(0, api_key=key, every_n_frames=5):  # webcam 0
        print(frame.frame_index, len(frame.result.detections))
```

- **Batches** (`detect_batch`, `classify_batch`, `extract_text_batch`,
  `detect_faces_batch`, `moderate_batch`) return one `BatchItem` per input.
  Each item has either a `.result` or an `.error` (a `Sphalma`), never both.
  Quota and rate limits are charged **per item**, but only after the item passes
  input validation (size, pixels, format, URL policy); items that fail validation
  get a `BatchItem.error` and cost nothing. An invalid API key aborts the whole batch.
- **Streams** (`client.detection.stream(source, every_n_frames=N, max_frames=...)`)
  yield a `FrameResult` per processed frame from a video file or camera index
  (needs `horasis[video]`). Each processed frame is charged as one detection call.
- **Lifecycle:**
  - Models load lazily, and `client.warmup()` loads them up front.
  - `Kanon(device="auto"|"cpu"|"cuda"|"mps")` picks the device for backends that
    can choose. An adapter's own `device=` takes precedence.
  - `Kanon(max_loaded_models=N)` unloads the least recently used backend beyond
    N. It reloads on next use. Granularity is per backend.

## Hooks and telemetry

```python
from horasis.orchestration import LoggingHook, Moment, TimingHook

LoggingHook().attach(client.kairos)
timing = TimingHook().attach(client.kairos)      # timing.summary() -> per-praxis count/mean/max
client.kairos.on(Moment.ON_ERROR, lambda **p: print("failed:", p["error"]))
```

Hook moments and their keyword payloads are documented in
`horasis.orchestration.kairos`. Hooks only observe. An exception in a
before/after hook propagates; one in an `ON_ERROR` hook is logged and swallowed.

`OpenTelemetryAisthesis` (extra `horasis[otel]`) emits a span per call, plus the
metrics `horasis.calls`, `horasis.errors` and `horasis.latency_ms`. Pass it as
`Skopos.build(aisthesis=...)`. It does nothing if OpenTelemetry isn't installed.

## Command line and HTTP server

```bash
horasis detect photo.jpg --json
horasis ocr scan.png --backend tesseract
horasis faces group.jpg --face-model yunet.onnx --draw faces.png
horasis classify|moderate|segment|pose IMAGE   # same flags: --backend --json --min-conf --draw --device
horasis backends                                # built-in and plugin backends (and plugins that failed to load)

HORASIS_API_KEY=change-me horasis serve --port 8000     # needs horasis[server]
curl -H "X-API-Key: change-me" -F image=@photo.jpg -F options='{"conf": 0.4}' localhost:8000/v1/detect
```

**Endpoints:**
- `POST /v1/{praxis}`, where praxis is one of `detect`, `classify`, `ocr`, `face`, `moderate`, `segment` or `pose`
- `GET /health`
- `GET /usage`

**Authentication:** an `X-API-Key` header.

**Errors** come back as `{"error": {"type", "message", ...}}` with status
401, 403, 404, 413, 422, 429, 501 or 500. A 429 carries `Retry-After`.

**Options** are whitelisted per praxis.

`RemoteAdapter` is the matching client, so one horasis instance can use another
as a backend.

## Configuration

Everything configurable lives in a `Kanon`:

- tiers, quotas, rate limits and praxis costs
- input policy
- backend order and fallbacks
- device and the model cap
- moderation thresholds

It can be loaded from YAML:

```python
from horasis import Kanon, Skopos

kanon = Kanon.from_yaml("kanon.yaml")
client = Skopos.build(kanon=kanon)
```

See `src/horasis/config/default.yaml` for the shipped defaults.

## Errors

All errors are `Sphalma` subclasses in `horasis.foundation.sphalma`, so you can
catch the base class or handle specific cases:

| Error | When |
|---|---|
| `InvalidApiKeyError` | Unknown key |
| `ExpiredApiKeyError` / `RevokedApiKeyError` | Key past expiry or revoked |
| `AccessDeniedError` | Tier or scope doesn't allow this praxis |
| `QuotaExceededError` | Quota window used up (`retry_after` set) |
| `RateLimitedError` | Too many calls per minute (`retry_after` set) |
| `ValidationError` | Input is too large, in the wrong format, or undecodable |
| `NoBackendError` | No registered backend supports the praxis |
| `BackendUnavailableError` | A backend's dependency or model can't be loaded |
| `RemoteError` | The remote server failed or couldn't be reached |
| `VisualizationUnavailableError` | Annotation requested without the needed extra |

## Writing a backend plugin

See [docs/plugins.md](docs/plugins.md) for the full guide.

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

## Architecture

There are five layers, and each depends only on the layers below it. A test
enforces this.

| Layer | Name | Responsibility |
|---|---|---|
| 5 | Public API | `Skopos` / `AsyncSkopos` clients + capability classes (`ObjectDetector`, `ImageClassifier`, `TextExtractor`, `FaceDetector`, `ContentModerator`, `ObjectSegmenter`, `PoseEstimator`), CLI, server |
| 4 | Access control | `Klimax` tiers, `Prosopon` identity, `Nomos` policy, `Metron` quotas, rate limits, `@limen` decorator |
| 3 | Orchestration | `Organon` pipeline, `Kairos` hooks (+ `LoggingHook`, `TimingHook`) |
| 2 | Backend abstraction | `Mechane` interface, `Taxis` registry and routing, adapters |
| 1 | Foundation | `Kanon` config, `Sphalma` exceptions, result schemas (`Theoria`, `Doxa`, `Praxis`, `Glyph`, `Physis`), imaging, `Aisthesis` telemetry |

Plugins (`Prosthesis`) register additional `Mechane` backends via the
`horasis.backends` entry-point group, independently of the core layers.

## Development

```bash
pip install -e ".[dev,server]"
pytest                     # offline; backends are faked
ruff check src tests
mypy
pylint $(git ls-files '*.py')
```

## Status and known limits

Version 1.1.0 (see [CHANGELOG.md](CHANGELOG.md)) supports Python 3.10–3.12.
Everything above is implemented and covered by offline tests that use fake
backends. The suite never downloads a model or touches the network. The
`integration` pytest marker is reserved for real-model tests, which are skipped
unless you pass `-m integration`. None ship yet.

Known limits:

- Rate limits (token buckets) are per process unless you use `RedisRateLimiter`. Quotas are shared via the SQLite and Redis stores.
- `max_loaded_models` evicts whole backends, not individual models.
- Ultralytics, Tesseract, EasyOCR, YuNet, the moderation model and onnxruntime have only been exercised through test doubles, never against the real engines or weights. Smoke-test each one you depend on.
- The YuNet model file isn't downloaded for you.
- `HashedApiKeyValidator` keeps its records in memory. Persist them with `export_records()`.
- Image URL fetching is off by default. When enabled it is SSRF-hardened: http(s) only, every resolved address must be globally routable, the connection is pinned to the validated IP (no DNS rebinding), redirects are followed manually (max 3, re-validated), bodies are streamed with a size cap and must be `image/*`. `InputPolicy.url_allow_hosts` / `url_deny_hosts` add a hostname policy. Video URLs (`Kinesis.from_url`, `stream`) use a simpler fetch without these checks, so keep them for trusted callers.
- There's no admin API for keys, no object tracking, no result caching and no Docker image yet. See [docs/guide.md](docs/guide.md) for the roadmap.

## License

Apache-2.0