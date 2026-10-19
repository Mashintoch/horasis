# Horasis

Horasis is a Python computer vision SDK. It puts object detection,
classification, segmentation, pose estimation, OCR, face detection and content
moderation behind one API, with API keys, tiers, quotas and rate limits built
in. Use it as a library, from the command line, or as a multi-tenant HTTP
service.

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

| Area | What you get |
|---|---|
| Vision | Object detection, classification, instance segmentation, pose estimation, OCR, face detection (no identification) and content moderation |
| Inputs | File paths, bytes, PIL images, numpy arrays, video files and webcams; image URLs if you opt in. Size, format and EXIF orientation are handled once at the API boundary |
| Results | Typed `Result` objects with filtering, drawing, JSON and COCO export, and video annotation |
| Access control | Hashed API keys with expiry, revocation and scopes; tiers with cost-weighted quotas and per-minute rate limits; memory, SQLite or Redis storage; per-key usage reports |
| Runtime | Sync and async clients, batches with per-item errors, frame-skipping video streams, lazy loading, CPU/CUDA/MPS selection, backend routing with fallbacks |
| Integration | A CLI, a FastAPI server, a remote backend so one horasis can call another, lifecycle hooks, OpenTelemetry, and third-party backends via entry points |

## Install

Requires Python 3.10–3.12.

```bash
pip install horasis
```

This installs horasis itself. The models and engines behind each capability
are optional extras, so you only download what you use:

| Extra | Adds |
|---|---|
| `ultralytics` | Detection, classification, segmentation and pose (YOLO) |
| `ocr` | OCR with EasyOCR and Tesseract |
| `face` | Face detection (YuNet) |
| `moderation` | Content moderation (transformers) |
| `onnx` | ONNX Runtime backend for your own models |
| `remote` | Client for a remote horasis server |
| `server` | HTTP server (`horasis serve`) |
| `redis` | Shared quotas and rate limits in Redis |
| `video` | Video and webcam streaming |
| `viz` | Annotated video output |
| `otel` | OpenTelemetry |
| `all` | Everything above |

```bash
pip install "horasis[ultralytics,ocr]"
```

Without the matching extra, a call raises `BackendUnavailableError` naming what
to install.

horasis is Apache-2.0. The `ultralytics` extra installs Ultralytics, which is
AGPL-3.0. If you run it behind a network service, the AGPL applies to that
service unless you have an Ultralytics enterprise licence. The other backends
don't have this restriction.

Tesseract also needs the `tesseract` binary. YuNet needs its `.onnx` model:
run `horasis models pull yunet`.

## Quick start

```python
from horasis import Client, InMemoryApiKeyValidator, Tier

validator = InMemoryApiKeyValidator()
validator.register_simple("demo-key", principal_id="alice", tier=Tier.FREE)

client = Client.build(validator=validator)

result = client.detection.detect("photo.jpg", api_key="demo-key")
for d in result.detections:
    print(d.label, float(d.confidence), d.box)
```

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

Each capability is an attribute of `Client`, such as `client.detection` or
`client.ocr`. `examples/` has a tested script for each: `detect`, `classify`,
`segment`, `pose`, `ocr`, `faces`, `moderate`, `async_batch`, `stream_video`,
`server` and `remote_client`.

```python
# OCR, with routing and a forced backend
client.ocr.extract_text("scan.png", api_key=key)
client.ocr.extract_text("scan.png", api_key=key, backend="tesseract")
```

## Inputs

Every capability accepts a file path (`str`/`Path`), raw `bytes`, a `PIL.Image`,
a numpy array (RGB order; `uint8`, or floats in `0..1`), or an `Image`. Images
are decoded once, at the API boundary:

- Size and format are checked against `Config.inputs` (`max_bytes`, `max_pixels`,
  `allowed_formats`). Bad input raises `ValidationError`.
- EXIF orientation is applied, so every backend sees upright pixels.
- URLs are off by default. Opt in with `Config(inputs=InputPolicy(allow_urls=True))`.

## Results

Every result has:

- `.to_dict()` and `.to_json()` for serialisation
- `.filter(min_conf=..., labels=[...])`, which returns a narrowed result
- `.draw(image)`, which returns an annotated PIL image and saves nothing

Detections also export COCO:

```python
result.to_coco(image_id=1, file_name="a.jpg", width=W, height=H)
```

OCR results have `result.lines`, which merges words into lines using each
`TextRegion`'s `line_index`.

## Access control

```python
from horasis import HashedApiKeyValidator, Tier, Client
from horasis.access import SQLiteQuotaStore   # or RedisQuotaStore (pip install horasis[redis])

validator = HashedApiKeyValidator()           # stores only salted hashes
key = validator.issue(principal_id="alice", tier=Tier.PRO,
                      expires_at=None, scopes=["detect", "ocr"])   # shown once; keep it
client = Client.build(validator=validator, quota_store=SQLiteQuotaStore("quota.db"))

client.usage(key)   # UsageReport: used/remaining overall and per task, seconds until reset
validator.revoke(key)
```

- A tier (`Tier`) sets a caller's quotas and rate limits. Scopes narrow what a
  key may call within its tier.
- Each call consumes `Config.task_cost[task]` quota units: 2 for OCR and
  moderation and 1 for everything else by default. Counters are atomic in every
  store. Add windowed quotas per task with `task_quotas`.
- Rate limits are per-minute token buckets per tier (`TierRule.rate`) and per
  task (`task_rates`). By default they live in-process, so each worker enforces
  its own buckets. To share them, use the Redis limiter (a Lua script on Redis
  server time): `store, limiter = redis_backends(url=...)`, then
  `Client.build(quota_store=store, rate_limiter=limiter)`; or set
  `Config.rate_limit_store: redis` with `redis_url`. `horasis serve --workers N`
  warns when the limiter is in-process, and `--redis-url` shares both.
- `QuotaExceededError` and `RateLimitedError` carry `limit`, `scope` and
  `retry_after` (seconds). Denied calls are refunded.
- `HashedApiKeyValidator` keeps its records in memory; persist them with
  `export_records()`. `InMemoryApiKeyValidator` keeps plaintext keys and is for
  tests and demos.

## Backends and routing

```python
from horasis import Config, Client
from horasis.backend import EasyOCRAdapter, TesseractAdapter, YuNetFaceAdapter

config = Config(
    default_backend="ultralytics",
    backend_order={"ocr": ["easyocr", "tesseract"]},  # first registered match wins
    fallback_order=["remote"],                         # tried after the default
)
client = Client.build(
    config=config,
    backends=[EasyOCRAdapter(), TesseractAdapter(), YuNetFaceAdapter(model_path="yunet.onnx")],
)
```

Each call resolves its backend in this order: an explicit `backend=`, then
`backend_order[task]`, `default_backend`, `fallback_order`, and finally any
registered backend. If nothing supports the task, `NoBackendError` lists what
is registered.

| Backend | Tasks | Notes |
|---|---|---|
| `UltralyticsAdapter` | detect, classify, segment, pose (+ video detect) | Default; weights download on first use |
| `OnnxAdapter` | detect or classify | Letterbox + NMS, CPU/CUDA, labels from a `<model>.labels.txt` / `.json` sidecar |
| `EasyOCRAdapter` / `TesseractAdapter` | ocr | `TextRegion`s with polygon and `line_index` |
| `YuNetFaceAdapter` | face | Boxes, confidence, 5-point landmarks; detection only, no recognition; model from `horasis models pull yunet` or `model_path=` |
| `TransformersModerationAdapter` | moderate | Per-category scores; `flagged` uses `Config.moderation_threshold(s)` |
| `RemoteAdapter` | all | httpx; timeouts, retries with backoff, errors mapped to `HorasisError` types |

## Async, batching, streaming

```python
from horasis import AsyncClient

async with AsyncClient.build(validator=validator) as client:      # blocking backends run on a thread pool
    result = await client.detection.detect("photo.jpg", api_key=key)
    items = await client.detection.detect_batch(paths, api_key=key)   # concurrent
    async for frame in client.detection.stream(0, api_key=key, every_n_frames=5):  # webcam 0
        print(frame.frame_index, len(frame.result.detections))
```

- Batch methods (`detect_batch`, `classify_batch`, `extract_text_batch`,
  `detect_faces_batch`, `moderate_batch`) return one `BatchItem` per input, with
  either a `.result` or an `.error` (a `HorasisError`). Quota and rate limits are
  charged per item, after the item passes input validation; items that fail
  validation cost nothing. An invalid API key aborts the whole batch.
- `client.detection.stream(source, every_n_frames=N, max_frames=...)` yields a
  `FrameResult` per processed frame from a video file or camera index (needs
  `horasis[video]`). Each processed frame is charged as one detection call.
- Models load lazily; `client.warmup()` loads them up front.
  `Config(device="auto"|"cpu"|"cuda"|"mps")` picks the device for backends that
  can choose, and an adapter's own `device=` takes precedence.
  `Config(max_loaded_models=N)` unloads the least recently used backend beyond
  N; it reloads on next use.

## Hooks and telemetry

```python
from horasis.orchestration import LoggingHook, HookEvent, TimingHook

LoggingHook().attach(client.hooks)
timing = TimingHook().attach(client.hooks)      # timing.summary() -> per-task count/mean/max
client.hooks.on(HookEvent.ON_ERROR, lambda **p: print("failed:", p["error"]))
```

Hook events and their payloads are documented in `horasis.orchestration.hooks`.
Hooks only observe. An exception in a before or after hook propagates; one in an
`ON_ERROR` hook is logged and swallowed.

`OTelTelemetry` (extra `horasis[otel]`) emits a span per call and the metrics
`horasis.calls`, `horasis.errors` and `horasis.latency_ms`. Pass it as
`Client.build(telemetry=...)`. It does nothing if OpenTelemetry isn't installed.

## Command line and HTTP server

```bash
horasis detect photo.jpg --json
horasis ocr scan.png --backend tesseract
horasis models pull yunet            # one-time, checksum-verified download into the cache
horasis faces group.jpg --draw faces.png
horasis classify|moderate|segment|pose IMAGE   # same flags: --backend --json --min-conf --draw --device
horasis backends                                # built-in and plugin backends, and plugins that failed to load

HORASIS_API_KEY=change-me horasis serve --port 8000     # needs horasis[server]
curl -H "X-API-Key: change-me" -F image=@photo.jpg -F options='{"conf": 0.4}' localhost:8000/v1/detect
```

The server has three endpoints: `POST /v1/{task}` (task is one of `detect`,
`classify`, `ocr`, `face`, `moderate`, `segment` or `pose`), `GET /health` and
`GET /usage`. Send the key in an `X-API-Key` header. Options are allow-listed
per task. Errors come back as `{"error": {"type", "message", ...}}` with status
401, 403, 404, 413, 422, 429, 501 or 500; a 429 carries `Retry-After`.

`RemoteAdapter` is the matching client, so one horasis instance can use another
as a backend.

## Configuration

`Config` holds everything configurable:

- tiers, quotas, rate limits and task costs
- input policy
- backend order and fallbacks
- device and the model cap
- moderation thresholds

Load it from YAML:

```python
from horasis import Config, Client

config = Config.from_yaml("config.yaml")
client = Client.build(config=config)
```

`src/horasis/config/default.yaml` has the shipped defaults.

## Errors

All errors are `HorasisError` subclasses in `horasis.foundation.errors`. Catch
the base class or a specific one:

| Error | When |
|---|---|
| `InvalidApiKeyError` | Unknown key |
| `ExpiredApiKeyError` / `RevokedApiKeyError` | Key past expiry or revoked |
| `AccessDeniedError` | Tier or scope doesn't allow this task |
| `QuotaExceededError` | Quota window used up (`retry_after` set) |
| `RateLimitedError` | Too many calls per minute (`retry_after` set) |
| `ValidationError` | Input is too large, in the wrong format, or undecodable |
| `NoBackendError` | No registered backend supports the task |
| `BackendUnavailableError` | A backend's dependency or model can't be loaded |
| `RemoteError` | The remote server failed or couldn't be reached |
| `VisualizationUnavailableError` | Annotation requested without the needed extra |

## Writing a backend plugin

See [docs/plugins.md](docs/plugins.md).

```toml
# your package's pyproject.toml
[project.entry-points."horasis.backends"]
my_backend = "my_package.adapter:MyAdapter"
```

```python
from horasis.backend.base import Backend

class MyAdapter(Backend):
    name = "my_backend"
    supported_tasks = frozenset({...})

    def infer(self, task, image, **options):
        ...
```

Then discover and register it:

```python
from horasis.plugins import PluginLoader

PluginLoader().discover(client.registry)
```

## Architecture

There are five layers, and each imports only from the layers below it. A test
enforces this.

| Layer | Name | Contents |
|---|---|---|
| 5 | Public API | `Client` / `AsyncClient` and the capability classes (`ObjectDetector`, `ImageClassifier`, `TextExtractor`, `FaceDetector`, `ContentModerator`, `ObjectSegmenter`, `PoseEstimator`), CLI, server |
| 4 | Access control | `Tier`, `Principal`, `Policy`, `QuotaManager`, rate limits, `@requires_access` |
| 3 | Orchestration | `Pipeline`, `Hooks` (+ `LoggingHook`, `TimingHook`) |
| 2 | Backend abstraction | `Backend` interface, `BackendRegistry` (routing), adapters |
| 1 | Foundation | `Config`, `HorasisError` hierarchy, result schemas (`Result`, `Confidence`, `Task`, `TextRegion`, `Image`), imaging, `Telemetry` |

`PluginLoader` registers third-party `Backend` implementations from the
`horasis.backends` entry-point group.

## Development

```bash
pip install -e ".[dev,server]"
pytest                     # offline; backends are faked
ruff check src tests
mypy
pylint $(git ls-files '*.py')
```

## Status and known limits

Version 1.0.0 (see [CHANGELOG.md](CHANGELOG.md)) supports Python 3.10–3.12.
The offline test suite uses fake backends and never downloads a model or touches
the network. The real-model tests in `tests/integration/real/` (Ultralytics,
ONNX, EasyOCR, Tesseract, YuNet, moderation, remote) run only with
`-m integration`, and each skips itself when its extra, binary or model is
missing. A weekly and manual workflow (`.github/workflows/integration.yml`) runs
them with everything installed.

Known limits:

- Rate limits are per process unless you use `RedisRateLimiter`. Quotas are shared through the SQLite and Redis stores.
- `max_loaded_models` evicts whole backends, not individual models.
- CI runs the real-model tests for Ultralytics (detect, classify, segment, pose), YuNet and the remote path. The ONNX, EasyOCR, Tesseract and moderation tests have not yet run against real weights; smoke-test the ones you depend on.
- `horasis models` manages only the YuNet model (cache dir `HORASIS_CACHE_DIR`, default the platform user cache). Ultralytics, EasyOCR and transformers fetch their own weights; you supply ONNX models. YuNet downloads automatically only with `Config(auto_download_models=True)`.
- `HashedApiKeyValidator` keeps its records in memory. Persist them with `export_records()`.
- Image URL fetching is off by default. When enabled: http(s) only, every resolved address must be globally routable, the connection is pinned to the validated IP, redirects are followed manually (at most 3, each re-validated), and bodies are size-capped and must be `image/*`. `InputPolicy.url_allow_hosts` and `url_deny_hosts` add a hostname policy. Video URLs (`Video.from_url`, `stream`) skip these checks, so accept them only from trusted callers.
- There is no admin API for keys, no object tracking, no result caching and no Docker image.

## License

Apache-2.0