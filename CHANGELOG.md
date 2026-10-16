# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0]

### Added
- **Backends:** `EasyOCRAdapter` and `TesseractAdapter` (OCR with polygons and
  line grouping), `YuNetFaceAdapter` (face boxes + 5-point landmarks, detection
  only), `TransformersModerationAdapter`, a real `OnnxAdapter` (detect/classify,
  letterbox, NMS, CUDA/CPU, label sidecars) and a real `RemoteAdapter` (httpx,
  timeouts, retries with backoff, error mapping).
- **New praxeis:** segmentation (`client.segmentation.segment`) and pose
  (`client.pose.estimate`) on the Ultralytics backend.
- **Routing:** `Kanon.backend_order` / `fallback_order`; `NoBackendError`.
- **Inputs:** PIL images and numpy arrays; size/pixel/format validation; EXIF
  orientation; `Kanon.inputs` (`InputPolicy`).
- **Access control:** `SQLiteQuotaStore`, `RedisQuotaStore`, atomic weighted
  quota consumption, per-minute token-bucket rate limits (per tier and per
  praxis), per-praxis quotas, `HashedApiKeyValidator` (salted hashes, expiry,
  revocation, scopes), `client.usage(api_key)`, `retry_after` on quota and
  rate errors (`RateLimitedError`).
- **Throughput and DX:** `AsyncSkopos`; `*_batch` methods with per-item results
  and errors; `detection.stream()` for video files and cameras; `client.warmup()`;
  `Kanon.device`; `Kanon.max_loaded_models` (LRU unloading).
- **Results:** `.to_dict()`, `.to_json()`, `.filter()`, `.draw()`, COCO export.
- **Hooks and telemetry:** `Moment.ON_ERROR`, `LoggingHook`, `TimingHook`,
  `OpenTelemetryAisthesis` (no-op without OpenTelemetry).
- **Surfaces:** `horasis` CLI, FastAPI server (`horasis.server`, `horasis serve`).
- Packaging: Python 3.10-3.12, `py.typed`, extras `ocr`, `face`, `moderation`,
  `redis`, `video`, `otel`, `server`; CI workflow; `docs/plugins.md`; examples.

### Changed (behaviour)
- **Image URLs are no longer fetched implicitly.** Opt in with
  `Kanon(inputs=InputPolicy(allow_urls=True))`. `Physis.from_path()` rejects URLs.
- Inputs are decoded and validated at the API boundary; undecodable bytes now
  raise `ValidationError` instead of failing inside the backend.
- Calls are charged `Kanon.praxis_cost[praxis]` units (OCR, moderation,
  segmentation and pose cost 2) instead of always 1.
- Rejected calls are refunded, so they no longer consume quota.
- `QuotaStore.increment_and_get` gained an `amount` argument.
- `FaceResult.faces` holds `Face` objects (a `Detection` with `landmarks`).
- `Glyph` gained `polygon`, `line_index`, `level`; `OcrResult.text` joins lines
  with newlines when line indices are present.

### Fixed
- Duplicate `FaceEstimate` class and dead duplicate methods in the Ultralytics adapter.
- `horasis.__version__` disagreed with the packaged version.
- mypy configuration crashed on numpy stubs; the strict type check now passes.

## [1.0.0]
- Initial layered SDK: detection and classification via Ultralytics, tiered
  access control, plugins.
