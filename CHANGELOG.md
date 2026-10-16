# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0]

### Breaking changes
- **URL inputs are opt-in.** Strings that look like URLs are only fetched when
  `Kanon.inputs.allow_urls` is True; otherwise they raise `ValidationError`.
- **Undecodable bytes raise `ValidationError`** (not a backend error), including
  oversize, wrong-format and too-many-pixel images.
- **Quotas are cost-weighted:** each call consumes `Kanon.praxis_cost[praxis]`
  units (OCR and moderation cost 2 by default), so usage numbers change.
- **Validation happens before charging.** Invalid input no longer consumes quota or
  rate-limit tokens, so usage counts differ from 1.x for callers that send bad input.
- `fetch_url` gains a `policy` keyword and refuses non-public destinations, so code
  that relied on fetching internal URLs must be rewritten (or use bytes).

### Added
- **Models:** `horasis.models` registry and cache manager (pinned URL + SHA-256,
  `HORASIS_CACHE_DIR`, atomic verified downloads), `horasis models
  list|pull|path|verify`, `Kanon.auto_download_models`, and `ModelDownloadError`.
  `YuNetFaceAdapter(model_path=...)` is now optional: it uses the cached model.
- **Rate limits:** `RateLimiter` protocol, `RedisRateLimiter` (atomic Lua token
  bucket on Redis server time), `redis_backends()` to share one client with
  `RedisQuotaStore`, `Skopos.build(rate_limiter=...)`, `Kanon.rate_limit_store` /
  `redis_url`, and `horasis serve --workers/--redis-url` with a warning when several
  workers use the in-process limiter.
- **URL policy:** `InputPolicy.url_allow_hosts` / `url_deny_hosts`.
- **Tests and CI:** real-model integration tests in `tests/integration/real/` and a
  manual/weekly `integration.yml` workflow.

### Fixed
- Malformed input and invalid batch items are no longer charged; the batch behaviour
  for an invalid API key (abort) and refund-on-denial is unchanged.
- **SSRF:** URL fetching allows only http(s), requires every resolved address to be
  globally routable (loopback, private, link-local incl. 169.254.169.254, CGNAT,
  multicast, reserved, IPv6 ULA and IPv4-mapped addresses are blocked), connects to the
  validated IP (no DNS-rebinding window), follows at most 3 manually re-validated
  redirects, streams with a size cap and timeouts, and requires `image/*`. Errors never
  echo the resolved address.
- Per-process rate limits can now be shared across workers (see Added).

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
