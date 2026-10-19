# Changelog

All notable changes to horasis are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-19

First public release.

### Capabilities

- Object detection, classification, segmentation and pose estimation with
  Ultralytics, plus ONNX Runtime for detection and classification.
- OCR with EasyOCR or Tesseract.
- Face detection with YuNet. It returns boxes and landmarks and does no recognition.
- Content moderation with transformers.
- A remote backend, so one horasis instance can call another over HTTP.

### Access control

- Hashed API keys with expiry, revocation and scopes.
- Tiers with quotas and per-minute rate limits.
- Quota and rate-limit storage in memory, SQLite or Redis.
- `client.usage()` for per-key usage reports.

### Everything else

- Sync and async clients, batch methods and video streaming.
- Accepts file paths, bytes, PIL images and numpy arrays. URL inputs are off by
  default and protected against SSRF when enabled.
- Results can be filtered, drawn on images, and exported to JSON or COCO.
- Backend routing with fallbacks, and third-party backends through plugins.
- `horasis` CLI, HTTP server, and a `horasis models` command for downloading
  model files.
- Lifecycle hooks and optional OpenTelemetry support.