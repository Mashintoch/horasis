# horasis

`horasis` is a computer vision SDK that gives you one simple API for common vision tasks like:

- object detection
- image classification
- OCR, or text extraction from images
- face detection
- content moderation

It is built to be easy to extend. The core library separates:

- the public API you call
- access control and API keys
- orchestration logic
- backend adapters
- shared data models and errors

That design makes it easier to add new backends later without breaking existing code.

## What this project is for

Use `horasis` when you want a small, consistent interface for vision tasks, but you also want:

- API-key based access control
- usage limits and tiered quotas
- plugin-based backend support
- structured result objects instead of raw backend responses

## Install

Install the base package:

```bash
pip install horasis
```

This installs the default Ultralytics-based backend, so object detection and image classification work right away.

Optional extras are available if you need them:

```bash
pip install horasis[viz]     # draw boxes on images and videos
pip install horasis[onnx]    # ONNX Runtime backend support
pip install horasis[remote]  # remote inference backend support
pip install horasis[all]     # install every optional extra
```

## Quick start

Here is the smallest useful example:

```python
from horasis import InMemoryApiKeyValidator, Klimax, Skopos

validator = InMemoryApiKeyValidator()
validator.register_simple("demo-key", principal_id="alice", tier=Klimax.FREE)

client = Skopos.build(validator=validator)

result = client.detection.detect("photo.jpg", api_key="demo-key")
for detection in result.detections:
    print(detection.label, float(detection.confidence), detection.box)
```

## What kinds of inputs can it read?

Most public methods accept any of these:

- a local file path
- a URL
- raw `bytes`
- a pre-built `Physis` or `Kinesis` object

That means you can use the same API whether the image already exists on disk, comes from a web link, or is already in memory.

Example with a URL:

```python
result = client.detection.detect(
    "https://example.com/photo.jpg",
    api_key="demo-key",
)
```

## What do you get back?

You never get a raw backend response directly. Instead, `horasis` returns typed result objects such as:

- `DetectionResult`
- `ClassificationResult`
- `OcrResult`
- `FaceResult`
- `ModerationResult`
- `VideoDetectionResult`

These objects are easier to inspect, test, and pass around in your code.

## Drawing results on images

If you install `horasis[viz]`, detection and face results can draw themselves onto the original image.

Draw boxes on an image:

```python
result = client.detection.detect("photo.jpg", api_key="demo-key")
annotated = result.annotate("photo.jpg", output_path="photo_annotated.jpg")
```

Run detection and annotation in one step:

```python
result, annotated = client.detection.detect_and_annotate(
    "photo.jpg",
    api_key="demo-key",
    output_path="photo_annotated.jpg",
)
```

If you only want boxes and no text labels:

```python
result.annotate("photo.jpg", show_labels=False, show_confidence=False)
```

For several images at once:

```python
batch = client.detection.detect_and_annotate_batch(
    ["photo1.jpg", "photo2.jpg"],
    api_key="demo-key",
    output_dir="annotated",
    filenames=["record_101.jpg", "record_102.jpg"],
)

for result, annotated in batch:
    print(len(result.detections), "objects found")
```

## Configuration

`horasis` uses a `Kanon` configuration object for tiers, quotas, and backend selection.

You can load it from YAML:

```python
from horasis import Kanon, Skopos

kanon = Kanon.from_yaml("kanon.yaml")
client = Skopos.build(kanon=kanon)
```

The built-in default configuration lives in:

```text
src/horasis/config/default.yaml
```

That default config currently defines three tiers:

- `free`
- `pro`
- `enterprise`

Each tier has its own quota and allowed tasks.

## API keys and access control

`InMemoryApiKeyValidator` is useful for local development and tests. It is not meant for production, but it shows the pattern a real validator should follow.

Example:

```python
import time

from horasis import InMemoryApiKeyValidator, Klimax

validator = InMemoryApiKeyValidator()
validator.register_simple(
    "demo-key",
    principal_id="john_doe",
    tier=Klimax.FREE,
    expires_at=time.time() + 3600,
)

validator.revoke("demo-key")
```

If a key fails, `horasis` raises a specific error so you can tell what happened:

- `InvalidApiKeyError` means the key does not exist
- `ExpiredApiKeyError` means the key existed but is past its expiry time
- `RevokedApiKeyError` means the key was intentionally disabled

If you want to build your own validator, implement the `ApiKeyValidator` protocol and use the same ideas:

- look up keys securely
- store hashed keys in real systems
- support expiry
- support revocation

## Plugins and backends

`horasis` is designed so backends can be added as plugins.

A backend plugin registers itself through the `horasis.backends` entry-point group:

```toml
[project.entry-points."horasis.backends"]
my_backend = "my_package.adapter:MyAdapter"
```

Backend adapters implement the `Mechane` interface:

```python
from horasis.backend.mechane import Mechane


class MyAdapter(Mechane):
    name = "my_backend"
    supported_praxeis = frozenset({...})

    def infer(self, praxis, image, **options):
        ...
```

If you want to discover plugins at runtime:

```python
from horasis.plugins import Prosthesis

Prosthesis().discover(client.taxis)
```

## Project structure

The codebase is organized into layers:

| Layer | Name | Purpose |
|---|---|---|
| 5 | Public API | The `Skopos` client and the user-facing task helpers |
| 4 | Access control | API keys, tiers, policies, and quotas |
| 3 | Orchestration | Pipelines and hooks that coordinate work |
| 2 | Backend abstraction | The adapter interface and backend registry |
| 1 | Foundation | Shared schemas, result types, errors, and config |

This structure keeps the public API simple while letting the internals stay flexible.

## Development

To work on the project locally:

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
mypy
```

## Current status

`horasis` is usable now, but not every task has a backend yet.

Works today:

- object detection through the Ultralytics backend
- image classification through the Ultralytics backend
- image annotation for detection and face results when `horasis[viz]` is installed

Available in the public API, but not backed by an implementation yet:

- OCR
- face detection
- content moderation
- ONNX backend
- remote inference backend

If you call one of the unimplemented tasks, you should expect a `BackendUnavailableError` for now.

## License

Apache-2.0



