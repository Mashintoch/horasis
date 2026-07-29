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
| 2 | Backend abstraction | `Mechane` interface, `Taxis` registry, `UltralyticsAdapter` (+ ONNX/remote stubs) |
| 1 | Foundation | `Kanon` config, `Sphalma` exceptions, result schemas (`Theoria`, `Doxa`, `Praxis`, `Glyph`, `Physis`), `Aisthesis` telemetry |

Plugins (`Prosthesis`) register additional `Mechane` backends via the
`horasis.backends` Python entry-point group, independent of the core layers.

## Install

```bash
pip install horasis[ultralytics]   # or [onnx], [remote], [all]
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

Each capability accepts a file path, raw `bytes`, or a pre-built `Physis`,
and always returns a `Theoria` subclass — never a raw backend object.

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

Early / v1 foundation. The Ultralytics backend covers detection and
classification; OCR, face detection, and content moderation ship with the
public API surface and result schemas but need a backend that implements
them. ONNX and remote-inference adapters are placeholder stubs.

## License

Apache-2.0



