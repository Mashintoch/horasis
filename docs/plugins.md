# Writing a backend plugin

A *backend* turns an image into a result. Anyone can ship one as a separate
package; `horasis` finds it through a Python entry point and routes work to it
exactly like a built-in backend. Your plugin depends only on the **foundation**
and **backend** layers (`Image`, `Result` subclasses, `Backend`, `HorasisError`) --
never on access control or the public API.

## 1. Implement `Backend`

```python
# my_package/adapter.py
from __future__ import annotations

import time

from horasis.backend.base import Backend
from horasis.foundation.imaging import load_rgb
from horasis.foundation.schemas import (
    BoundingBox, Detection, DetectionResult, Confidence, Image, Task, Result,
)
from horasis.foundation.errors import AdapterError, BackendUnavailableError, HorasisError


class MyDetector(Backend):
    name = "my_detector"                              # registry name; must be unique
    supported_tasks = frozenset({Task.DETECT})    # what you can serve

    def __init__(self, model_path: str = "model.bin") -> None:
        self._model_path = model_path
        self._model = None                            # load lazily, not here

    # --- required -------------------------------------------------------
    def infer(self, task: Task, image: Image, **options: object) -> Result:
        if not self.supports(task):
            raise BackendUnavailableError(f"{self.name} does not support {task.value!r}")
        try:
            started = time.perf_counter()
            pixels = load_rgb(image)                  # upright RGB PIL image, already validated
            model = self._load()
            boxes = model.run(pixels, **options)      # your inference
            elapsed_ms = (time.perf_counter() - started) * 1000
            return DetectionResult(
                backend=self.name,
                latency_ms=elapsed_ms,                # measure it yourself (time.perf_counter)
                detections=[
                    Detection(
                        label=b.label,
                        confidence=Confidence(value=b.score),            # must be within 0..1
                        box=BoundingBox(x_min=b.x1, y_min=b.y1, x_max=b.x2, y_max=b.y2),
                    )
                    for b in boxes
                ],
            )
        except HorasisError:
            raise                                     # already a horasis error
        except Exception as exc:                      # wrap everything else
            raise AdapterError(f"{self.name} failed: {exc}") from exc

    # --- optional -------------------------------------------------------
    def warm_up(self) -> None:                        # called by client.warmup()
        self._load()

    def is_loaded(self) -> bool:                      # lets Config.max_loaded_models work
        return self._model is not None

    def unload(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            self._model = ...                         # open model_path; honour self.effective_device()
        return self._model
```

### Rules of the contract

* **Return the right result type** for the task: `DetectionResult`,
  `ClassificationResult`, `OcrResult`, `FaceResult`, `ModerationResult`,
  `SegmentationResult` or `PoseResult`. Set `backend=self.name`.
* **Read pixels with `load_rgb(image)`**, never from `image.path` / `image.data`.
  That is what guarantees size limits, format checks and EXIF orientation.
* **Raise `HorasisError` subclasses only.** Use `BackendUnavailableError` for a
  missing dependency, model file or unsupported task (say how to fix it), and
  `AdapterError` for failures during inference. Never let a raw exception or a
  library object escape: backend objects must not leak past the backend layer.
* **Import heavy libraries inside methods** so that merely registering your
  backend is cheap and a missing optional dependency fails with a clear message.
* **`options`** are caller-supplied keyword arguments (e.g. `conf=0.4`).
  Ignore ones you do not know, or validate them -- but never forward them blindly
  to code that could write files or load arbitrary paths.
* **Device:** `self.effective_device()` is `"auto"`, `"cpu"`, `"cuda"` or `"mps"`
  (your own `self.device` wins over `Config.device`).
* **Thread safety:** the async client calls `infer` from several threads at once.
* **Video:** to serve video, also set `supported_video_tasks` and implement
  `infer_video` (and `supported_video_annotate_tasks` / `infer_video_and_annotate`).
* **Moderation:** return raw scores. `Config.moderation_threshold(s)` decides
  `flagged` for every backend, so thresholds behave the same everywhere.

## 2. Register the entry point

```toml
# your package's pyproject.toml
[project.entry-points."horasis.backends"]
my_detector = "my_package.adapter:MyDetector"
```

The entry point must be callable with **no arguments** (give constructor
parameters defaults, or point at a small factory function).

## 3. Use it

```python
from horasis import Config, Client
from horasis.plugins import PluginLoader

client = Client.build(config=Config(backend_order={"detect": ["my_detector"]}))
PluginLoader().discover(client.registry)                 # skips plugins that fail to load
client.detection.detect("photo.jpg", api_key=key)   # routed to my_detector
client.detection.detect("photo.jpg", api_key=key, backend="ultralytics")  # per-call override
```

Pass `strict=True` to `discover` if a broken plugin should be an error instead of
being skipped. From a shell, `horasis backends` lists built-in and plugin
backends and reports plugins that failed to load.

Routing order is: explicit `backend=` -> `Config.backend_order[task]` ->
`Config.default_backend` -> `Config.fallback_order` -> any backend that supports
the task. If none does, `NoBackendError` lists what is registered.

## 4. Test it without real models

Keep tests offline: inject a fake model object (see how the built-in adapters
accept `reader=`, `session=`, `detector=`, `classifier=` for exactly this), and
assert on the result schema. A minimal contract check:

```python
def test_conforms_to_backend():
    backend = MyDetector()
    assert isinstance(backend, Backend)
    assert backend.name and all(isinstance(p, Task) for p in backend.supported_tasks)
```

Mark tests that need real weights or the network `@pytest.mark.integration`.
