# Writing a backend plugin

A backend turns an image into a result. You can ship one as a separate
package; horasis finds it through a Python entry point and routes work to it
like a built-in backend. A plugin imports only from the foundation and backend
layers (`Image`, the `Result` subclasses, `Backend`, `HorasisError`), never from
access control or the public API.

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
    supported_tasks = frozenset({Task.DETECT})

    def __init__(self, model_path: str = "model.bin") -> None:
        self._model_path = model_path
        self._model = None                            # load lazily, not here

    def infer(self, task: Task, image: Image, **options: object) -> Result:
        if not self.supports(task):
            raise BackendUnavailableError(f"{self.name} does not support {task.value!r}")
        try:
            started = time.perf_counter()
            pixels = load_rgb(image)                  # upright RGB PIL image, already validated
            model = self._load()
            boxes = model.run(pixels, **options)
            elapsed_ms = (time.perf_counter() - started) * 1000
            return DetectionResult(
                backend=self.name,
                latency_ms=elapsed_ms,
                detections=[
                    Detection(
                        label=b.label,
                        confidence=Confidence(value=b.score),  # must be within 0..1
                        box=BoundingBox(x_min=b.x1, y_min=b.y1, x_max=b.x2, y_max=b.y2),
                    )
                    for b in boxes
                ],
            )
        except HorasisError:
            raise
        except Exception as exc:
            raise AdapterError(f"{self.name} failed: {exc}") from exc

    # Optional
    def warm_up(self) -> None:                        # called by client.warmup()
        self._load()

    def is_loaded(self) -> bool:                      # needed for Config.max_loaded_models
        return self._model is not None

    def unload(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            self._model = ...                         # open model_path; honour self.effective_device()
        return self._model
```

### Rules

* Return the result type for the task: `DetectionResult`,
  `ClassificationResult`, `OcrResult`, `FaceResult`, `ModerationResult`,
  `SegmentationResult` or `PoseResult`. Set `backend=self.name`.
* Read pixels with `load_rgb(image)`, never from `image.path` or `image.data`,
  so size limits, format checks and EXIF orientation apply.
* Raise only `HorasisError` subclasses: `BackendUnavailableError` for a missing
  dependency, model file or unsupported task (say how to fix it), and
  `AdapterError` for failures during inference. Do not let a raw exception or a
  library object escape the backend layer.
* Import heavy libraries inside methods, so registering the backend is cheap and
  a missing optional dependency fails with a clear message.
* `options` are caller-supplied keyword arguments such as `conf=0.4`. Ignore the
  ones you do not know, or validate them. Never forward them to code that could
  write files or load arbitrary paths.
* `self.effective_device()` is `"auto"`, `"cpu"`, `"cuda"` or `"mps"`; your own
  `self.device` overrides `Config.device`.
* The async client calls `infer` from several threads at once.
* To serve video, also set `supported_video_tasks` and implement `infer_video`
  (and `supported_video_annotate_tasks` with `infer_video_and_annotate`).
* Moderation backends return raw scores. `Config.moderation_threshold(s)`
  decides `flagged` for every backend.

## 2. Register the entry point

```toml
# your package's pyproject.toml
[project.entry-points."horasis.backends"]
my_detector = "my_package.adapter:MyDetector"
```

The entry point must be callable with no arguments: give constructor
parameters defaults, or point it at a small factory function.

## 3. Use it

```python
from horasis import Config, Client
from horasis.plugins import PluginLoader

client = Client.build(config=Config(backend_order={"detect": ["my_detector"]}))
PluginLoader().discover(client.registry)                 # skips plugins that fail to load
client.detection.detect("photo.jpg", api_key=key)   # routed to my_detector
client.detection.detect("photo.jpg", api_key=key, backend="ultralytics")  # per-call override
```

Pass `strict=True` to `discover` to make a broken plugin an error instead of
skipping it. `horasis backends` lists built-in and plugin backends and reports
plugins that failed to load.

Routing order is: explicit `backend=` -> `Config.backend_order[task]` ->
`Config.default_backend` -> `Config.fallback_order` -> any backend that supports
the task. If none does, `NoBackendError` lists what is registered.

## 4. Test it without real models

Keep tests offline: inject a fake model object, as the built-in adapters do
with `reader=`, `session=`, `detector=` and `classifier=`, and assert on the
result schema. A minimal contract check:

```python
def test_conforms_to_backend():
    backend = MyDetector()
    assert isinstance(backend, Backend)
    assert backend.name and all(isinstance(p, Task) for p in backend.supported_tasks)
```

Mark tests that need real weights or the network `@pytest.mark.integration`.
