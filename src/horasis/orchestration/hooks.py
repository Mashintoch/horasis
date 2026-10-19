"""Pipeline hooks.

Hooks observe a request as it moves through the pipeline; they cannot change
it. Register a callable for a :class:`HookEvent` and it is called with these
keyword arguments:

==================  ==============================================================
Event               Keyword arguments
==================  ==============================================================
``BEFORE_RESOLVE``  ``task``, and ``image`` or ``video``
``AFTER_RESOLVE``   ``task``, ``backend`` (the chosen :class:`Backend`)
``BEFORE_INFER``    ``task``, ``backend``, and ``image`` or ``video``
``AFTER_INFER``     ``task``, ``backend``, ``result``
``ON_ERROR``        ``task``, ``error`` (nothing else is guaranteed)
==================  ==============================================================

Write hooks as ``def hook(**payload): ...``, or name the arguments you want
and add ``**_``, so they keep working when payloads grow.

An exception raised by a before or after hook propagates to the caller (and
fires ``ON_ERROR``). An exception raised by an ``ON_ERROR`` hook is logged and
swallowed, so it cannot hide the real error. Built-in hooks live in :mod:`horasis.orchestration.builtin_hooks`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

from horasis.backend.base import Backend
from horasis.foundation.schemas import Image, Result, Task

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    """A named point in the orchestration pipeline that hooks can attach to."""

    BEFORE_RESOLVE = "before_resolve"
    AFTER_RESOLVE = "after_resolve"
    BEFORE_INFER = "before_infer"
    AFTER_INFER = "after_infer"
    ON_ERROR = "on_error"


HookFn = Callable[..., None]


class Hooks:
    """Registers and fires hooks for each :class:`HookEvent`."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookFn]] = {event: [] for event in HookEvent}

    def on(self, event: HookEvent, fn: HookFn) -> None:
        self._hooks[event].append(fn)

    def off(self, event: HookEvent, fn: HookFn) -> None:
        if fn in self._hooks[event]:
            self._hooks[event].remove(fn)

    def fire(self, event: HookEvent, **payload: Any) -> None:
        """Call every hook for `event`; exceptions propagate (except ``ON_ERROR``)."""
        if event is HookEvent.ON_ERROR:
            for hook in list(self._hooks[event]):
                try:
                    hook(**payload)
                except Exception:
                    logger.exception("on_error hook %r failed", hook)
            return
        for hook in list(self._hooks[event]):
            hook(**payload)

    def before_resolve(self, *, task: Task, image: Image) -> None:
        self.fire(HookEvent.BEFORE_RESOLVE, task=task, image=image)

    def after_resolve(self, *, task: Task, backend: Backend) -> None:
        self.fire(HookEvent.AFTER_RESOLVE, task=task, backend=backend)

    def before_infer(self, *, task: Task, backend: Backend, image: Image) -> None:
        self.fire(HookEvent.BEFORE_INFER, task=task, backend=backend, image=image)

    def after_infer(self, *, task: Task, backend: Backend, result: Result) -> None:
        self.fire(HookEvent.AFTER_INFER, task=task, backend=backend, result=result)

    def on_error(self, *, task: Task, error: BaseException) -> None:
        self.fire(HookEvent.ON_ERROR, task=task, error=error)
