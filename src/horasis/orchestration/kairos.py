"""Kairos -- the pipeline hook system."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from horasis.backend.mechane import Mechane
from horasis.foundation.schemas import Physis, Praxis, Theoria


class Moment(str, Enum):
    """A named point in the orchestration pipeline that hooks can attach to."""

    BEFORE_RESOLVE = "before_resolve"
    AFTER_RESOLVE = "after_resolve"
    BEFORE_INFER = "before_infer"
    AFTER_INFER = "after_infer"


HookFn = Callable[..., None]


class Kairos:
    """Registers and fires hooks at named :class:`Moment`s.

    Hooks are advisory: they cannot mutate the pipeline's outcome, only
    observe it (e.g. for logging, metrics, or debugging). Exceptions raised
    by a hook propagate to the caller of `Organon.run`.
    """

    def __init__(self) -> None:
        self._hooks: dict[Moment, list[HookFn]] = {moment: [] for moment in Moment}

    def on(self, moment: Moment, fn: HookFn) -> None:
        self._hooks[moment].append(fn)

    def off(self, moment: Moment, fn: HookFn) -> None:
        if fn in self._hooks[moment]:
            self._hooks[moment].remove(fn)

    def fire(self, moment: Moment, **payload: Any) -> None:
        for hook in self._hooks[moment]:
            hook(**payload)


    def before_resolve(self, *, praxis: Praxis, image: Physis) -> None:
        self.fire(Moment.BEFORE_RESOLVE, praxis=praxis, image=image)

    def after_resolve(self, *, praxis: Praxis, backend: Mechane) -> None:
        self.fire(Moment.AFTER_RESOLVE, praxis=praxis, backend=backend)

    def before_infer(self, *, praxis: Praxis, backend: Mechane, image: Physis) -> None:
        self.fire(Moment.BEFORE_INFER, praxis=praxis, backend=backend, image=image)

    def after_infer(self, *, praxis: Praxis, backend: Mechane, result: Theoria) -> None:
        self.fire(Moment.AFTER_INFER, praxis=praxis, backend=backend, result=result)
