"""The engine abstraction the service depends on (and nothing else).

Dependency inversion: `server.app` depends only on this protocol and the registry. Concrete
engines depend on it too. Nothing depends on a concrete engine.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ..schemas import Decision, SensorFrame


@runtime_checkable
class DecisionEngine(Protocol):
    """One flight decision per sensor frame.

    Implementations must be safe to call repeatedly from one asyncio task. They may hold
    state (an engine that remembers its last action is fine). `decide` must always return a
    Decision; if it cannot decide, return BRAKE with a reason rather than raising.
    """

    name: str

    def warmup(self) -> None:
        """Do expensive one-time work (load a model, JIT a kernel) before the first frame."""
        ...

    def decide(self, frame: SensorFrame, decision_id: int) -> Decision:
        ...

    def describe(self) -> dict:
        """Static facts for the HUD/README: device, model, expected latency, notes."""
        ...


EngineFactory = Callable[[], DecisionEngine]
ENGINES: dict[str, EngineFactory] = {}


def register(name: str) -> Callable[[EngineFactory], EngineFactory]:
    def deco(factory: EngineFactory) -> EngineFactory:
        if name in ENGINES:
            raise ValueError(f"engine {name!r} already registered")
        ENGINES[name] = factory
        return factory

    return deco


def available() -> list[str]:
    return sorted(ENGINES)


def create(name: str) -> DecisionEngine:
    try:
        return ENGINES[name]()
    except KeyError:
        raise KeyError(f"unknown engine {name!r}; available: {available()}") from None
