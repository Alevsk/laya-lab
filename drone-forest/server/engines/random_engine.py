"""The floor: a uniformly random action every frame.

Any engine that claims to fly must beat this one on collisions per minute. Seeded so a run is
reproducible (RANDOM_ENGINE_SEED, default 0).
"""
from __future__ import annotations

import random
from time import perf_counter

from ..config import get_settings
from ..schemas import SPEED_MIN, Action, Decision, SensorFrame
from .base import register

_ACTIONS = list(Action)
_UNIFORM = {a.value: 1.0 / len(_ACTIONS) for a in _ACTIONS}


class RandomEngine:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = get_settings().random_engine_seed if seed is None else seed
        self._rng = random.Random(self.seed)

    def warmup(self) -> None:
        return None

    def decide(self, frame: SensorFrame, decision_id: int) -> Decision:
        t0 = perf_counter()
        action = self._rng.choice(_ACTIONS)
        return Decision(
            action=action,
            engine=self.name,
            frame_id=frame.frame_id,
            decision_id=decision_id,
            latency_ms=(perf_counter() - t0) * 1000.0,
            probabilities=dict(_UNIFORM),
            target_speed=round(self._rng.uniform(SPEED_MIN, frame.bounds.speed_max), 2),
            reason="uniform random",
        )

    def describe(self) -> dict:
        return {
            "kind": "baseline",
            "model": None,
            "device": "cpu",
            "expected_latency_ms": 0.01,
            "seed": self.seed,
            "notes": "Uniform over the six actions; the floor every real engine must beat.",
        }


@register("random")
def _make_random() -> RandomEngine:
    return RandomEngine()
