"""The Laya-backed DecisionEngine - the engine this simulator exists to test.

One `agent.predict()` per frame carries three questions in a single forward pass: `move` (a choice
over the six rendered actions), `collision_imminent` (a calibrated yes/no) and `urgency` (a 4-level
score). The action returned is Laya's argmax over the move options - nothing else. This engine never
consults the heuristic, never applies a safety override and never re-ranks by clearance; combining
Laya with anything is a separate engine's job. What it does do is give Laya an honest best chance:
semantic framing by default, six short consequence-stated options, and a per-decision shuffle of
option order (recorded in the Decision) so positional bias is measurable instead of mistaken for
preference.

Latency is measured with the device synchronised around the predict call; MPS and CUDA dispatch
asynchronously and an unsynchronised timer under-reports by about half.

Configuration is read from the environment, because the registry creates engines with no
arguments: LAYA_CHECKPOINT (english|multilingual|typed-decisions), LAYA_DEVICE (cuda|mps|cpu; else
cuda > mps > cpu), FRAMING (semantic|numeric), SHUFFLE_OPTIONS (1|0), LAYA_SEED (int).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from ..schemas import Action, Decision, SensorFrame
from . import framing
from .base import DecisionEngine, register

ENGINE_NAME = "laya"
TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LayaEngineConfig:
    checkpoint: str = "english"
    device: str | None = None
    framing: framing.FramingKind = "semantic"
    shuffle_options: bool = True
    seed: int = 0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LayaEngineConfig:
        e = os.environ if env is None else env
        kind = e.get("FRAMING", "semantic").strip().lower()
        if kind not in framing.FRAMINGS:
            raise ValueError(f"FRAMING={kind!r}; expected one of {framing.FRAMINGS}")
        device = e.get("LAYA_DEVICE", "").strip().lower()
        return cls(
            checkpoint=e.get("LAYA_CHECKPOINT", "english").strip().lower(),
            device=None if device in ("", "auto") else device,
            framing=kind,  # type: ignore[arg-type]  # validated against FRAMINGS above
            shuffle_options=e.get("SHUFFLE_OPTIONS", "1").strip().lower() in TRUTHY,
            seed=int(e.get("LAYA_SEED", "0")),
        )


def resolve_device(requested: str | None = None) -> str:
    """LAYA_DEVICE wins; otherwise the fastest device torch can see."""
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    """Block until queued GPU work has finished, so a timer around predict() tells the truth."""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _warmup_frame() -> SensorFrame:
    """A plausible frame for the throwaway predict that pays kernel compilation up front."""
    from ..schemas import RAY_SPEC, Bounds, DroneState, Nearest, Ray, Vec3

    rays = [Ray(name=n, bearing_deg=b, elevation_deg=e, distance=60.0) for n, b, e in RAY_SPEC]
    rays[0] = Ray(name="forward", bearing_deg=0.0, elevation_deg=0.0, distance=8.0, hit="tree")
    return SensorFrame(
        frame_id=0,
        t=0.0,
        drone=DroneState(position=Vec3(x=0, y=12, z=0), velocity=Vec3(x=0, y=0, z=6), heading_deg=0.0,
                         altitude=12.0, speed=6.0),
        rays=rays,
        nearest=Nearest(kind="tree", distance=8.0, bearing_deg=0.0, elevation_deg=0.0, closing_speed=6.0),
        bounds=Bounds(),
        last_action=Action.FORWARD,
    )


class LayaEngine:
    name = ENGINE_NAME

    def __init__(self, config: LayaEngineConfig | None = None) -> None:
        self.config = config or LayaEngineConfig.from_env()
        self.device = resolve_device(self.config.device)
        self.agent: Any = None
        self.load_ms: float | None = None
        self.warmup_ms: float | None = None
        self.decisions = 0

    # ---- lifecycle ----

    def warmup(self) -> None:
        if self.agent is not None:
            return
        from ..fastload import load_agent

        t0 = time.perf_counter()
        self.agent = load_agent(self.config.checkpoint, self.device)
        self.device = str(self.agent.device)
        self.load_ms = (time.perf_counter() - t0) * 1000.0

        prompt = framing.build_prompt(_warmup_frame(), self.config.framing, self.config.shuffle_options,
                                      self.config.seed)
        sync(self.device)
        t0 = time.perf_counter()
        self.agent.predict(prompt.state, prompt.questions)
        sync(self.device)
        self.warmup_ms = (time.perf_counter() - t0) * 1000.0

    # ---- one decision ----

    def decide(self, frame: SensorFrame, decision_id: int) -> Decision:
        self.decisions += 1
        try:
            if self.agent is None:
                self.warmup()
            prompt = framing.build_prompt(frame, self.config.framing, self.config.shuffle_options,
                                          self.config.seed)
            sync(self.device)
            t0 = time.perf_counter()
            result = self.agent.predict(prompt.state, prompt.questions)
            sync(self.device)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return self._to_decision(frame, decision_id, prompt, result["answers"], latency_ms)
        except Exception as e:  # the protocol asks for BRAKE with a reason rather than a raise
            return Decision(
                action=Action.BRAKE,
                engine=self.name,
                frame_id=frame.frame_id,
                decision_id=decision_id,
                latency_ms=0.0,
                reason=f"laya failed ({type(e).__name__}: {e}); braking",
            )

    def _to_decision(
        self,
        frame: SensorFrame,
        decision_id: int,
        prompt: framing.Prompt,
        answers: dict[str, dict[str, Any]],
        latency_ms: float,
    ) -> Decision:
        move = answers["move"]
        action, index = framing.resolve_choice(move["choice"], prompt.options)
        probs = move["probabilities"]
        chosen = prompt.options[index]
        return Decision(
            action=action,
            engine=self.name,
            frame_id=frame.frame_id,
            decision_id=decision_id,
            latency_ms=latency_ms,
            confidence=float(move["confidence"]),
            probabilities={a.value: float(probs[a.value]) for a in framing.ACTION_ORDER},
            collision_imminent=float(answers["collision_imminent"]["noul"]),
            urgency=float(answers["urgency"]["score"]),
            reason=f'laya picked "{chosen.label}: {chosen.description}"',
            option_index=index,
            option_order=prompt.order,
        )

    # ---- facts for the HUD / README ----

    def describe(self) -> dict:
        return {
            "engine": self.name,
            "model": "convaiinnovations/laya",
            "checkpoint": self.config.checkpoint,
            "device": self.device,
            "framing": self.config.framing,
            "shuffle_options": self.config.shuffle_options,
            "seed": self.config.seed,
            "loaded": self.agent is not None,
            "load_ms": None if self.load_ms is None else round(self.load_ms, 1),
            "warmup_latency_ms": None if self.warmup_ms is None else round(self.warmup_ms, 1),
            "decisions": self.decisions,
            "questions_per_forward_pass": 3,
            "notes": "action is Laya's argmax over 6 consequence-stated options; no overrides, no re-ranking",
        }


@register(ENGINE_NAME)
def create_laya_engine() -> DecisionEngine:
    return LayaEngine()
