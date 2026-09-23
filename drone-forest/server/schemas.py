"""The wire contract between the game and the decision service.

These models are the single source of truth for what a sensor frame, a decision and an event
look like. `web/src/core/types.ts` mirrors them field for field; if you change one, change
the other and bump PROTOCOL_VERSION.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 2

SPEED_MIN = 4.0
"""Metres per second: the slowest the drone will be asked to fly (a hover cannot progress)."""

# The fixed ray fan every frame carries, in this order. Bearing is degrees left(-)/right(+)
# of the drone's heading; elevation is degrees up(+)/down(-). The game casts exactly these.
RAY_SPEC: list[tuple[str, float, float]] = [
    ("forward", 0.0, 0.0),
    ("left_15", -15.0, 0.0),
    ("right_15", 15.0, 0.0),
    ("left_30", -30.0, 0.0),
    ("right_30", 30.0, 0.0),
    ("left_45", -45.0, 0.0),
    ("right_45", 45.0, 0.0),
    ("left_60", -60.0, 0.0),
    ("right_60", 60.0, 0.0),
    ("up", 0.0, 35.0),
    ("down", 0.0, -35.0),
]
RAY_NAMES = [r[0] for r in RAY_SPEC]


class Action(str, Enum):
    """The discrete flight commands a decision engine may return.

    Six options on purpose: Laya renders every option into a shared 192-token head budget, and
    past ~8 options descriptions start being truncated (see the root repo's scenario 06).
    """

    FORWARD = "forward"
    BANK_LEFT = "bank_left"
    BANK_RIGHT = "bank_right"
    CLIMB = "climb"
    DESCEND = "descend"
    BRAKE = "brake"


ObstacleKind = Literal["tree", "rock", "bird", "ground", "wall"]


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class Ray(BaseModel):
    name: str
    bearing_deg: float
    elevation_deg: float
    distance: float = Field(description="Metres to the first hit, or max_range when nothing was hit")
    hit: ObstacleKind | None = None
    max_range: float = 60.0

    @property
    def clear(self) -> bool:
        return self.hit is None


class DroneState(BaseModel):
    position: Vec3
    velocity: Vec3
    heading_deg: float = Field(description="Compass heading, degrees")
    altitude: float = Field(description="Metres above ground")
    speed: float = Field(description="Metres per second, magnitude of velocity")


class Nearest(BaseModel):
    kind: ObstacleKind
    distance: float
    bearing_deg: float
    elevation_deg: float
    closing_speed: float = Field(0.0, description="m/s, positive when the gap is shrinking")


class Bounds(BaseModel):
    altitude_min: float = 2.0
    altitude_max: float = 40.0
    speed_max: float = Field(18.0, description="m/s; an engine's target_speed is clamped to [SPEED_MIN, speed_max]")


class SensorFrame(BaseModel):
    """Everything the drone knows at one instant. This is what an engine decides from."""

    protocol: int = PROTOCOL_VERSION
    frame_id: int
    t: float = Field(description="Simulation time, seconds")
    drone: DroneState
    rays: list[Ray] = Field(description=f"Exactly the {len(RAY_SPEC)} rays in RAY_SPEC, in order")
    nearest: Nearest | None = None
    bounds: Bounds = Field(default_factory=Bounds)
    last_action: Action | None = None

    def ray(self, name: str) -> Ray:
        for r in self.rays:
            if r.name == name:
                return r
        raise KeyError(name)


class Decision(BaseModel):
    """What an engine returns. `latency_ms` is the engine's own measured think time."""

    action: Action
    engine: str
    frame_id: int
    decision_id: int
    latency_ms: float
    confidence: float | None = Field(None, description="Engine-defined; for Laya, its `confidence` field")
    probabilities: dict[str, float] | None = Field(None, description="Per-action, when the engine has them")
    collision_imminent: float | None = Field(None, description="P(collision soon), when the engine estimates it")
    urgency: float | None = Field(None, description="0..3 expected level, when the engine estimates it")
    target_speed: float | None = Field(None, description="m/s the engine wants the drone to fly at, in [SPEED_MIN, bounds.speed_max]; None = keep cruise")
    reason: str = Field("", description="One line a human can read on the HUD")
    option_index: int | None = Field(None, description="Index of the chosen option as presented to the model, for positional-bias audits")
    option_order: list[str] | None = Field(None, description="Action order as presented to the model, when shuffled")


class GameEvent(BaseModel):
    """Things the game tells the service about, so decisions can be scored later."""

    type: Literal["collision", "near_miss", "reset", "engine_switch", "score"]
    frame_id: int
    t: float
    obstacle_kind: ObstacleKind | None = None
    details: dict = Field(default_factory=dict)


# ---- WebSocket envelope ------------------------------------------------------------------

class FrameMessage(BaseModel):
    type: Literal["frame"] = "frame"
    frame: SensorFrame


class EventMessage(BaseModel):
    type: Literal["event"] = "event"
    event: GameEvent


class SetEngineMessage(BaseModel):
    type: Literal["set_engine"] = "set_engine"
    engine: str


class DecisionMessage(BaseModel):
    type: Literal["decision"] = "decision"
    decision: Decision


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str


class InfoMessage(BaseModel):
    type: Literal["info"] = "info"
    engine: str
    engines: list[str]
    protocol: int = PROTOCOL_VERSION
