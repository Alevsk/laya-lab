"""The geometric baseline every other engine is measured against.

Pure arithmetic over the 11-ray fan, a few microseconds per frame, fully deterministic. The
policy scores each action by the clearance in the sectors that action steers into, then
applies four corrections a pilot would: prefer going straight when straight is fine, do not
reverse the previous manoeuvre (no left-right weaving), never leave the altitude band, and
brake only when everything ahead is close and there is still speed to shed. The scores are
exposed as normalised probabilities so the telemetry can compare this engine's ranking with
Laya's on identical frames.

Distances are saturated at PLANNING_HORIZON metres: a ray that is clear for 40 m is as good as
one clear for 60 m, which keeps the drone from weaving toward marginally longer clearances.
"""
from __future__ import annotations

import math
from time import perf_counter

from ..schemas import Action, Decision, Ray, SensorFrame
from .base import register

PLANNING_HORIZON = 40.0
"""Metres beyond which extra clearance no longer matters."""

TTC_HORIZON = 4.0
"""Seconds; time-to-collision at which collision_imminent starts rising above 0."""

ALTITUDE_MARGIN = 1.0
"""Metres kept inside the band before an altitude change is fully allowed again."""

ALTITUDE_RAMP = 4.0
"""Metres over which climb/descend fade in from 0 to full as headroom/floor grows."""

# Each action's sector: (ray name, weight). Weights sum to 1. The first entry is the primary
# ray - the one the action steers straight into - and it gates the whole score.
SECTORS: dict[Action, tuple[tuple[str, float], ...]] = {
    Action.FORWARD: (("forward", 0.5), ("left_15", 0.25), ("right_15", 0.25)),
    Action.BANK_LEFT: (("left_30", 0.4), ("left_45", 0.25), ("left_15", 0.2), ("left_60", 0.15)),
    Action.BANK_RIGHT: (("right_30", 0.4), ("right_45", 0.25), ("right_15", 0.2), ("right_60", 0.15)),
    Action.CLIMB: (("up", 0.6), ("forward", 0.2), ("left_15", 0.1), ("right_15", 0.1)),
    Action.DESCEND: (("down", 0.6), ("forward", 0.2), ("left_15", 0.1), ("right_15", 0.1)),
}

PREFERENCE: dict[Action, float] = {
    Action.FORWARD: 1.15,
    Action.BANK_LEFT: 1.0,
    Action.BANK_RIGHT: 1.0,
    Action.CLIMB: 0.85,
    Action.DESCEND: 0.8,
    Action.BRAKE: 0.9,
}

REVERSE: dict[Action, Action] = {
    Action.BANK_LEFT: Action.BANK_RIGHT,
    Action.BANK_RIGHT: Action.BANK_LEFT,
    Action.CLIMB: Action.DESCEND,
    Action.DESCEND: Action.CLIMB,
}
REVERSE_PENALTY = 0.6
CONTINUE_BONUS = 1.05

FRONT_CONE_DEG = 30.0
"""Half-angle of the cone in which `nearest` is treated as being on the flight path."""

AHEAD = ("forward", "left_15", "right_15")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _clearance(ray: Ray) -> float:
    """0..1: how much of the planning horizon is free along this ray."""
    return _clamp(ray.distance / PLANNING_HORIZON)


def _describe(ray: Ray, label: str) -> str:
    if ray.hit is None:
        return f"{label} clear {ray.distance:.0f} m"
    return f"{label} {ray.hit} {ray.distance:.0f} m"


class HeuristicEngine:
    name = "heuristic"

    def warmup(self) -> None:
        return None

    def decide(self, frame: SensorFrame, decision_id: int) -> Decision:
        t0 = perf_counter()
        try:
            rays = {r.name: r for r in frame.rays}
            scores, notes = self._score(frame, rays)
        except KeyError as e:
            return Decision(
                action=Action.BRAKE,
                engine=self.name,
                frame_id=frame.frame_id,
                decision_id=decision_id,
                latency_ms=(perf_counter() - t0) * 1000.0,
                reason=f"invalid frame: missing ray {e.args[0]!r} -> brake",
            )

        action = max(Action, key=lambda a: scores[a])  # ties -> first in Action order
        total = sum(scores.values())
        probabilities = (
            {a.value: scores[a] / total for a in Action}
            if total > 0
            else {a.value: 1.0 / len(Action) for a in Action}
        )
        ttc = self._time_to_collision(frame, rays)
        collision_imminent = _clamp(1.0 - ttc / TTC_HORIZON)
        urgency = 0.0 if ttc > 4.0 else 1.0 if ttc > 2.5 else 2.0 if ttc > 1.2 else 3.0

        return Decision(
            action=action,
            engine=self.name,
            frame_id=frame.frame_id,
            decision_id=decision_id,
            latency_ms=(perf_counter() - t0) * 1000.0,
            confidence=probabilities[action.value],
            probabilities=probabilities,
            collision_imminent=collision_imminent,
            urgency=urgency,
            reason=f"{_describe(rays['forward'], 'forward')}; {notes[action]} -> {action.value}",
        )

    # ---- scoring -------------------------------------------------------------------------

    def _score(self, frame: SensorFrame, rays: dict[str, Ray]) -> tuple[dict[Action, float], dict[Action, str]]:
        scores: dict[Action, float] = {}
        notes: dict[Action, str] = {}
        drone, bounds = frame.drone, frame.bounds

        for action, sector in SECTORS.items():
            primary = rays[sector[0][0]]
            mean = sum(w * _clearance(rays[name]) for name, w in sector)
            # A blocked primary ray gates the sector even when its neighbours are open.
            scores[action] = mean * (0.5 + 0.5 * _clearance(primary))

        headroom = bounds.altitude_max - drone.altitude - ALTITUDE_MARGIN
        floor = drone.altitude - bounds.altitude_min - ALTITUDE_MARGIN
        scores[Action.CLIMB] *= _clamp(headroom / ALTITUDE_RAMP)
        scores[Action.DESCEND] *= _clamp(floor / ALTITUDE_RAMP)

        # Something closing in inside the front cone makes going straight worse even if the
        # forward ray has not caught it yet (a bird crossing, say).
        nearest = frame.nearest
        if nearest is not None and nearest.closing_speed > 0 and abs(nearest.bearing_deg) <= FRONT_CONE_DEG:
            ttc_nearest = nearest.distance / nearest.closing_speed
            scores[Action.FORWARD] *= _clamp(ttc_nearest / 3.0, 0.3, 1.0)

        # Brake wants everything ahead close and speed still worth shedding; a hovering drone
        # that keeps braking never leaves the box, so at low speed the climb/bank take over.
        ahead = sum(_clearance(rays[n]) for n in AHEAD) / len(AHEAD)
        speed_factor = 0.35 + 0.65 * _clamp(drone.speed / 8.0)
        scores[Action.BRAKE] = (1.0 - ahead) ** 2 * speed_factor

        for action in Action:
            scores[action] *= PREFERENCE[action]
        last = frame.last_action
        if last in REVERSE:  # banks and altitude changes carry momentum; reversing them weaves
            scores[REVERSE[last]] *= REVERSE_PENALTY
            scores[last] *= CONTINUE_BONUS

        notes[Action.FORWARD] = _describe(rays["forward"], "ahead")
        notes[Action.BANK_LEFT] = _describe(rays["left_30"], "left")
        notes[Action.BANK_RIGHT] = _describe(rays["right_30"], "right")
        notes[Action.CLIMB] = f"{_describe(rays['up'], 'up')}, headroom {max(headroom + ALTITUDE_MARGIN, 0):.0f} m"
        notes[Action.DESCEND] = f"{_describe(rays['down'], 'down')}, floor {max(floor + ALTITUDE_MARGIN, 0):.0f} m"
        notes[Action.BRAKE] = f"boxed in (ahead {ahead * PLANNING_HORIZON:.0f} m avg, {drone.speed:.1f} m/s)"
        return scores, notes

    def _time_to_collision(self, frame: SensorFrame, rays: dict[str, Ray]) -> float:
        speed = max(frame.drone.speed, 0.1)
        fwd = rays["forward"]
        ttc = fwd.distance / speed if fwd.hit is not None else math.inf
        nearest = frame.nearest
        if nearest is not None and nearest.closing_speed > 0 and abs(nearest.bearing_deg) <= FRONT_CONE_DEG:
            ttc = min(ttc, nearest.distance / nearest.closing_speed)
        return ttc

    def describe(self) -> dict:
        return {
            "kind": "baseline",
            "model": None,
            "device": "cpu",
            "expected_latency_ms": 0.05,
            "planning_horizon_m": PLANNING_HORIZON,
            "notes": (
                "Sector clearance scoring over the 11 rays with forward preference, "
                "anti-oscillation, altitude-band limits and time-to-collision braking."
            ),
        }


@register("heuristic")
def _make_heuristic() -> HeuristicEngine:
    return HeuristicEngine()
