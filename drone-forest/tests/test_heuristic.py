"""The heuristic is the baseline Laya is measured against, so its behaviour is pinned here."""
from __future__ import annotations

import time

import pytest
from conftest import make_frame

from server.engines.heuristic_engine import HeuristicEngine
from server.schemas import Action, Bounds, Nearest, SensorFrame


@pytest.fixture
def engine() -> HeuristicEngine:
    return HeuristicEngine()


def _valid(decision, frame: SensorFrame) -> None:
    assert decision.engine == "heuristic"
    assert decision.frame_id == frame.frame_id
    assert decision.probabilities is not None
    assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-9
    assert set(decision.probabilities) == {a.value for a in Action}
    assert decision.collision_imminent is not None and 0.0 <= decision.collision_imminent <= 1.0
    assert decision.urgency is not None and 0.0 <= decision.urgency <= 3.0
    assert decision.reason.endswith(f"-> {decision.action.value}")


def test_all_clear_goes_forward(engine: HeuristicEngine) -> None:
    frame = make_frame()
    d = engine.decide(frame, 1)
    _valid(d, frame)
    assert d.action is Action.FORWARD
    assert d.collision_imminent == 0.0
    assert d.urgency == 0.0


def test_tree_ahead_clear_left_banks_left(engine: HeuristicEngine) -> None:
    frame = make_frame({"forward": (6.0, "tree"), "right_15": (7.0, "tree"), "right_30": (9.0, "tree")})
    d = engine.decide(frame, 1)
    _valid(d, frame)
    assert d.action is Action.BANK_LEFT
    assert "forward tree 6 m" in d.reason and "left clear 60 m" in d.reason
    assert d.collision_imminent > 0.8
    assert d.urgency == 3.0


def test_tree_ahead_clear_right_banks_right(engine: HeuristicEngine) -> None:
    frame = make_frame({"forward": (6.0, "tree"), "left_15": (7.0, "tree"), "left_30": (9.0, "tree")})
    assert engine.decide(frame, 1).action is Action.BANK_RIGHT


def test_symmetric_obstacle_breaks_tie_deterministically(engine: HeuristicEngine) -> None:
    frame = make_frame({"forward": (6.0, "tree")})
    first = engine.decide(frame, 1)
    assert first.action in (Action.BANK_LEFT, Action.BANK_RIGHT)
    for i in range(5):
        assert engine.decide(frame, i).action is first.action


def test_boxed_in_at_speed_brakes(engine: HeuristicEngine) -> None:
    close = dict.fromkeys(("forward", "left_15", "right_15", "left_30", "right_30", "left_45", "right_45", "left_60", "right_60"), (5.0, "tree"))
    frame = make_frame({**close, "up": (5.0, "tree")}, speed=12.0)
    d = engine.decide(frame, 1)
    _valid(d, frame)
    assert d.action is Action.BRAKE
    assert d.urgency == 3.0


def test_boxed_in_with_up_clear_climbs_once_slow(engine: HeuristicEngine) -> None:
    close = dict.fromkeys(("forward", "left_15", "right_15", "left_30", "right_30", "left_45", "right_45", "left_60", "right_60"), (5.0, "tree"))
    fast = engine.decide(make_frame(close, speed=12.0), 1)
    slow = engine.decide(make_frame(close, speed=0.5), 2)
    assert fast.action in (Action.BRAKE, Action.CLIMB)
    assert slow.action is Action.CLIMB


def test_never_climbs_at_altitude_max(engine: HeuristicEngine) -> None:
    sides = dict.fromkeys(("forward", "left_15", "right_15", "left_30", "right_30", "left_45", "right_45", "left_60", "right_60"), (5.0, "tree"))
    frame = make_frame(sides, altitude=40.0, speed=0.5, bounds=Bounds(altitude_min=2.0, altitude_max=40.0))
    d = engine.decide(frame, 1)
    assert d.action is not Action.CLIMB
    assert d.probabilities is not None and d.probabilities["climb"] == 0.0


def test_never_descends_at_altitude_min(engine: HeuristicEngine) -> None:
    sides = dict.fromkeys(("forward", "left_15", "right_15", "left_30", "right_30", "left_45", "right_45", "left_60", "right_60", "up"), (5.0, "tree"))
    frame = make_frame(sides, altitude=2.0, speed=0.5)
    d = engine.decide(frame, 1)
    assert d.action is not Action.DESCEND
    assert d.probabilities is not None and d.probabilities["descend"] == 0.0


def test_does_not_reverse_a_bank_when_alternatives_exist(engine: HeuristicEngine) -> None:
    # Forward blocked, both sides equally open: the drone was banking left, so it keeps banking left.
    frame = make_frame({"forward": (8.0, "rock")}, last_action=Action.BANK_LEFT)
    assert engine.decide(frame, 1).action is Action.BANK_LEFT
    frame = make_frame({"forward": (8.0, "rock")}, last_action=Action.BANK_RIGHT)
    assert engine.decide(frame, 1).action is Action.BANK_RIGHT


def test_distant_obstacle_does_not_cause_weaving(engine: HeuristicEngine) -> None:
    frame = make_frame({"forward": (38.0, "tree")})
    assert engine.decide(frame, 1).action is Action.FORWARD


def test_closing_bird_in_front_cone_raises_urgency(engine: HeuristicEngine) -> None:
    bird = Nearest(kind="bird", distance=6.0, bearing_deg=10.0, elevation_deg=0.0, closing_speed=12.0)
    calm = engine.decide(make_frame(), 1)
    threatened = engine.decide(make_frame(nearest=bird), 2)
    assert calm.urgency == 0.0
    assert threatened.urgency == 3.0
    assert threatened.collision_imminent > calm.collision_imminent
    assert threatened.probabilities["forward"] < calm.probabilities["forward"]


def test_missing_ray_falls_back_to_brake(engine: HeuristicEngine) -> None:
    frame = make_frame()
    frame.rays = frame.rays[:-1]
    d = engine.decide(frame, 7)
    assert d.action is Action.BRAKE
    assert d.decision_id == 7
    assert "missing ray" in d.reason


def test_is_deterministic_and_fast(engine: HeuristicEngine) -> None:
    frame = make_frame({"forward": (12.0, "tree"), "left_30": (20.0, "rock")}, last_action=Action.FORWARD)
    a = engine.decide(frame, 1)
    b = engine.decide(frame, 1)
    assert a.probabilities == b.probabilities and a.action is b.action
    n = 2000
    t0 = time.perf_counter()
    for i in range(n):
        engine.decide(frame, i)
    per_call_us = (time.perf_counter() - t0) / n * 1e6
    assert per_call_us < 500, f"{per_call_us:.0f} us per decision"
