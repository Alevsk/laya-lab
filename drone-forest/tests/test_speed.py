"""The engine-recommended speed: max whenever the way is open, held down near obstacles."""
from __future__ import annotations

from conftest import make_frame

from server.engines import framing
from server.engines.heuristic_engine import HeuristicEngine
from server.engines.random_engine import RandomEngine
from server.schemas import SPEED_MIN, Action, Bounds, Nearest


def test_heuristic_flies_at_max_speed_when_clear():
    d = HeuristicEngine().decide(make_frame({}), 1)
    assert d.target_speed == make_frame({}).bounds.speed_max
    assert "speed 18 m/s -> forward" in d.reason


def test_heuristic_slows_toward_an_obstacle_ahead():
    close = HeuristicEngine().decide(make_frame({"forward": (6.0, "tree")}), 1)
    mid = HeuristicEngine().decide(make_frame({"forward": (20.0, "tree")}), 1)
    assert close.target_speed == SPEED_MIN
    assert SPEED_MIN < mid.target_speed < 18.0
    assert mid.target_speed <= 20.0 / 2.5 + 1e-9        # keeps 2.5 s of time-to-collision


def test_heuristic_brake_means_minimum_speed():
    frame = make_frame({"forward": (3.0, "tree"), "left_15": (3.0, "tree"), "right_15": (3.0, "tree"),
                        "left_30": (3.0, "tree"), "right_30": (3.0, "tree"), "left_45": (3.0, "tree"),
                        "right_45": (3.0, "tree"), "left_60": (3.0, "tree"), "right_60": (3.0, "tree"),
                        "up": (3.0, "tree"), "down": (3.0, "ground")}, speed=12.0)
    d = HeuristicEngine().decide(frame, 1)
    assert d.action is Action.BRAKE and d.target_speed == SPEED_MIN


def test_heuristic_respects_a_closing_bird_in_the_front_cone():
    frame = make_frame({}, nearest=Nearest(kind="bird", distance=5.0, bearing_deg=10.0, elevation_deg=0.0, closing_speed=10.0))
    d = HeuristicEngine().decide(frame, 1)
    assert d.target_speed <= max(SPEED_MIN, 5.0 / 2.5) + 1e-9


def test_random_speed_is_inside_the_band():
    e = RandomEngine()
    speeds = {e.decide(make_frame({}), i).target_speed for i in range(30)}
    assert all(SPEED_MIN <= s <= 18.0 for s in speeds) and len(speeds) > 5


def test_framing_asks_a_speed_question_and_maps_levels_both_ways():
    opts = framing.render_options(make_frame({}), "semantic", shuffle=False)
    qs = framing.build_questions(opts)
    assert qs["speed"]["type"] == "score" and len(qs["speed"]["criteria"]) == 4
    b = Bounds()
    assert framing.speed_from_level(3.0, b) == b.speed_max
    assert framing.speed_from_level(0.0, b) == SPEED_MIN
    for lvl in range(4):
        assert framing.speed_level(framing.speed_from_level(lvl, b), b) == lvl
    assert framing.speed_level(b.speed_max + 5, b) == 3      # clamped
