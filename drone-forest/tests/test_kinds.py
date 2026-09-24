"""New obstacle kinds travel through the contract and into Laya's prompt."""
from __future__ import annotations

from conftest import make_frame

from server.engines import framing
from server.engines.heuristic_engine import HeuristicEngine
from server.schemas import PROTOCOL_VERSION, Action, Nearest, SensorFrame


def test_protocol_is_3_and_frames_accept_ogre_and_projectile():
    assert PROTOCOL_VERSION == 4
    f = make_frame({"forward": (9.0, "projectile"), "left_30": (14.0, "ogre")},
                   nearest=Nearest(kind="projectile", distance=9.0, bearing_deg=0.0, elevation_deg=-10.0, closing_speed=30.0))
    assert SensorFrame.model_validate(f.model_dump()).ray("forward").hit == "projectile"


def test_semantic_framing_names_the_new_kinds():
    f = make_frame({"forward": (9.0, "projectile"), "left_30": (14.0, "ogre")},
                   threat=Nearest(kind="projectile", distance=9.0, bearing_deg=0.0, elevation_deg=-10.0, closing_speed=30.0))
    state = framing.render_state(f, "semantic")
    assert "a thrown rock" in state and "an ogre" in state and "Incoming: a thrown rock" in state and "from below" in state
    opts = {o.action.value: o.description for o in framing.render_options(f, "semantic", shuffle=False)}
    assert "a thrown rock" in opts["forward"]


def test_heuristic_reacts_to_a_fast_closing_rock():
    f = make_frame({}, threat=Nearest(kind="projectile", distance=8.0, bearing_deg=5.0, elevation_deg=-15.0, closing_speed=30.0))
    d = HeuristicEngine().decide(f, 1)
    assert d.action in (Action.CLIMB, Action.BANK_LEFT), f"expected a dodge up/away from a rock from below-right, got {d.action}"
    clear = HeuristicEngine().decide(make_frame({}), 1)
    assert clear.action is Action.FORWARD and clear.target_speed == 18.0  # scenery-free frame: no dodge, full speed
