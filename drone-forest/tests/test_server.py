"""HTTP + WebSocket behaviour of the decision service, with the baseline engines only."""
from __future__ import annotations

import json
from pathlib import Path

from conftest import make_frame
from fastapi.testclient import TestClient

from server.schemas import Action, Decision, GameEvent


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["engine"] == "heuristic"
    assert {"random", "heuristic"} <= set(body["engines"])
    assert "heuristic" in body["loaded"]
    assert body["telemetry_file"].endswith(".jsonl")


def test_engines_list(client: TestClient) -> None:
    r = client.get("/engines")
    assert r.status_code == 200
    by_name = {e["name"]: e["describe"] for e in r.json()}
    assert {"random", "heuristic"} <= set(by_name)
    assert by_name["random"]["kind"] == "baseline"
    assert "notes" in by_name["heuristic"]


def test_decide_http_uses_requested_engine(client: TestClient) -> None:
    frame = make_frame({"forward": (6.0, "tree"), "right_30": (8.0, "tree")})
    r = client.post("/decide", json={"frame": frame.model_dump(mode="json"), "engine": "heuristic"})
    assert r.status_code == 200
    d = Decision.model_validate(r.json())
    assert d.engine == "heuristic" and d.action is Action.BANK_LEFT and d.frame_id == frame.frame_id

    r = client.post("/decide", json={"frame": frame.model_dump(mode="json"), "engine": "random"})
    d2 = Decision.model_validate(r.json())
    assert d2.engine == "random" and d2.reason == "uniform random"
    assert d2.decision_id == d.decision_id + 1


def test_decide_http_unknown_engine_is_404(client: TestClient) -> None:
    r = client.post("/decide", json={"frame": make_frame().model_dump(mode="json"), "engine": "nope"})
    assert r.status_code == 404


def test_decide_http_malformed_frame_is_422(client: TestClient) -> None:
    r = client.post("/decide", json={"frame": {"frame_id": 1}})
    assert r.status_code == 422


def test_event_http(client: TestClient) -> None:
    ev = GameEvent(type="collision", frame_id=3, t=1.5, obstacle_kind="tree")
    r = client.post("/event", json={"event": ev.model_dump(mode="json")})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert client.get("/stats").json()["engines"]["heuristic"]["collisions"] == 1


def test_ws_round_trip(client: TestClient) -> None:
    with client.websocket_connect("/ws?engine=heuristic") as ws:
        info = ws.receive_json()
        assert info["type"] == "info" and info["engine"] == "heuristic"
        assert {"random", "heuristic"} <= set(info["engines"])

        for i in range(1, 4):
            ws.send_json({"type": "frame", "frame": make_frame(frame_id=i).model_dump(mode="json")})
            msg = ws.receive_json()
            assert msg["type"] == "decision"
            d = Decision.model_validate(msg["decision"])
            assert d.frame_id == i and d.decision_id == i and d.engine == "heuristic"
            assert d.action is Action.FORWARD


def test_ws_set_engine_switches(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["engine"] == "heuristic"
        ws.send_json({"type": "set_engine", "engine": "random"})
        info = ws.receive_json()
        assert info["type"] == "info" and info["engine"] == "random"
        ws.send_json({"type": "frame", "frame": make_frame(frame_id=9).model_dump(mode="json")})
        msg = ws.receive_json()
        assert msg["decision"]["engine"] == "random" and msg["decision"]["frame_id"] == 9

        ws.send_json({"type": "set_engine", "engine": "does-not-exist"})
        err = ws.receive_json()
        assert err["type"] == "error" and "unknown engine" in err["message"]
        ws.send_json({"type": "frame", "frame": make_frame(frame_id=10).model_dump(mode="json")})
        assert ws.receive_json()["decision"]["engine"] == "random"


def test_ws_unknown_engine_on_connect_falls_back(client: TestClient) -> None:
    with client.websocket_connect("/ws?engine=ghost") as ws:
        err = ws.receive_json()
        assert err["type"] == "error"
        info = ws.receive_json()
        assert info["type"] == "info" and info["engine"] == "heuristic"


def test_ws_malformed_frame_keeps_socket_open(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text("this is not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "frame", "frame": {"frame_id": 1}})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "teleport", "to": "mars"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "frame", "frame": make_frame(frame_id=2).model_dump(mode="json")})
        msg = ws.receive_json()
        assert msg["type"] == "decision" and msg["decision"]["frame_id"] == 2


def test_ws_event_is_recorded(client: TestClient) -> None:
    with client.websocket_connect("/ws?engine=random") as ws:
        ws.receive_json()
        ev = GameEvent(type="near_miss", frame_id=1, t=0.2, obstacle_kind="bird")
        ws.send_json({"type": "event", "event": ev.model_dump(mode="json")})
        ws.send_json({"type": "frame", "frame": make_frame().model_dump(mode="json")})
        ws.receive_json()
    st = client.get("/stats").json()["engines"]["random"]
    assert st["near_misses"] == 1 and st["decisions"] == 1 and st["events"] == 1


def test_telemetry_file_gets_lines(client: TestClient, telemetry_dir: Path) -> None:
    frame = make_frame({"forward": (5.0, "rock")}, frame_id=42)
    client.post("/decide", json={"frame": frame.model_dump(mode="json")})
    client.post("/event", json={"event": GameEvent(type="reset", frame_id=42, t=3.0).model_dump(mode="json")})
    files = list(telemetry_dir.glob("telemetry-*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(line) for line in files[0].read_text().splitlines()]
    kinds = [rec["kind"] for rec in lines]
    assert kinds == ["decision", "event"]
    decision_rec = lines[0]
    assert decision_rec["engine"] == "heuristic"
    assert decision_rec["frame"]["frame_id"] == 42 and len(decision_rec["frame"]["rays"]) == 11
    assert decision_rec["decision"]["action"] in {a.value for a in Action}
    assert lines[1]["event"]["type"] == "reset"

    stats = client.get("/stats").json()
    assert stats["records"] == 2
    h = stats["engines"]["heuristic"]
    assert h["decisions"] == 1 and h["latency_p50_ms"] >= 0 and sum(h["actions"].values()) == 1


def test_stats_option_index_histogram_when_engine_reports_it(client: TestClient) -> None:
    from server.telemetry import Telemetry

    tel: Telemetry = client.app.state.telemetry
    frame = make_frame()
    for idx in (0, 0, 2):
        d = Decision(action=Action.FORWARD, engine="fake", frame_id=1, decision_id=1, latency_ms=1.0, option_index=idx)
        tel.record_decision(frame, d, "fake")
    st = client.get("/stats").json()["engines"]["fake"]
    assert st["option_index"] == {"0": 2, "2": 1}


def test_health_reports_laya_consistently(client: TestClient) -> None:
    """Laya is not required here: whichever way its import went, /health must say so honestly."""
    body = client.get("/health").json()
    if body["laya_import_error"] is None:
        assert "laya" in body["engines"]
    else:
        assert "laya" not in body["engines"]
