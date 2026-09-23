"""The offline summary must agree with the live stats for the same records."""
from __future__ import annotations

import json
from pathlib import Path

from server.schemas import Action, Decision, GameEvent, SensorFrame
from server.telemetry import Telemetry, latest_file, main, summarize


def _decision(frame_id: int, action: Action, latency: float, idx: int | None = None) -> Decision:
    return Decision(action=action, engine="e", frame_id=frame_id, decision_id=frame_id, latency_ms=latency, option_index=idx)


def test_summary_matches_live_stats(tmp_path: Path, frame: SensorFrame) -> None:
    t = Telemetry(tmp_path)
    t.record_decision(frame, _decision(1, Action.FORWARD, 2.0, 0), "laya")
    t.record_decision(frame, _decision(2, Action.BANK_LEFT, 4.0, 3), "laya")
    t.record_decision(frame, _decision(3, Action.FORWARD, 0.1), "heuristic")
    t.record_event(GameEvent(type="collision", frame_id=2, t=0.2, obstacle_kind="tree"), "laya")
    t.record_event(GameEvent(type="near_miss", frame_id=3, t=0.3, obstacle_kind="bird"), "heuristic")
    live = t.stats()
    t.close()

    offline = summarize(t.path)
    assert offline["records"] == live["records"] == 5
    assert offline["engines"] == live["engines"]
    assert offline["engines"]["laya"]["option_index"] == {"0": 1, "3": 1}
    assert offline["engines"]["laya"]["collisions"] == 1
    assert offline["engines"]["heuristic"]["near_misses"] == 1


def test_cli_picks_newest_file(tmp_path: Path, capsys, monkeypatch) -> None:
    older = tmp_path / "telemetry-20260101T000000Z.jsonl"
    newer = tmp_path / "telemetry-20260102T000000Z.jsonl"
    older.write_text("")
    newer.write_text(
        json.dumps({"kind": "event", "ts": "x", "engine": "random", "event": {"type": "reset", "frame_id": 0, "t": 0}}) + "\n"
    )
    assert latest_file(tmp_path) == newer
    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path))
    assert main([]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["file"] == str(newer)
    assert out["engines"]["random"]["events"] == 1


def test_cli_reports_missing_file(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path / "empty"))
    assert main([]) == 1
    assert "no telemetry file" in capsys.readouterr().err
