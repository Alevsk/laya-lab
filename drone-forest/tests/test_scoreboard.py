"""The scoreboard aggregates arena results and lets a re-run replace an older one."""
from __future__ import annotations

import json

from server.scoreboard import aggregate, load_results, markdown, parse_seeds


def _score(engine, mode, seed, difficulty, ts, **kw):
    d = {"engine": engine, "mode": mode, "seed": seed, "difficulty": difficulty, "collisions": 1, "collisions_by_kind": {},
         "near_misses": 2, "distance": 500.0, "mean_speed": 14.0, "ticks_met": 450, "ticks_missed": 0, "think_p50_ms": 60.0}
    d.update(kw)
    return {"kind": "event", "ts": ts, "engine": engine, "event": {"type": "score", "frame_id": 1, "t": 45.0, "details": d}}


def test_dedupe_and_aggregate(tmp_path):
    f = tmp_path / "telemetry-1.jsonl"
    lines = [
        _score("laya-ft", "quality", 7, 5, 1.0, collisions=5, collisions_by_kind={"projectile": 5}),
        _score("laya-ft", "quality", 7, 5, 2.0, collisions=3, collisions_by_kind={"projectile": 3}),  # re-run: replaces
        _score("laya-ft", "quality", 8, 5, 3.0, collisions=1, collisions_by_kind={"tree": 1}),
        _score("heuristic", "realtime", 7, 5, 4.0, collisions=0, ticks_met=400, ticks_missed=50, think_p50_ms=0.05),
        {"kind": "event", "ts": 5.0, "engine": "heuristic", "event": {"type": "score", "frame_id": 9, "t": 3.0,
                                                                        "details": {"reason": "collision", "episode": 2}}},  # play mode: ignored
    ]
    f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    results = load_results([str(f)])
    assert len(results) == 3
    rows = aggregate(results)
    ft = next(r for r in rows if r["engine"] == "laya-ft")
    assert ft["runs"] == 2 and ft["seeds"] == [7, 8]
    assert ft["collisions_mean"] == 2.0 and ft["rock_hits_mean"] == 1.5
    assert ft["ticks_met_pct"] == 100.0
    h = next(r for r in rows if r["engine"] == "heuristic")
    assert h["mode"] == "realtime" and abs(h["ticks_met_pct"] - 88.888) < 0.01
    md = markdown(rows)
    assert "| quality | `laya-ft` | 5 | 2 | 2.00 | 1.50 |" in md and "89%" in md


def test_seed_and_seconds_filters(tmp_path):
    f = tmp_path / "telemetry-2.jsonl"
    lines = [_score("heuristic", "quality", 7, 3, 1.0, seconds=45), _score("heuristic", "quality", 8, 3, 2.0, seconds=45),
             _score("heuristic", "quality", 100, 3, 3.0, seconds=60), _score("heuristic", "quality", 7, 3, 4.0, seconds=60)]
    f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert len(load_results([str(f)])) == 3                                    # seed 7 twice: last wins
    assert {r["seed"] for r in load_results([str(f)], seeds=(7, 8))} == {7, 8}
    assert [r["seed"] for r in load_results([str(f)], seeds=(7, 8), seconds=45)] == [7, 8]
    assert parse_seeds("7-8") == (7, 8) and parse_seeds("7") == (7, 7) and parse_seeds(None) is None
