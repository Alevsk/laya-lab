"""Append-only JSONL telemetry plus in-memory running stats.

Every decision record carries the full SensorFrame and the full Decision, so the file is a
self-contained dataset: a future fine-tune can pair each frame with what the heuristic chose,
what Laya chose, and (via the event records) what happened next. One file per service run,
flushed on every write so a crash loses nothing already decided.

The running stats are what the HUD and /stats show: per engine, decision count, latency
percentiles over a bounded window, collision and near-miss counts, the action histogram and -
when the engine reports which option slot it picked - the option_index histogram. That last
one is the positional-bias audit: a model that always picks slot 0 shows up there immediately.
"""
from __future__ import annotations

import json
import threading
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from .schemas import Decision, GameEvent, SensorFrame

LATENCY_WINDOW = 10_000
"""Decisions kept per engine for percentile computation."""


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


class EngineStats:
    __slots__ = ("actions", "collisions", "decisions", "events", "latencies", "near_misses", "option_index")

    def __init__(self) -> None:
        self.decisions = 0
        self.latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self.collisions = 0
        self.near_misses = 0
        self.events = 0
        self.actions: Counter[str] = Counter()
        self.option_index: Counter[int] = Counter()

    def snapshot(self) -> dict:
        lat = list(self.latencies)
        return {
            "decisions": self.decisions,
            "latency_p50_ms": _percentile(lat, 0.50),
            "latency_p95_ms": _percentile(lat, 0.95),
            "collisions": self.collisions,
            "near_misses": self.near_misses,
            "events": self.events,
            "actions": dict(sorted(self.actions.items())),
            "option_index": {str(k): v for k, v in sorted(self.option_index.items())},
        }


class Telemetry:
    """One instance per service run. Safe to call from any thread."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now(UTC)
        stamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        self.path = directory / f"telemetry-{stamp}.jsonl"
        self._file: IO[str] = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._stats: dict[str, EngineStats] = {}
        self._records = 0

    def record_decision(self, frame: SensorFrame, decision: Decision, engine: str) -> None:
        record = {
            "kind": "decision",
            "ts": _now(),
            "engine": engine,
            "frame": frame.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
        }
        with self._lock:
            st = self._stats.setdefault(engine, EngineStats())
            st.decisions += 1
            st.latencies.append(decision.latency_ms)
            st.actions[decision.action.value] += 1
            if decision.option_index is not None:
                st.option_index[decision.option_index] += 1
            self._write(record)

    def record_event(self, event: GameEvent, engine: str) -> None:
        record = {"kind": "event", "ts": _now(), "engine": engine, "event": event.model_dump(mode="json")}
        with self._lock:
            st = self._stats.setdefault(engine, EngineStats())
            st.events += 1
            if event.type == "collision":
                st.collisions += 1
            elif event.type == "near_miss":
                st.near_misses += 1
            self._write(record)

    def stats(self) -> dict:
        with self._lock:
            return {
                "file": str(self.path),
                "started_at": self.started_at.isoformat(),
                "records": self._records,
                "engines": {name: st.snapshot() for name, st in sorted(self._stats.items())},
            }

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record, separators=(",", ":")))
        self._file.write("\n")
        self._file.flush()
        self._records += 1


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# ---- offline summary (CLI) -------------------------------------------------------------------


def summarize(path: Path) -> dict:
    """Recompute `stats()` from a telemetry file, so a finished run can be inspected offline.

    Reads the JSONL back through the same `EngineStats` accumulator the live service uses, so
    `python -m server.telemetry` and `GET /stats` agree on every number.
    """
    by_engine: dict[str, EngineStats] = {}
    records = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records += 1
            st = by_engine.setdefault(rec["engine"], EngineStats())
            if rec["kind"] == "decision":
                d = rec["decision"]
                st.decisions += 1
                st.latencies.append(float(d["latency_ms"]))
                st.actions[d["action"]] += 1
                if d.get("option_index") is not None:
                    st.option_index[int(d["option_index"])] += 1
            elif rec["kind"] == "event":
                ev = rec["event"]
                st.events += 1
                if ev["type"] == "collision":
                    st.collisions += 1
                elif ev["type"] == "near_miss":
                    st.near_misses += 1
    return {
        "file": str(path),
        "records": records,
        "engines": {name: st.snapshot() for name, st in sorted(by_engine.items())},
    }


def latest_file(directory: Path) -> Path | None:
    files = sorted(directory.glob("telemetry-*.jsonl"))
    return files[-1] if files else None


def main(argv: list[str] | None = None) -> int:
    """`python -m server.telemetry [file]` — summarise a telemetry file (default: the newest in data/)."""
    import argparse
    import sys

    from .config import get_settings

    parser = argparse.ArgumentParser(description="Summarise a drone-forest telemetry JSONL file.")
    parser.add_argument("file", nargs="?", type=Path, help="telemetry-*.jsonl (default: newest in TELEMETRY_DIR)")
    args = parser.parse_args(argv)
    path = args.file or latest_file(get_settings().telemetry_dir)
    if path is None or not path.exists():
        print("no telemetry file found", file=sys.stderr)
        return 1
    print(json.dumps(summarize(path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
