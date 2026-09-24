"""Shared fixtures: hand-built sensor frames and an app client with telemetry in a temp dir.

The project is not installed as a package (uv treats it as a virtual project); pyproject's
`pythonpath = ["."]` puts the root on sys.path so `server` imports here and in every test.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.schemas import RAY_SPEC, Action, Bounds, DroneState, Nearest, Ray, SensorFrame, Vec3

RAY_MAX = 60.0


def make_frame(
    distances: dict[str, tuple[float, str]] | None = None,
    *,
    frame_id: int = 1,
    t: float = 0.0,
    altitude: float = 10.0,
    speed: float = 10.0,
    last_action: Action | None = None,
    nearest: Nearest | None = None,
    threat: Nearest | None = None,
    bounds: Bounds | None = None,
) -> SensorFrame:
    """A frame with every ray clear except those in `distances` ({ray: (metres, kind)})."""
    hits = distances or {}
    rays = []
    for name, bearing, elevation in RAY_SPEC:
        if name in hits:
            d, kind = hits[name]
            rays.append(Ray(name=name, bearing_deg=bearing, elevation_deg=elevation, distance=d, hit=kind, max_range=RAY_MAX))
        else:
            rays.append(Ray(name=name, bearing_deg=bearing, elevation_deg=elevation, distance=RAY_MAX, hit=None, max_range=RAY_MAX))
    return SensorFrame(
        frame_id=frame_id,
        t=t,
        drone=DroneState(
            position=Vec3(x=0.0, y=altitude, z=0.0),
            velocity=Vec3(x=0.0, y=0.0, z=-speed),
            heading_deg=0.0,
            altitude=altitude,
            speed=speed,
        ),
        rays=rays,
        nearest=nearest,
        threat=threat,
        bounds=bounds or Bounds(),
        last_action=last_action,
    )


@pytest.fixture
def frame() -> SensorFrame:
    return make_frame()


@pytest.fixture
def telemetry_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "telemetry"
    monkeypatch.setenv("TELEMETRY_DIR", str(d))
    monkeypatch.setenv("DEFAULT_ENGINE", "heuristic")
    return d


@pytest.fixture
def client(telemetry_dir: Path) -> Iterator[TestClient]:
    from server.app import app

    with TestClient(app) as c:
        yield c
