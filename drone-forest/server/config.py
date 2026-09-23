"""Service settings, read from the environment with sane local-dev defaults.

Kept as a plain frozen dataclass rather than pydantic-settings so the service has no extra
dependency and the whole configuration surface fits on one screen. `get_settings()` re-reads
the environment on every call; the app calls it once at startup, tests call it after
monkeypatching.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_TRUE = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in _TRUE


@dataclass(frozen=True, slots=True)
class Settings:
    default_engine: str
    host: str
    port: int
    telemetry_dir: Path
    laya_checkpoint: str
    laya_device: str
    shuffle_options: bool
    random_engine_seed: int


def get_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    return Settings(
        default_engine=os.environ.get("DEFAULT_ENGINE", "heuristic"),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8765")),
        telemetry_dir=Path(os.environ.get("TELEMETRY_DIR", str(root / "data"))),
        laya_checkpoint=os.environ.get("LAYA_CHECKPOINT", "english"),
        laya_device=os.environ.get("LAYA_DEVICE", "auto"),
        shuffle_options=_env_bool("SHUFFLE_OPTIONS", True),
        random_engine_seed=int(os.environ.get("RANDOM_ENGINE_SEED", "0")),
    )
