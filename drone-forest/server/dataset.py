"""Turn recorded telemetry into a labelled dataset: (SensorFrame, teacher action).

Every telemetry decision record carries the full SensorFrame the engine saw. The teacher is the
geometric heuristic, re-run offline on each frame, so frames flown by any engine (including the
ones Laya crashed on) get a consistent label. This is the dataset a fine-tune learns from and the
offline evaluator scores against.
"""
from __future__ import annotations

import glob
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .engines import create, load_builtin_engines
from .schemas import Action, SensorFrame

ACTIONS = [a.value for a in Action]


@dataclass(frozen=True)
class Example:
    frame: SensorFrame
    teacher: Action
    teacher_probs: dict[str, float]     # heuristic's normalised scores per action
    teacher_collision: float            # heuristic's P(collision imminent)
    teacher_urgency: float              # heuristic's urgency 0..3
    source_engine: str                  # who was flying when this frame was recorded
    source_file: str
    teacher_speed: float = 0.0          # heuristic's target speed, m/s

    def acceptable(self, ratio: float = 0.8) -> set[str]:
        """Actions the teacher scores within `ratio` of its best - 'reasonable' rather than 'exact'."""
        best = max(self.teacher_probs.values())
        return {a for a, p in self.teacher_probs.items() if p >= ratio * best}


def load_examples(pattern: str = "data/telemetry-*.jsonl", limit: int | None = None) -> list[Example]:
    load_builtin_engines()
    teacher = create("heuristic")
    teacher.warmup()
    out: list[Example] = []
    seen: set[tuple] = set()
    for path in sorted(glob.glob(pattern)):
        name = Path(path).name
        for line in open(path):
            r = json.loads(line)
            if r.get("kind") != "decision":
                continue
            frame = SensorFrame.model_validate(r["frame"])
            # collapse exact duplicates (the same frame re-sent while a decision was pending)
            key = (round(frame.drone.altitude, 2), round(frame.drone.speed, 2),
                   tuple((ry.hit, round(ry.distance, 1)) for ry in frame.rays), frame.last_action)
            if key in seen:
                continue
            seen.add(key)
            d = teacher.decide(frame, len(out))
            out.append(Example(frame, d.action, dict(d.probabilities or {}),
                               float(d.collision_imminent or 0.0), float(d.urgency or 0.0), r["engine"], name,
                               teacher_speed=float(d.target_speed or 0.0)))
            if limit and len(out) >= limit:
                return out
    return out


def stratified(examples: list[Example], per_class: int, seed: int = 0) -> list[Example]:
    """Up to `per_class` examples per teacher action, shuffled - so rare actions are represented."""
    rng = random.Random(seed)
    by: dict[str, list[Example]] = {}
    for e in examples:
        by.setdefault(e.teacher.value, []).append(e)
    picked: list[Example] = []
    for a in ACTIONS:
        xs = by.get(a, [])
        rng.shuffle(xs)
        picked.extend(xs[:per_class])
    rng.shuffle(picked)
    return picked


def split(examples: list[Example], val_frac: float = 0.2, seed: int = 0) -> tuple[list[Example], list[Example]]:
    """Train/val split by source FILE (a run), so consecutive near-identical frames never straddle the split."""
    files = sorted({e.source_file for e in examples})
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * val_frac)))
    val_files = set(files[:n_val])
    train = [e for e in examples if e.source_file not in val_files]
    val = [e for e in examples if e.source_file in val_files]
    return train, val


def block_split(examples: list[Example], val_frac: float = 0.2, block: int = 40, seed: int = 0
                ) -> tuple[list[Example], list[Example]]:
    """Split by contiguous BLOCKS of recorded frames (in file order), not by single frames.

    Consecutive frames are near-duplicates; a per-frame split would leak them across the sets.
    Blocks are large enough that neighbouring blocks share little, and every run contributes
    to both sets.
    """
    rng = random.Random(seed)
    train: list[Example] = []
    val: list[Example] = []
    for i in range(0, len(examples), block):
        (val if rng.random() < val_frac else train).extend(examples[i:i + block])
    return train, val


def summary(examples: list[Example]) -> dict:
    return {
        "n": len(examples),
        "teacher_actions": dict(Counter(e.teacher.value for e in examples).most_common()),
        "source_engines": dict(Counter(e.source_engine for e in examples).most_common()),
        "files": len({e.source_file for e in examples}),
    }
