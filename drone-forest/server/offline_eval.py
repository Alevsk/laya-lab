"""Score question FORMULATIONS for the Laya engine against the heuristic teacher, offline.

A formulation is: how a frame becomes (state text, questions) and how the answers become an
Action. Every formulation is scored on the same frames with the same checkpoint, so the only
variable is how the question is asked. Metrics:

  agree      exact agreement with the teacher's action
  reasonable the chosen action is within 80% of the teacher's best score (a "fine" move)
  danger     agreement restricted to frames where the teacher does NOT say forward - the frames
             that decide whether the drone lives
  forward%   how often the formulation says forward on danger frames (the crash rate proxy)
  hist       the formulation's action histogram (a constant predictor shows up here)

    uv run python -m server.offline_eval --per-class 60 --formulations choice_semantic,noul_dirs
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .dataset import ACTIONS, Example, load_examples, stratified, summary
from .engines import framing
from .schemas import Action, SensorFrame

# ---- formulations ------------------------------------------------------------------------

Decode = Callable[[dict[str, Any]], tuple[Action, dict[str, float]]]


@dataclass
class Formulation:
    name: str
    build: Callable[[SensorFrame], tuple[str, dict[str, Any], Decode]]


def _choice(kind: framing.FramingKind, instruction: str | None = None, shuffle: bool = True):
    def build(frame: SensorFrame):
        opts = framing.render_options(frame, kind, shuffle=shuffle, seed=0)
        q = framing.move_question(opts)
        if instruction:
            q["instructions"] = instruction
        state = framing.render_state(frame, kind)

        def decode(ans):
            a, _ = framing.resolve_choice(ans["move"]["choice"], opts)
            return a, {k: float(v) for k, v in ans["move"]["probabilities"].items()}

        return state, {"move": q}, decode

    return build


DIRS = {
    Action.FORWARD: ("straight ahead", ("forward",)),
    Action.BANK_LEFT: ("to the left", ("left_15", "left_30", "left_45")),
    Action.BANK_RIGHT: ("to the right", ("right_15", "right_30", "right_45")),
    Action.CLIMB: ("upward", ("up",)),
    Action.DESCEND: ("downward", ("down",)),
}


def _noul_dirs(kind: framing.FramingKind = "semantic", with_consequence: bool = True, brake_below: float = 0.35):
    """Five independent yes/no questions in ONE forward pass; pick the direction Laya calls safest."""

    def build(frame: SensorFrame):
        state = framing.render_state(frame, kind)
        qs: dict[str, Any] = {}
        for a, (phrase, _rays) in DIRS.items():
            ins = f"Is it safe for the drone to fly {phrase} right now?"
            if with_consequence:
                ins += f" ({framing.describe_action(a, frame, kind)})"
            qs[a.value] = {"type": "noul", "instructions": ins,
                           "criteria": {"true": "yes, that way is clear enough", "false": "no, it would hit something"}}

        def decode(ans):
            p = {a.value: float(ans[a.value]["noul"]) for a in DIRS}
            best = max(p, key=p.get)
            if p[best] < brake_below:
                return Action.BRAKE, {**p, "brake": 1.0 - p[best]}
            return Action(best), {**p, "brake": 1.0 - p[best]}

        return state, qs, decode

    return build


def _score_dirs(kind: framing.FramingKind = "semantic"):
    """Five ordinal clearance questions; pick the direction with the highest expected clearance."""
    levels = ["blocked, an obstacle is very close", "tight, an obstacle is close", "open, some room", "wide open"]

    def build(frame: SensorFrame):
        state = framing.render_state(frame, kind)
        qs = {a.value: {"type": "score", "criteria": levels,
                        "instructions": f"How much clear space does the drone have {phrase}?"}
              for a, (phrase, _) in DIRS.items()}

        def decode(ans):
            s = {a.value: float(ans[a.value]["score"]) for a in DIRS}
            best = max(s, key=s.get)
            return Action(best), {k: v / 3.0 for k, v in s.items()}

        return state, qs, decode

    return build


FORMULATIONS: dict[str, Formulation] = {
    "choice_semantic": Formulation("choice_semantic", _choice("semantic")),
    "choice_numeric": Formulation("choice_numeric", _choice("numeric")),
    "choice_clearspace": Formulation("choice_clearspace", _choice(
        "semantic", "Which direction has the most clear space for the drone to fly into?")),
    "noul_dirs": Formulation("noul_dirs", _noul_dirs("semantic", True)),
    "noul_dirs_plain": Formulation("noul_dirs_plain", _noul_dirs("semantic", False)),
    "noul_dirs_numeric": Formulation("noul_dirs_numeric", _noul_dirs("numeric", True)),
    "score_dirs": Formulation("score_dirs", _score_dirs("semantic")),
}

# ---- scoring -----------------------------------------------------------------------------


def evaluate(agent, form: Formulation, examples: list[Example], device: str) -> dict:
    from .engines.laya_engine import sync

    agree = reasonable = danger_n = danger_agree = danger_forward = 0
    hist: Counter = Counter()
    confusion: dict[str, Counter] = {a: Counter() for a in ACTIONS}
    lat: list[float] = []
    for e in examples:
        state, qs, decode = form.build(e.frame)
        sync(device)
        t0 = time.perf_counter()
        ans = agent.predict(state, qs)["answers"]
        sync(device)
        lat.append((time.perf_counter() - t0) * 1000)
        a, _ = decode(ans)
        hist[a.value] += 1
        confusion[e.teacher.value][a.value] += 1
        agree += a == e.teacher
        reasonable += a.value in e.acceptable()
        if e.teacher is not Action.FORWARD:
            danger_n += 1
            danger_agree += a == e.teacher
            danger_forward += a is Action.FORWARD
    lat.sort()
    n = len(examples)
    return {
        "formulation": form.name, "n": n,
        "agree": agree / n, "reasonable": reasonable / n,
        "danger_n": danger_n,
        "danger_agree": danger_agree / max(1, danger_n),
        "danger_forward": danger_forward / max(1, danger_n),
        "hist": dict(hist.most_common()),
        "confusion": {k: dict(v) for k, v in confusion.items() if v},
        "latency_p50_ms": lat[len(lat) // 2], "latency_p95_ms": lat[int(len(lat) * 0.95) - 1],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--checkpoint", default=os.environ.get("LAYA_CHECKPOINT", "english"))
    ap.add_argument("--formulations", default="choice_semantic,noul_dirs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write full results here")
    args = ap.parse_args(argv)

    from .engines.laya_engine import resolve_device
    from .fastload import load_agent

    examples = load_examples()
    print("dataset:", summary(examples))
    sample = stratified(examples, args.per_class, args.seed)
    print("sample :", summary(sample))
    device = resolve_device(None)
    t0 = time.perf_counter()
    agent = load_agent(args.checkpoint, device)
    print(f"loaded {args.checkpoint} on {agent.device} in {time.perf_counter() - t0:.1f}s\n")

    rows = []
    for name in args.formulations.split(","):
        form = FORMULATIONS[name.strip()]
        r = evaluate(agent, form, sample, device)
        rows.append(r)
        print(f"{r['formulation']:<20} agree={r['agree']:.2f} reasonable={r['reasonable']:.2f} "
              f"danger_agree={r['danger_agree']:.2f} danger_forward={r['danger_forward']:.2f} "
              f"(danger n={r['danger_n']})  p50={r['latency_p50_ms']:.0f}ms")
        print(f"{'':<20} hist={r['hist']}")
    if args.json:
        import json
        json.dump({"checkpoint": args.checkpoint, "sample": summary(sample), "rows": rows}, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
