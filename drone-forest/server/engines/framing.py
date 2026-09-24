"""Turn a SensorFrame into text Laya can score - and back.

Laya is an encoder that scores options; it does not read numbers well (the parent repo measured a
constant "forward" on raw raycast JSON). So the default framing is SEMANTIC: distances become words
("a tree, close"), sectors become bearings a person would say ("slightly left"), and every action
is rendered as an option whose description states its CONSEQUENCE IN THIS FRAME, derived from the
rays. A NUMERIC framing (compact JSON, distances in metres) is kept alongside so the two can be
A/B tested through the same engine.

Option order is shuffled per decision with a seeded RNG so a positional bias shows up in telemetry
instead of being mistaken for a preference. Shuffling changes the order only: labels are the
Action values and descriptions are computed from the frame, so mapping a chosen label back is
exact and the semantics never move.

Nothing here ranks, filters or recommends. The consequence text is a faithful reading of the rays
for each action; whether "steer left, which is clear for a long way" beats "continue straight
into a tree that is close" is Laya's call, and the whole point of the simulator is to find out.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Literal

from ..schemas import SPEED_MIN, Action, Bounds, Ray, SensorFrame

FramingKind = Literal["semantic", "numeric"]
FRAMINGS: tuple[FramingKind, ...] = ("semantic", "numeric")

# Upper bound (exclusive) -> word. The last entry catches everything, including "no hit".
DISTANCE_BUCKETS: tuple[tuple[float, str], ...] = (
    (2.0, "touching"),
    (5.0, "very close"),
    (10.0, "close"),
    (20.0, "medium"),
    (40.0, "far"),
    (float("inf"), "clear"),
)

SECTOR_LABELS: dict[str, str] = {
    "forward": "directly ahead",
    "left_15": "slightly left",
    "right_15": "slightly right",
    "left_30": "left",
    "right_30": "right",
    "left_45": "hard left",
    "right_45": "hard right",
    "left_60": "far left",
    "right_60": "far right",
    "up": "above",
    "down": "below",
}

# Rays an action carries the drone through. Banking sweeps the near-side rays; the worst of them
# describes the consequence, because that is the one the drone would meet first.
ACTION_RAYS: dict[Action, tuple[str, ...]] = {
    Action.FORWARD: ("forward",),
    Action.BANK_LEFT: ("left_15", "left_30", "left_45"),
    Action.BANK_RIGHT: ("right_15", "right_30", "right_45"),
    Action.CLIMB: ("up",),
    Action.DESCEND: ("down",),
    Action.BRAKE: ("forward",),
}

ACTION_ORDER: tuple[Action, ...] = tuple(Action)

OBSTACLE_NOUN: dict[str, str] = {
    "tree": "a tree",
    "rock": "a rock",
    "bird": "a bird",
    "ogre": "an ogre",
    "projectile": "a thrown rock",
    "ground": "the ground",
    "wall": "a wall",
}

# How close to an altitude bound counts as "at" it, in metres.
BOUND_MARGIN = 3.0

MOVE_INSTRUCTION = "A drone is flying through a forest and must not crash. Which flight action is safest now?"
COLLISION_INSTRUCTION = "Is the drone about to hit something if it keeps flying straight?"
URGENCY_INSTRUCTION = "How urgently does the drone need to change course?"
SPEED_INSTRUCTION = "How fast should the drone fly right now? Fly at full speed whenever the way ahead is open."
SPEED_LEVELS: tuple[str, ...] = (
    "crawl, something is very close ahead",
    "slow, obstacles are near",
    "fast, the way ahead is mostly open",
    "full speed, the way ahead is wide open",
)
URGENCY_LEVELS: tuple[str, ...] = (
    "no danger, cruise on",
    "mild, adjust soon",
    "serious, act now",
    "critical, collision imminent",
)


# ---- words for numbers -------------------------------------------------------------------


def distance_word(distance: float, hit: bool = True) -> str:
    """Bucket a clearance into a word. A ray that hit nothing is always "clear"."""
    if not hit:
        return "clear"
    for upper, word in DISTANCE_BUCKETS:
        if distance < upper:
            return word
    return "clear"


def proximity(distance: float) -> str:
    """The bucket word as a predicate: "close", "a medium distance away", "far away"."""
    word = distance_word(distance)
    return {"medium": "a medium distance away", "far": "far away"}.get(word, word)


def describe_ray(ray: Ray) -> str:
    """One sector, e.g. "directly ahead: a tree, close." or "slightly left: clear.". """
    label = SECTOR_LABELS.get(ray.name, ray.name.replace("_", " "))
    if ray.hit is None:
        return f"{label}: clear."
    noun = OBSTACLE_NOUN.get(ray.hit, ray.hit)
    return f"{label}: {noun}, {distance_word(ray.distance)}."


def altitude_words(frame: SensorFrame) -> str:
    alt, lo, hi = frame.drone.altitude, frame.bounds.altitude_min, frame.bounds.altitude_max
    if alt <= lo + BOUND_MARGIN:
        return "flying low, near the floor"
    if alt >= hi - BOUND_MARGIN:
        return "flying high, near the ceiling"
    span = hi - lo
    if alt < lo + span / 3:
        return "flying fairly low"
    if alt > hi - span / 3:
        return "flying fairly high"
    return "flying at a comfortable height"


def speed_words(speed: float) -> str:
    if speed < 0.5:
        return "hovering"
    if speed < 4.0:
        return "moving slowly"
    if speed < 10.0:
        return "cruising"
    return "moving fast"


def bearing_words(bearing_deg: float, elevation_deg: float) -> str:
    """Where the nearest obstacle sits, in words, for the "nearest" line."""
    if elevation_deg > 20:
        vertical = " and above"
    elif elevation_deg < -20:
        vertical = " and below"
    else:
        vertical = ""
    mag = abs(bearing_deg)
    if mag < 8:
        horizontal = "directly ahead"
    else:
        side = "left" if bearing_deg < 0 else "right"
        if mag < 22:
            horizontal = f"slightly {side}"
        elif mag < 50:
            horizontal = f"to the {side}"
        else:
            horizontal = f"far {side}"
    return horizontal + vertical


# ---- state rendering ---------------------------------------------------------------------


def render_state_semantic(frame: SensorFrame) -> str:
    lines = [
        f"Drone status: {speed_words(frame.drone.speed)}, {altitude_words(frame)}.",
        "Sensors: " + " ".join(describe_ray(r) for r in frame.rays),
    ]
    if frame.nearest is not None:
        n = frame.nearest
        closing = ", closing in" if n.closing_speed > 0.5 else ""
        lines.append(
            f"Nearest obstacle: {OBSTACLE_NOUN.get(n.kind, n.kind)}, {distance_word(n.distance)}, "
            f"{bearing_words(n.bearing_deg, n.elevation_deg)}{closing}."
        )
    if frame.threat is not None:
        t = frame.threat
        vertical = " from below" if t.elevation_deg < -8 else " from above" if t.elevation_deg > 8 else ""
        lines.append(
            f"Incoming: {OBSTACLE_NOUN.get(t.kind, t.kind)}, {distance_word(t.distance)}, "
            f"{bearing_words(t.bearing_deg, 0.0)}{vertical}, closing fast."
        )
    if frame.last_action is not None:
        lines.append(f"Last action: {frame.last_action.value.replace('_', ' ')}.")
    return "\n".join(lines)


def render_state_numeric(frame: SensorFrame) -> str:
    """Compact JSON; metres, one decimal. The A/B partner of the semantic framing."""
    payload: dict[str, Any] = {
        "speed_mps": round(frame.drone.speed, 1),
        "altitude_m": round(frame.drone.altitude, 1),
        "altitude_range_m": [frame.bounds.altitude_min, frame.bounds.altitude_max],
        "rays_m": {r.name: [round(r.distance, 1), r.hit] for r in frame.rays},
    }
    if frame.nearest is not None:
        n = frame.nearest
        payload["nearest"] = {
            "kind": n.kind,
            "distance_m": round(n.distance, 1),
            "bearing_deg": round(n.bearing_deg),
            "elevation_deg": round(n.elevation_deg),
        }
    if frame.last_action is not None:
        payload["last_action"] = frame.last_action.value
    return json.dumps(payload, separators=(",", ":"))


def render_state(frame: SensorFrame, framing: FramingKind) -> str:
    if framing == "semantic":
        return render_state_semantic(frame)
    if framing == "numeric":
        return render_state_numeric(frame)
    raise ValueError(f"unknown framing {framing!r}; choose one of {FRAMINGS}")


# ---- options: each action's consequence in this frame -------------------------------------


def _worst_ray(frame: SensorFrame, names: tuple[str, ...]) -> Ray:
    """The nearest hit among the given rays, or the first of them when none hit."""
    hits = [frame.ray(n) for n in names if frame.ray(n).hit is not None]
    if not hits:
        return frame.ray(names[0])
    return min(hits, key=lambda r: r.distance)


def _clearance(ray: Ray) -> str:
    return "clear" if ray.hit is None else f"{ray.distance:.0f} m"


def describe_action(action: Action, frame: SensorFrame, framing: FramingKind = "semantic") -> str:
    """The consequence of taking `action` from this frame, in about a dozen words."""
    ray = _worst_ray(frame, ACTION_RAYS[action])
    alt, lo, hi = frame.drone.altitude, frame.bounds.altitude_min, frame.bounds.altitude_max
    noun = OBSTACLE_NOUN.get(ray.hit, ray.hit) if ray.hit is not None else ""
    numeric = framing == "numeric"

    if action is Action.FORWARD:
        if ray.hit is None:
            return "continue straight, the way ahead is clear"
        if numeric:
            return f"continue straight toward {noun} {ray.distance:.0f} m ahead"
        return f"continue straight into {noun} that is {proximity(ray.distance)}"

    if action in (Action.BANK_LEFT, Action.BANK_RIGHT):
        side = "left" if action is Action.BANK_LEFT else "right"
        if ray.hit is None:
            return f"steer {side}, which is clear for a long way"
        if numeric:
            return f"steer {side}, where {noun} is {ray.distance:.0f} m away"
        return f"steer {side}, where {noun} is {proximity(ray.distance)}"

    if action is Action.CLIMB:
        if alt >= hi - BOUND_MARGIN:
            return "climb, but the drone is already at the ceiling"
        if ray.hit is None:
            return "climb, the air above is clear"
        if numeric:
            return f"climb toward {noun} {ray.distance:.0f} m above"
        return f"climb toward {noun}, {proximity(ray.distance)} above"

    if action is Action.DESCEND:
        if alt <= lo + BOUND_MARGIN:
            return "descend, but the drone is already at the floor"
        if ray.hit is None:
            return "descend, the space below is clear"
        if numeric:
            return f"descend toward {noun} {ray.distance:.0f} m below"
        return f"descend toward {noun}, {proximity(ray.distance)} below"

    # BRAKE
    if ray.hit is None:
        return "slow down and hover, nothing is ahead"
    if numeric:
        return f"slow down and hover, {noun} is {ray.distance:.0f} m ahead"
    return f"slow down and hover before {noun} that is {proximity(ray.distance)} ahead"


@dataclass(frozen=True)
class RenderedOption:
    action: Action
    description: str

    @property
    def label(self) -> str:
        return self.action.value


def shuffle_seed(seed: int, frame_id: int) -> int:
    """Per-decision seed that depends on the frame, not on how many decisions came before, so a
    replayed run presents the same order to the model even when frames were dropped."""
    return (seed & 0xFFFFFFFF) << 32 | (frame_id & 0xFFFFFFFF)


def render_options(
    frame: SensorFrame,
    framing: FramingKind = "semantic",
    shuffle: bool = True,
    seed: int = 0,
) -> list[RenderedOption]:
    """Every Action as an option, in the order the model will see them."""
    options = [RenderedOption(a, describe_action(a, frame, framing)) for a in ACTION_ORDER]
    if shuffle:
        random.Random(shuffle_seed(seed, frame.frame_id)).shuffle(options)
    return options


def option_order(options: list[RenderedOption]) -> list[str]:
    return [o.label for o in options]


def resolve_choice(label: str, options: list[RenderedOption]) -> tuple[Action, int]:
    """Map the label Laya picked back to (Action, index as presented)."""
    for i, o in enumerate(options):
        if o.label == label:
            return o.action, i
    raise KeyError(f"label {label!r} is not one of the presented options {option_order(options)}")


# ---- the three questions -----------------------------------------------------------------


def move_question(options: list[RenderedOption]) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": MOVE_INSTRUCTION,
        "criteria": {o.label: o.description for o in options},
    }


def build_questions(options: list[RenderedOption]) -> dict[str, dict[str, Any]]:
    """The one predict() payload: move (choice), collision_imminent (noul), urgency (score), speed (score)."""
    return {
        "move": move_question(options),
        "collision_imminent": {"type": "noul", "instructions": COLLISION_INSTRUCTION},
        "urgency": {"type": "score", "instructions": URGENCY_INSTRUCTION, "criteria": list(URGENCY_LEVELS)},
        "speed": {"type": "score", "instructions": SPEED_INSTRUCTION, "criteria": list(SPEED_LEVELS)},
    }


def speed_from_level(level: float, bounds: Bounds) -> float:
    """Expected speed level (0..3) -> m/s in [SPEED_MIN, bounds.speed_max]. Level 3 is exactly max."""
    frac = max(0.0, min(1.0, level / (len(SPEED_LEVELS) - 1)))
    return round(SPEED_MIN + frac * (bounds.speed_max - SPEED_MIN), 2)


def speed_level(target_speed: float, bounds: Bounds) -> int:
    """m/s -> nearest speed level 0..3 (the inverse of speed_from_level; used for training labels)."""
    span = max(1e-6, bounds.speed_max - SPEED_MIN)
    frac = max(0.0, min(1.0, (target_speed - SPEED_MIN) / span))
    return int(round(frac * (len(SPEED_LEVELS) - 1)))


@dataclass(frozen=True)
class Prompt:
    """Everything that goes into one forward pass, plus what is needed to read the answer back."""

    state: str
    options: list[RenderedOption]
    questions: dict[str, dict[str, Any]]

    @property
    def order(self) -> list[str]:
        return option_order(self.options)


def build_prompt(
    frame: SensorFrame,
    framing: FramingKind = "semantic",
    shuffle: bool = True,
    seed: int = 0,
) -> Prompt:
    options = render_options(frame, framing, shuffle, seed)
    return Prompt(render_state(frame, framing), options, build_questions(options))


# ---- head-budget check ------------------------------------------------------------------


def _internal(question: dict[str, Any]) -> dict[str, Any]:
    """The shape laya.common.build_sequence expects (what Agent._to_internal produces)."""
    crit = question.get("criteria")
    if question["type"] == "choice" and isinstance(crit, list):
        crit = dict.fromkeys(crit)
    return {"t": question["type"], "ins": question["instructions"], "crit": crit}


@dataclass(frozen=True)
class HeadBudgetReport:
    """What laya's build_sequence would do with this prompt. `fits` is the invariant the engine
    relies on: no option was cut (48-token cap), the options were not re-truncated to share the
    head (needs >= 16 tokens left), the instruction was not cut, and the state was not cut."""

    question: str
    n_options: int
    option_tokens: list[int]
    option_tokens_total: int
    instruction_tokens: int
    head_budget: int
    tokens_left_for_instruction: int
    sequence_tokens: int
    max_len: int
    option_capped: bool
    options_re_truncated: bool
    instruction_truncated: bool
    state_truncated: bool

    @property
    def fits(self) -> bool:
        return not (self.option_capped or self.options_re_truncated
                    or self.instruction_truncated or self.state_truncated)


def head_budget_report(
    tok: Any,
    state: str,
    question: dict[str, Any],
    name: str = "move",
    head_max_len: int = 192,
    max_len: int = 512,
) -> HeadBudgetReport:
    """Tokenise exactly the way laya does and report every truncation point.

    `tok` is the agent's tokenizer (`agent.tok`), or any AutoTokenizer loaded from the checkpoint's
    tokenizer directory - the check needs no model weights.
    """
    from laya.common import build_sequence
    from laya.common import render_options as laya_render_options

    q = _internal(question)
    opts = laya_render_options(q)
    raw = [tok(" " + o, add_special_tokens=False)["input_ids"] for o in opts]
    per_option = [1 + min(len(ids), 48) for ids in raw]
    left = head_max_len - sum(per_option)
    ins_tokens = len(tok(f"{q['t']} question: {q['ins']}", add_special_tokens=False)["input_ids"])
    ids, _markers = build_sequence(tok, state, q, max_len, head_max_len)
    return HeadBudgetReport(
        question=name,
        n_options=len(opts),
        option_tokens=per_option,
        option_tokens_total=sum(per_option),
        instruction_tokens=ins_tokens,
        head_budget=head_max_len,
        tokens_left_for_instruction=left,
        sequence_tokens=len(ids),
        max_len=max_len,
        option_capped=any(len(ids_) > 48 for ids_ in raw),
        options_re_truncated=left < 16,
        instruction_truncated=ins_tokens > max(8, left),
        state_truncated=len(ids) >= max_len,
    )


def assert_head_budget(
    tok: Any,
    prompt: Prompt,
    head_max_len: int = 192,
    max_len: int = 512,
) -> list[HeadBudgetReport]:
    """Raise if any of the prompt's questions would be silently truncated by laya."""
    reports = [
        head_budget_report(tok, prompt.state, q, name, head_max_len, max_len)
        for name, q in prompt.questions.items()
    ]
    bad = [r for r in reports if not r.fits]
    if bad:
        raise AssertionError("prompt exceeds laya's head budget: " + "; ".join(
            f"{r.question}: options={r.option_tokens_total}/{r.head_budget} tokens, "
            f"instruction={r.instruction_tokens} (room {max(8, r.tokens_left_for_instruction)}), "
            f"sequence={r.sequence_tokens}/{r.max_len}" for r in bad))
    return reports
