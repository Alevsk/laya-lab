"""Tests for the Laya engine and its framing.

Framing tests need no model. Model tests skip unless the checkpoint is in the local HF cache
(this repo runs offline). The honesty probe at the end PRINTS what Laya picked on twelve frames
whose obvious answer differs; it asserts nothing, because the simulator exists to measure Laya,
not to assume it is competent. Run with `-s` to see it.
"""
from __future__ import annotations

import collections
import json
import os
import statistics
import time
from typing import Any

import pytest

from server.engines import framing
from server.engines.laya_engine import LayaEngine, LayaEngineConfig, resolve_device, sync
from server.schemas import RAY_SPEC, Action, Bounds, DroneState, Nearest, Ray, SensorFrame, Vec3

RAY_MAX = 60.0
Hit = tuple[float, str]


def frame(
    hits: dict[str, Hit] | None = None,
    *,
    frame_id: int = 1,
    altitude: float = 12.0,
    speed: float = 6.0,
    last_action: Action | None = Action.FORWARD,
    nearest: Nearest | None = None,
) -> SensorFrame:
    """Every ray clear except those in `hits` ({ray: (metres, kind)})."""
    hits = hits or {}
    rays = []
    for name, bearing, elevation in RAY_SPEC:
        if name in hits:
            d, kind = hits[name]
            rays.append(Ray(name=name, bearing_deg=bearing, elevation_deg=elevation, distance=d, hit=kind, max_range=RAY_MAX))
        else:
            rays.append(Ray(name=name, bearing_deg=bearing, elevation_deg=elevation, distance=RAY_MAX, max_range=RAY_MAX))
    return SensorFrame(
        frame_id=frame_id,
        t=float(frame_id) / 10,
        drone=DroneState(position=Vec3(x=0, y=altitude, z=0), velocity=Vec3(x=0, y=0, z=speed),
                         heading_deg=0.0, altitude=altitude, speed=speed),
        rays=rays,
        nearest=nearest,
        bounds=Bounds(),
        last_action=last_action,
    )


TREE_AHEAD: dict[str, Hit] = {"forward": (7.0, "tree")}


# ---- framing: no model needed --------------------------------------------------------------


@pytest.mark.parametrize("distance,word", [
    (0.5, "touching"), (1.99, "touching"), (2.0, "very close"), (4.9, "very close"),
    (5.0, "close"), (9.9, "close"), (10.0, "medium"), (19.9, "medium"),
    (20.0, "far"), (39.9, "far"), (40.0, "clear"), (60.0, "clear"),
])
def test_distance_buckets(distance: float, word: str) -> None:
    assert framing.distance_word(distance) == word


def test_unhit_ray_is_clear_whatever_its_distance() -> None:
    assert framing.distance_word(1.0, hit=False) == "clear"


def test_sector_wording() -> None:
    f = frame({"forward": (7.0, "tree"), "left_45": (25.0, "rock"), "down": (11.0, "ground")})
    text = framing.render_state_semantic(f)
    assert "directly ahead: a tree, close." in text
    assert "hard left: a rock, far." in text
    assert "below: the ground, medium." in text
    assert "slightly left: clear." in text
    assert "above: clear." in text
    assert "cruising" in text
    assert "Last action: forward." in text


@pytest.mark.parametrize("altitude,phrase", [
    (3.0, "near the floor"), (38.5, "near the ceiling"), (8.0, "fairly low"),
    (34.0, "fairly high"), (20.0, "comfortable height"),
])
def test_altitude_words(altitude: float, phrase: str) -> None:
    assert phrase in framing.altitude_words(frame(altitude=altitude))


def test_nearest_line_and_closing() -> None:
    f = frame(TREE_AHEAD, nearest=Nearest(kind="tree", distance=7.0, bearing_deg=-12, elevation_deg=0, closing_speed=6))
    assert "Nearest obstacle: a tree, close, slightly left, closing in." in framing.render_state_semantic(f)


def test_numeric_state_is_compact_json_with_metres() -> None:
    payload = json.loads(framing.render_state_numeric(frame(TREE_AHEAD)))
    assert payload["rays_m"]["forward"] == [7.0, "tree"]
    assert payload["rays_m"]["up"] == [60.0, None]
    assert payload["altitude_range_m"] == [2.0, 40.0]
    assert payload["last_action"] == "forward"


def test_option_descriptions_state_consequences() -> None:
    f = frame({"forward": (7.0, "tree"), "right_30": (12.0, "rock"), "down": (11.0, "ground")})
    d = {a: framing.describe_action(a, f) for a in Action}
    assert d[Action.FORWARD] == "continue straight into a tree that is close"
    assert d[Action.BANK_LEFT] == "steer left, which is clear for a long way"
    assert d[Action.BANK_RIGHT] == "steer right, where a rock is a medium distance away"
    assert d[Action.CLIMB] == "climb, the air above is clear"
    assert d[Action.DESCEND] == "descend toward the ground, a medium distance away below"
    assert d[Action.BRAKE] == "slow down and hover before a tree that is close ahead"
    assert all(len(text.split()) <= 12 for text in d.values())


def test_bank_uses_worst_ray_in_its_sector() -> None:
    f = frame({"left_15": (30.0, "tree"), "left_45": (3.0, "bird")})
    assert framing.describe_action(Action.BANK_LEFT, f) == "steer left, where a bird is very close"


def test_altitude_bounds_show_in_climb_and_descend() -> None:
    assert framing.describe_action(Action.CLIMB, frame(altitude=38.0)) == "climb, but the drone is already at the ceiling"
    assert framing.describe_action(Action.DESCEND, frame(altitude=3.0)) == "descend, but the drone is already at the floor"


def test_numeric_descriptions_carry_metres() -> None:
    f = frame({"forward": (7.0, "tree"), "up": (4.0, "bird")})
    assert framing.describe_action(Action.FORWARD, f, "numeric") == "continue straight toward a tree 7 m ahead"
    assert framing.describe_action(Action.CLIMB, f, "numeric") == "climb toward a bird 4 m above"


def test_unshuffled_order_is_action_order() -> None:
    opts = framing.render_options(frame(), shuffle=False)
    assert [o.action for o in opts] == list(Action)


def test_shuffle_is_seeded_per_frame_and_round_trips() -> None:
    f = frame(TREE_AHEAD, frame_id=42)
    a = framing.render_options(f, seed=7)
    b = framing.render_options(f, seed=7)
    assert framing.option_order(a) == framing.option_order(b)
    assert framing.option_order(framing.render_options(f, seed=8)) != framing.option_order(a)
    assert sorted(o.action for o in a) == sorted(Action)
    unshuffled = {o.action: o.description for o in framing.render_options(f, shuffle=False)}
    for i, o in enumerate(a):
        assert o.description == unshuffled[o.action]      # shuffling never changes the semantics
        assert framing.resolve_choice(o.label, a) == (o.action, i)
    with pytest.raises(KeyError):
        framing.resolve_choice("hover", a)


def test_orders_differ_across_frames() -> None:
    orders = {tuple(framing.option_order(framing.render_options(frame(frame_id=i)))) for i in range(50)}
    assert len(orders) > 10


def test_questions_are_one_choice_one_noul_two_scores() -> None:
    p = framing.build_prompt(frame(TREE_AHEAD))
    assert set(p.questions) == {"move", "collision_imminent", "urgency", "speed"}
    assert p.questions["speed"]["type"] == "score" and len(p.questions["speed"]["criteria"]) == 4
    assert p.questions["move"]["type"] == "choice"
    assert list(p.questions["move"]["criteria"]) == p.order
    assert p.questions["collision_imminent"]["type"] == "noul"
    assert p.questions["urgency"]["type"] == "score"
    assert len(p.questions["urgency"]["criteria"]) == 4


def test_config_from_env() -> None:
    cfg = LayaEngineConfig.from_env({"LAYA_DEVICE": "auto", "FRAMING": "numeric", "SHUFFLE_OPTIONS": "0", "LAYA_SEED": "5"})
    assert cfg.device is None and cfg.framing == "numeric" and not cfg.shuffle_options and cfg.seed == 5
    assert LayaEngineConfig.from_env({}).framing == "semantic"
    assert LayaEngineConfig.from_env({"LAYA_DEVICE": "cpu"}).device == "cpu"
    with pytest.raises(ValueError):
        LayaEngineConfig.from_env({"FRAMING": "poetry"})


def test_resolve_device_honours_request() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device(None) in ("cuda", "mps", "cpu")


# ---- with the checkpoint (tokenizer only, then the model) ------------------------------------


def cached_checkpoint_dir() -> str | None:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download("convaiinnovations/laya", local_files_only=True,
                                 allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
    except Exception:
        return None


needs_checkpoint = pytest.mark.skipif(cached_checkpoint_dir() is None, reason="Laya checkpoint not in the local HF cache")


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    d = cached_checkpoint_dir()
    assert d is not None
    _fix_tokenizer_config(d)
    return AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))


def stress_frames() -> list[SensorFrame]:
    """The longest prompts the framing can produce: every ray hit, long nouns, nearest, bounds."""
    everything = {n: (3.0 + i, "ground" if n == "down" else "bird") for i, (n, _, _) in enumerate(RAY_SPEC)}
    return [
        frame(everything, nearest=Nearest(kind="bird", distance=3.0, bearing_deg=-55, elevation_deg=30, closing_speed=9)),
        frame(everything, altitude=39.0, speed=14.0),
        frame(everything, altitude=2.5, speed=0.2),
        frame(TREE_AHEAD),
        frame(),
    ]


@needs_checkpoint
@pytest.mark.parametrize("kind", framing.FRAMINGS)
def test_prompts_fit_head_budget(tokenizer: Any, kind: framing.FramingKind) -> None:
    worst = 0
    for f in stress_frames():
        reports = framing.assert_head_budget(tokenizer, framing.build_prompt(f, kind))
        move = next(r for r in reports if r.question == "move")
        assert move.n_options == 6
        worst = max(worst, move.option_tokens_total)
    assert worst < 192 - 16
    print(f"\n[{kind}] worst-case move option tokens: {worst}/192")


@pytest.fixture(scope="module")
def engine() -> LayaEngine:
    e = LayaEngine(LayaEngineConfig(checkpoint="english", device=None, framing="semantic", shuffle_options=True, seed=0))
    e.warmup()
    return e


def check_decision(d: Any, f: SensorFrame, engine: LayaEngine) -> None:
    assert d.engine == "laya" and d.frame_id == f.frame_id
    assert d.action in Action
    assert d.latency_ms > 0
    assert d.probabilities is not None and set(d.probabilities) == {a.value for a in Action}
    assert abs(sum(d.probabilities.values()) - 1.0) < 0.01
    assert d.option_order is not None and sorted(d.option_order) == sorted(a.value for a in Action)
    assert d.option_index is not None and d.option_order[d.option_index] == d.action.value
    assert d.action.value == max(d.probabilities, key=d.probabilities.get)     # argmax, nothing else
    assert d.confidence is not None and 0.0 <= d.confidence <= 1.0
    assert d.collision_imminent is not None and 0.0 <= d.collision_imminent <= 1.0
    assert d.urgency is not None and 0.0 <= d.urgency <= 3.0
    assert d.reason.startswith('laya picked "')


@needs_checkpoint
def test_warmup_and_describe(engine: LayaEngine) -> None:
    info = engine.describe()
    assert info["loaded"] and info["warmup_latency_ms"] > 0 and info["load_ms"] > 0
    assert info["framing"] == "semantic" and info["shuffle_options"] is True
    print(f"\nlaya loaded on {info['device']} in {info['load_ms']} ms; warmup predict {info['warmup_latency_ms']} ms")


@needs_checkpoint
def test_decide_on_hand_built_frames(engine: LayaEngine) -> None:
    frames = [
        frame(TREE_AHEAD, frame_id=1, nearest=Nearest(kind="tree", distance=7, bearing_deg=0, elevation_deg=0, closing_speed=6)),
        frame({"forward": (3.0, "rock"), "left_15": (4.0, "rock"), "right_30": (6.0, "tree")}, frame_id=2, altitude=3.0),
        frame(frame_id=3, altitude=38.0, speed=12.0),
    ]
    for i, f in enumerate(frames):
        d = engine.decide(f, decision_id=i + 1)
        check_decision(d, f, engine)
        print(f"\nframe {f.frame_id}: {d.action.value} idx={d.option_index} conf={d.confidence} "
              f"collision={d.collision_imminent} urgency={d.urgency} {d.latency_ms:.1f} ms")


@needs_checkpoint
def test_numeric_framing_also_decides(tokenizer: Any) -> None:
    e = LayaEngine(LayaEngineConfig(framing="numeric", shuffle_options=False))
    e.warmup()
    d = e.decide(frame(TREE_AHEAD, frame_id=9), decision_id=1)
    check_decision(d, frame(TREE_AHEAD, frame_id=9), e)
    assert d.option_order == [a.value for a in Action]
    assert e.describe()["framing"] == "numeric"


@needs_checkpoint
def test_failure_returns_brake_not_raise(engine: LayaEngine) -> None:
    broken = LayaEngine(engine.config)
    broken.agent = object()                   # has no predict(); decide must still answer
    d = broken.decide(frame(frame_id=5), decision_id=1)
    assert d.action is Action.BRAKE and "laya failed" in d.reason


@pytest.fixture(scope="module")
def numeric_engine() -> LayaEngine:
    e = LayaEngine(LayaEngineConfig(checkpoint="english", device=None, framing="numeric", shuffle_options=True, seed=0))
    e.warmup()
    return e


@needs_checkpoint
@pytest.mark.parametrize("engine_fixture", ["engine", "numeric_engine"])
def test_honesty_probe_prints_picks(request: pytest.FixtureRequest, engine_fixture: str) -> None:
    """Twelve frames where the obvious answer differs, under both framings. Printed, never asserted."""
    engine: LayaEngine = request.getfixturevalue(engine_fixture)
    cases: list[tuple[str, SensorFrame]] = []
    fid = 100
    for dist in (3.0, 6.0, 9.0):
        fid += 1
        cases.append(("tree ahead, left clear", frame({"forward": (dist, "tree"), "right_15": (dist, "tree"), "right_30": (dist + 2, "tree"), "right_45": (dist + 4, "tree"), "up": (dist, "tree"), "down": (dist, "ground")}, frame_id=fid)))
        fid += 1
        cases.append(("tree ahead, right clear", frame({"forward": (dist, "tree"), "left_15": (dist, "tree"), "left_30": (dist + 2, "tree"), "left_45": (dist + 4, "tree"), "up": (dist, "tree"), "down": (dist, "ground")}, frame_id=fid)))
        fid += 1
        cases.append(("tree ahead, up clear", frame({"forward": (dist, "tree"), "left_15": (dist, "tree"), "right_15": (dist, "tree"), "left_30": (dist, "tree"), "right_30": (dist, "tree"), "left_45": (dist, "tree"), "right_45": (dist, "tree"), "down": (dist, "ground")}, frame_id=fid)))
        fid += 1
        cases.append(("all clear", frame(frame_id=fid, speed=dist * 2)))

    picks: list[tuple[str, str, int, float]] = []
    for i, (label, f) in enumerate(cases):
        d = engine.decide(f, decision_id=200 + i)
        check_decision(d, f, engine)
        picks.append((label, d.action.value, d.option_index or 0, d.collision_imminent or 0.0))

    by_case: dict[str, list[str]] = collections.defaultdict(list)
    for label, action, _, _ in picks:
        by_case[label].append(action)
    idx_hist = collections.Counter(i for _, _, i, _ in picks)
    action_hist = collections.Counter(a for _, a, _, _ in picks)

    print(f"\n=== Laya honesty probe ({engine.config.framing} framing, shuffled options, {engine.config.checkpoint} checkpoint) ===")
    for label, action, idx, coll in picks:
        print(f"  {label:26s} -> {action:10s} option_index={idx} collision_imminent={coll:.2f}")
    print("  per-scenario picks:")
    for label, actions in by_case.items():
        print(f"    {label:26s}: {actions}")
    print(f"  action histogram:        {dict(action_hist)}")
    print(f"  chosen-index histogram:  {dict(sorted(idx_hist.items()))}")
    print(f"  distinct actions: {len(action_hist)}/{len(picks)} decisions; "
          f"most common action share: {action_hist.most_common(1)[0][1] / len(picks):.0%}; "
          f"most common index share: {idx_hist.most_common(1)[0][1] / len(picks):.0%}")


@needs_checkpoint
def test_latency_distribution(engine: LayaEngine) -> None:
    frames = [frame({"forward": (2.0 + i, "tree"), "left_30": (60.0 - i * 2, "rock") if i % 2 else (5.0, "bird")},
                    frame_id=300 + i, altitude=4.0 + i * 1.5, speed=2.0 + i * 0.5) for i in range(20)]
    samples = []
    for i, f in enumerate(frames):
        d = engine.decide(f, decision_id=300 + i)
        assert d.latency_ms > 0
        samples.append(d.latency_ms)
    samples.sort()
    p50 = statistics.median(samples)
    p95 = samples[int(round(0.95 * (len(samples) - 1)))]
    print(f"\nlaya decide() latency on {engine.device}, n={len(samples)}: p50={p50:.1f} ms p95={p95:.1f} ms "
          f"min={samples[0]:.1f} max={samples[-1]:.1f} (device-synchronised)")
    assert p95 < 5000


@needs_checkpoint
def test_unsynchronised_timer_would_lie(engine: LayaEngine) -> None:
    """Why decide() synchronises anyway. Laya's predict() ends with a .cpu() copy of the logits, which
    already blocks on the GPU, so today the two timers agree; the explicit sync keeps the measurement
    honest if that implementation detail ever changes. Printed for the record, not asserted."""
    p = framing.build_prompt(frame(TREE_AHEAD, frame_id=400))
    bare, synced = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        engine.agent.predict(p.state, p.questions)
        bare.append((time.perf_counter() - t0) * 1000)
        sync(engine.device)
        t0 = time.perf_counter()
        engine.agent.predict(p.state, p.questions)
        sync(engine.device)
        synced.append((time.perf_counter() - t0) * 1000)
    print(f"\nbare timer p50={statistics.median(bare):.1f} ms vs synchronised p50={statistics.median(synced):.1f} ms on {engine.device}")
    assert statistics.median(synced) > 0
