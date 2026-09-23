"""Dataset and fine-tune plumbing: no model weights needed (the tokenizer is enough)."""
from __future__ import annotations

import os
import types

import pytest
from conftest import make_frame

from server.dataset import ACTIONS, Example, block_split, stratified, summary
from server.engines import framing
from server.schemas import Action, Decision


def _example(i: int, teacher: Action, source_file: str = "f0") -> Example:
    frame = make_frame({"forward": (6.0 if teacher is not Action.FORWARD else 60.0, "tree")}, frame_id=i)
    probs = dict.fromkeys(ACTIONS, 0.05)
    probs[teacher.value] = 0.75
    return Example(frame, teacher, probs, 0.9, 2.0, "heuristic", source_file)


def test_block_split_keeps_blocks_whole_and_never_leaks():
    ex = [_example(i, Action.FORWARD, f"f{i // 100}") for i in range(400)]
    train, val = block_split(ex, val_frac=0.25, block=40, seed=3)
    assert len(train) + len(val) == 400
    assert val, "some blocks must land in val"
    ids_tr = {e.frame.frame_id for e in train}
    ids_va = {e.frame.frame_id for e in val}
    assert not ids_tr & ids_va
    # whole blocks: every 40-aligned block is entirely in one set
    for start in range(0, 400, 40):
        block = set(range(start, start + 40))
        assert block <= ids_tr or block <= ids_va


def test_stratified_caps_each_class_and_shuffles():
    ex = [_example(i, Action.FORWARD) for i in range(50)] + [_example(100 + i, Action.BANK_LEFT) for i in range(5)]
    s = stratified(ex, per_class=10, seed=0)
    counts = summary(s)["teacher_actions"]
    assert counts == {"forward": 10, "bank_left": 5}
    assert [e.frame.frame_id for e in s] != sorted(e.frame.frame_id for e in s)


def test_acceptable_set_uses_ratio_of_best():
    e = _example(1, Action.BANK_LEFT)
    e = Example(e.frame, e.teacher, {"forward": 0.4, "bank_left": 0.5, "brake": 0.1}, 0.5, 1.0, "h", "f")
    assert e.acceptable(0.8) == {"forward", "bank_left"}
    assert e.acceptable(0.95) == {"bank_left"}


@pytest.fixture(scope="module")
def tok_agent():
    """A stand-in for a laya Agent: tokenizer + cfg is all build_items needs."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    try:
        snap = snapshot_download("convaiinnovations/laya", local_files_only=True, allow_patterns=["tokenizer/*"])
    except Exception:
        pytest.skip("laya tokenizer not cached")
    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    return types.SimpleNamespace(tok=tok, cfg={"max_len": 512, "head_max_len": 192})


def test_build_items_labels_follow_the_shuffle(tok_agent):
    from server.finetune import build_items

    e = _example(7, Action.BANK_LEFT)
    for seed in range(5):
        items = build_items(tok_agent, e, seed)
        assert [it["qid"] for it in items] == ["move", "collision_imminent", "urgency", "speed"]
        opts = framing.render_options(e.frame, "semantic", shuffle=True, seed=seed)
        move = items[0]
        assert opts[move["label"]].action is Action.BANK_LEFT       # label index points at the teacher's action
        assert len(move["markers"]) == 6                            # every option survived build_sequence
        assert items[1]["label"] == 1 and items[2]["label"] == 2    # collision 0.9 -> yes; urgency 2.0 -> level 2
        assert items[3]["label"] == 0                               # teacher_speed 0.0 -> slowest level
    orders = {tuple(o.action for o in framing.render_options(e.frame, "semantic", True, s)) for s in range(5)}
    assert len(orders) > 1, "shuffle must actually vary the order across seeds"


def test_ft_engine_is_registered_and_points_at_a_checkpoint_dir():
    from server.engines import available, load_builtin_engines
    from server.engines.laya_ft_engine import DEFAULT_CHECKPOINT, LayaFtEngine

    load_builtin_engines()
    assert "laya-ft" in available()
    eng = LayaFtEngine()
    assert eng.name == "laya-ft"
    assert eng.config.checkpoint.endswith("laya-drone-ft") or os.path.isdir(eng.config.checkpoint)
    assert DEFAULT_CHECKPOINT.endswith(os.path.join("data", "checkpoints", "laya-drone-ft"))
    d = eng.describe()
    assert d["fine_tuned"] is True and d["engine"] == "laya-ft"
    assert isinstance(Decision, type)
