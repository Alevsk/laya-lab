"""Fine-tune Laya on this simulator's own telemetry, with the geometric heuristic as teacher.

Why: zero-shot, every question formulation scores near chance against the teacher
(server/offline_eval.py). Upstream says the base checkpoints are a base to fine-tune, and the
telemetry the service writes is exactly the dataset for it.

What: supervised multi-task training on the SAME three questions the engine asks at inference
(move: 6-way choice, collision_imminent: noul, urgency: 4-level score, speed: 4-level score), built with laya's own
build_sequence so train and inference see byte-identical prompts. Option order is re-shuffled
for every example every epoch, so the model cannot learn a slot; it has to read the words.
By default the encoder is frozen and only the decision head (2 transformer layers + scorer +
type embedding, ~26.5M params) trains; --unfreeze-top N also trains the top N encoder layers.

Output: a checkpoint directory in laya's own layout, loadable by laya.load(path) and served by
the `laya-ft` engine with no other change.

    uv run python -m server.finetune --epochs 3 --max-minutes 7
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from .dataset import Example, block_split, load_examples, summary
from .engines import framing
from .schemas import Action

ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = ROOT / "data" / "checkpoints" / "laya-drone-ft"


def build_items(agent, e: Example, seed: int) -> list[dict]:
    """The three question sequences for one example, with integer labels, laya-style."""
    from laya.agent import Agent
    from laya.common import QTYPES, build_sequence, render_options

    opts = framing.render_options(e.frame, "semantic", shuffle=True, seed=seed)
    prompt = framing.Prompt(framing.render_state(e.frame, "semantic"), opts, framing.build_questions(opts))
    labels = {
        "move": [o.action for o in opts].index(e.teacher),
        "collision_imminent": 1 if e.teacher_collision >= 0.5 else 0,
        "urgency": max(0, min(3, int(round(e.teacher_urgency)))),
        "speed": framing.speed_level(e.teacher_speed, e.frame.bounds),
    }
    items = []
    for qid, q in prompt.questions.items():
        qi = Agent._to_internal(q)
        ids, markers = build_sequence(agent.tok, prompt.state, qi, agent.cfg.get("max_len", 512),
                                      agent.cfg.get("head_max_len", 192))
        assert len(markers) == len(render_options(qi)), f"{qid}: options truncated out of the sequence"
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[qi["t"]], "label": labels[qid], "qid": qid})
    return items


def head_logits(model, h, attention_mask, marker_pos, marker_mask, qtype):
    """Exactly DecisionModel.forward from the encoder output onward (laya/common.py), so a head
    trained here behaves identically inside laya at inference."""
    import torch

    h = h + model.type_emb(qtype)[:, None, :]
    if model.head is not None:
        pad = ~attention_mask.bool()
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    m = torch.gather(h, 1, idx)
    logits = model.scorer(m).squeeze(-1).float()
    return logits.masked_fill(~marker_mask, -1e4)


def forward_logits(model, b, device: str, half_encoder: bool, train_encoder: bool):
    """Encoder (frozen, optionally fp16, no graph) -> fp32 hidden states -> head (with graph)."""
    import torch

    ids, att = b["input_ids"].to(device), b["attention_mask"].to(device)
    mpos, mmask, qt = b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device)
    if train_encoder:
        h = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
    else:
        with torch.no_grad():
            h = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        h = h.float() if half_encoder else h.detach()
    return head_logits(model, h, att, mpos, mmask, qt)


def run_eval(agent, examples: list[Example], device: str, seed: int = 12345, batch: int = 16,
             half_encoder: bool = False) -> dict:
    import torch
    from laya.common import collate_items

    model = agent.model
    model.eval()
    n = agree = danger = danger_agree = danger_fwd = 0
    hist: Counter = Counter()
    coll_ok = urg_ok = spd_ok = 0
    with torch.no_grad():
        for i in range(0, len(examples), batch):
            chunk = examples[i:i + batch]
            groups = [build_items(agent, e, seed + j) for j, e in enumerate(chunk, start=i)]
            b = collate_items(groups, agent.tok.pad_token_id)
            logits = forward_logits(model, b, device, half_encoder, train_encoder=False)
            pred = logits.argmax(-1).cpu().tolist()
            k = 0
            for e, items in zip(chunk, groups, strict=True):
                for it in items:
                    p = pred[k]
                    k += 1
                    if it["qid"] == "move":
                        opts = framing.render_options(e.frame, "semantic", shuffle=True, seed=seed + i + chunk.index(e))
                        a = opts[p].action
                        hist[a.value] += 1
                        n += 1
                        agree += a == e.teacher
                        if e.teacher is not Action.FORWARD:
                            danger += 1
                            danger_agree += a == e.teacher
                            danger_fwd += a is Action.FORWARD
                    elif it["qid"] == "collision_imminent":
                        coll_ok += p == it["label"]
                    elif it["qid"] == "urgency":
                        urg_ok += p == it["label"]
                    else:
                        spd_ok += p == it["label"]
    return {"n": n, "agree": agree / max(1, n), "danger_n": danger, "danger_agree": danger_agree / max(1, danger),
            "danger_forward": danger_fwd / max(1, danger), "collision_acc": coll_ok / max(1, n),
            "urgency_acc": urg_ok / max(1, n), "speed_acc": spd_ok / max(1, n), "hist": dict(hist.most_common())}


def save_checkpoint(agent, out: Path, base_checkpoint: str, meta: dict) -> None:
    """Writes fp16 tensors in laya's layout. The frozen encoder is byte-identical to the base."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file

    base = Path(snapshot_download("convaiinnovations/laya", local_files_only=True,
                                  allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*"]))
    sub = {"english": "", "multilingual": "multilingual", "typed-decisions": "typed-decisions"}[base_checkpoint]
    src = base / sub if sub else base
    final = out
    out = final.with_name(final.name + ".tmp")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for d in ("tokenizer", "encoder"):
        shutil.copytree(src / d, out / d)
    cfg = json.load(open(src / "rl_agent_config.json"))
    cfg.update({"model_name": "laya-drone-ft", "fine_tuned": True, "fine_tuned_from": f"convaiinnovations/laya/{sub or 'english'}",
                "drone_forest_training": meta})
    json.dump(cfg, open(out / "rl_agent_config.json", "w"), indent=2)
    sd = {k: v.detach().to("cpu").to(__import__("torch").float16).contiguous() for k, v in agent.model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))
    if final.exists():
        shutil.rmtree(final)
    os.replace(out, final)          # atomic: a running server never sees a half-written checkpoint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="english")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=12, help="examples per step (x3 sequences)")
    ap.add_argument("--balance", type=int, default=1, help="1: each epoch is 50/50 forward vs non-forward frames")
    ap.add_argument("--half-encoder", type=int, default=1, help="1: run the frozen encoder in fp16 (head stays fp32)")
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate for the decision head")
    ap.add_argument("--encoder-lr", type=float, default=2e-5, help="learning rate for unfrozen encoder layers")
    ap.add_argument("--unfreeze-top", type=int, default=0, help="also train the top N encoder layers")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="0 = no time box")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args(argv)

    import torch

    from .engines.laya_engine import resolve_device
    from .fastload import load_agent

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(None)

    examples = load_examples()
    train, val = block_split(examples, args.val_frac, seed=args.seed)
    print("dataset:", summary(examples))
    print("train  :", summary(train))
    print("val    :", summary(val))

    agent = load_agent(args.base, device)
    model = agent.model
    for p in model.encoder.parameters():
        p.requires_grad = False
    if args.unfreeze_top > 0:
        for layer in model.encoder.layers[-args.unfreeze_top:]:
            for p in layer.parameters():
                p.requires_grad = True
    half = bool(args.half_encoder) and args.unfreeze_top == 0 and device != "cpu"
    if half:
        model.encoder.half()
    trainable = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("act_head")]
    n_tr = sum(p.numel() for p in trainable)
    print(f"device={device} trainable params={n_tr / 1e6:.1f}M (encoder frozen except top {args.unfreeze_top})")

    before = run_eval(agent, val, device, half_encoder=half)
    print("val BEFORE:", json.dumps(before))

    from laya.common import collate_items
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(("act_head", "encoder."))]
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    groups = [{"params": head_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.encoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    n_epoch = 2 * sum(1 for e in train if e.teacher is not Action.FORWARD) if args.balance else len(train)
    steps_per_epoch = (n_epoch + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    warm = min(30, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.05, 1 - s / max(1, total)))
    ce = torch.nn.CrossEntropyLoss()

    best = before
    best_key = (before["danger_agree"], before["agree"])
    t_start = time.time()
    step = 0
    stopped_early = False
    for epoch in range(args.epochs):
        model.train()
        model.encoder.eval()               # frozen BN/dropout state stays inference-like
        rng = random.Random(args.seed + epoch)
        if args.balance:
            fwd = [i for i, e in enumerate(train) if e.teacher is Action.FORWARD]
            other = [i for i, e in enumerate(train) if e.teacher is not Action.FORWARD]
            rng.shuffle(fwd)
            order = other + fwd[:len(other)]
        else:
            order = list(range(len(train)))
        rng.shuffle(order)
        run_loss = 0.0
        for bi in range(0, len(order), args.batch):
            idx = order[bi:bi + args.batch]
            groups = [build_items(agent, train[i], seed=args.seed * 1000 + epoch * 100000 + i) for i in idx]
            b = collate_items(groups, agent.tok.pad_token_id)
            logits = forward_logits(model, b, device, half, train_encoder=args.unfreeze_top > 0)
            loss = ce(logits, b["label"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            if step % 25 == 0:
                print(f"  epoch {epoch + 1} step {step}/{total} loss {run_loss / 25:.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t_start:.0f}s", flush=True)
                run_loss = 0.0
            if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
                print(f"  time box hit at step {step}; stopping")
                stopped_early = True
                break
        ev = run_eval(agent, val, device, half_encoder=half)
        print(f"val after epoch {epoch + 1}:", json.dumps(ev), flush=True)
        key = (ev["danger_agree"], ev["agree"])
        if key > best_key:
            best, best_key = ev, key
            save_checkpoint(agent, Path(args.out), args.base, {
                "epochs_completed": epoch + 1, "steps": step, "train_n": len(train), "val_n": len(val),
                "unfreeze_top": args.unfreeze_top, "lr": args.lr, "encoder_lr": args.encoder_lr,
                "batch": args.batch, "seed": args.seed,
                "val_before": before, "val_after": ev, "minutes": round((time.time() - t_start) / 60, 1),
            })
            print(f"  saved best -> {args.out}", flush=True)
        if stopped_early:
            break
    print("\nBEST val:", json.dumps(best))
    print("BEFORE  :", json.dumps(before))
    return 0


if __name__ == "__main__":
    sys.exit(main())
