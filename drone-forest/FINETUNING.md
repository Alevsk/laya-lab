# Fine-tuning Laya to fly the drone — the handoff document

This is the document to read before producing `laya-ft` **v4** or any later version. It says
what exists, how it fits together, exactly which commands produced v1–v3 and what they scored,
what went wrong along the way, and where the remaining headroom is. It is written for an agent
or a human who has never opened this folder; everything it claims is either in the code it
points at or in `data/finetune-v*.log` and the README's scoreboards.

If you only read one paragraph: **the loop is collect → label → train → swap → evaluate →
record**, every step is one command, the arena scoreboard is the only verdict that counts, and
the thing that has not been solved after three fine-tunes is the 6-way *steering* choice — the
model learns to *read* the scene in an epoch and learns to *pick the right move* slowly.

---

## 1. Where things stand

| version | base | data | recipe | val: danger agreement | arena (quality, L5, seeds 7–8) |
|---|---|---|---|---:|---|
| zero-shot `english` | – | – | – | 0.42–0.57 (constant-action predictor) | 2.5 collisions/run, 13 m/s |
| v1 | `english` | 2.3k frames, level 3 | head only, 2 epochs | **0.34 (worse)** | not flown |
| v2 | `english` | 23.6k frames, levels 3–5, **before ogres** | top-2 encoder layers, 3 epochs, ~114 min | 0.57 → 0.61 | 2.0/run, **all rocks**, 9.2 m/s, 411 m |
| v3 | **v2** | 80.2k frames incl. 72 teacher runs L3–5 with ogres + 9 student runs | top-2 layers, 2 × 6,000 examples, dodges ×3, 60 min | 0.535 → 0.598 (rock-dodge 0.59 → 0.66) | 2.5/run (1.5 rocks + 1 tree), 13.6 m/s, 611 m |
| teacher (`heuristic`) | – | – | – | 1.00 by definition | **0.0/run**, 15.5 m/s, 692 m |

v3 is what `laya-ft` serves today (`data/checkpoints/laya-drone-ft`); v2 is kept next to it as
`laya-drone-ft-v2`. Checkpoints and telemetry are git-ignored, so on a fresh clone there is no
`laya-ft` until you train one. The training logs *are* committed (`data/finetune-v1.log`,
`-v2.log`, `-v3.log`) — read them, they are the ground truth for the table above.

What the three versions established:

1. **A frozen encoder cannot do it.** v1 (head only) got worse than zero-shot. Unfreezing the
   top 2 encoder layers (v2, v3) is the minimum that moves the steering question.
2. **State-reading is easy, steering is hard.** collision-imminent / urgency / speed go from
   chance to 0.78–0.87 within an epoch; move-agreement on danger frames climbs a few points per
   version (0.535 → 0.598 in v3). This is the bottleneck. See §8.
3. **Train on the world you test in.** v2 had never seen a thrown rock and every level-5
   collision was a rock. v3 saw 3,119 teacher dodges and halved its rock hits per metre flown —
   but it is still hit. The dataset must be re-collected after every world change (new
   obstacle kind, new sensor field, new framing line).
4. **Speed is learned from the label.** The teacher's `target_speed` is the fourth question's
   label; v3 flies at 13.6 m/s because it learned it, v2 crawled at 9.2 because its data was
   older and its speed answers were 0.60 accurate on the new distribution.

---

## 2. The loop, and what each piece is

```
 game (web/)  ──frames @10 Hz──▶  service (server/app.py)  ──▶  data/telemetry-<UTC>.jsonl
   ▲   arena / matrix runs                 │ engine.decide()          kind: decision {frame, decision, engine}
   │                                        │                          kind: event    {score = ArenaResult, ...}
   │                                        ▼
   │                           engines/heuristic_engine.py   ← the TEACHER (geometric, exact, 0.02 ms)
   │                           engines/laya_engine.py        ← zero-shot Laya, 4 questions / forward pass
   │                           engines/laya_ft_engine.py     ← same class, LAYA_FT_CHECKPOINT dir
   │
   │   server/dataset.py   load_examples(): every recorded frame → re-run the teacher offline → Example
   │   server/finetune.py  multi-task CE on the 4 questions, laya's own build_sequence, shuffled options
   │                        → data/checkpoints/<name>/ (laya layout: rl_agent_config.json, model.safetensors,
   │                          tokenizer/, encoder/)
   │   server/offline_eval.py  score any checkpoint / question formulation against the teacher on frames
   │   server/scoreboard.py    aggregate arena `score` events per (mode, engine, level)
   │
   └── make evaluate → web/src/decision/matrix.ts chains one arena per (mode, engine, level, seed)
```

**The model.** Laya (`convaiinnovations/laya`, ModernBERT-large encoder, 395M params) plus a
26.5M-parameter decision head (`head` = 2 transformer layers, `type_emb`, `scorer`; `act_head`
is unused). It does not generate text: for a *choice* question it scores each option at its
marker token in one forward pass and returns the argmax; *noul* is a calibrated yes/no,
*score* is an ordinal level. The engine asks four questions per frame from one rendered state
(`engines/framing.py`): `move` (6 options with consequences), `collision_imminent`, `urgency`
(4 levels), `speed` (4 levels → `[SPEED_MIN=4, bounds.speed_max=18]` m/s). Option order is
shuffled per decision with a seed derived from `LAYA_SEED` and the frame id, and the order is
recorded in the Decision so positional bias is measurable.

**The teacher.** `HeuristicEngine` scores sectors from the 11 rays, brakes on time-to-collision,
dodges thrown rocks (`frame.threat.kind == "projectile"`, TTC < 2 s: climb/descend by elevation,
bank away by bearing) and chooses speed (max when ≥ 30 m clear, floor 8 m, keep 2.5 s TTC).
It is deterministic and exact, which is why relabelling is free: **the label of every recorded
frame is recomputed at dataset-build time**, so improving the teacher improves the whole dataset
retroactively, and frames flown by *any* engine (Laya's own crashes included) get consistent
labels.

**The dataset.** `load_examples()` reads every `data/telemetry-*.jsonl`, keeps `kind: decision`
records, validates the frame, collapses exact duplicates on
`(altitude, speed, ray hits+distances, last_action)`, runs the teacher, and yields `Example`
(frame, teacher action, teacher probabilities, collision, urgency, speed, source engine, file).
`block_split()` splits by contiguous 40-frame blocks so near-duplicate neighbours never straddle
train/val. `stratified()` is for offline evaluation samples.

**Training.** `finetune.py` builds, for each example, the four question sequences with laya's
own `build_sequence` (train and serve are byte-identical prompts) and a fresh option shuffle per
example per epoch; runs the encoder (frozen, or top-N layers trainable), then exactly
`DecisionModel.forward`'s head math (`head_logits`) so the trained head behaves identically inside
laya; cross-entropy on the four labels; AdamW with two parameter groups (`--lr` head,
`--encoder-lr` unfrozen layers), linear warm-up (≤ 30 steps) then linear decay to 5%, gradient
clipping at 1.0. Each *balanced* epoch is `2 × danger frames` (danger = teacher ≠ forward), with
rock-dodge frames repeated `--dodge-repeat` times, optionally capped by `--epoch-examples`.
Validation runs before training and after every epoch; the checkpoint is saved **only when
`(danger_agree, agree)` improves**, atomically, in laya's layout.

**Serving.** `LayaFtEngine` is `LayaEngine` with `checkpoint=` a directory. `fastload.load_agent`
loads it in ~0.5–2 s (meta-device construction, `assign=True`, rotary buffers rebuilt,
`model.float()`); it falls back to the stock loader on any surprise. The service loads engines
lazily on first use and **caches them for the life of the process**.

**Evaluation.** Quality-mode arena steps the world at a fixed 1/60 s and awaits every decision
(policy only); realtime mode keeps the wall clock and counts 10 Hz slots met/missed/skipped
(`web/src/decision/loop.ts`). Each run reports an `ArenaResult` as a `score` event.
`make evaluate` prints one URL that runs the whole matrix; `make scoreboard` tabulates.

---

## 3. File map

| path | what it is | touch it when |
|---|---|---|
| `server/engines/framing.py` | state rendering (semantic / numeric), the 4 questions, option texts, `speed_level`/`speed_from_level`, head-budget checks | changing what Laya is asked — **invalidates every earlier checkpoint** |
| `server/engines/heuristic_engine.py` | the teacher | improving labels (then rebuild the dataset for free) |
| `server/engines/laya_engine.py` | zero-shot engine, synchronised timing, `_warmup_frame` | inference-side behaviour (framing kind, shuffle, device) |
| `server/engines/laya_ft_engine.py` | `laya-ft` = same engine + checkpoint dir | never, unless the checkpoint location scheme changes |
| `server/fastload.py` | fast checkpoint loader | never; `LAYA_NO_FAST_INIT=1` disables it if suspected |
| `server/dataset.py` | `Example`, `load_examples`, `block_split`, `stratified`, `summary` | new label fields, new dedup keys, filtering by file/level |
| `server/finetune.py` | the trainer (`build_items`, `head_logits`, `forward_logits`, `run_eval`, `cap_val`, `save_checkpoint`, `main`) | new losses, new knobs, new eval slices |
| `server/offline_eval.py` | formulation/checkpoint scoring on frames without the game | comparing checkpoints or prompt formulations cheaply |
| `server/scoreboard.py` | arena aggregation with `--seeds`/`--seconds`/`--all`/`--files` | new columns |
| `server/telemetry.py` | the recorder + `make telemetry` summary | new record fields |
| `server/schemas.py` ↔ `web/src/core/types.ts` | the wire contract (`PROTOCOL_VERSION`), `SensorFrame`, `Decision`, `Bounds` | any new sensor field (bump the version, then re-collect) |
| `web/src/decision/matrix.ts` | URL-driven evaluation matrix | new dimensions |
| `web/src/decision/arena.ts`, `loop.ts` | arena runner / decision loop and tick accounting | new result fields |
| `Makefile` | `serve web dev test lint check-web probe collect finetune eval-offline evaluate scoreboard telemetry` | new targets |
| `data/finetune-v*.log` | committed training logs | every run: `... > data/finetune-vN.log` |
| `README.md` → *Evaluation protocol* and *Teaching it to fly* | the recorded results | every version |

---

## 4. The recipe for the next version (v4)

Every command runs from `drone-forest/`. `HF_HUB_OFFLINE=1` is exported by the Makefile; set it
yourself when calling `python -m server.…` directly. Nothing here needs the network.

### 4.0 Preflight

```bash
make setup                                   # uv sync --group dev; npm install
ls ~/.cache/huggingface/hub | grep laya      # the base checkpoint must be cached (make link-models in the repo root)
ls data/checkpoints/                         # laya-drone-ft (current), laya-drone-ft-v2 ...
ls -la data/telemetry-*.jsonl | wc -l        # the training data; git-ignored — back it up, `make clean` DELETES it
sysctl -n vm.loadavg                         # GPU/CPU must be quiet for latency numbers (see §7)
make dev                                     # service :8765 + game :5173; or `make serve` and `make web` separately
curl -s :8765/health | jq .engines           # ["heuristic","laya","laya-ft","random"]
```

### 4.1 Decide what data v4 needs, then collect it

Ask first: *what did the last version get wrong in the arena?* Read the newest scoreboard rows
and the `collisions_by_kind` of the `laya-ft` cells (`make telemetry`, or the `score` events).
Then collect frames of exactly that situation:

```bash
make collect            # prints: heuristic × levels 3,4,5 × seeds 100-123 × 60 s, quality mode (~3 min, ~40k frames)
# student runs (DAgger): the current laya-ft flying its own path; the teacher relabels them
#   http://127.0.0.1:5173/?arena=quality&seconds=60&engine=laya-ft&difficulties=3,4,5&difficulty=3&seed=100&seeds=100-102
```

Rules of thumb:

- Teacher-flown frames are plentiful and clean; student-flown frames are the ones that fix the
  student's *own* mistakes (it flies through places the teacher never goes). Keep ~10% student.
- Use **new seeds** for collection (100+); seeds 7–8 (and 7–16) are the evaluation seeds. Never
  train on the evaluation seeds.
- A quality-mode heuristic run of 60 s takes ~1 s wall-clock; a `laya-ft` run takes ~75 s.
- Keep the browser tab alone and do not run tests or a training job while collecting `laya-ft`
  runs (GPU contention makes them slower, though the *frames* are still valid).
- Count what you got before training (rock-dodge frames are the rare class):

```bash
HF_HUB_OFFLINE=1 uv run python - <<'PY'
from collections import Counter
from server.dataset import load_examples, summary
from server.schemas import Action
ex = load_examples(); print(summary(ex))
rock = [e for e in ex if e.frame.threat and e.frame.threat.kind == "projectile"]
dodge = [e for e in rock if e.teacher is not Action.FORWARD]
print("projectile frames", len(rock), "teacher dodges", len(dodge), Counter(e.teacher.value for e in dodge))
PY
```

(v3 trained on 80,189 frames: forward 56,019 / bank 18,356 / climb 5,353 / brake 291 / descend
170; 3,119 rock-dodges in the train split.)

### 4.2 Train

The exact v3 command, which is the starting point for v4:

```bash
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 nohup uv run python -m server.finetune \
  --base data/checkpoints/laya-drone-ft \        # continue from the current best (v3); or `english` to start over
  --out  data/checkpoints/laya-drone-ft-v4 \     # NEVER train into the served directory
  --epochs 2 --epoch-examples 6000 \             # 2 × 500 steps of 12 examples (48 sequences) — ~3.3 s/step on M4 Max
  --val-examples 1200 \                          # every rock-dodge val frame first, then danger, then forward
  --dodge-repeat 3 --unfreeze-top 2 \
  --batch 12 --balance 1 --max-minutes 85 \
  > data/finetune-v4.log 2>&1 &
tail -f data/finetune-v4.log | grep --line-buffered -E "val|step [0-9]*00/|saved|Traceback"
```

Wall-clock budget on an M4 Max: dataset build ~2 min (80k frames), each validation pass ≈ 90 s
per 1,000 sequences (1,200 frames × 4 questions ≈ 7 min), training ≈ 3.1–3.4 s per step. v3 =
60 min total (3 validation passes + 1,000 steps). `--max-minutes` is a hard time box that still evaluates and saves.

What to expect in the log: `val BEFORE`, `train danger frames N, of which rock dodges M (xK)`,
`epoch e step s/total loss … lr … Ns` every 25 steps, `val after epoch e: {…}`, `saved best ->`
when it improved, `BEST val` at the end. **If you never see `saved best`, nothing was written**
(the run did not beat its own `val BEFORE` on `(danger_agree, agree)`).

### 4.3 Read the validation numbers

| field | meaning | v3 |
|---|---:|---:|
| `agree` | move = teacher, all val frames | 0.68 |
| `danger_agree` | move = teacher on frames where the teacher does **not** say forward — the number that matters | 0.598 |
| `danger_forward` | says forward on danger frames (the crash proxy; lower is better) | 0.26 |
| `rock_agree` / `rock_n` | move = teacher on frames where the teacher dodges a thrown rock | 0.66 / 400 |
| `collision_acc`, `urgency_acc`, `speed_acc` | the state-reading questions | 0.87 / 0.77 / 0.78 |
| `hist` | predicted action histogram — a collapsed histogram means a constant predictor | — |

Validation sets differ between versions (v3's is harder than v2's by construction), so compare
`val BEFORE` → `val after` *within* a run, and compare versions **in the arena**.

For a cheaper checkpoint-vs-checkpoint comparison on a fixed stratified sample without the
game: `make eval-offline CHECKPOINT=data/checkpoints/laya-drone-ft-v4 PER_CLASS=80`.

### 4.4 Swap it in and restart the service

```bash
mv data/checkpoints/laya-drone-ft data/checkpoints/laya-drone-ft-v3     # keep every version
mv data/checkpoints/laya-drone-ft-v4 data/checkpoints/laya-drone-ft
# restart the service: it caches the engine it loaded
curl -s :8765/engines | jq '.[] | select(.name=="laya-ft") | .describe | {checkpoint, load_ms}'
```

Or point at it without moving: `LAYA_FT_CHECKPOINT=data/checkpoints/laya-drone-ft-v4 make serve`.
Restarting the service **rotates the telemetry file** (`data/telemetry-<UTC>.jsonl` is opened
per process) — anything that watches or aggregates by file must follow.

### 4.5 Evaluate — the only verdict

```bash
sysctl -n vm.loadavg                      # want load1 < 5 before realtime cells; see §7
make evaluate                             # EVAL_ENGINES=laya,laya-ft EVAL_LEVELS=3,5 EVAL_SEEDS=7-8 EVAL_SECONDS=45
# open the printed URL in ONE tab and leave it alone until the console logs ARENA_MATRIX_DONE
make scoreboard SCOREBOARD_ARGS="--files data/telemetry-<build1>.jsonl data/telemetry-<now>.jsonl --seeds 7-8 --seconds 45"
```

- Quality-mode `laya` (zero-shot) decisions are deterministic: if its quality rows differ from
  the last build, the protocol changed (framing, world, sensors) and the tables are no longer
  comparable — fix that before reading anything else.
- `heuristic`/`random` rows need no re-run (no GPU, no checkpoint).
- 2 seeds is a smoke test; `EVAL_SEEDS=7-16` gives ten per cell (4 engines × 2 levels × 10 × 2
  modes ≈ 160 cells ≈ 2 h, mostly the realtime half).
- Realtime rows are only meaningful on a quiet machine (§7); quality rows are always meaningful.

### 4.6 Record it

Add a *Scoreboard, build N* subsection under *Evaluation protocol* in `README.md` (table from
`make scoreboard`, the machine conditions, what changed and what did not), a row in §1 of this
document, the v-N settings in the README's *Teaching it to fly* results, and commit
`data/finetune-vN.log`. Never commit `data/checkpoints/`, `data/*.jsonl`, `out/`, `web/dist`,
`web/.check`, `.playwright-mcp/`.

---

## 5. Knob reference (`server/finetune.py`)

| flag | default | effect |
|---|---|---|
| `--base` | `english` | `english` / `multilingual` / `typed-decisions`, or a checkpoint **directory** to continue from (lineage recorded in `rl_agent_config.json → fine_tuned_from`) |
| `--out` | `data/checkpoints/laya-drone-ft` | output dir; written atomically only on improvement |
| `--epochs` | 3 | epochs; the LR schedule spans `steps_per_epoch × epochs` |
| `--batch` | 12 | examples per step = 4× that many sequences; 12 fits comfortably on a 128 GB M4 Max with top-2 unfrozen |
| `--balance` | 1 | each epoch = all danger frames + an equal number of forward frames |
| `--dodge-repeat` | 1 | repeat rock-dodge frames K× inside the danger pool (v3: 3) |
| `--epoch-examples` | 0 (all) | cap the balanced epoch (v3: 6,000 ≈ 500 steps) |
| `--val-examples` | 0 (all) | cap validation: rock-dodge frames first (≤ n/3), other danger (≤ n/3), forward fills |
| `--val-frac` | 0.2 | block-split fraction before capping |
| `--unfreeze-top` | 0 | train the top N encoder layers too (v2/v3: 2). 0 = head only, which **does not work** (v1) |
| `--lr` | 1e-4 | head learning rate |
| `--encoder-lr` | 2e-5 | unfrozen-layer learning rate |
| `--half-encoder` | 1 | fp16 frozen encoder — only applies when `--unfreeze-top 0` and not on CPU |
| `--max-minutes` | 0 | hard time box; evaluates + saves at the box |
| `--seed` | 0 | split, shuffles, option order |

`make finetune` wraps this with `--epochs $(EPOCHS)` (8) `--batch 12 --balance 1 --half-encoder 1
$(FINETUNE_ARGS)`; for anything beyond a quick run call the module directly as in §4.2.

---

## 6. Techniques that are in place (and how to use them well)

- **Teacher relabelling.** Labels are computed at dataset-build time, not recorded. If you make
  the heuristic better (or fix a bug in it), every past frame is relabelled by rerunning
  `load_examples()`. Keep the teacher deterministic and stateless per frame.
- **Byte-identical train/serve prompts.** `build_items` uses `framing.render_options`,
  `framing.build_questions`, `Agent._to_internal` and laya's `build_sequence` — the same path
  the engine takes. If you change framing, change it once and retrain from `english` or accept
  that continued training starts from a distribution the checkpoint never saw.
- **Per-example option shuffling** removes slot bias; verify with the recorded `option_index`
  histogram in `make telemetry` (flat = good).
- **Head budget.** Options are scored at marker tokens inside a 192-token head window
  (`head_max_len`). `build_items` asserts no option was truncated; `framing.assert_head_budget`
  checks the longest possible state. Keep ≤ 6–8 short options.
- **Block split + capped, dodge-first validation** keeps evaluation honest and affordable.
- **Continue-from-checkpoint** (`--base <dir>`) lets each version be a short, targeted run on
  fresh data instead of a multi-hour retrain (v3 was 60 min on top of v2).
- **Determinism as a regression test.** The zero-shot `laya` quality rows must not change
  between builds; the heuristic's arena rows must not change unless you changed the teacher.
- **Student-distribution data (DAgger).** Fly the current `laya-ft`, relabel with the teacher,
  retrain. v3 used 9 such runs; this is the cheapest lever for "it keeps making the same mistake".

---

## 7. Pitfalls (each of these cost real time)

1. **`make clean` deletes `data/telemetry-*.jsonl`** — the whole training set. Copy the files
   somewhere first if you ever need `clean`.
2. **The service caches engines.** Swapping the checkpoint directory does nothing to a running
   service; restart it, then check `/engines … describe.checkpoint`.
3. **Restarting the service rotates the telemetry file.** A waiter or `make scoreboard` looking
   at the old file sees nothing new. Pass both files with `--files`, or `--all`.
4. **Nothing is saved unless validation improves.** Look for `saved best`; a smoke test that
   trains 2 steps writes nothing. `save_checkpoint` accepts a dir base since v3 (it used to
   `KeyError` on anything but the three named checkpoints).
5. **Latency numbers are only valid on a quiet machine.** Build 1's `laya-ft` cells ran while
   tests and a dataset build were running (125 → 173 ms); build 2's ran during a Time Machine
   backup and a container VM (load 15–94; identical frames: 94 ms quiet vs 164 ms loaded).
   Check `sysctl -n vm.loadavg`, use one browser tab, run nothing else. Quality-mode *decisions*
   are unaffected; their `think p50` column is.
6. **`laya` and `laya-ft` are the same speed.** Measured to within 1 ms on identical frames
   through `/decide`. Prompt length is what moves latency (85 ms at level 3 → 125 ms at level 5
   as more rays report hits and the `Incoming:` line appears).
7. **MPS is asynchronous.** Any timing must `torch.mps.synchronize()` first (the engine does).
   Loading fp16 tensors with `assign=True` without `model.float()` crashes MPS matmuls.
8. **Do not train and evaluate at the same time.** Both use the GPU; the arena will be slow and
   the realtime rows meaningless.
9. **Validation is 4 forward passes per frame.** Uncapped val on 80k frames is ~1.5 h per pass.
   Always cap (`--val-examples`).
10. **The balanced epoch grows with the dataset** (2 × danger frames + repeats). Cap with
    `--epoch-examples` so the LR schedule and the wall clock stay what you planned.
11. **The matrix URL must start on the first value of every dimension** (`engine=`,
    `difficulty=`, `seed=` equal to the first entries of `engines=`, `difficulties=`,
    `seeds=A-B`) and `seconds` must be > 0 (0 = endless flight, which never chains).
12. **Never train on the evaluation seeds** (7–8, 7–16). Collect on 100+.
13. **A missing checkpoint makes `laya-ft` answer `brake`** with `reason: laya failed
    (FileNotFoundError …)`; `/health → loaded` tells you what actually loaded.
14. **Changing framing or the sensor frame invalidates prior checkpoints and prior telemetry
    comparisons.** Bump `PROTOCOL_VERSION` in both `schemas.py` and `types.ts`, re-collect,
    retrain, and start a new scoreboard build.
15. **`--half-encoder` is silently ignored when layers are unfrozen** (it only applies to a
    fully frozen encoder). That is intended; do not expect fp16 speed with `--unfreeze-top`.
16. **Node cannot type-strip this TypeScript** (constructor parameter properties); use
    `make check-web` (esbuild bundle) for headless checks, `npm run typecheck` for types.
17. **Commits are signed via the user's 1Password SSH agent**; a locked agent fails the commit
    (`failed to fill whole buffer`). Retry after it is unlocked; never change signing config.

---

## 8. Where the headroom is — ideas for v4 and beyond, ranked

The measured gap: teacher 0.0 collisions/run vs v3 1.0 (L3) and 2.5 (L5); steering agreement
0.60 on danger frames, rock-dodge 0.66; state questions already 0.78–0.87. The steering choice
is the bottleneck, and the levers below are ordered by expected value per hour.

1. **Weight the move loss.** All four questions share one cross-entropy today. Add
   `--move-weight W` (multiply the loss of `qid == "move"` items) so the encoder's unfrozen
   layers spend their capacity on steering once the state questions have saturated. Cheap:
   ~10 lines in `main()` (the batch carries `qid` per item via `build_items`).
2. **Soft targets from the teacher.** `Example.teacher_probs` holds the heuristic's normalised
   scores for all six actions and `Example.acceptable(ratio)` the set within 80% of the best.
   Training on the argmax alone punishes "bank left" when "bank right" was 95% as good.
   Replace the move CE with KL against `teacher_probs` (or CE against the acceptable set) —
   the offline evaluator already reports `reasonable` agreement.
3. **Iterated DAgger.** After each version: fly `laya-ft` at levels 3–5 on ~10 fresh seeds,
   relabel, continue-train for 30–60 min, re-evaluate. Three short iterations beat one long
   run; v2 → v3 was one iteration.
4. **Unfreeze more.** Top 4–6 layers at `--encoder-lr 1e-5` (memory is not the constraint on a
   128 GB M4 Max; time is ~+15% per extra pair of layers). Full-encoder training at 5e-6 is the
   next step after that. Watch `val BEFORE` on the state questions for forgetting.
5. **Make the threat unmistakable in the prompt.** Today the state carries one line
   (`Incoming: a thrown rock, close, from below-left, closing fast`). Add the time-to-impact in
   words ("will hit in about one second") and put it first. Any framing change → retrain from
   `english` or accept a distribution jump; bump nothing in the protocol, this is server-side.
6. **Rebalance the rare classes.** Teacher `climb` (5.3k), `brake` (291) and `descend` (170)
   are rare; `--dodge-repeat` covers rock frames only. A per-action cap/repeat (`stratified`
   exists in `dataset.py`) or a class-weighted CE would stop the model defaulting to banks.
7. **Latency: shorten the state.** Eleven ray sentences plus altitude/speed/nearest/incoming
   is ~440–480 characters at level 5. Merge rays into sectors ("left: clear 60 m; ahead: tree
   close") and measure with `make probe` / `/decide` — the goal is the level-5 prompt under
   100 ms so the realtime rows stop missing slots. Same caveat: retrain after changing it.
8. **More seeds, then error bars.** `EVAL_SEEDS=7-16` before claiming any improvement smaller
   than ~0.5 collisions/run.
9. **Hyperparameter sweeps in time boxes.** `--max-minutes 15 --val-examples 600` runs with a
   fixed `--seed` and the same `--base` compare `rock_agree`/`danger_agree` quickly; promote
   the best to a full run.
10. **Reward-based training (unexplored).** Laya's checkpoint ships an `act_head` and an
    `rl_agent_config.json`; whether upstream supports an RL-style update from arena outcomes
    was not investigated. Verify in `../upstream` before spending time on it — the supervised
    loop above has not been exhausted.

**What "done" would look like.** On seeds 7–16, quality mode: ≤ 0.5 collisions/run at level 3
and ≤ 1.0 at level 5 with rock hits ≤ 0.3/run at ≥ 13 m/s; realtime mode on a quiet machine:
≥ 90% of 10 Hz slots met at level 3. The teacher is the ceiling (0.0, 15.5 m/s); random the floor
(3.5–9).

---

## 9. Quick reference

```bash
# stack
make dev                      # or: make serve  +  make web
make probe ENGINE=laya-ft     # one frame through the service (tree 6 m ahead, left clear)
make telemetry [FILE=…]       # summarise a telemetry file: decisions, latency, actions, option_index histogram
# data
make collect                  # teacher runs URL (levels 3-5, seeds 100-123, 60 s)
# train / compare
make finetune EPOCHS=2 FINETUNE_ARGS="--base data/checkpoints/laya-drone-ft --out data/checkpoints/laya-drone-ft-v4 --unfreeze-top 2 --epoch-examples 6000 --val-examples 1200 --dodge-repeat 3"
make eval-offline CHECKPOINT=data/checkpoints/laya-drone-ft-v4 PER_CLASS=80
# evaluate / record
make evaluate EVAL_ENGINES=laya,laya-ft EVAL_LEVELS=3,5 EVAL_SEEDS=7-8
make scoreboard SCOREBOARD_ARGS="--all --seeds 7-8 --seconds 45"
# hygiene
make lint && make test && make check-web
```

Environment variables that matter: `HF_HUB_OFFLINE=1` (always), `LAYA_FT_CHECKPOINT` (serve a
checkpoint without moving it), `LAYA_CHECKPOINT` (zero-shot base: english|multilingual|typed-decisions),
`LAYA_DEVICE` (cuda|mps|cpu), `FRAMING` (semantic|numeric — the fine-tunes are semantic),
`SHUFFLE_OPTIONS`, `LAYA_SEED`, `LAYA_NO_FAST_INIT=1`, `DEFAULT_ENGINE`.
