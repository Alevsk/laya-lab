# Laya lab

A self-hosted, reproducible setup of **[Laya](https://github.com/NandhaKishorM/laya)** — a
non-autoregressive "System 1" decision engine — plus runnable scenarios that show what it can
and cannot do.

Laya is *not* a chat model. It answers **typed questions about a state** in a single forward
pass: no tokens are generated, so there is nothing to parse and nothing to hallucinate. You hand
it a document and a schema of questions; it hands back labels, ordinal scores and calibrated
probabilities.

```python
agent.predict(ticket, {
    "department":  {"type": "choice", "instructions": "Who handles this?", "criteria": {...}},
    "urgency":     {"type": "score",  "instructions": "How urgent?", "criteria": [...]},
    "churn_risk":  {"type": "noul",   "instructions": "Might they cancel?"},
})
```

| primitive | returns | good for |
|---|---|---|
| `choice` | top label + probability per option + confidence | routing, intent, categorisation |
| `score` | expected level on an ordinal rubric + distribution | urgency, severity, frustration |
| `noul` | a single calibrated `P(true)` | phishing, jailbreak, churn, spam |

---

## Quick start

```bash
make setup     # create .venv and install pinned deps with uv
make warm      # download the three checkpoints (~2.3 GB, once)
make doctor    # verify interpreter, deps, device, checkpoints, one real forward pass
make scenarios # run every scenario end to end
```

Everything after `make warm` runs **fully offline** — the Makefile exports `HF_HUB_OFFLINE=1`.

---

## What is in this folder

```
Makefile              every workflow below
pyproject.toml        uv project; laya installed editable from ./upstream
uv.lock               pinned, reproducible resolution
LICENSE / NOTICE      MIT for this lab; Laya itself stays Apache-2.0
.github/workflows/    CI: lint + the weight-free suites and scenario
upstream/             the cloned NandhaKishorM/laya repo (git-ignored, fetched by `make upstream`)
scenarios/            one runnable capability demo per file
  _common.py          device resolution, fast loading, timing, table output
tools/
  warm_cache.py          prefetch checkpoints
  doctor.py              environment health gate
  gpu_check.py           proves the Mac GPU agrees with the CPU, per checkpoint
  verify_loader.py       proves the fast loader matches stock laya.load()
  run_upstream_tests.py  runs upstream's own suite the way its CI does
  link_local_models.py   exposes cached checkpoints as plain dirs for the e2e suite
  fetch_upstream.sh      clones/pins the library into upstream/
UPSTREAM_PIN          the upstream commit this lab is pinned to
out/                  scenario artefacts
```

---

## The three checkpoints

All three live in one Hugging Face repo, `convaiinnovations/laya`, as subfolders. Only the
subfolder you ask for is downloaded.

| name | encoder | params | context | `head_max_len` | use for |
|---|---|---|---|---|---|
| `english` | ModernBERT-large | 421M | 512 | 192 | English text |
| `multilingual` | mmBERT-base | 322M | 1024 | 256 | 100+ languages |
| `typed-decisions` | ModernBERT-large | 421M | 1024 | 256 | the four fine-tuned workflows |

`laya.Router` picks between them **before** any forward pass, using pure-Python script and
language detection that costs well under a millisecond.

---

## Verified on this machine

Measured here, on an Apple Silicon Mac (128 GB, macOS 15.7, no CUDA) with
`torch 2.14.0 / transformers 5.17.0 / laya 0.3.6` — not copied from the upstream README.

### Speed

Latency of one `predict()` call answering *N* questions about one short ticket, english
checkpoint, p50 of 7 runs after 3 warm-up runs, with the device synchronised before the clock
stops. **Synchronising matters**: MPS dispatches asynchronously, so a bare `perf_counter()`
around `predict()` can stop before the GPU has finished and report a number that is too good.
`_common.timed()` always synchronises; if you roll your own timing loop, call `_common.sync()`.

| questions per call | MPS (Apple GPU) | ms / question | CPU | ms / question |
|---:|---:|---:|---:|---:|
| 1 | 21.0 ms | 21.0 | 128.0 ms | 128.0 |
| 5 | 60.6 ms | 12.1 | 317.6 ms | 63.5 |
| 10 | 148.5 ms | 14.9 | 541.5 ms | 54.1 |
| 20 | 267.5 ms | 13.4 | 892.0 ms | 44.6 |
| 50 | 685.8 ms | 13.7 | 2389.8 ms | 47.8 |

Upstream's README quotes 33 ms for one question and 7.2 ms/question batched — those are
**Tesla T4** figures and do not describe this machine. On Apple Silicon expect roughly
**13–21 ms per question** — 3.5× faster than CPU on a 50-question batch, 6× on a single question.

### Routing costs nothing

`Router.route()` is pure Python. It downloads nothing, loads no weights, and returns a
`reason` string explaining itself:

```
router.route({"body": "मुझसे दो बार शुल्क लिया गया"})
# model='multilingual'
# reason='non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it'
```

Measured here: **0.06–0.16 ms** per decision, with `router.loaded == []` afterwards.

### Checkpoint loading is 10–40× faster here than stock

`laya.load()` spends 20–40 seconds inside `AutoModel.from_config`, which allocates and randomly
initialises the full 421M-parameter encoder immediately before `load_state_dict(..., strict=True)`
overwrites every one of those values. The work is never read.

`scenarios/_common.py` builds the model on PyTorch's **meta device** (no allocation at all) and
installs the checkpoint tensors with `assign=True` instead:

| checkpoint | stock `laya.load()` | via `_common.load_agent()` | speed-up |
|---|---:|---:|---:|
| english | 20.9 – 23.9 s | 2.1 s | ~10× |
| multilingual | 19.9 – 20.4 s | 1.3 – 1.4 s | ~15× |
| typed-decisions | 20.6 – 33.5 s | 0.5 – 0.8 s | ~40× |

Two safeguards, because this is the one place the lab departs from stock behaviour:

* The load stays **strict**, and the build is **abandoned in favour of the stock path** if any
  tensor is left on the meta device — so a checkpoint it does not fully understand can never
  yield a half-initialised model.
* `make verify-loader` loads every checkpoint *both* ways and asserts the predictions are
  identical. It passes on all three. `LAYA_NO_FAST_INIT=1` disables the optimisation entirely.

One subtlety it has to handle: `assign=True` installs the checkpoint's fp16 tensors verbatim,
whereas the stock path copies them *into* fp32 tensors and upcasts. Without matching that, fp16
weights meet fp32 activations and MPS aborts inside `MPSNDArrayMatrixMultiplication`.

### It runs on the Mac GPU — verified, not assumed

This machine is an **Apple M4 Max, 40-core GPU, Metal 3, 128 GB unified memory**. PyTorch's MPS
backend is built and available, and Laya's `Agent` picks it automatically (`cuda` → `mps` → `cpu`).

A GPU that is fast and *wrong* is worse than no GPU, and MPS does have op-level gaps, so
`make gpu-check` loads every checkpoint on both devices and compares every returned value:

| checkpoint | MPS | CPU | speed-up | answers |
|---|---:|---:|---:|---|
| english | 30.7 ms | 173.9 ms | 5.7× | identical |
| multilingual | 14.4 ms | 99.1 ms | 6.9× | identical |
| typed-decisions | 32.9 ms | 228.4 ms | 7.0× | identical |

One representative run; across runs the speed-up moved between **4.8× and 7.0×** depending on what
else the machine was doing. Eight values compared per checkpoint (label, expected score,
probability vector, confidence for each primitive); **zero differences, every run**. The whole upstream suite also passes under
`LAYA_DEVICE=mps`.

Two caveats worth knowing:

* Laya forces **fp32** on MPS and CPU and only uses autocast on CUDA
  (`agent.py`), so you are not getting fp16/bf16 throughput here even though the M4 supports
  bf16. That is the library's choice, not a hardware limit.
* MPS dispatches asynchronously — see the note under **Speed** about synchronising before you
  trust any timing you take yourself.

Set `LAYA_DEVICE=cpu` to force CPU for any target (e.g. `LAYA_DEVICE=cpu make s4`).

### Running it on a CPU-only server

No GPU is required. Measured here with `torch.set_num_threads(4)` to approximate a 4-vCPU box,
one checkpoint per process, CPU only:

| checkpoint | peak process RSS | 1 question | 5 questions | per question |
|---|---:|---:|---:|---:|
| `english` (ModernBERT-large, 421M) | ~3.0 GB | 165 ms | 642 ms | 128 ms |
| `multilingual` (mmBERT-base, 322M) | ~3.0 GB | **75 ms** | **304 ms** | **61 ms** |

**On CPU, prefer `multilingual` even for English-only work.** It is a smaller encoder, so it is
roughly **2× faster** and holds fewer weights — at the cost of English accuracy (upstream measures
0.657 vs 0.783 on MASSIVE intent). That trade is usually worth it on a CPU box.

Sizing, for something like an 8 GB droplet:

| | |
|---|---|
| checkpoints on disk | **2.37 GB** for all three (843 + 644 + 843 MB of weights plus tokenizers) |
| RAM, one checkpoint resident | ~3 GB peak |
| RAM, two checkpoints | ~5–6 GB — fits, with little headroom |
| RAM, all three | **don't**, not on 8 GB |

So 4 cores / 8 GB is comfortable for **one** checkpoint, workable for two.

Two things that matter much more on a small server than on a laptop:

**1. Do not install the CUDA build of torch.** On Linux, `torch` from PyPI drags in 43
`nvidia-*` packages — gigabytes of CUDA libraries that a CPU-only droplet can never use. Point
torch at the CPU index for Linux and they disappear (verified: 55 packages → 37, 43 nvidia → 0):

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

This is deliberately **not** the default here, because it would force CPU wheels on GPU servers
too. Add it to `pyproject.toml` when you deploy to a CPU-only host.

**2. Preload at boot, never lazily per request.** A stock `laya.load()` costs 20–40 s *and*
allocates 1.7 GB of throwaway random initialisation — real memory pressure on an 8 GB box. Use
`Router(preload=True, max_loaded=1)` at startup, or the meta-device loader in
`scenarios/_common.py`. Also set `torch.set_num_threads(<your core count>)`.

A caveat on the numbers above: they were taken on an M4 Max's CPU cores, which are considerably
faster than a shared-vCPU cloud instance. Expect roughly **2–3× slower** on a basic droplet —
call it ~150–200 ms per question on `multilingual`, ~400 ms on `english`. Benchmark on the actual
instance before committing to a latency budget; `make s4` gives you that curve.

### Footprint

| | |
|---|---|
| english weights | 843 MB |
| multilingual weights | 644 MB |
| typed-decisions weights | 843 MB |
| total HF cache | ~2.3 GB |
| network needed after `make warm` | none — everything runs with `HF_HUB_OFFLINE=1` |


---

## Upstream test suite

`make test` runs the library's own suite the way its CI does. The files in `upstream/tests/`
are standalone scripts, not pytest modules — `pytest upstream/tests` fails with an
`INTERNALERROR` because each one calls `sys.exit()` at import time. `tools/run_upstream_tests.py`
runs each as a subprocess instead.

```
$ make test
  [pass] test_criteria.py           0.6s  34 passed, 0 failed
  [pass] test_decision_model.py     2.6s  all decision model tests passed
  [pass] test_download.py           4.2s  OK
  [pass] test_email.py              0.8s  11 passed, 0 failed
  [pass] test_local_e2e.py        116.3s  23 passed, 0 failed
  [pass] test_packaging.py          0.0s  all packaging tests passed
  [pass] test_router.py             0.7s  all routing tests passed
  [pass] test_shortlist.py          0.5s  81 passed, 0 failed
  8/8 passed, 0 skipped
```

`test_local_e2e.py` is the only suite that exercises real weights across all three checkpoints,
and it wants them as three sibling directories rather than as the single HF snapshot. `make test`
symlinks them into `~/laya_models` first (`tools/link_local_models.py`), so it runs here rather
than being skipped.

---

## Scenarios

Seven scenarios, each a standalone script that prints a readable report, writes its raw numbers to
`out/<name>.json`, and **exits non-zero if its core claim stops holding** — so they double as
regression tests. `make scenarios` runs all seven in about 100 seconds on this machine.

| target | scenario | what you learn |
|---|---|---|
| `make s1` / `routing` | routing table | the model family and its decision logic, with zero weights in memory |
| `make s2` / `guard` | prompt-injection firewall | the capability that most justifies running Laya |
| `make s3` / `confidence` | confidence gate | why you must fit your own threshold |
| `make s4` / `throughput` | batching economics | what asking 20 questions instead of 1 actually costs |
| `make s5` / `limits-language` | language limit | where routing is blind, and the price of a mis-route |
| `make s6` / `limits-footguns` | silent footguns | four ways your input is destroyed without an error |
| `make s7` / `tetris` | Tetris arena | latency budget vs decision quality — two different things |

### 1 — Routing table (`make s1`, ~10 s, no checkpoints)

Nineteen tickets in ten languages through `Router.route()`, which picks a checkpoint and explains
itself — while downloading, building and holding **nothing**. `r.loaded == []` after 6,424 calls.

Routing is cheap for Latin text and emphatically not free for CJK, because `analyse()` scans the
string twice and `han` is the *last* of 25 entries in `_SCRIPT_RANGES`:

| state | `route()` p50 |
|---|---:|
| digits only, 10 chars | 1.1 µs |
| English ticket, 66 chars | 11.5 µs |
| English, 4,000 chars | 443 µs |
| Chinese, 4,000 chars | **27.3 ms** |

The scenario proves that is *caused* by probe depth rather than merely correlated with it: it
monkeypatches `_SCRIPT_RANGES` so `han` is probed first, and the Han cost collapses from 62× Latin
to 5×. Budget for this if you route long CJK states on a hot path.

### 2 — Prompt-injection firewall (`make s2`, ~35 s, english)

The shipped `laya.guard_questions()` preset, unmodified, over 11 hand-written prompts. It measures
the *separation* between benign and attack rather than asserting a threshold:

| signal | benign mean | attack mean | margin (weakest attack − worst benign) |
|---|---:|---:|---:|
| `jailbreak` | 0.053 | 0.999 | **+0.728** |
| `prompt_injection` | 0.075 | 0.945 | +0.423 |

No overlap, so a single 0.5 threshold classifies all 11 correctly. Five safety decisions in one
pass: **p50 35.6 ms**, offline, on a laptop. This is the clearest reason to run Laya.

### 3 — Confidence gate (`make s3`, ~20 s, english)

Eighteen deliberately unambiguous tickets. The model gets **18/18 right** while reporting **mean
confidence 0.594** and **ECE 0.406**. The miscalibration runs the *opposite* way from the one
people guard against — Laya is badly **under**-confident here.

Applying upstream's own suggested gate (`README.md:202`, `if conf >= 0.85: route_automatically`):

| | |
|---|---:|
| automated | 3 of 18 (17%) |
| accuracy of the automated subset | 1.000 |
| **correct answers sent to a human anyway** | **15** |

The lesson is not "confidence is broken", it is that the threshold is a property of *your* data.
Fit it, the way this scenario does, before you trust one.

### 4 — Batching economics (`make s4`, ~25 s, english)

Instruments the model rather than asserting: a forward pre-hook on `agent.model.encoder` shows a
twenty-question `predict()` fires the encoder **exactly once**, over a 20-row batch — and the
batched answers are identical to twenty separate calls.

| questions | whole call p50 | per question |
|---:|---:|---:|
| 1 | 24.0 ms | 24.0 ms |
| 5 | 47.9 ms | 9.6 ms |
| 20 | 110.1 ms | 5.5 ms |
| 40 | 204.5 ms | 5.1 ms |

Least-squares fit: **~22 ms fixed overhead + ~4.6 ms per additional question**.

The counterweight, and the reason this is a latency win rather than a free lunch: `usage.
input_tokens` goes **45 → 900 from N=1 to N=20, exactly 20×**. Laya builds one sequence per
question, each containing the full state, so the document is re-encoded once per question.

### 5 — Language limit (`make s5`, ~30 s, english + multilingual)

The stopword table covers exactly eight languages. Swahili, Indonesian, Malay, Somali,
ASCII-folded Turkish and Tagalog all report `language=None, is_english=True` and go to the
**English** checkpoint. Then it prices that mis-route on the same Swahili ticket:

| | top probability | confidence |
|---|---:|---:|
| English checkpoint, English control ticket | 0.942 | 0.844 |
| English checkpoint, **Swahili** ticket | 0.379 | **0.144** |
| Multilingual checkpoint, same Swahili ticket | 0.931 | 0.804 |

A 0.70 confidence collapse. The good news is that here the model does *not* stay confident while
being wrong — so on this input a confidence gate would have caught it.

### 6 — Silent footguns (`make s6`, ~15 s, english)

Four behaviours that destroy or ignore your input with **no exception, warning or log line**:

1. **`clean_email_body()` on a forward** — 294 chars in, 38 out (13% kept); only the
   `---------- Forwarded message ---------` separator survives. A mid-body `From:` line cuts
   everything after it. A polite opener can reduce a short email to `Hi,`.
2. **Right-side truncation** — a 3,652-char thread with the decisive `UPDATE` line last scores
   urgency 0.99; the same line first scores 2.01. The fix (`truncate_left=True`) exists in
   `build_sequence` but is unreachable through `predict()`.
3. **Option shrinkage** — tokens per option fall 25 → 4 between 2 and 44 options and descriptions
   vanish entirely. Then a cliff: 126 options work, 127 raises. The error names `head_max_len=192`,
   but the binding limit is `max_len=512`.
4. **`action.act_probability`** — 30 answers across 10 very different states produced exactly one
   distinct value: `[1.0]`. Raw act logit gaps measured in the thousands; a gap of 17 already
   saturates a float32 softmax.

Each block ends with the guard to add instead.

### 7 — Tetris arena (`make s7`, ~30 s, english)

A [viral demo](https://x.com/atomic_chat_hq/status/2102160983409955244) put two Tetris boards side
by side — a cloud decision model at 316–326 ms against Laya locally at 45–51 ms — and the cloud
board filled up under **"GAME OVER — COULD NOT KEEP UP"**. This scenario reproduces that, and then
asks the question the demo doesn't.

It ships a real Tetris engine (7 pieces, all rotations, collision, multi-line clears, top-out,
seeded 7-bag so every decider faces a byte-identical sequence) with 8 in-file self-tests. Four
deciders choose from the **same** candidate placements, shuffled per decision with a shared seed
so any index bias is positional rather than a board preference.

**The demo's claim reproduces cleanly.** At a 110 ms tick derived from this run's own measured
latency, `laya-local` met **24/24** ticks while a *simulated* 300 ms round trip met **0/11** and
topped out. Both boards ran the identical decision procedure on identical pieces — the cloud
board's answers simply arrived after the tick had passed.

**And then the other half.** With latency removed entirely, over 11 paired seeds × 80 pieces:

| decider | lines cleared | pieces survived | cost per decision |
|---|---:|---:|---|
| heuristic (4-term linear score) | **7.6** | 59.4 | ~6 µs |
| laya-numeric | 0.3 | 28.5 | ~35 ms |
| random | 0.1 | 26.6 | ~0 |
| laya-semantic | 0.0 | 25.6 | ~35 ms |

Laya does not measurably beat a uniform random pick (exact paired permutation test, **p = 0.62**
on lines), while the heuristic clears the same bar at p = 0.001 — so the test isn't simply blind.

**The framing experiment.** Laya has no numeric grounding but was trained on semantic typed
decisions, so the same candidates were rendered twice: numerically (`"col 3 rot 1: clears 0,
holes +2, height 7"`) and qualitatively (`"clears two rows and leaves the surface flat with no new
gaps"`). Words did not rescue it — but the two framings **fail differently**. `laya-numeric`
collapses onto option slot 0 on **83%** of decisions (4.5× uniform, using only 5 of 8 slots);
`laya-semantic` spreads across all 8 with its mode at 33%, and still clears zero lines. A diverse
output distribution is not evidence of judgement.

The scenario's invariant table marks every row `measured` or `by construction` — the latter are
arithmetic consequences of simulating the cloud as +300 ms, kept as regression guards and
explicitly **not** offered as evidence.

Two honest limits it prints itself: the 8-candidate cap that Laya's 192-token option budget forces
also costs the heuristic most of its strength (27.4 lines unconstrained vs 7.6 here), so nobody in
the file plays at full strength; and the 300 ms is an assumption from the demo's own report, not a
measurement of any service. If the machine is too busy to establish the latency window, the
scenario exits **2** (inconclusive) rather than 1, so a loaded laptop doesn't read as a false claim.

---

## Beyond benchmarks: the Drone Forest simulator

[`drone-forest/`](drone-forest/README.md) is a separate application, not a scenario: a 3D
Three.js game in which a quadcopter flies an endless forest of trees, rocks and wandering birds,
and every flight decision comes from a pluggable engine behind a WebSocket microservice — Laya,
a geometric heuristic, or random — on identical seeded worlds. It exists to test Laya in a
closed loop rather than on a single prompt, and to generate the telemetry a fine-tune would
train on.

The honest headline, measured through the real game: **zero-shot Laya is a constant-action
predictor in the forest** (`bank_left` on 450/450 decisions in one 45-second arena, with a flat
positional-bias histogram, so it is a label preference and not slot bias), and given a frame
with a tree 6 m dead ahead and the left clear for 60 m it picks *"forward: continue straight
into a tree that is close"* at P = 0.996. The heuristic flew the same worlds with 1 collision in
960 decisions. `make probe` in that folder reproduces the one-frame result in seconds.

Obstacle kinds, obstacle behaviours (birds wander randomly today; an AI policy plugs in at the
same seam) and decision engines are each one new module plus one registration line.

It has since grown difficulty levels 1–5 (denser forests with a narrower, faster-wandering gap
that a full-speed drone cannot follow — the engine must trade speed for safety), an
engine-recommended `target_speed` in the protocol (the same seed went from 546 m to 708 m in
45 s once the heuristic could choose its speed), a chained-seed arena that collects thousands
of teacher-labelled frames per minute, and a fine-tune loop (`make finetune`) that trains
Laya on that telemetry and serves the result as the `laya-ft` engine. Whether the fine-tune
makes it fly is reported honestly in that folder's README as the runs complete.

---

## Notes, caveats and honest limits

Everything below was reproduced on this machine against laya 0.3.6. These are not reasons to
avoid Laya — they are the things that will bite a pipeline, because **none of them raises an
error**.

### `confidence` is not P(correct)

Two different formulas share one field name:

* `choice` and `score` use normalised entropy, `1 - H(p)/ln(k)` — floor 0.0.
* `noul` uses `max(p, 1-p)` — hard floor **0.5**.

So a single threshold across question types compares incomparable numbers, and a `noul` answer
can never report confidence below 0.5 no matter how uncertain it is. On an easy hand-written
routing set the model was right nearly every time while reporting mean confidence around 0.59 —
the error is *under*-confidence, so the README's `if conf > 0.85: automate` pattern escalates
many correct answers. **Fit your own threshold on your own data** (scenario 3 does exactly this).

Also note `laya-multilingual` ships **no fitted calibration at all** (`temperature = [1.0, 1.0,
1.0]`, `temperature_by_options = {}`), so its confidences are sharper than english's on every
input without being better justified.

### `action.act_probability` is a dead field

Every answer carries `action: {act_probability: ...}`, which reads like an escalate-or-act head.
It is **1.0 for every input on every checkpoint** — the underlying logit gap is in the thousands,
so the softmax saturates. Verified here across benign, jailbreak, angry, empty and 2,400-word
inputs: the set of distinct values observed was `{1.0}`. Do not branch on it.

### Locale strings with an underscore mis-route

```
route(..., lang="en")     -> english
route(..., lang="en-GB")  -> english
route(..., lang="en_GB")  -> multilingual   # and en_US, en_AU ...
```

`router.py` splits on `-` only, so the POSIX/Java/.NET spelling every backend actually emits
falls through to the multilingual checkpoint. Normalise to `en` before passing `lang=`.

### The language detector knows exactly eight languages

Stopword lists exist for `en, fr, de, es, pt, it, nl, ro`. Any other Latin-script language with
no diacritics reports `language=None, is_english=True` and is sent to the **English** checkpoint.
Reproduced here:

| input | detected language | `is_english` | routed to |
|---|---|---|---|
| Swahili | `None` | `True` | `english` |
| Indonesian | `None` | `True` | `english` |
| German | `de` | `False` | `multilingual` |
| French | `fr` | `False` | `multilingual` |

Script detection is also a *plurality* vote, so a state that is up to ~49% non-Latin still goes
to English, and `state_text()` truncates to 4,000 characters — a long English preamble can hide a
non-English tail. Pass `lang=` explicitly when you already know the language.

### Long state is truncated from the *right*

`build_sequence` keeps `state[:room]`. The `truncate_left=True` parameter that would keep the tail
exists but is **unreachable** through `predict()`. For an email thread or chat log where the
decisive line is at the bottom, the model reads the oldest content and silently drops the newest.
Trim the state yourself before calling.

### `clean_email_body()` destroys forwarded mail

The helper the library recommends for email treats a forward separator as a quote boundary.
Measured here on a standard Gmail forward:

```
in  223 chars  ->  out 38 chars (17% kept)
out = '---------- Forwarded message ---------'
```

The entire request is gone. Any line starting `From: ` anywhere in the body truncates everything
after it. If your pipeline forwards mail into Laya, do not use this function unmodified.

### High-cardinality `choice` degrades silently, then throws

Options share a fixed token budget, so descriptions get truncated with **no warning** as the
option count grows — by ~77 labels each one has about three real tokens and the labels become
indistinguishable. Past a hard limit it raises:

```
n=127 options -> ok
n=128 options -> ValueError: question 'q' options exceed head_max_len=192
```

The message names the wrong parameter: the binding constraint is `max_len` (512 here), not
`head_max_len`. Raising `head_max_len` without also raising `max_len` makes it worse.

A note on the documented workaround: `predict_shortlist` with `embed_fn_from_agent` did **not**
improve accuracy in our testing — the mean-pooled encoder is a weak retriever (recall@20 ≈ 0.67
over 77 labels, which caps end-to-end accuracy at that figure) and it re-embeds every option on
every call. Split the label set into a coarse and a fine question instead.

### `criteria` as a list silently collapses duplicates

`criteria=["a", "a", "b"]` becomes a two-option question, and `probabilities` returns two keys
where the caller expects three. Use a dict.

### Upstream benchmark provenance

Checked against the repository as pinned here (`c752770`), so you can re-run these yourself:

* `BENCHMARKS.md:9` cites `research/results/app_benchmark.json` as the source for the applications
  table. **That file is not in the repository** — `research/results/` contains only
  `t4_colab_benchmark.json` and `cpu_51_language_sweep.json`.
* The committed T4 results do measure the typed-decisions *task*, but only on two checkpoints:
  `laya` at **0.362** and `laya-multilingual` at **0.3495**. The `laya-typed-decisions` checkpoint —
  the one the headline **0.766** figure is about — appears in no committed result file.
* The T4 latency figures are real, but describe a Tesla T4, not this Mac.

To be fair to upstream: it is **explicit** that the Jev comparison numbers are third-party
published figures rather than its own measurements (`README.md:276-277`, "Jev figures are
third-party published, never measured here"), and its "Honest limits" section states plainly that
the base checkpoints sit near chance on typed decisions. The gap is in the provenance of the
*fine-tuned* checkpoint's headline number, not in the project's candour.

This README quotes only what was measured on this machine.

### Upstream tests are not pytest modules

`upstream/tests/*.py` call `sys.exit()` at module scope, so `pytest upstream/tests` dies with an
`INTERNALERROR` and collects nothing. Use `make test`.

---

## Licence and attribution

This repository is **MIT** (see `LICENSE`). That covers the original work here: the scenarios, the
tools, the Makefile and this README.

**Laya itself is not covered by that, and is not redistributed here.** Laya is a separate project
by **Convai Innovations**, licensed **Apache-2.0**:

* source — <https://github.com/NandhaKishorM/laya>
* weights — <https://huggingface.co/convaiinnovations/laya>

`make upstream` clones it into `./upstream/` at the commit in `UPSTREAM_PIN`, and that directory
is git-ignored, so no Apache-2.0 code ships in this repository. See `NOTICE` for detail.

The measured criticisms in this README are offered in the spirit the upstream project set itself —
its own README carries an "Honest limits" section and openly labels the Jev comparison as
third-party figures. Everything here was re-measured rather than repeated, and every claim names
the file and line it came from so you can check it.
