"""Confidence is not accuracy: measure your own gate before you trust one.

Upstream's README (upstream/README.md:194-208, "Automated Confidence Gating") says
Laya's probabilities are "statistically meaningful" and shows this gate:

    if conf >= 0.85:
        route_automatically(dept)
    else:
        escalate_to_human_agent(dept, ...)

On 18 deliberately unambiguous support tickets this checkpoint is **18/18 correct**
while reporting a **mean confidence of 0.59** -- so that gate automates 3 answers and
escalates 15 perfectly correct ones. The miscalibration is real (ECE 0.41) but it
points the *other* way from the one everyone defends against: Laya is under-confident
here, not over-confident. Raising the threshold -- the instinctive reaction -- makes it
strictly worse: it buys no accuracy and costs human hours.

How to read the output
----------------------
Setup   the RuntimeWarning this checkpoint raises on load, the shipped vs. applied
        temperature table behind it, and a computed demonstration of what the rejected
        temperature would have done to a probability vector.
A       18 tickets, gold label, prediction, and the confidence attached to each. The
        headline line is accuracy vs. mean confidence vs. ECE, plus the measured cost
        of collecting exactly this much labelled evidence yourself.
B       what six gate thresholds -- including the README's own 0.85 -- would actually
        have done to those 18 answers. The last column is the cost: correct answers
        sent to a human.
C       why `score` confidences look like coin flips. `score` is an expected value,
        not an argmax, and normalized entropy puts a hard ceiling on how confident a
        4-level question can ever look. Computed here, not quoted.
D       the punchline: one response, three question types, two different formulas for
        the field called `confidence`, with two different floors -- and a decisive
        `choice` answer scoring *lower* than a near-coin-flip `noul`.
E       the formula's two fixed points -- the only part of `confidence` that is exact.

Every number printed below is computed in this process on this machine. Nothing is
quoted from upstream's README or BENCHMARKS.md, whose figures come from a T4 GPU.

Run:  uv run python scenarios/03_confidence_gate.py
Exits 0 when the invariant holds, 1 when it does not.
"""
from __future__ import annotations

import math
import os
import statistics
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402

# --------------------------------------------------------------------------- data

TICKETS: list[tuple[str, str]] = [
    ("billing", "I was charged twice for my subscription this month"),
    ("billing", "My credit card was declined but the invoice says paid"),
    ("billing", "Please refund the duplicate payment of 49 dollars"),
    ("billing", "Can you send me a VAT invoice for last month"),
    ("billing", "I want to upgrade my plan and be billed annually"),
    ("billing", "Please issue a credit note for invoice 4471"),
    ("auth", "I cannot log in, SSO redirects me back to the login page"),
    ("auth", "My MFA device was lost and I am locked out of my account"),
    ("auth", "Password reset email never arrives in my inbox"),
    ("auth", "SAML assertion is rejected with an invalid signature error"),
    ("auth", "Our session tokens expire after five minutes instead of an hour"),
    ("auth", "Single sign on with Okta stopped working this morning"),
    ("infra", "The API is returning 503 for every request since 09:00"),
    ("infra", "Latency on the eu-west endpoint jumped to 8 seconds"),
    ("infra", "The deploy pipeline is stuck and the service is down"),
    ("infra", "We are seeing 500 errors on the webhook delivery endpoint"),
    ("infra", "The status page says operational but nothing is responding"),
    ("infra", "Database connections are being refused in production"),
]

DEPT_Q = {
    "dept": {
        "type": "choice",
        "instructions": "Which support team should own this ticket?",
        "criteria": {
            "billing": "payments, invoices, refunds, subscriptions",
            "auth": "login, passwords, SSO, MFA, sessions",
            "infra": "outages, latency, errors, deploys",
        },
    }
}

SEV_Q = {
    "sev": {
        "type": "score",
        "instructions": "Severity of this incident?",
        "criteria": ["none", "minor", "major", "critical"],
    }
}

INCIDENTS = [
    "everything is on fire, total outage, all customers down",
    "one user reports a typo in the footer",
    "checkout is failing for about 5% of users",
]

MIXED_TICKET = "The API is returning 503 for every request since 09:00"
NOUL_QUESTIONS = {
    "outage": "Is this ticket reporting a production outage?",
    "usable": "Is the customer currently unable to use the product?",
}

# The README's own gate is `conf >= 0.85` (upstream/README.md:202), so the sweep uses
# `>=` throughout and carries 0.85 as a row rather than inventing a threshold.
README_GATE = 0.85
README_GATE_SOURCE = "upstream/README.md:202  ->  if conf >= 0.85: route_automatically(dept)"
GATES = [0.3, 0.4, 0.5, 0.6, 0.7, README_GATE]
TOP_PROBS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

ENTROPY_FORMULA = "1 - H(p)/ln(k)  (common.py:216)"
NOUL_FORMULA = "max(p, 1-p)  (agent.py:360)"

# Largest k in each temperature bucket name, used to rebuild the probability vector the
# rejected temperature would have been applied to.
BUCKET_K = {"2": 2, "3-5": 5, "6-10": 10, "11+": 11}


def clip(text: str, width: int) -> str:
    """Trim `text` to `width` columns so a table column stays aligned."""
    return text if len(text) <= width else text[: width - 3] + "..."


def note(*lines: str, style: str = "dim") -> None:
    """Print explanatory prose under a table, indented to match it."""
    for line in lines:
        print("  " + C.c(line, style))


def softmax(z: np.ndarray) -> np.ndarray:
    """The same softmax laya applies to temperature-scaled logits (agent.py:330-332)."""
    e = np.exp(z - z.max())
    return e / e.sum()


# --------------------------------------------------------------------------- setup


def load_with_warning():
    """Load the english checkpoint, returning (agent, warnings raised during the load).

    `_common` silences laya's RuntimeWarnings so the other scenarios stay readable.
    This scenario is about calibration, so it re-enables them for the load and prints
    what the checkpoint says about itself.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = C.load_agent("english")
    return agent, [str(w.message) for w in caught]


def sharpening_demo(shipped: float, k: int, top_at_t1: float = 0.24) -> dict:
    """What `shipped` would do to a k-option vector whose top probability is `top_at_t1`.

    Builds the logit vector that softmaxes to exactly `top_at_t1` at T=1, then divides it
    by `shipped` the way laya does at agent.py:330 and reports the published top
    probability. This is computed here, not quoted from upstream's source comment.
    """
    lead = math.log(top_at_t1 * (k - 1) / (1.0 - top_at_t1))
    z = np.array([lead] + [0.0] * (k - 1))
    return {
        "k": k,
        "multiplier": 1.0 / shipped,
        "top_at_t1": float(softmax(z)[0]),
        "top_at_shipped": float(softmax(z / shipped)[0]),
    }


def print_setup(agent, messages: list[str]) -> dict:
    C.section("setup. what the checkpoint says about itself on load")
    if messages:
        for msg in messages:
            print("  " + C.c("RuntimeWarning:", "yellow") + " " + msg)
    else:
        note("(no warning raised on this load)")

    raw, applied = agent.temperature_by_options_raw, agent.temperature_by_options
    rows, clamped = [], []
    for bucket in sorted(raw, key=lambda b: (b.split(":")[0], b)):
        shipped, used = float(raw[bucket]), float(applied[bucket])
        changed = abs(shipped - used) > 1e-9
        if changed:
            clamped.append((bucket, shipped, used))
        rows.append([bucket, f"{shipped:.5f}", f"{used:.5f}",
                     C.c("CLAMPED", "red") if changed else C.c("as shipped", "dim")])
    print()
    C.table(rows, ["BUCKET", "SHIPPED", "APPLIED", ""], align="lrr")
    print()

    demos = []
    for bucket, shipped, used in clamped:
        k = BUCKET_K.get(bucket.split(":")[-1])
        if k is None or not 0.0 < shipped < 1.0:
            continue
        d = sharpening_demo(shipped, k)
        d.update(bucket=bucket, shipped=shipped, applied=used)
        demos.append(d)
        note("A temperature below 1 sharpens logits instead of softening them. "
             f"{bucket} ships {shipped:.5f},",
             f"a {d['multiplier']:.2f}x logit multiplier. Computed here, not quoted: a k={k} "
             f"vector whose top",
             f"probability is {d['top_at_t1']:.3f} at T=1 would be published as "
             f"{d['top_at_shipped']:.4f} at the shipped temperature.",
             f"laya refuses it and applies {used:.5f}, so confidence from that bucket is "
             "uncalibrated",
             "either way -- but at least it is not a fabricated certainty.")
    if not clamped:
        note("(this checkpoint ships no temperature outside [0.5, 5])")
    return {"clamped": [{"bucket": b, "shipped": s, "applied": u} for b, s, u in clamped],
            "sharpening_demos": demos}


# --------------------------------------------------------------------------- A


def _classify_all(agent, per_ticket_ms: list[float] | None = None) -> list[dict]:
    """One pass over every ticket, optionally recording per-ticket milliseconds.

    `C.sync()` brackets each timed call: MPS and CUDA dispatch asynchronously, so a bare
    `perf_counter()` around `system_one` stops before the device has finished and reports
    a number that is far too good (see `_common.sync`).
    """
    results = []
    for gold, text in TICKETS:
        C.sync()
        t0 = time.perf_counter()
        answer = agent.system_one(text, DEPT_Q)["answers"]["dept"]
        C.sync()
        if per_ticket_ms is not None:
            per_ticket_ms.append((time.perf_counter() - t0) * 1000.0)
        results.append({
            "ticket": text,
            "gold": gold,
            "predicted": answer["choice"],
            "correct": answer["choice"] == gold,
            "confidence": float(answer["confidence"]),
            "probabilities": answer["probabilities"],
        })
    return results


def run_tickets(agent) -> tuple[list[dict], dict]:
    """Classify every ticket twice, returning (results of the second pass, timings).

    The first pass is warm-up only and is not timed: every ticket is a new token length,
    and on MPS the first forward pass at a new tensor shape pays kernel compilation. The
    second pass is timed per ticket, and the reported figure is the **median**, which is
    what a wall-clock cost on a shared laptop can honestly support -- a mean or a single
    total swings by 2-3x with whatever else the machine is doing. No cause is attributed
    to the spread beyond that.

    Running twice also establishes determinism: the caller compares the two passes, and
    every claim below rests on the run being reproducible.
    """
    first = _classify_all(agent)

    per_ticket: list[float] = []
    second = _classify_all(agent, per_ticket)

    deterministic = ([r["predicted"] for r in first] == [r["predicted"] for r in second]
                     and [r["confidence"] for r in first] == [r["confidence"] for r in second])
    ordered = sorted(per_ticket)
    p50 = statistics.median(per_ticket)
    return second, {
        "ms_per_ticket_p50": p50,
        "ms_per_ticket_min": ordered[0],
        "ms_per_ticket_max": ordered[-1],
        "estimated_set_ms": p50 * len(TICKETS),
        "measured_pass_ms": sum(per_ticket),
        "per_ticket_ms": per_ticket,
        "deterministic_across_passes": deterministic,
    }


def list_input_error() -> str:
    """Establish, rather than assert, what `ece_score` does with Python lists."""
    from laya.common import ece_score

    try:
        ece_score([0.5, 0.6], [1.0, 0.0], 15)
    except Exception as exc:  # noqa: BLE001 - the exception type is the finding
        return f"{type(exc).__name__}: {exc}"
    return "(no error: lists are accepted on this version)"


def section_a(results: list[dict], timing: dict) -> dict:
    C.section("A. 18 unambiguous tickets: accuracy against reported confidence")
    rows = []
    for r in results:
        rows.append([
            clip(r["ticket"], 52),
            r["gold"],
            r["predicted"],
            C.c("ok", "green") if r["correct"] else C.c("MISS", "red"),
            f"{r['confidence']:.4f} " + C.bar(r["confidence"], 16),
        ])
    C.table(rows, ["TICKET", "GOLD", "PREDICTED", "OK", "CONFIDENCE"])

    confs = np.array([r["confidence"] for r in results], dtype=float)
    corrects = np.array([float(r["correct"]) for r in results], dtype=float)
    n_ok, n = int(corrects.sum()), len(results)
    accuracy, mean_conf = float(corrects.mean()), float(confs.mean())

    # ece_score is strict about its inputs; `list_input_error()` shows what lists do.
    from laya.common import ece_score

    ece = float(ece_score(confs, corrects, 15))

    print()
    note(f"accuracy {n_ok}/{n} = {accuracy:.3f}"
         f" | mean confidence {mean_conf:.3f}"
         f" | ECE(15 bins) = {ece:.3f}", style="bold")
    print(f"  confidence range {confs.min():.3f} .. {confs.max():.3f}"
          f"     gap (accuracy - mean confidence) = {accuracy - mean_conf:+.3f}")

    missed = n - n_ok
    headline = ("Every answer is right." if missed == 0
                else f"{missed} of {n} answers are wrong.")
    direction = "under-confident" if mean_conf < accuracy else "over-confident"
    note(f"{headline} Mean reported confidence {mean_conf:.2f} against an observed hit rate",
         f"of {accuracy:.2f}: on this set the model is {direction} by "
         f"{abs(accuracy - mean_conf):.2f}.")

    device = C.resolve_device()
    note(f"Cost of collecting this evidence, measured on {device} with the device synchronised",
         f"around every call: median {timing['ms_per_ticket_p50']:.1f} ms per ticket "
         f"(min {timing['ms_per_ticket_min']:.1f}, max {timing['ms_per_ticket_max']:.1f}),",
         f"so about {timing['estimated_set_ms'] / 1000:.1f} s to label all {n}. The spread is "
         "wall-clock noise on a shared machine;",
         "the median is the only part of it worth quoting.",
         f"Both passes over the set returned identical labels and confidences: "
         f"{timing['deterministic_across_passes']}.")
    return {"accuracy": accuracy, "mean_confidence": mean_conf, "ece": ece,
            "n": n, "n_correct": n_ok,
            "confidence_min": float(confs.min()), "confidence_max": float(confs.max()),
            "device": device, **timing}


# --------------------------------------------------------------------------- B


def section_b(results: list[dict]) -> list[dict]:
    C.section("B. what a confidence gate would actually have done to these 18 answers")
    print("  gate under test  " + C.c(README_GATE_SOURCE, "cyan"))
    note("the sweep uses >= to match the README exactly; other rows are shown for contrast")
    print()

    total = len(results)
    sweep = []
    for t in GATES:
        passed = [r for r in results if r["confidence"] >= t]
        n = len(passed)
        sweep.append({
            "threshold": t,
            "is_readme_gate": t == README_GATE,
            "automated": n,
            "automated_pct": 100.0 * n / total,
            "accuracy_of_automated": (sum(r["correct"] for r in passed) / n) if n else None,
            "correct_sent_to_human": sum(1 for r in results
                                         if r["confidence"] < t and r["correct"]),
        })

    rows = []
    for s in sweep:
        cost = s["correct_sent_to_human"]
        acc = s["accuracy_of_automated"]
        label = f"{s['threshold']:.2f}" + (" <- README" if s["is_readme_gate"] else "")
        rows.append([
            C.c(label, "bold") if s["is_readme_gate"] else label,
            f"{s['automated']}/{total}",
            f"{s['automated_pct']:.0f}%",
            "n/a" if acc is None else f"{acc:.3f}",
            C.c(str(cost), "red") if cost else C.c("0", "green"),
        ])
    C.table(rows, ["THRESHOLD", "AUTOMATED", "AUTOMATED %", "ACCURACY OF AUTOMATED",
                   "CORRECT ANSWERS SENT TO A HUMAN"], align="rrrrr")
    print()

    accs = [s["accuracy_of_automated"] for s in sweep if s["accuracy_of_automated"] is not None]
    spread = max(accs) - min(accs) if accs else 0.0
    if spread < 1e-9:
        note(f"The accuracy column never moves ({accs[0]:.3f} at every threshold): there is",
             "nothing here for the gate to catch. Raising the threshold buys no accuracy and",
             "costs human hours. That is exactly what under-confidence looks like in production.")
    else:
        note(f"The accuracy of the automated subset ranges over {spread:.3f} across these",
             "thresholds, so on this set the gate does separate right answers from wrong ones.",
             "Read the cost column against that gain before picking a threshold.")
    return sweep


# --------------------------------------------------------------------------- C


def section_c(agent) -> tuple[list[dict], list[dict], list[dict]]:
    from laya.common import confidence_from_probs

    C.section("C. why score confidences look like coin flips")
    score_rows = []
    for text in INCIDENTS:
        answer = agent.system_one(text, SEV_Q)["answers"]["sev"]
        probs = answer["probabilities"]
        top_key = max(probs, key=lambda k: probs[k])
        score_rows.append({
            "incident": text,
            "score": float(answer["score"]),
            "top_prob": float(probs[top_key]),
            "top_level": answer["legend"][top_key],
            "confidence": float(answer["confidence"]),
            "probabilities": probs,
            "legend": answer["legend"],
        })

    C.table([[clip(r["incident"], 48),
              f"{r['score']:.3f}",
              f"{r['top_prob']:.3f} ({r['top_level']})",
              f"{r['confidence']:.4f}",
              " ".join(f"{k}:{v:.3f}" for k, v in r["probabilities"].items())]
             for r in score_rows],
            ["INCIDENT", "score", "top prob", "confidence", "distribution"], align="lrrrl")

    legend = score_rows[0]["legend"]
    outage = score_rows[0]
    lo_i, hi_i = int(math.floor(outage["score"])), int(math.ceil(outage["score"]))
    print()
    print("  legend  " + " ".join(f"{k}={v}" for k, v in legend.items())
          + "   " + C.c("(score is the only primitive that ships one)", "dim"))
    note("score = sum(i * p_i), an EXPECTED VALUE, not an argmax. The total outage scores",
         f"{outage['score']:.2f} -- between {legend[str(lo_i)]} and {legend[str(hi_i)]} -- because "
         f"real mass sits on both levels",
         f"({legend[str(lo_i)]} {outage['probabilities'][str(lo_i)]:.3f}, "
         f"{legend[str(hi_i)]} {outage['probabilities'][str(hi_i)]:.3f}).")

    k_levels = len(score_rows[0]["probabilities"])
    C.section(f"C2. the ceiling: highest confidence a {k_levels}-level score can report (computed)")
    ceiling = []
    for top in TOP_PROBS:
        flat = np.full(k_levels, (1.0 - top) / (k_levels - 1))
        flat[0] = top
        ceiling.append({"top_prob": top,
                        "confidence": round(confidence_from_probs(flat, k_levels), 4)})
    C.table([[f"{r['top_prob']:.2f}", f"{r['confidence']:.4f}", C.bar(r["confidence"], 28)]
             for r in ceiling],
            ["TOP PROBABILITY", "ACHIEVABLE CONFIDENCE",
             f"residual mass spread flat over k={k_levels}"],
            align="rr")

    C.section("C3. ordinal spill: the same mass on ONE adjacent level instead (computed)")
    ordinal = []
    for top in TOP_PROBS[:-1]:
        p = np.zeros(k_levels)
        p[0], p[1] = top, 1.0 - top
        ordinal.append({"top_prob": top,
                        "confidence": round(confidence_from_probs(p, k_levels), 4)})
    C.table([[f"{r['top_prob']:.2f}", f"{r['confidence']:.4f}", C.bar(r["confidence"], 28)]
             for r in ordinal],
            ["TOP PROBABILITY", "ACHIEVABLE CONFIDENCE", "residual mass on one neighbour"],
            align="rr")

    needed = next((r["top_prob"] for r in ceiling if r["confidence"] >= 0.8), None)
    at_same_top = {r["top_prob"]: r["confidence"] for r in ordinal}
    worst_flat = ceiling[0]
    print()
    if needed is not None:
        note(f"A {k_levels}-level score needs a top probability of {needed:.2f} to reach 0.80 "
             f"confidence:",
             f"at {needed:.2f} the flat-residual vector scores "
             f"{next(r['confidence'] for r in ceiling if r['top_prob'] == needed):.4f}, "
             f"at {ceiling[-2]['top_prob']:.2f} only {ceiling[-2]['confidence']:.4f}.")
    else:
        note(f"No top probability in this sweep reaches 0.80 confidence at k={k_levels}; "
             f"the best is {max(r['confidence'] for r in ceiling):.4f}.")
    note("An ordinal question legitimately spills mass onto the level next door. At a "
         f"{worst_flat['top_prob']:.2f} top",
         f"probability that scores {at_same_top[worst_flat['top_prob']]:.4f}, against "
         f"{worst_flat['confidence']:.4f} for the same mass spread flat. Normalized",
         "entropy cannot tell the two apart, so low score-confidence often means ordinal,",
         "not unsure.")
    return score_rows, ceiling, ordinal


# --------------------------------------------------------------------------- D


def section_d(agent) -> dict:
    C.section("D. one response, one field name, two formulas, two floors")
    questions = {"dept": DEPT_Q["dept"], "sev": SEV_Q["sev"]}
    for qid, instructions in NOUL_QUESTIONS.items():
        questions[qid] = {"type": "noul", "instructions": instructions}
    answers = agent.system_one(MIXED_TICKET, questions)["answers"]

    print("  state   " + C.c(MIXED_TICKET, "cyan"))
    note(f"one system_one() call, {len(questions)} questions, one forward pass "
         f"(agent.py:295-301 batches them)")
    print()

    choice, score = answers["dept"], answers["sev"]
    k_choice, k_score = len(choice["probabilities"]), len(score["probabilities"])
    nouls = {qid: answers[qid] for qid in NOUL_QUESTIONS}
    least_decisive = min(nouls.values(), key=lambda a: abs(a["noul"] - 0.5))

    C.table([[f"choice (k={k_choice})", f"{choice['confidence']:.4f}", ENTROPY_FORMULA, "0.0"],
             [f"score  (k={k_score})", f"{score['confidence']:.4f}", ENTROPY_FORMULA, "0.0"],
             ["noul   (k=2)", f"{answers['outage']['confidence']:.4f}", NOUL_FORMULA, "0.5"]],
            ["TYPE", "confidence", "formula", "floor"], align="lr")

    print()
    C.table([[qid, f"{a['noul']:.4f}", f"{a['confidence']:.4f}",
              f"{abs(a['noul'] - 0.5):.4f}", NOUL_QUESTIONS[qid]]
             for qid, a in nouls.items()],
            ["NOUL", "noul", "confidence", "|noul-0.5|", "QUESTION"], align="lrrr")

    top_label = max(choice["probabilities"], key=lambda k: choice["probabilities"][k])
    top_p = float(choice["probabilities"][top_label])
    print()
    note(f"The choice answer picks {top_label} at p={top_p:.3f} and reports confidence "
         f"{choice['confidence']:.2f}.",
         f"The noul answer nearest a coin flip (noul={least_decisive['noul']:.4f}, "
         f"{abs(least_decisive['noul'] - 0.5):.4f} from 0.5) reports",
         f"{least_decisive['confidence']:.2f}, because max(p, 1-p) cannot go below 0.5. "
         f"The more decisive answer looks less confident.")
    note("Practical rule: for noul use abs(noul - 0.5) as decisiveness, never confidence;",
         "and never compare confidence across two different k.", style="bold")
    return {"ticket": MIXED_TICKET, "choice": choice, "score": score, "noul": nouls,
            "choice_top_label": top_label, "choice_top_prob": top_p,
            "k_choice": k_choice, "k_score": k_score,
            "least_decisive_noul": {"noul": float(least_decisive["noul"]),
                                    "confidence": float(least_decisive["confidence"]),
                                    "distance_from_half": abs(float(least_decisive["noul"]) - 0.5)}}


# --------------------------------------------------------------------------- E


def section_e() -> list[dict]:
    """The two fixed points of normalized entropy -- the only exact part of `confidence`."""
    from laya.common import confidence_from_probs

    C.section("E. the two fixed points of the entropy formula")
    rows = []
    for k in (2, 3, 4, 6):
        one_hot = np.zeros(k)
        one_hot[0] = 1.0
        rows.append({
            "k": k,
            "uniform": confidence_from_probs(np.full(k, 1.0 / k), k),
            "one_hot": confidence_from_probs(one_hot, k),
        })
    C.table([[r["k"], f"{r['uniform']:.3e}", f"{r['one_hot']:.6f}"] for r in rows],
            ["k", "confidence(uniform)", "confidence(one-hot)"], align="rrr")
    print()
    note("Confidence is a pure function of the probability vector and its width k. It measures",
         "how peaked the distribution is. It never saw a label, so it cannot know accuracy.")
    return rows


# --------------------------------------------------------------------------- main


def main() -> None:
    C.header("03 - Confidence is not accuracy",
             "measure your own gate before you trust one  |  checkpoint: english")
    C.require_cached("english")

    agent, warning_messages = load_with_warning()
    setup = print_setup(agent, warning_messages)

    results, timing = run_tickets(agent)
    stats = section_a(results, timing)
    sweep = section_b(results)
    score_rows, ceiling, ordinal = section_c(agent)
    mixed = section_d(agent)
    fixed_points = section_e()

    accuracy, mean_conf, ece = stats["accuracy"], stats["mean_confidence"], stats["ece"]
    n, n_ok = stats["n"], stats["n_correct"]
    readme_gate = next(s for s in sweep if s["is_readme_gate"])
    noul_confs = [float(a["confidence"]) for a in mixed["noul"].values()]
    least = mixed["least_decisive_noul"]
    choice_conf = float(mixed["choice"]["confidence"])

    C.section("verdict")
    direction = "UNDER-confidence" if mean_conf < accuracy else "OVER-confidence"
    note(f"{n_ok}/{n} correct at mean confidence {mean_conf:.2f}, ECE {ece:.2f}. "
         f"The miscalibration is {direction}.",
         f"Do not inherit the README's threshold -- fit one with ece_score() on ~{n}",
         f"labelled items of your own. Measured here: about "
         f"{stats['estimated_set_ms'] / 1000:.1f} s of inference, once.",
         style="bold")
    print(f"\n  At the README's {README_GATE:.2f} gate this set automates "
          f"{readme_gate['automated']}/{n} answers and hands "
          f"{C.c(str(readme_gate['correct_sent_to_human']), 'red')} correct ones to a human.")
    print(f"  A near-coin-flip noul ({least['noul']:.4f}) reports {least['confidence']:.2f} while "
          f"the {mixed['k_choice']}-way choice that picked")
    print(f"  {mixed['choice_top_label']} at p={mixed['choice_top_prob']:.3f} reports "
          f"{choice_conf:.2f}. Confidence is not comparable across question types.")

    print()
    note("Footnotes",
         f"- n={n}, hand-written and deliberately unambiguous. 15 bins over {n} points is not a",
         "  stable ECE. This shows the DIRECTION of the error on easy items; it is not a",
         "  benchmark of Laya calibration on real traffic. It says nothing about how the",
         "  checkpoint is calibrated on hard or adversarial inputs.",
         "- ece_score() needs numpy arrays. Checked in this run: Python lists raise",
         f"  {list_input_error()}",
         "- ece_score() bins with (conf > lo) & (conf <= hi), so a sample at exactly 0.0 falls",
         "  into no bin and is silently dropped. np.clip(conf, 1e-6, 1.0) if that matters to you.",
         "- confidence(uniform) is floating-point zero, not literally 0.0: 1 - H/ln(k) leaves",
         "  ~2e-16 for k=3 and k=6, just above the clip. Compare with a tolerance, not ==.",
         "- the noul >= 0.5 floor below is structural (max(p, 1-p)), so that clause of the",
         "  assertion guards the library's formula, not this checkpoint's behaviour.")

    C.save_artifact("03_confidence_gate", {
        "env": C.env_summary(),
        "load_warnings": warning_messages,
        "temperature_by_options_raw": agent.temperature_by_options_raw,
        "temperature_by_options_applied": agent.temperature_by_options,
        "temperature_setup": setup,
        "tickets": results,
        "ticket_stats": stats,
        "readme_gate": {"threshold": README_GATE, "source": README_GATE_SOURCE, **readme_gate},
        "gate_sweep": sweep,
        "score_answers": score_rows,
        "confidence_ceiling_k4_flat": ceiling,
        "confidence_ceiling_k4_ordinal": ordinal,
        "mixed_response": mixed,
        "formula_fixed_points": fixed_points,
    })

    uniform_is_zero = all(r["uniform"] <= 1e-9 for r in fixed_points)
    one_hot_is_one = all(r["one_hot"] == 1.0 for r in fixed_points)
    # The load-bearing behavioural claim of section D: a noul well inside coin-flip
    # territory outranks a choice that committed to a clear winner. Both sides are model
    # outputs, so this can genuinely fail; the observed gap is ~0.24 with the margins below.
    cross_type_inversion = (least["distance_from_half"] < 0.25
                            and mixed["choice_top_prob"] > 0.6
                            and least["confidence"] > choice_conf + 0.1)

    C.require(
        accuracy >= 0.85
        and ece > 0.15
        and mean_conf < accuracy - 0.05
        and readme_gate["correct_sent_to_human"] >= 5
        and cross_type_inversion
        and stats["deterministic_across_passes"]
        and all(cf >= 0.5 for cf in noul_confs)
        and uniform_is_zero
        and one_hot_is_one,
        f"accuracy {accuracy:.3f} >= 0.85 while mean confidence {mean_conf:.3f} sits below it "
        f"(under-confident, ECE {ece:.3f} > 0.15); the README's {README_GATE:.2f} gate escalates "
        f"{readme_gate['correct_sent_to_human']} correct answers; a noul "
        f"{least['distance_from_half']:.3f} from a coin flip reports {least['confidence']:.3f} "
        f"against {choice_conf:.3f} for a choice that picked "
        f"{mixed['choice_top_label']} at p={mixed['choice_top_prob']:.3f}; two passes over the "
        f"set agreed exactly; and confidence_from_probs pins uniform to 0 and one-hot to 1 "
        f"for k in 2,3,4,6",
    )


if __name__ == "__main__":
    main()
