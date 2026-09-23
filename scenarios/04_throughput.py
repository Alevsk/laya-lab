"""Five decisions, one forward pass: what batching actually buys.

Laya's `predict(state, questions)` takes a *dictionary* of questions, not one
question, and answers all of them in a single call. This scenario measures what
that is really worth, and what it is not.

What it demonstrates, in order:

  A  A forward pre-hook on the encoder records how many times the transformer
     actually runs *and the shape of the tensor it is handed*. Twenty questions
     -> ONE encoder call over a twenty-row batch. Four questions of different
     lengths -> ONE four-row batch whose width is the longest row, the shorter
     rows made up with padding that the attention mask switches off.
  B  Batching changes nothing about the answers: the three-in-one result equals
     the three separate calls, is invariant to the order the questions are listed
     in, and a long fourth question sitting next to short ones does not leak into
     them.
  C  The latency curve. Per-question cost falls several-fold between N=1 and
     N=40, because a fixed per-call overhead is amortised over more rows. The
     absolute milliseconds move with machine load, so the script measures them
     fresh every run and asserts only the direction.
  D  The counterweight nobody advertises: `usage.input_tokens` is the sum of the
     per-question row lengths -- every row re-encodes the whole state, and
     nothing is shared between them. Batching buys latency, not compute.
  E  What that means for a real workload: the shipped 5-question triage preset,
     converted into tickets per second on this machine.

How to read the output: Section C's "ms PER QUESTION" column is the number that
matters for throughput planning; its "input_tokens" column is the number that
matters for capacity planning. They move in opposite directions on purpose.

Stable invariants asserted at the end: one encoder call per `predict`, with one
row per question and a padded width equal to the longest row; batched answers
equal to per-question answers and invariant to question order; `input_tokens`
additive over distinct questions and exactly 20x for twenty identical ones; and
per-question p50 at N=20 strictly below p50 at N=1. No absolute millisecond
figure is asserted -- absolute latency on this machine swings with load, the
shape of the curve does not.
"""
from __future__ import annotations

import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

SHAPES = (1, 3, 5, 10, 20, 40)
REPS = 12
WARMUP = 5
SEED = 0

# laya rounds every number it returns to 4 decimals (see Agent.system_one), so one
# unit in the last place is the finest distinction the API can express.
ROUND_UNIT = 5e-4

STATE = "I cannot log in with SSO and I was also charged twice this month, please help."

TICKET = (
    "Subject: still locked out, third email this week\n\n"
    "Our whole team has been unable to log in since the SSO change on Monday. "
    "We have a customer demo on Thursday and right now nobody can get into the "
    "dashboard. We were also billed twice for October. If this is not fixed "
    "today we are going to start looking at alternatives."
)

Q3 = {
    "dept": {
        "type": "choice",
        "instructions": "Which team owns this ticket?",
        "criteria": {"billing": "payments", "auth": "login", "infra": "outages"},
    },
    "urg": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["low", "med", "high", "critical"],
    },
    "ang": {"type": "noul", "instructions": "Is the customer angry?"},
}

# Deliberately much longer than the three above, so the batch has to pad.
LONG_SIBLING = {
    "type": "choice",
    "instructions": (
        "Taking into account the full lifecycle of this account, its contract tier, "
        "the regional data-residency commitments we have made to this customer, and "
        "the follow-the-sun rota currently in effect, which regional support desk "
        "should be given ownership of this ticket?"
    ),
    "criteria": {"emea": "europe", "amer": "americas", "apac": "asia-pacific"},
}


def mkq(n: int) -> dict:
    """`n` copies of one identical choice question -- a clean, constant-width batch."""
    return {
        f"q{i}": {
            "type": "choice",
            "instructions": "Which team owns this ticket?",
            "criteria": {"billing": "payments", "auth": "login", "infra": "outages"},
        }
        for i in range(n)
    }


def percentile(samples: list[float], q: float) -> float:
    """Nearest-rank percentile; with 12 samples the quantile helpers are noisier."""
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def close(x, y, tol: float = ROUND_UNIT) -> bool:
    """Structural equality, with numbers compared to within one rounding unit.

    Exact float equality across two different batch widths is a stronger claim
    than laya's own output precision supports: the same row computed inside a
    3-row batch and inside a 1-row batch goes through differently shaped kernel
    reductions. laya rounds to 4 decimals, so `tol` is one unit in the last
    place laya reports -- tight enough that any real difference in the decision
    fails, loose enough that a rounding boundary does not make the scenario
    flaky. Exact equality is reported separately, as an observation.
    """
    if isinstance(x, dict) and isinstance(y, dict):
        return set(x) == set(y) and all(close(x[k], y[k], tol) for k in x)
    if isinstance(x, bool) or isinstance(y, bool):
        return x == y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(float(x) - float(y)) <= tol
    return x == y


class EncoderProbe:
    """Context manager recording every call into the transformer encoder.

    `DecisionModel.forward` invokes `self.encoder(input_ids=..., attention_mask=...)`
    exactly once per forward, so a forward pre-hook here counts real transformer
    invocations -- if laya looped over questions in Python, this would fire once
    per question instead of once per `predict`.
    """

    def __init__(self, agent):
        self.agent = agent
        self.calls: list[dict] = []
        self._hook = None

    def _record(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        rows, width = tuple(ids.shape)
        self.calls.append({
            "rows": rows,
            "width": width,
            "real_tokens": int(mask.sum()) if mask is not None else None,
        })

    def __enter__(self):
        self._hook = self.agent.model.encoder.register_forward_pre_hook(
            self._record, with_kwargs=True
        )
        return self

    def __exit__(self, *exc):
        self._hook.remove()
        return False

    def run(self, state, questions) -> tuple[dict, list[dict]]:
        """One `predict`, returning its result and the encoder calls it caused."""
        before = len(self.calls)
        result = self.agent.predict(state, questions)
        return result, self.calls[before:]


def time_shapes(agent, shapes, device) -> dict[int, list[float]]:
    """Time `predict` at each batch width, interleaving the shapes at random.

    This is the one place in the project where a naive benchmark lies. The
    backend re-specialises its kernels per tensor shape, so timing each N in a
    contiguous block lets one graph stay hot for the whole block and produces a
    curve whose shape is an artefact of the measurement order. Shuffling the
    samples across shapes gives every width the same cache conditions.
    """
    payloads = {n: mkq(n) for n in shapes}
    for n in shapes:
        for _ in range(WARMUP):
            agent.predict(STATE, payloads[n])
    C.sync(device)

    order = [n for n in shapes for _ in range(REPS)]
    random.Random(SEED).shuffle(order)

    samples: dict[int, list[float]] = {n: [] for n in shapes}
    for n in order:
        C.sync(device)
        t0 = time.perf_counter()
        agent.predict(STATE, payloads[n])
        C.sync(device)
        samples[n].append((time.perf_counter() - t0) * 1000)
    return samples


def main() -> None:
    C.require_cached("english")
    device = C.resolve_device()

    C.header(
        "04 - Five decisions, one forward pass: what batching actually buys",
        f"english checkpoint on {device} - one encoder call, N answers, N x the tokens",
    )
    agent = C.load_agent("english")

    # ---------------------------------------------------------------- A
    C.section("A. How many times does the encoder actually run, and on what tensor?")

    mixed = {**Q3, "region": LONG_SIBLING}
    with EncoderProbe(agent) as probe:
        result20, calls20 = probe.run(STATE, mkq(20))
        result1, calls1 = probe.run(STATE, mkq(1))
        result_mixed, calls_mixed = probe.run(STATE, mixed)
        singles = {k: probe.run(STATE, {k: v}) for k, v in mixed.items()}

    # Every predict() in the probe must have produced exactly one encoder call, the four
    # single-question calls included -- otherwise the widths below are not what they claim.
    call_counts = [len(calls20), len(calls1), len(calls_mixed)] + [len(cs) for _, cs in singles.values()]
    one_call_each = set(call_counts) == {1}

    single_widths = {k: (cs[0]["width"] if cs else -1) for k, (_, cs) in singles.items()}
    single_tokens = {k: r["usage"]["input_tokens"] for k, (r, _) in singles.items()}

    rows_match_questions = (
        calls20[0]["rows"] == 20
        and calls1[0]["rows"] == 1
        and calls_mixed[0]["rows"] == len(mixed)
    )

    C.table(
        [
            ["mkq(20)", len(calls20), calls20[0]["rows"], calls20[0]["width"],
             calls20[0]["real_tokens"], calls20[0]["rows"] * calls20[0]["width"]],
            ["mkq(1)", len(calls1), calls1[0]["rows"], calls1[0]["width"],
             calls1[0]["real_tokens"], calls1[0]["rows"] * calls1[0]["width"]],
            ["3 short + 1 long", len(calls_mixed), calls_mixed[0]["rows"],
             calls_mixed[0]["width"], calls_mixed[0]["real_tokens"],
             calls_mixed[0]["rows"] * calls_mixed[0]["width"]],
        ],
        ["predict() CALL", "ENCODER CALLS", "ROWS", "PADDED WIDTH", "REAL TOKENS", "TENSOR CELLS"],
        align="lrrrrr",
    )
    print(C.c("  Rows come from questions, not from Python iteration: twenty questions are twenty rows\n"
              "  of ONE tensor handed to the transformer once. The loop is inside the tensor.", "dim"))

    widest = max(single_widths, key=single_widths.get)
    padded_to_longest = calls_mixed[0]["width"] == single_widths[widest]
    pad_cells = (calls_mixed[0]["rows"] * calls_mixed[0]["width"]) - calls_mixed[0]["real_tokens"]
    print(f"\n  the mixed batch is {calls_mixed[0]['rows']} x {calls_mixed[0]['width']}; "
          f"alone, {widest!r} is {single_widths[widest]} tokens wide and is the longest of the four "
          f"({', '.join(f'{k}={v}' for k, v in single_widths.items())})")
    print(f"  -> padded width == longest row: {'yes' if padded_to_longest else 'NO'}; "
          f"{pad_cells} of the {calls_mixed[0]['rows'] * calls_mixed[0]['width']} cells are padding "
          f"that the attention mask switches off")

    # ---------------------------------------------------------------- B
    C.section("B. Batching does not change the answers")

    batched = agent.predict(STATE, Q3)["answers"]
    per_question = {k: agent.predict(STATE, {k: v})["answers"][k] for k, v in Q3.items()}
    same_as_singles = close(batched, per_question)
    exact_as_singles = batched == per_question

    reversed_q3 = {k: Q3[k] for k in reversed(list(Q3))}
    reversed_answers = agent.predict(STATE, reversed_q3)["answers"]
    order_invariant = close(reversed_answers, batched)
    exact_reversed = reversed_answers == batched

    with_sibling = result_mixed["answers"]
    padding_clean = close({k: with_sibling[k] for k in Q3}, batched)
    exact_sibling = {k: with_sibling[k] for k in Q3} == batched

    def mark(ok: bool) -> str:
        return C.c("PASS", "green") if ok else C.c("FAIL", "red")

    C.table(
        [
            ["3 questions batched == 3 separate calls", mark(same_as_singles),
             "exact" if exact_as_singles else "within 5e-4",
             "choice + score + noul, every field compared"],
            ["question order reversed == original", mark(order_invariant),
             "exact" if exact_reversed else "within 5e-4",
             "answers keep insertion order; the values do not move"],
            ["long 4th question leaves the other 3 alone", mark(padding_clean),
             "exact" if exact_sibling else "within 5e-4",
             "the padding mask holds; rows do not leak into each other"],
        ],
        ["CHECK", "RESULT", "AGREEMENT", "WHAT IT RULES OUT"],
    )
    all_same = same_as_singles and order_invariant and padding_clean
    print(f"\n  dept={batched['dept']['choice']!r} conf={batched['dept']['confidence']}   "
          f"urg={batched['urg']['score']}   ang={batched['ang']['noul']}   "
          f"({'the same in all four calls above' if all_same else 'DIFFERED between calls -- see above'})")
    print(C.c("  Compared to within one unit of the 4 decimals laya reports. Bit-exact equality across "
              "two\n  different batch widths is a stronger claim than the API's own precision supports; "
              "the\n  AGREEMENT column says which one this run actually achieved.", "dim"))

    # ---------------------------------------------------------------- C
    C.section("C. What batching costs, per question")

    print(f"  protocol: {WARMUP} untimed warmups per width, then {REPS} timed calls per width "
          f"issued in one\n  shuffled interleaved order (seed {SEED}), device synchronised "
          f"immediately before and after each call.")
    print(C.c("  Interleaving matters: timing each width in a contiguous block lets the backend "
              "keep one\n  shape's kernels hot and produces a curve that is an artefact of the "
              "measurement order.", "dim"))
    print()

    samples = time_shapes(agent, SHAPES, device)
    tokens = {n: agent.predict(STATE, mkq(n))["usage"]["input_tokens"] for n in SHAPES}

    rows, curve = [], {}
    for n in SHAPES:
        p50 = statistics.median(samples[n])
        p90 = percentile(samples[n], 0.90)
        per_q = p50 / n
        curve[n] = {"p50_ms": round(p50, 2), "p90_ms": round(p90, 2),
                    "ms_per_question": round(per_q, 3), "input_tokens": tokens[n]}
        rows.append([n, f"{p50:7.1f}", f"{p90:7.1f}", f"{per_q:7.2f}", tokens[n]])

    C.table(rows, ["QUESTIONS", "p50 ms", "p90 ms", "ms PER QUESTION", "input_tokens"],
            align="rrrrr")

    per_q_values = [curve[n]["ms_per_question"] for n in SHAPES]
    monotonic = all(b <= a for a, b in zip(per_q_values, per_q_values[1:]))
    speedup = per_q_values[0] / per_q_values[-1]
    print(f"\n  per-question cost {per_q_values[0]:.1f} ms at N=1 -> "
          f"{per_q_values[-1]:.1f} ms at N={SHAPES[-1]}, a {speedup:.1f}x improvement  "
          f"({'monotonically decreasing' if monotonic else 'NOT monotonic on this run'})")

    # Least-squares fit of p50 against N: intercept is the fixed per-call overhead,
    # slope is the marginal cost of one more question. Both are measured here, every
    # run; neither is asserted, because both move with machine load.
    xs = list(SHAPES)
    ys = [curve[n]["p50_ms"] for n in SHAPES]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
             / sum((x - mean_x) ** 2 for x in xs))
    intercept = mean_y - slope * mean_x
    print(f"  least-squares fit over the six widths, measured on this run: {intercept:.1f} ms fixed "
          f"overhead per call\n  + {slope:.1f} ms per additional question. Both figures move with "
          f"machine load; only the direction\n  of the per-question column is asserted below.")

    # ---------------------------------------------------------------- D
    C.section("D. The counterweight: tokens do NOT amortise")

    usage1, usage20 = result1["usage"], result20["usage"]
    ratio = usage20["input_tokens"] / usage1["input_tokens"]
    mixed_tokens = result_mixed["usage"]["input_tokens"]
    singles_total = sum(single_tokens.values())
    additive = mixed_tokens == singles_total

    C.table(
        [
            ["1 question", usage1["input_tokens"], usage1["output_tokens"], "-"],
            ["20 identical questions", usage20["input_tokens"], usage20["output_tokens"],
             f"{ratio:.1f}x the single-question cost"],
            ["4 different questions, batched", mixed_tokens,
             result_mixed["usage"]["output_tokens"],
             f"= {' + '.join(str(single_tokens[k]) for k in mixed)} = {singles_total} asked separately"],
        ],
        ["predict() CALL", "input_tokens", "output_tokens", "AGAINST ASKING ONE AT A TIME"],
        align="lrrl",
    )
    print(f"\n  Batching {len(mixed)} different questions costs "
          f"{'exactly the same' if additive else 'a different number of'} input tokens as asking them "
          f"one at a time\n  ({mixed_tokens} vs {singles_total}): the shared state is re-encoded once "
          f"per question and nothing is deduplicated.\n  Padding is not billed either -- "
          f"`input_tokens` is the attention mask's sum, so the "
          f"{pad_cells} pad cells\n  in that batch do not appear in the count.")
    print()
    print(C.c("  Mechanism (upstream/laya/agent.py, Agent.system_one): one sequence is built PER "
              "QUESTION,\n  each containing the FULL serialised state, and collate_items stacks "
              "them into one batch. The\n  encoder therefore runs once -- over N copies of your "
              "state. Latency amortises; compute does not.", "dim"))
    print(C.c("  Two more things the response will not tell you: output_tokens is the literal "
              "constant 0 (there\n  is no decoder, the heads emit probabilities), and "
              f"result['model'] is the literal "
              f"{result1['model']!r}\n  in Agent.system_one -- the same string whichever checkpoint "
              "you loaded, so record the subfolder\n  yourself. This run loaded the english "
              "checkpoint.", "dim"))

    # ---------------------------------------------------------------- E
    C.section("E. Applied: the shipped 5-question triage preset on one real ticket")

    import laya

    preset = laya.triage_questions()
    for _ in range(WARMUP):
        agent.predict(TICKET, preset)
    C.sync(device)
    triage_samples = []
    for _ in range(20):
        C.sync(device)
        t0 = time.perf_counter()
        triage = agent.predict(TICKET, preset)
        C.sync(device)
        triage_samples.append((time.perf_counter() - t0) * 1000)

    triage_p50 = statistics.median(triage_samples)
    tickets_per_s = 1000.0 / triage_p50
    decisions_per_s = tickets_per_s * len(preset)

    C.table(
        [[k,
          v["type"],
          v.get("choice", v.get("score", v.get("noul"))),
          v["confidence"],
          C.bar(v["confidence"], width=18)]
         for k, v in triage["answers"].items()],
        ["QUESTION", "TYPE", "ANSWER", "CONF", ""],
    )
    print(f"\n  {len(preset)} structured decisions about one support ticket in "
          f"{triage_p50:.1f} ms (p50 of 20), {triage['usage']['input_tokens']} input tokens")
    print(f"  -> {tickets_per_s:.1f} tickets/second, {decisions_per_s:.1f} decisions/second "
          f"({len(preset)} per ticket) on one {device} process:\n     no GPU cluster, no network call, "
          f"no per-token bill. Single-threaded and un-batched across tickets; this is the floor, "
          f"not the ceiling.")

    # ---------------------------------------------------------------- verdict
    latency_win = curve[20]["ms_per_question"] < curve[1]["ms_per_question"]

    C.save_artifact("04_throughput", {
        "env": C.env_summary(),
        "checkpoint": "english",
        "encoder_calls": {
            "n20": calls20, "n1": calls1, "mixed": calls_mixed,
            "single_widths": single_widths,
        },
        "equivalence": {
            "batched_equals_singles": same_as_singles,
            "batched_equals_singles_exactly": exact_as_singles,
            "order_invariant": order_invariant,
            "order_invariant_exactly": exact_reversed,
            "long_sibling_does_not_perturb": padding_clean,
            "long_sibling_does_not_perturb_exactly": exact_sibling,
            "tolerance": ROUND_UNIT,
        },
        "protocol": {"warmup_per_shape": WARMUP, "timed_reps_per_shape": REPS,
                     "interleaved": True, "seed": SEED, "device": device},
        "latency_curve": curve,
        "per_question_speedup_n1_to_n40": round(speedup, 2),
        "fit": {"fixed_overhead_ms": round(intercept, 2), "marginal_ms_per_question": round(slope, 2)},
        "token_scaling": {
            "n1": usage1, "n20": usage20, "ratio": ratio,
            "mixed_batched": mixed_tokens, "mixed_separately": singles_total,
            "per_question": single_tokens,
        },
        "triage": {
            "questions": len(preset),
            "p50_ms": round(triage_p50, 2),
            "tickets_per_second": round(tickets_per_s, 2),
            "decisions_per_second": round(decisions_per_s, 2),
            "input_tokens": triage["usage"]["input_tokens"],
            "answers": triage["answers"],
        },
        "per_question_monotonic": monotonic,
    })

    checks = [
        ("exactly one encoder call per predict()", one_call_each,
         f"{call_counts} for N=20, N=1, N=4 and the four singles"),
        ("one encoder row per question", rows_match_questions,
         f"{calls20[0]['rows']}, {calls1[0]['rows']}, {calls_mixed[0]['rows']} rows"),
        ("batch padded to the longest row", padded_to_longest,
         f"width {calls_mixed[0]['width']} == longest single {single_widths[widest]}"),
        ("batched answers == per-question answers", same_as_singles,
         "exact" if exact_as_singles else "within 5e-4"),
        ("answers invariant to question order", order_invariant,
         "exact" if exact_reversed else "within 5e-4"),
        ("long sibling leaves the others alone", padding_clean,
         "exact" if exact_sibling else "within 5e-4"),
        ("input_tokens additive over 4 different questions", additive,
         f"{mixed_tokens} batched == {singles_total} separately"),
        ("input_tokens(20) == 20 x input_tokens(1)", ratio == 20.0,
         f"{usage1['input_tokens']} -> {usage20['input_tokens']} ({ratio:.1f}x)"),
        ("per-question p50 at N=20 < at N=1", latency_win,
         f"{curve[20]['ms_per_question']:.1f} ms < {curve[1]['ms_per_question']:.1f} ms"),
    ]

    print()
    C.table([[name, mark(ok), observed] for name, ok, observed in checks],
            ["INVARIANT", "RESULT", "OBSERVED"])

    C.require(
        all(ok for _, ok, _ in checks),
        "Batching is one encoder pass, one row per question and identical answers at lower "
        "per-question latency -- but the input tokens add up row by row.",
    )


if __name__ == "__main__":
    main()
