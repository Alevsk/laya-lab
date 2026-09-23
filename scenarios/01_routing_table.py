"""Laya's routing brain, exercised on 19 inputs without loading a single checkpoint.

`Router.route()` is pure Python. It reads the state, decides which of the three Laya
checkpoints should answer, and returns a `RouteDecision` explaining itself -- all before
any weights exist in memory. That makes it the cheapest way to understand the model
family: this scenario needs no HF cache, no GPU and no download.

How to read the output
----------------------
Section A  One row per input. MODEL is the checkpoint `route()` picked, REASON is the
           string it generated for that pick (truncated), and the four right-hand columns
           are the raw detection signals behind it. Rows marked `*` are the ones whose
           answer is not the one you would guess; `make s5` shows what the checkpoints
           actually do with them.
Section B  `r.loaded` after all the decisions. It is empty: routing constructed no Agent.
Section C  What a decision costs. Routing is nearly free for Latin text and emphatically
           not free for CJK. The table is sorted by the script's position in
           `lang._SCRIPT_RANGES`, which is the mechanism: Latin short-circuits before the
           probe loop, Greek exits on the first entry, Han only on the last -- and every
           character pays that probe twice, because `analyse()` scans the string once for
           `script_profile()` and again for `detect_script()`.

           Two estimators are printed. `min` is the uncontended cost of the call and is
           what every ratio below is computed from; `p50` is the median of the same
           samples and carries whatever else the machine was doing. On a busy laptop p50
           drifts by 3-5x between runs while min barely moves, which is why min -- the
           standard estimator for a pure-Python microbenchmark -- is the one asserted on.

Invariants asserted: route() loads nothing; every non-Latin script and every identified
non-English Latin language goes to `multilingual`; English/emoji/digits/empty/None go to
`english`; routing cost tracks position in `_SCRIPT_RANGES` (Han > 10x Latin and > 3x
Greek at 4000 chars); and the two string scans account for most of a Han decision.
"""
from __future__ import annotations

import inspect
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

from laya import Router, lang  # noqa: E402

REASON_WIDTH = 50

# (label, state, expected model, expected script, surprising?)
# The `True` rows are the interesting failures of intuition, not bugs: Laya's detector is a
# stopword/diacritic heuristic and says so in its own docstring.
CASES = [
    ("english", "I was charged twice for my subscription and I want a refund please",
     "english", "latin", False),
    ("german", "Ich wurde zweimal belastet und moechte eine Rueckerstattung, bitte pruefen Sie das",
     "multilingual", "latin", False),
    ("spanish", "Me cobraron dos veces por la suscripcion y quiero que me devuelvan el dinero",
     "multilingual", "latin", False),
    ("french", "Je ne peux pas me connecter a mon compte depuis ce matin",
     "multilingual", "latin", False),
    ("hindi", "मुझसे दो बार पैसे लिए गए हैं और मुझे रिफंड चाहिए",
     "multilingual", "devanagari", False),
    ("arabic", "لقد تم خصم المبلغ مرتين وأريد استرداد أموالي",
     "multilingual", "arabic", False),
    ("chinese", "客户被重复扣款要求退款",
     "multilingual", "han", False),
    ("japanese", "アカウントから二回引き落とされました",
     "multilingual", "kana", False),
    ("korean", "두 번 청구되었습니다 환불해 주세요",
     "multilingual", "hangul", False),
    ("russian", "С меня дважды списали деньги, я хочу возврат",
     "multilingual", "cyrillic", False),
    ("emoji", "🔥🔥🔥 👍", "english", "unknown", False),
    ("digits", "12345 6789", "english", "unknown", False),
    ("empty", "", "english", "unknown", False),
    ("none", None, "english", "unknown", False),
    ("dict state", {"subject": "Rechnung falsch",
                    "body": "Ich wurde zweimal belastet und moechte eine Rueckerstattung bitte"},
     "multilingual", "latin", False),
    ("swahili", "Mteja alitozwa mara mbili na anataka kurudishiwa pesa yake sasa",
     "english", "latin", True),
    ("mixed", "Hello I need help with my account " + "क" * 20, "english", "latin", True),
    ("short_es", "quiero reembolso", "english", "latin", True),
    ("accented_en", "Please contact José García and Müller about the refund on the invoice 4411 today",
     "multilingual", "latin", True),
]

# Scripts the English checkpoint physically cannot tokenise; all must reach `multilingual`.
NON_LATIN = ("devanagari", "arabic", "han", "kana", "hangul", "cyrillic")

# Position (1-based) of each script in lang._SCRIPT_RANGES. Latin is absent on purpose: it is
# short-circuited by a codepoint test before the probe loop is ever entered.
SCRIPT_POSITION = {name: i + 1 for i, (name, _) in enumerate(lang._SCRIPT_RANGES)}


def repeat_to(seed: str, chars: int = 4000) -> str:
    """A state whose detection text is exactly `chars` long (state_text truncates at 4000)."""
    return seed * (chars // len(seed) + 1)


# Single-script 4000-char states, chosen to sit at opposite ends of _SCRIPT_RANGES:
# latin never enters the probe loop, greek exits on entry 1, hangul on 23, han on 25.
EN_4000 = repeat_to("the quick brown fox jumps over the lazy dog ")
EL_4000 = repeat_to("η υπηρεσια δεν λειτουργει σωστα σημερα ")
RU_4000 = repeat_to("с меня дважды списали деньги сегодня утром ")
KO_4000 = repeat_to("두 번 청구되었습니다 환불해 주세요 오늘 ")
ZH_4000 = repeat_to("客户被重复扣款要求退款")

# (label, state, repetitions) -- cheap calls get more samples, expensive ones fewer.
LATENCY_CASES = [
    ("digits only", "12345 6789", 2000),
    ("English ticket", CASES[0][1], 2000),
    ("Chinese ticket", CASES[6][1], 1000),
    ("English, 4000 chars", EN_4000, 300),
    ("Greek, 4000 chars", EL_4000, 150),
    ("Cyrillic, 4000 chars", RU_4000, 150),
    ("Korean, 4000 chars", KO_4000, 50),
    ("Chinese, 4000 chars", ZH_4000, 50),
]


def truncate(text: str, width: int = REASON_WIDTH) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def summarise(samples_us: list[float]) -> dict[str, float]:
    """{min, p50, p99} of microsecond samples.

    `min` is the estimator every ratio in this scenario uses: these calls are pure Python
    and CPU-bound, so the fastest sample is the one least polluted by the scheduler.
    """
    ordered = sorted(samples_us)
    idx = min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))
    return {"min": ordered[0], "p50": statistics.median(ordered), "p99": ordered[idx]}


def bench(fn, arg, reps: int) -> dict[str, float]:
    """Time a pure-Python callable. No device sync needed: nothing here touches torch."""
    fn(arg)  # warm the interpreter's caches on this input shape
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(arg)
        samples.append((time.perf_counter() - t0) * 1e6)
    return summarise(samples)


def bench_interleaved(labelled_calls, reps: int) -> dict[str, dict[str, float]]:
    """Time several no-arg callables round-robin so they share the same machine conditions.

    Attribution is a ratio between separate measurements, and on a contended machine two
    timings taken minutes apart can differ by 3-5x -- enough to make a share look like 67%
    or 120% purely from drift. Sampling them in one interleaved loop removes that drift.
    """
    for _, fn in labelled_calls:
        fn()
    samples = {label: [] for label, _ in labelled_calls}
    for _ in range(reps):
        for label, fn in labelled_calls:
            t0 = time.perf_counter()
            fn()
            samples[label].append((time.perf_counter() - t0) * 1e6)
    return {label: summarise(vals) for label, vals in samples.items()}


def analyse_call_sites() -> dict[str, int]:
    """Line numbers of the script_profile/detect_script calls inside `lang.analyse`.

    Read out of the installed source rather than hardcoded, so the mechanism paragraph
    below cannot quote a line number that upstream has since moved.
    """
    lines, first = inspect.getsourcelines(lang.analyse)
    found = {}
    for offset, line in enumerate(lines):
        for name in ("script_profile", "detect_script"):
            if f"= {name}(text)" in line:
                found[name] = first + offset
    return found


def main() -> None:
    C.header(f"Laya routing table — {len(CASES)} inputs, zero models loaded",
             "Router.route() is pure Python: no download, no weights, a reason string every time.")

    router = Router()

    # ---------------------------------------------------------------- Section A
    C.section("A. Every decision, and the detection signals behind it")
    rows, decisions, wrong = [], {}, []
    for label, state, want_model, want_script, surprising in CASES:
        d = router.route(state)
        det = d.get("detection") or {}
        decisions[label] = {
            "model": d.model, "reason": d.reason, "expected_model": want_model,
            "script": det.get("script"), "language": det.get("language"),
            "is_english": det.get("is_english"),
            "non_latin_fraction": det.get("non_latin_fraction"),
            "diacritic_rate": det.get("diacritic_rate"),
            "surprising": surprising,
        }
        if d.model != want_model or det.get("script") != want_script:
            wrong.append(f"{label}: got {d.model}/{det.get('script')}, "
                         f"expected {want_model}/{want_script}")
        rows.append([
            f"{label} *" if surprising else label,
            d.model,
            truncate(d.reason),
            det.get("script"),
            det.get("language") or "-",
            f"{det.get('non_latin_fraction', 0.0):.4f}",
            f"{det.get('diacritic_rate', 0.0):.4f}",
        ])

    C.table(rows, ["INPUT", "MODEL", "REASON", "script", "lang", "non-latin", "diacr"],
            align="lllllrr")
    surprising_n = sum(1 for case in CASES if case[4])
    print(f"\n  {C.c('*', 'yellow')} the {surprising_n} surprising ones — `make s5` shows what the "
          "checkpoints actually do with these.")

    # ---------------------------------------------------------------- Section B
    C.section("B. What all of that cost in memory")
    print(f"  r.loaded == {router.loaded}  (no checkpoint was constructed)")
    nothing_loaded = router.loaded == []

    # ---------------------------------------------------------------- Section C
    C.section("C. What a decision costs in time")
    latency = {}
    route_calls = len(CASES)
    for label, state, reps in LATENCY_CASES:
        text = lang.state_text(state)
        stats = bench(router.route, state, reps)
        route_calls += reps + 1  # +1 for the warm-up call
        latency[label] = {"chars": len(text), "reps": reps,
                          "script": lang.detect_script(text),
                          "ranges_pos": SCRIPT_POSITION.get(lang.detect_script(text)),
                          **{k: round(v, 1) for k, v in stats.items()}}

    en4000 = latency["English, 4000 chars"]["min"]
    el4000 = latency["Greek, 4000 chars"]["min"]
    zh4000 = latency["Chinese, 4000 chars"]["min"]
    C.table(
        [[label, latency[label]["chars"], latency[label]["script"],
          f"#{latency[label]['ranges_pos']}" if latency[label]["ranges_pos"] else "—",
          f"{latency[label]['min']:,.1f}", f"{latency[label]['p50']:,.1f}",
          f"{latency[label]['min'] / en4000:.3f}x"]
         for label, _, _ in LATENCY_CASES],
        ["STATE", "CHARS", "script", "_SCRIPT_RANGES", "min (us)", "p50 (us)", "vs English-4000ch"],
        align="lrlrrrr")
    print("\n  CHARS is what detection sees: lang.state_text() truncates every state at 4000.")
    print("  _SCRIPT_RANGES is how deep into the probe list that script sits; Latin (—) never "
          "enters it.")
    print("  Ratios use min, the uncontended cost; p50 is shown so you can see the machine's noise.")

    # Where the Han cost goes. Measured round-robin against route() itself, on the same state.
    big = lang.state_text(ZH_4000)
    attribution = bench_interleaved(
        [("route", lambda: router.route(big)),
         ("script_profile", lambda: lang.script_profile(big)),
         ("detect_script", lambda: lang.detect_script(big))], 30)
    route_calls += 30 + 1
    route_us = attribution["route"]["min"]
    prof_us = attribution["script_profile"]["min"]
    script_us = attribution["detect_script"]["min"]
    scan_share = 100 * (prof_us + script_us) / route_us

    print(f"\n  Where a 4000-char Han decision goes (round-robin samples, min of {30}):")
    print(f"    lang.script_profile(zh)  {prof_us / 1000:7.2f} ms   "
          f"{100 * prof_us / route_us:4.0f}% of route()")
    print(f"    lang.detect_script(zh)   {script_us / 1000:7.2f} ms   "
          f"{100 * script_us / route_us:4.0f}% of route()")
    print(f"    {C.c('the two scans together', 'bold')}   {(prof_us + script_us) / 1000:7.2f} ms   "
          f"{scan_share:4.0f}% of route() ({route_us / 1000:.2f} ms)")
    print("    route() is state_text() + these two scans + a few dict operations, so a share at\n"
          "    (or a little over) 100% is what the mechanism below predicts; the excess is noise.")

    sites = analyse_call_sites()
    last_script, n_ranges = lang._SCRIPT_RANGES[-1][0], len(lang._SCRIPT_RANGES)
    mechanism_ok = (set(sites) == {"script_profile", "detect_script"}
                    and abs(sites["script_profile"] - sites["detect_script"]) == 1
                    and last_script == "han")
    site_line = "-".join(str(sites[k]) for k in sorted(sites, key=sites.get)) if sites else "?"
    print(f"\n  {C.c('Mechanism:', 'bold')} analyse() at lang.py:{site_line} calls BOTH functions, "
          f"each doing an independent\n  full-string scan, and {last_script!r} is entry "
          f"{SCRIPT_POSITION[last_script]} of {n_ranges} in _SCRIPT_RANGES — so every CJK\n  "
          f"character pays the full linear probe twice. The table above is that list order: "
          f"Greek\n  (#{SCRIPT_POSITION['greek']}) costs "
          f"{el4000 / en4000:.0f}x Latin, Han (#{SCRIPT_POSITION['han']}) costs "
          f"{zh4000 / en4000:.0f}x Latin and {zh4000 / el4000:.0f}x Greek.")

    # ---------------------------------------------------------------- checks
    C.section("Checks")
    checks = [
        ("route() constructed no Agent (r.loaded == [])", nothing_loaded, False),
        (f"all {len(CASES)} inputs routed to the expected model and script", not wrong, False),
        (f"non-Latin scripts ({', '.join(NON_LATIN)}) -> multilingual",
         all(decisions[k]["model"] == "multilingual" for k in
             ("hindi", "arabic", "chinese", "japanese", "korean", "russian")), True),
        ("German / Spanish / French -> multilingual",
         all(decisions[k]["model"] == "multilingual" for k in ("german", "spanish", "french")), True),
        ("English / emoji / digits / empty / None -> english",
         all(decisions[k]["model"] == "english" for k in
             ("english", "emoji", "digits", "empty", "none")), True),
        (f"Han-4000 costs >10x Latin-4000 (measured {zh4000 / en4000:.0f}x)",
         zh4000 > 10 * en4000, False),
        (f"Latin short-circuits the probe: >1.5x cheaper than Greek(#{SCRIPT_POSITION['greek']}) "
         f"(measured {el4000 / en4000:.1f}x)", el4000 > 1.5 * en4000, False),
        (f"cost then grows with probe depth: Han(#{SCRIPT_POSITION['han']}) >3x Greek"
         f"(#{SCRIPT_POSITION['greek']}) (measured {zh4000 / el4000:.1f}x)",
         zh4000 > 3 * el4000, False),
        (f"the two scans are >=60% of a Han decision (measured {scan_share:.0f}%)",
         scan_share >= 60, False),
        (f"source says so too: analyse() calls both scans on adjacent lines, "
         f"{last_script!r} is last of {n_ranges}", mechanism_ok, False),
        (f"still nothing loaded after {route_calls:,} route() calls", router.loaded == [], False),
    ]
    for i, (name, ok, detail) in enumerate(checks):
        mark = C.c('ok  ', 'green') if ok else C.c('FAIL', 'red')
        print(f"  {'    ' if detail else ''}{mark}  {name}")
        if detail and not (i + 1 < len(checks) and checks[i + 1][2]):
            print("  " + C.c("    (indented lines are the readable breakdown of the check above "
                             "them; they cannot fail on their own)", "dim"))
    for problem in wrong:
        print(f"  {C.c('  ->', 'red')} {problem}")

    print(f"\n  {C.c('VERDICT:', 'bold')} Routing is near-free for Latin text "
          f"({latency['English ticket']['min']:.0f} us for a real ticket)\n"
          f"  and expensive for CJK ({zh4000 / 1000:.0f} ms for 4000 chars). Trim the state or "
          f"pass lang= if you serve CJK at volume.")

    C.save_artifact("01_routing_table", {
        "env": {**C.env_summary(), "note": "this scenario never touches torch; device is unused"},
        "decisions": decisions,
        "loaded_after_routing": router.loaded,
        "route_calls": route_calls,
        "estimator": "min of N samples (ratios); p50/p99 reported for noise",
        "latency_us": latency,
        "attribution_us": {k: {m: round(v, 1) for m, v in stats.items()}
                           for k, stats in attribution.items()},
        "scan_share_pct": round(scan_share, 1),
        "ratios": {"han_vs_latin_4000": round(zh4000 / en4000, 1),
                   "greek_vs_latin_4000": round(el4000 / en4000, 1),
                   "han_vs_greek_4000": round(zh4000 / el4000, 1)},
        "script_ranges": {"count": n_ranges, "last": last_script,
                          "analyse_call_lines": sites},
        "checks": {name: ok for name, ok, _ in checks},
    })

    C.require(all(ok for _, ok, _ in checks),
              "route() loads nothing, sends every non-Latin and non-English state to "
              "multilingual,\n         and costs >10x more on Han than on Latin at 4000 chars.")


if __name__ == "__main__":
    main()
