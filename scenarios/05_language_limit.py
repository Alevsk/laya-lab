"""Where Laya's router is blind, and what a mis-route costs.

Laya routes between checkpoints with a dependency-free detector: exact script
detection, plus a stopword/diacritic heuristic for Latin-script text. The
stopword table covers eight languages. Everything else written in the Latin
alphabet without accents is, to this detector, indistinguishable from English --
so Swahili, Indonesian, Malay and Somali tickets are handed to the English
checkpoint, silently and with a confident-sounding reason string.

The scenario enumerates the blind spots first (Sections A-C, pure Python, no
model), then prices one of them on real inferences (Section D), then discloses
the calibration asymmetry that keeps that price honest (Section E).

How to read the output
----------------------
A  Six Latin-script tickets no stopword list covers. `language` is what
   `laya.lang.analyse` decided: None means "undecided", 'en' means "actively
   labelled English". Both end up at the English checkpoint.
B  English text with N Devanagari characters appended. Watch `script` flip only
   once the non-Latin share passes 50%: `detect_script` is a plurality vote.
C  Four edges where the routing decision turns on something that has nothing to
   do with the language of the text.
D  The cost. Same ticket, same question, both checkpoints. '>' marks the cell
   the router actually chose. Read the `conf` column, not `choice`.
E  The counter-caveat: laya-multilingual ships no fitted temperatures, and on
   every ticket measured here -- the English control included -- it is the
   sharper of the two. Section D therefore prices the English checkpoint's
   collapse on Swahili; it is not evidence that multilingual is the better model.

Invariant: sixteen checks. Routing facts (A-C), calibration facts (E), a >= 0.30
confidence collapse on the English checkpoint between the English and Swahili
tickets, and three CONTROL checks whose job is to make the others falsifiable --
if the detector simply answered "english" to everything, or if no checkpoint
shipped fitted temperatures, the controls would fail and the story would be wrong.
"""
from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

# One question, five options. Wide enough that the uniform prior is a visible
# 0.20, narrow enough to land in a fitted temperature bucket on the English
# checkpoint (choice:3-5).
QUESTION = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
            "refund": "money back for a charge",
            "cancel": "end the subscription",
            "technical_help": "something is broken",
            "address_change": "update delivery address",
            "complaint": "unhappy but no specific ask",
        },
    }
}

# Six ways of writing "I was charged twice, I want my money back". None of these
# six languages has a stopword list in laya.lang.
UNDETECTABLE = [
    ("Swahili", "Mteja alitozwa mara mbili na anataka kurudishiwa pesa yake sasa"),
    ("Indonesian", "Saya ditagih dua kali dan saya ingin uang saya dikembalikan sekarang"),
    ("Malay", "Saya dicaj dua kali dan saya mahu wang saya dikembalikan sekarang"),
    ("Somali", "Macmiilka waxaa laga qaaday laba jeer wuxuuna rabaa lacagtiisa"),
    ("Turkish (folded)", "Hesabimdan iki kez para cekildi ve iadesini istiyorum lutfen"),
    ("Tagalog", "Nasingil ako ng dalawang beses at gusto ko ang pera ko ngayon"),
]

# The same Turkish sentence with its diacritics intact. Only the accents differ.
TURKISH_ACCENTED = "Hesabımdan iki kez para çekildi ve iadesini istiyorum lütfen"

SWAHILI = UNDETECTABLE[0][1]
INDONESIAN = UNDETECTABLE[1][1]
ENGLISH_CTL = "I was charged twice and I want my money back please"
GERMAN_CTL = "Ich wurde zweimal belastet und moechte mein Geld zurueck bitte"

MIXED_BASE = "Hello I need help with my account "
DEVANAGARI = "क"

ACCENTED_EN = "Please contact José García and Müller about the refund on the invoice 4411 today"
FOLDED_EN = "Please contact Jose Garcia and Muller about the refund on the invoice 4411 today"
SHORT_ES = "quiero reembolso"

# Each edge case, plus whether its routing is the wrong one for the text.
EDGES = [
    ("accented English (real names)", ACCENTED_EN, {}, True),
    ("the same line, ASCII-folded", FOLDED_EN, {}, False),
    ("Spanish, 2 words", SHORT_ES, {}, True),
    ("English, lang='en_GB'", "hello there", {"lang": "en_GB"}, True),
    ("English, lang='en-US'", "hello there", {"lang": "en-US"}, False),
]

# (label, text, the checkpoint that can actually read it)
TICKETS = [
    ("Swahili", SWAHILI, "multilingual"),
    ("Indonesian", INDONESIAN, "multilingual"),
    ("English (control)", ENGLISH_CTL, "english"),
    ("German (control)", GERMAN_CTL, "multilingual"),
]


def clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main() -> None:
    C.require_cached("english", "multilingual")

    from laya import Router
    from laya.lang import (_NON_EN_DIACRITICS, _STOP, _WORD, NON_EN_DIACRITIC_RATE, analyse,
                           latin_profile)
    import laya.lang as lang_mod

    C.header(
        "05 - The limit: where the router is blind, and what a mis-route costs",
        "Laya's language detector knows eight languages. This is the bill for the ninth.",
    )

    artifact: dict = {"env": C.env_summary(), "question": QUESTION}

    # A bare Router() is enough for every routing decision below: route() only
    # calls analyse(), it never touches a checkpoint.
    probe = Router()

    # ---------------------------------------------------------------- Section A
    C.section("A. Six Latin-script tickets the detector cannot tell from English")

    rows, section_a = [], []
    for name, text in UNDETECTABLE:
        det = analyse(text)
        decision = probe.route(text)
        model = decision["model"]
        rows.append([clip(text, 44), name, str(det["language"]),
                     f"{det['diacritic_rate']:.4f}", str(det["is_english"]),
                     C.c(model, "red")])
        section_a.append({"language": name, "text": text,
                          "detected_language": det["language"],
                          "diacritic_rate": det["diacritic_rate"],
                          "is_english": det["is_english"],
                          "routed_to": model, "reason": decision["reason"]})
    C.table(rows, ["TEXT (44 chars)", "ACTUALLY", "language", "diacritic", "is_english",
                   "ROUTED TO"])

    reasons_a = sorted({r["reason"] for r in section_a})
    all_one_reason = len(reasons_a) == 1
    # Whichever row analyse() actively labelled 'en' -- found by the label, not hard-coded,
    # so the sentence below always describes the row the table actually shows.
    labelled_en = next((r for r in section_a if r["detected_language"] == "en"), None)

    print()
    print("  Columns 3-5 are analyse() fields: ['language'], ['diacritic_rate'], ['is_english'].")
    if all_one_reason:
        print(f"  All six route with reason={reasons_a[0]!r}. None of them is English.")
    else:
        print(f"  Reasons observed across the six: {reasons_a}. None of them is English.")
    print("  'None' means undecided: no stopword list scored and no accents to argue otherwise.")
    if labelled_en is not None:
        hit_words = sorted({w.lower() for w in _WORD.findall(labelled_en["text"])} & _STOP["en"])
        hit_count = int(latin_profile(labelled_en["text"])["english_hits"])
        print("  'en' is worse: it is a positive claim. It needs >=1 English function word and a")
        print("  diacritic rate under the threshold (lang.py:241) -- no margin, no second hit.")
        print(f"  The {labelled_en['language']} line is labelled English on english_hits={hit_count}, "
              f"the word(s) {hit_words}.")
    else:
        print("  No row was actively labelled 'en' on this run; all six were merely undecided.")

    stop_sizes = {k: len(v) for k, v in sorted(_STOP.items())}
    print()
    print("  The whole ceiling, laya.lang._STOP:")
    print("    " + "  ".join(f"{k}={n}" for k, n in stop_sizes.items()))
    print(f"  Eight languages. Everything else is caught only by a {len(_NON_EN_DIACRITICS)}-character")
    print(f"  diacritic set at a {NON_EN_DIACRITIC_RATE:.0%} rate threshold (lang.py:199) -- which tests")
    print("  for European accents, not for being non-English.")

    turkish_det = analyse(TURKISH_ACCENTED)
    turkish_model = probe.route(TURKISH_ACCENTED)["model"]
    print()
    print("  Row 5 is Turkish with its accents stripped, as a ticketing system would store it.")
    print(f"  The same sentence WITH diacritics scores {turkish_det['diacritic_rate']:.4f} and routes to "
          f"{C.c(turkish_model, 'green')}.")
    print("  The accents are what saved it, not the language. Any pipeline that folds to ASCII")
    print("  deletes the only evidence the detector had.")

    # Quote the upstream docstring rather than transcribing it, so this claim cannot
    # silently rot if upstream edits the file.
    # Split on sentence ends only (period + whitespace), so the decimals survive.
    doc = " ".join((lang_mod.__doc__ or "").split())
    sentences = re.split(r"(?<=\.)\s+", doc)
    swahili_quote = next((s.strip() for s in sentences if "Swahili" in s), "")
    print()
    print("  laya.lang's own module docstring names this exact case. Quoted from the installed")
    print("  source, not from any README:")
    for line in _wrap(swahili_quote, 88):
        print(f"    {C.c(line, 'dim')}")
    print("  Those are upstream's MASSIVE numbers on a T4, not measured here -- what is measured")
    print("  here is Section D. The point is that the router walks into a failure its own source")
    print("  file documents, because the benchmark and the detector were never wired together.")

    # ---------------------------------------------------------------- Section B
    C.section("B. Mixed script: the vote is a plurality, so 49% non-Latin is still 'latin'")

    rows, section_b = [], []
    for n in (5, 10, 20, 30, 40):
        text = MIXED_BASE + DEVANAGARI * n
        det = analyse(text)
        model = probe.route(text)["model"]
        rows.append([f"{n} chars", f"{det['non_latin_fraction']:.4f}", det["script"],
                     C.c(model, "red" if model == "english" else "green"),
                     C.bar(float(det["non_latin_fraction"]), width=20)])
        section_b.append({"devanagari_chars": n,
                          "non_latin_fraction": det["non_latin_fraction"],
                          "script": det["script"], "routed_to": model})
    C.table(rows, ["DEVANAGARI CHARS ADDED", "non_latin_fraction", "script", "ROUTED TO",
                   "non-Latin share"], align="rr")

    english_rows = [r for r in section_b if r["routed_to"] == "english"]
    worst = max(english_rows, key=lambda r: r["non_latin_fraction"], default=None)
    print()
    print("  The flip is at 50%, not at 'there is Devanagari in here': detect_script is a plurality")
    print("  vote over character counts (lang.py:174).")
    if worst is not None:
        print(f"  The worst row still sent to the English checkpoint is {worst['devanagari_chars']} "
              f"appended characters:")
        print(f"  {worst['non_latin_fraction']:.2%} of its letters are Devanagari, and the English "
              "checkpoint's 50k English BPE")
        print("  tokenizer cannot read a single one of them.")
    else:
        print("  On this run every row crossed the 50% line, so none stayed on the English")
        print("  checkpoint -- the base sentence must have changed.")
    print()
    print("  The information IS in the payload -- decision['detection']['non_latin_fraction'] carries")
    print("  the exact figure. route() just never reads it: router.py:295 tests only")
    print("  det['script'] != 'latin'. The workaround is to read that field and override yourself.")

    # ---------------------------------------------------------------- Section C
    C.section("C. Four edges that turn on something other than the language (last one is a pair)")

    rows, section_c, reasons = [], [], []
    for label, text, kwargs, wrong in EDGES:
        decision = probe.route(text, **kwargs)
        det = analyse(text)
        model = decision["model"]
        rows.append([label, clip(text, 36), f"{det['diacritic_rate']:.4f}",
                     C.c(model, "red" if wrong else "green")])
        reasons.append((label, decision["reason"]))
        section_c.append({"case": label, "text": text, "kwargs": kwargs, "routed_to": model,
                          "reason": decision["reason"],
                          "diacritic_rate": det["diacritic_rate"],
                          "wrong_for_this_text": wrong})
    C.table(rows, ["CASE", "TEXT", "diacritic rate", "ROUTED TO"])

    print("\n  Reason strings, verbatim:")
    for label, reason in reasons:
        print(f"    {label:<30s} {clip(reason, 70)}")

    print()
    print("  Rows 1-2: two European surnames push ordinary English over the 2% threshold and send it")
    print("  to the multilingual checkpoint; folding the accents sends the identical sentence back.")
    print("  Row 3: 'quiero reembolso' is Spanish, but latin_profile short-circuits at")
    print("  len(words) < 4 (lang.py:218) before scoring a single stopword, so it routes to English.")
    print("  Rows 4-5: 'en_GB' is a tag every ticketing system emits. router.py:287 splits it on '-'")
    print("  only, so 'en_gb' is not in ('en','eng','english') and an English ticket is sent to the")
    print("  multilingual checkpoint by an underscore.")

    # ---------------------------------------------------------------- Section D
    C.section("D. The bill: same ticket, same question, both checkpoints")
    print("  Loading both checkpoints (the only slow step in this scenario) ...")
    t0 = time.perf_counter()
    router = C.load_router(["english", "multilingual"], max_loaded=3)
    # Weight transfers to MPS/CUDA are queued asynchronously; without this the preload
    # figure below would be a dispatch time, not a load time.
    C.sync()
    preload_s = time.perf_counter() - t0

    # A pure-Python dict hit: no device work is queued, so no sync is needed inside the
    # loop. The sync above guarantees nothing is left pending from the preload either.
    warm_calls = 2000
    t_warm = time.perf_counter()
    for _ in range(warm_calls):
        router.load("english")
    warm_us = (time.perf_counter() - t_warm) / warm_calls * 1e6

    rows, section_d, answers = [], [], {}
    for label, text, readable_by in TICKETS:
        chosen = router.route(text)["model"]
        cells = {}
        for model in ("english", "multilingual"):
            ans = router.load(model).system_one(text, QUESTION)["answers"]["intent"]
            cells[model] = {"choice": ans["choice"],
                            "top_p": max(ans["probabilities"].values()),
                            "confidence": ans["confidence"]}

        def cell(model: str, cells=cells, chosen=chosen) -> str:
            v = cells[model]
            body = f"{v['choice']:<7s} p={v['top_p']:.3f}  conf={v['confidence']:.3f}"
            if model != chosen:
                return C.c("  " + body, "dim")
            return C.c("> " + body, "red" if v["confidence"] < 0.5 else "green")

        misrouted = chosen != readable_by
        # The '!' is deliberate: colour disappears under a pipe or NO_COLOR, and this
        # column is the whole point of the table.
        mark = "! " if misrouted else "  "
        rows.append([label, C.c(mark + chosen, "red" if misrouted else "cyan"),
                     cell("english"), cell("multilingual")])
        answers[label] = cells
        section_d.append({"ticket": label, "text": text, "routed_to": chosen,
                          "readable_by": readable_by, "misrouted": misrouted, **cells})

    C.table(rows, ["TICKET", "ROUTED TO", "english:  choice / top p / conf",
                   "multilingual:  choice / top p / conf"])

    sw = answers["Swahili"]["english"]
    en_conf = answers["English (control)"]["english"]["confidence"]
    collapse = en_conf - sw["confidence"]

    print("\n  '>' marks the cell the router actually chose; '!' in ROUTED TO means it chose the")
    print("  checkpoint that cannot read the ticket. The two control rows are there so the table")
    print("  cannot be explained by 'the router always says english': German, ASCII-folded and")
    print("  in the stopword table, goes to multilingual; English goes to english.")
    print(f"  English checkpoint on the English ticket: confidence {en_conf:.3f}")
    print(f"  English checkpoint on the Swahili ticket: confidence {sw['confidence']:.3f}   "
          + C.c(f"collapse {collapse:.3f}", "red"))
    print(f"  Its top probability there is {sw['top_p']:.3f} against a 0.200 uniform prior over five")
    print("  options. The model is barely off the fence, and nothing in the response says so")
    print("  except that number.")

    # ---------------------------------------------------------------- Section E
    C.section("E. Calibration disclosure: why 'multilingual is more confident' is NOT the claim")

    ml_agent = router.load("multilingual")
    calib = {}
    for name in ("english", "multilingual"):
        agent = router.load(name)
        buckets = {k: float(v) for k, v in agent.temperature_by_options.items()}
        # temperature_raw is what the checkpoint's rl_agent_config.json literally ships;
        # .temperature is that after laya's [0.5, 5] clamp. The "ships nothing" claim is
        # about the raw value, so read the raw value.
        calib[name] = {"temperature": [round(float(t), 4) for t in agent.temperature],
                       "temperature_raw": [round(float(t), 4) for t in agent.temperature_raw],
                       "temperature_by_options": buckets,
                       "temperature_by_options_raw": {k: float(v) for k, v
                                                      in agent.temperature_by_options_raw.items()}}
        print(f"\n  laya-{name}")
        print(f"    .temperature             {calib[name]['temperature']}"
              f"   (shipped: {calib[name]['temperature_raw']})")
        if buckets:
            print(f"    .temperature_by_options  {len(buckets)} fitted buckets")
            for key, value in buckets.items():
                print(f"                               {key:<12s} {value:.4f}")
        else:
            print("    .temperature_by_options  {}"
                  + C.c("   <- nothing fitted, at all", "yellow"))

    ml_en_conf = answers["English (control)"]["multilingual"]["confidence"]
    sharper = [label for label, cells in answers.items()
               if cells["multilingual"]["confidence"] > cells["english"]["confidence"]]
    paragraphs = [
        "laya-multilingual ships no fitted temperature calibration: an all-1.0 vector and an "
        "empty bucket map, read above straight off its rl_agent_config.json. The 'shipped' "
        "column is the pre-clamp value, so this is not an artefact of laya's own clamping.",

        f"An uncalibrated softmax is systematically sharper, and on {len(sharper)} of the "
        f"{len(answers)} tickets measured here laya-multilingual is indeed the sharper of the two "
        f"-- including the English control, where it reports {ml_en_conf:.3f} against english's "
        f"fitted {en_conf:.3f} on a ticket english reads perfectly well. Four tickets is not a "
        "calibration study; it is enough to show that Section D is not a like-for-like comparison.",

        "english's own numbers are not spotless either: laya clamps its choice:11+ bucket into "
        "[0.5, 5] at load time and warns on stderr about it. That warning is the honest kind.",

        "So the defensible reading of Section D is NOT 'multilingual is the better model'. It is: "
        "the English checkpoint, which the router chose unprompted, sits near the uniform prior "
        "on text it was never meant to read.",
    ]
    for para in paragraphs:
        print()
        for line in _wrap(para, 94):
            print(f"  {line}")

    # Every ticket says "I was charged twice, I want my money back", so 'refund' is the
    # right answer for all of them; report what actually came back rather than asserting it.
    all_calls = [v["choice"] for cells in answers.values() for v in cells.values()]
    n_calls = len(all_calls)
    n_refund = sum(1 for ch in all_calls if ch == "refund")
    print()
    if n_refund == n_calls:
        print(f"  And note what did NOT break: all {n_calls} calls returned 'refund', which is the")
        print("  right answer for all four tickets. The router did not produce a wrong answer here --")
        print("  it produced an unusable confidence, which is worse in any pipeline that thresholds")
        print("  on one.")
    else:
        print(f"  {n_refund} of {n_calls} calls returned 'refund' (the right answer); the others returned")
        print(f"  {sorted(set(all_calls) - {'refund'})}. So here the mis-route cost an answer, not only")
        print("  a confidence.")

    C.section("Router lifecycle, since this is the scenario that touches it")
    print("  Router() defaults to max_loaded=1, so a stream alternating English and non-English")
    print("  tickets rebuilds a 400M-parameter checkpoint on every single request. preload() fixes")
    print(f"  it: both checkpoints came up in {preload_s:.1f}s here, and a warm load() is then a plain")
    print(f"  dictionary lookup, measured at {warm_us:.2f} us over {warm_calls} calls.")

    # ---------------------------------------------------------------- verdict
    swahili_det = analyse(SWAHILI)
    en_agent = router.load("english")
    ml_temp = [float(t) for t in ml_agent.temperature]
    ml_temp_raw = [float(t) for t in ml_agent.temperature_raw]
    checks = [
        ("Swahili ticket routes to 'english'", router.route(SWAHILI)["model"] == "english"),
        ("Indonesian ticket routes to 'english'", router.route(INDONESIAN)["model"] == "english"),
        ("all 6 undetectable tickets give reason='English Latin text'",
         reasons_a == ["English Latin text"]),
        ("CONTROL: ASCII-folded German routes to 'multilingual'",
         router.route(GERMAN_CTL)["model"] == "multilingual"),
        ("CONTROL: the English ticket routes to 'english'",
         router.route(ENGLISH_CTL)["model"] == "english"),
        ("analyse(swahili)['language'] is None", swahili_det["language"] is None),
        ("analyse(swahili)['is_english'] is True", swahili_det["is_english"] is True),
        ("lang='en_GB' routes to 'multilingual'",
         router.route("hello there", lang="en_GB")["model"] == "multilingual"),
        ("lang='en-US' routes to 'english'",
         router.route("hello there", lang="en-US")["model"] == "english"),
        ("mixed script: 20 Devanagari chars -> 'english'",
         router.route(MIXED_BASE + DEVANAGARI * 20)["model"] == "english"),
        ("mixed script: 30 Devanagari chars -> 'multilingual'",
         router.route(MIXED_BASE + DEVANAGARI * 30)["model"] == "multilingual"),
        ("multilingual .temperature_by_options is empty", not ml_agent.temperature_by_options),
        # The length guard matters: `all()` over an empty vector is vacuously true, which
        # would make this check pass on a checkpoint that ships no temperature field at all.
        (f"multilingual ships {len(ml_temp_raw)} temperatures, all 1.0 pre-clamp",
         len(ml_temp_raw) == 3 and all(abs(t - 1.0) < 1e-6 for t in ml_temp_raw)
         and ml_temp == ml_temp_raw),
        # Non-vacuity control for the two checks above: an empty bucket map only means
        # something if a fitted one is what the other checkpoint has.
        (f"CONTROL: english HAS fitted buckets ({len(en_agent.temperature_by_options)})",
         len(en_agent.temperature_by_options) > 0),
        (f"english conf collapses >= 0.30 on Swahili (measured {collapse:.3f})", collapse >= 0.30),
        # Guards the sentence "barely off the fence": 5 options, uniform prior 0.200.
        (f"english top p on Swahili < 0.60 (measured {sw['top_p']:.3f}, prior 0.200)",
         sw["top_p"] < 0.60),
    ]

    C.section("VERDICT")
    C.table([[C.c("ok", "green") if ok else C.c("FAILED", "red"), name] for name, ok in checks],
            ["", "CHECK"])

    artifact.update({
        "section_a_undetectable": section_a,
        "stop_word_lists": stop_sizes,
        "diacritic_set_size": len(_NON_EN_DIACRITICS),
        "diacritic_rate_threshold": NON_EN_DIACRITIC_RATE,
        "turkish_with_diacritics": {"text": TURKISH_ACCENTED, "routed_to": turkish_model,
                                    "diacritic_rate": turkish_det["diacritic_rate"]},
        "section_b_mixed_script": section_b,
        "section_c_edges": section_c,
        "section_d_cost": section_d,
        "section_e_calibration": calib,
        "confidence_collapse_english_checkpoint": {
            "english_ticket": en_conf,
            "swahili_ticket": sw["confidence"],
            "delta": collapse,
        },
        "section_a_reasons": reasons_a,
        "actively_labelled_english": (
            None if labelled_en is None
            else {"language": labelled_en["language"], "english_hits": hit_count,
                  "words": hit_words}),
        "multilingual_sharper_on": sharper,
        "choices_returned": {"refund": n_refund, "calls": n_calls},
        "router_lifecycle": {"preload_seconds": round(preload_s, 2),
                             "warm_load_microseconds": round(warm_us, 3),
                             "warm_load_samples": warm_calls},
        "checks": dict(checks),
    })
    C.save_artifact("05_language_limit", artifact)

    print()
    print("  The detector covers 8 languages. Swahili, Indonesian, Malay and Somali read as English,")
    print(f"  go to the English checkpoint, and come back at confidence {sw['confidence']:.2f}. For non-European")
    print("  Latin-script traffic, pass lang= or model= explicitly, or gate on")
    print("  analyse()['language_undecided'].")

    C.require(all(ok for _, ok in checks),
              "Eight known languages; the ninth is routed to a checkpoint that cannot read it, "
              f"at confidence {sw['confidence']:.2f} instead of {en_conf:.2f}.")


if __name__ == "__main__":
    main()
