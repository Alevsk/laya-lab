"""Four ways Laya loses your input without saying a word.

Every failure in this scenario is silent: no exception, no warning, no log line. The
call returns, the JSON looks well formed, the confidence looks plausible -- and the
part of the input that actually mattered is gone. That is what makes these dangerous
in a pipeline, and why each block here reproduces the failure live rather than quoting
it, and pairs it with a one-line guard you can put in your own code.

  BLOCK 1  laya.email.clean_email_body -- exported at the top level of the package and
           applied by default inside email_state() -- deletes the body of a standard
           forwarded message, drops whatever follows a mid-body "From:" line, truncates
           on a polite sentence near the top, and turns a literal backslash-n inside a
           JSON payload into a real newline. Pure Python, so it runs before the model
           loads.

  BLOCK 2  Long inputs are truncated from the RIGHT. `build_sequence` keeps as much of
           the FRONT of your state as fits the token budget and discards the rest, so in
           an email thread or chat log ordered oldest first, the newest line -- usually
           the decisive one -- is the first thing dropped. The same thread scored with
           the decisive line first vs. last gives two very different answers.

  BLOCK 3  Choice options are silently shortened as the option count grows, until the
           model sees bare labels with no descriptions at all. Past a hard cliff the
           call finally does raise -- and the error message names the wrong limit.

  BLOCK 4  Every answer carries `action.act_probability`. It is always exactly 1.0.
           The head that produces it emits logits in the thousands, which saturates a
           float32 softmax, so the field carries no information about any input.

HOW TO READ IT: each block ends in a `FOOTGUN REPRODUCED` line and a `WORKAROUND` line.
The summary table at the end restates the four in one place. The script exits non-zero
if any of the four stops reproducing -- which would mean upstream fixed it.

Run:  make s6        (or: HF_HUB_OFFLINE=1 uv run python scenarios/06_silent_footguns.py)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

# --------------------------------------------------------------------------- block 1 inputs

FORWARD = (
    "---------- Forwarded message ---------\n"
    "From: Finance <finance@vendor.example>\n"
    "Date: Mon, 22 Sep 2025 at 09:14\n"
    "Subject: Overdue invoice INV-4471\n"
    "To: Alex <alex@northwind.example>\n"
    "\n"
    "Hi, invoice INV-4471 for $12,400 is 30 days overdue. Please arrange payment "
    "this week or we will pause the service."
)

MID_BODY_FROM = (
    "Hi team,\n"
    "I got a bounce on the alias.\n"
    "From: noreply@acme.example was rejected by our MX.\n"
    "Can you whitelist it?"
)

POLITE_OPENER = "Hi,\nThank you so much!\nThe issue is fixed now.\nClosing the ticket."

JSON_PAYLOAD = "Our job failed with: " + json.dumps(
    {"error": "timeout", "trace": "at foo()\nFrom: worker-7 aborted\nat bar()", "ticket": "OPS-91"}
)

# --------------------------------------------------------------------------- block 2 inputs

PAD_REPEATS = 40
PAD = "Thanks for reaching out. We have received your message and will get back to you shortly. " * PAD_REPEATS
TAIL = "\n\nUPDATE: production is completely down, every customer is affected, this is a Sev-1 outage."
URGENCY = {
    "urg": {
        "type": "score",
        "instructions": "How urgent is this thread?",
        "criteria": ["not urgent", "normal", "urgent", "drop everything"],
    }
}

# --------------------------------------------------------------------------- block 3 inputs

LONGDESC = ("everything that concerns the billing lifecycle including invoices refunds "
            "chargebacks and dunning for all customers")
INS = "Read the whole ticket carefully and decide which of these teams should own it."
DESC_WORDS = {w for w in LONGDESC.split() if len(w) > 4}
OPTION_COUNTS = (2, 6, 10, 20, 44, 45)
NARROW_N = 44          # the row the assertion is aimed at: descriptions are gone by here
CLIFF_OK, CLIFF_FAIL = 126, 127

# --------------------------------------------------------------------------- block 4 inputs

SWEEP_STATES = [
    "I cannot log in with SSO.",
    "",
    "asdkjh qwe zxcv 1029384 !!!",
    "a" * 3000,
    "Maybe. Not sure. 50/50.",
    "Je ne peux pas me connecter",
    "\N{SLIGHTLY SMILING FACE}",
    "EVERYTHING IS DOWN FIX IT NOW",
    "thanks!",
    "refund",
]
SWEEP_QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should own this message?",
             "criteria": {"billing": "invoices and payments", "tech": "bugs and outages",
                          "other": "anything else"}},
    "urg": {"type": "score", "instructions": "How urgent is this message?",
            "criteria": ["no rush", "soon", "immediately"]},
    "reply": {"type": "noul", "instructions": "Does the sender expect a reply?"},
}


UPSTREAM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "upstream")


def read_upstream(relpath: str) -> str:
    """Read a file out of the vendored upstream checkout, or "" if it is not there."""
    try:
        with open(os.path.join(UPSTREAM, relpath), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def short_repr(text: str, limit: int) -> str:
    r = repr(text)
    return r if len(r) <= limit else r[: limit - 3] + "..."


def tail_repr(text: str, limit: int) -> str:
    """repr kept from the END, so a truncation shows where the text was cut off."""
    r = repr(text)
    return r if len(r) <= limit else "..." + r[-(limit - 3):]


def decode(agent, ids) -> str:
    """Decode without the BPE clean-up pass (which strips spaces and warns on every call)."""
    return agent.tok.decode(ids, clean_up_tokenization_spaces=False)


# =========================================================================== block 1


def block_one() -> dict:
    """clean_email_body deletes content it cannot recognise, and never says so."""
    from laya.email import clean_email_body

    C.section("BLOCK 1 / 4  -  clean_email_body() silently deletes the message")
    print("  laya.email.clean_email_body is a top-level export (it is in laya.__all__) and runs by")
    print("  default inside email_state(subject, body), whose signature is `clean: bool = True`, so")
    print("  you get it whether or not you asked. It returns a string; it never reports what it cut.\n")

    cases = [
        ("forwarded thread", FORWARD),
        ("mid-body From:", MID_BODY_FROM),
        ("polite line 2", POLITE_OPENER),
        ("JSON payload", JSON_PAYLOAD),
    ]
    rows, results = [], {}
    for label, raw in cases:
        cleaned = clean_email_body(raw)
        pct = 100.0 * len(cleaned) / len(raw)
        results[label] = {"raw_chars": len(raw), "clean_chars": len(cleaned),
                          "pct_kept": round(pct, 1), "survived": cleaned}
        rows.append([label, len(raw), len(cleaned), "%.0f%%" % pct, short_repr(cleaned, 60)])
    C.table(rows, ["CASE", "RAW CHARS", "CLEAN CHARS", "% KEPT", "WHAT SURVIVED (repr)"], align="lrrr")

    fwd = results["forwarded thread"]
    print("\n  Why the forward loses its body (email.py:42):")
    print("    `if any(p.match(line) for p in _QUOTE_HEADERS) and lines: break`")
    print("    The `and lines` guard means a quote header on line 0 cannot trigger the break, so the")
    print("    '---------- Forwarded message ---------' separator is appended instead. The very next")
    print("    line matches _QUOTE_HEADERS[3] (^From:) with `lines` now non-empty, and breaks --")
    print("    discarding the entire forwarded body, invoice number, amount and deadline included.")
    print("    Result: %d chars in, %d chars out (%.0f%% kept), and the separator is all that is left."
          % (fwd["raw_chars"], fwd["clean_chars"], fwd["pct_kept"]))

    print("\n  Why the mid-body case loses the request:")
    print("    The same ^From: rule fires on a line that is prose, not a quote header. Everything")
    print("    after it goes, so 'Can you whitelist it?' -- the only actionable sentence -- is dropped.")

    n_values = (4, 12, 16, 30, 50)
    windows = [max(1, min(int(n * 0.6), n - 8)) for n in n_values]
    print("\n  Why a polite line near the top truncates the message (email.py:48):")
    print("    `start = max(1, min(int(n*0.6), n-8))` -- the signature scan starts at:")
    print("      n lines = %s" % "  ".join("%d->%d" % (n, s) for n, s in zip(n_values, windows)))
    print("    For a short email the window starts at line 1, so 'Thank you so much!' at index 1 is")
    print("    read as a signature. Where the scan begins depends on the LINE COUNT, not on the text.")

    print("\n  Why the JSON payload is mangled (email.py:39):")
    print("    `.replace(\"\\\\n\", \"\\n\")` rewrites the literal two-character backslash-n that json.dumps")
    print("    emits into a real newline. The payload stops being one line, the new lines become")
    print("    eligible for quote and signature matching, and here the embedded 'From:' truncates it.")
    parsed_ok = True
    body = results["JSON payload"]["survived"]
    try:
        json.loads(body[body.index("{"):])
    except Exception as exc:  # noqa: BLE001 - the failure itself is the result
        parsed_ok = False
        print("    json.loads() on what comes back: %s: %s" % (type(exc).__name__, exc))

    tests = read_upstream("tests/test_email.py")
    n_cases = tests.count("\ncheck(") + tests.count("\ncheck_true(")
    forwarded_cases = "Forwarded" in tests
    if tests:
        print("\n  %s upstream does test this function -- %d cases in upstream/tests/test_email.py,"
              % (C.c("credit where it is due:", "dim"), n_cases))
        print("  counted in the file just now, and they are about disclaimer footers. The word")
        print("  'Forwarded' appears anywhere in that file: %s. That is why this block reproduces the"
              % forwarded_cases)
        print("  behaviour live instead of citing a test: the tested cases are not these cases.")

    kept_fwd = fwd["pct_kept"] < 25.0
    lost_request = "whitelist" not in results["mid-body From:"]["survived"]
    print("\n  %s  forward kept %.0f%% of its characters (<25%%), and the mid-body 'From:' case"
          % (C.c("FOOTGUN REPRODUCED", "yellow"), fwd["pct_kept"]))
    print("  dropped the request: 'whitelist' in output = %s. No exception, no warning." % (not lost_request))
    print("  %s  `if len(clean_email_body(b)) < 0.2 * len(b): use the raw body` -- and pass"
          % C.c("WORKAROUND", "green"))
    print("  clean=False to email_state() for machine output: runs of spaces and tabs are collapsed")
    print("  too (email.py:54), which destroys ASCII tables and indentation.")

    return {"cases": results, "signature_window": dict(zip(map(str, n_values), windows)),
            "json_still_parses": parsed_ok,
            "ok": bool(kept_fwd and lost_request)}


# =========================================================================== block 2


def block_two(agent) -> dict:
    """State is truncated from the right, so the newest line in a thread is dropped first."""
    from laya.common import build_sequence

    C.section("BLOCK 2 / 4  -  long input is truncated from the END, newest line first")
    print("  The same %s-character thread, scored three ways. Only the ORDER of its two halves"
          % format(len(PAD + TAIL), ","))
    print("  changes: %d boilerplate acknowledgements, and one Sev-1 update.\n" % PAD_REPEATS)

    layouts = [
        ("boilerplate then UPDATE (oldest first)", PAD + TAIL),
        ("UPDATE then boilerplate (newest first)", TAIL + PAD),
        ("UPDATE alone", TAIL),
    ]
    rows, measured = [], {}
    for label, state in layouts:
        out = agent.predict(state, URGENCY)
        ans = out["answers"]["urg"]
        measured[label] = {"score": ans["score"], "confidence": ans["confidence"],
                           "input_tokens": out["usage"]["input_tokens"], "chars": len(state)}
        rows.append([label, "%.4f" % ans["score"], "%.4f" % ans["confidence"],
                     out["usage"]["input_tokens"]])
    C.table(rows, ["THREAD LAYOUT", "urgency score", "confidence", "input_tokens"], align="lrrr")

    last = measured["boilerplate then UPDATE (oldest first)"]
    first = measured["UPDATE then boilerplate (newest first)"]
    drop = first["score"] - last["score"]
    same_tokens = first["input_tokens"] == last["input_tokens"]
    top = len(URGENCY["urg"]["criteria"]) - 1
    print("\n  Both long layouts report the same %d input_tokens (identical: %s), so nothing in the"
          % (first["input_tokens"], same_tokens))
    print("  response signals that one of them threw the outage away. The scale runs 0..%d; moving the" % top)
    print("  decisive line from the front to the back costs %.4f points of urgency and %.4f of confidence."
          % (drop, first["confidence"] - last["confidence"]))

    q = agent._to_internal(URGENCY["urg"])
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    seq_default, _ = build_sequence(agent.tok, PAD + TAIL, q, max_len, head_max_len)
    seq_left, _ = build_sequence(agent.tok, PAD + TAIL, q, max_len, head_max_len, None, True)
    in_default = "UPDATE" in decode(agent, seq_default)
    in_left = "UPDATE" in decode(agent, seq_left)

    print("\n  Straight from the tokeniser, on the oldest-first layout:")
    print("    'UPDATE' in build_sequence(..., %d, %d)                 -> %s"
          % (max_len, head_max_len, C.c(str(in_default), "red" if not in_default else "green")))
    print("    'UPDATE' in build_sequence(..., %d, %d, None, True)     -> %s   (truncate_left=True)"
          % (max_len, head_max_len, C.c(str(in_left), "green" if in_left else "red")))
    print("\n  common.py:84 is `st = st[-room:] if truncate_left else st[:room]`, so the fix exists.")
    print("  But Agent.system_one calls build_sequence with only FIVE positional arguments")
    print("  (agent.py:286), so truncate_left can never be reached through predict() -- there is no")
    print("  keyword argument for it on the public API either.")

    print("\n  %s  the word UPDATE is absent from the sequence predict() actually builds,"
          % C.c("FOOTGUN REPRODUCED", "yellow"))
    print("  and the score falls from %.4f to %.4f with no warning and no drop in token count."
          % (first["score"], last["score"]))
    print("  %s  order your state so the decisive content comes FIRST, or pre-truncate it"
          % C.c("WORKAROUND", "green"))
    print("  yourself (keep the newest ~1,500 characters of a thread) before handing it to predict().")

    return {"layouts": measured, "score_drop": round(drop, 4), "same_input_tokens": same_tokens,
            "update_in_default_sequence": in_default, "update_in_truncate_left_sequence": in_left,
            "ok": bool((not in_default) and in_left and drop >= 0.5 and same_tokens)}


# =========================================================================== block 3


def block_three(agent) -> dict:
    """Option descriptions are trimmed away as the option count rises, then the call falls off a cliff."""
    from laya.common import build_sequence

    C.section("BLOCK 3 / 4  -  option descriptions shrink to nothing as the option count grows")
    print("  One choice question, one %d-word description per option, only the OPTION COUNT changes."
          % len(LONGDESC.split()))
    print("  'what the model sees' is the decoded slice between the first two [MASK] markers, shown")
    print("  from its END so you can see exactly where each row was cut off.\n")

    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    rows, measured = [], {}
    for n in OPTION_COUNTS:
        crit = {"team_%02d" % i: LONGDESC for i in range(n)}
        q = agent._to_internal({"type": "choice", "instructions": INS, "criteria": crit})
        seq, markers = build_sequence(agent.tok, "Customer disputes an invoice.", q, max_len, head_max_len)
        per = markers[1] - markers[0]
        seen = decode(agent, seq[markers[0]:markers[1]])
        measured[n] = {"tokens_per_option": per, "option0": seen,
                       "desc_words_left": sorted(DESC_WORDS & set(seen.split()))}
        rows.append([n, per, tail_repr(seen, 78)])
    C.table(rows, ["OPTIONS", "TOKENS/OPTION", "WHAT THE MODEL SEES FOR option[0] (repr, from the end)"],
            align="rr")

    wide, narrow = measured[OPTION_COUNTS[0]], measured[NARROW_N]
    print("\n  common.py:70-75 is the mechanism:")
    print("    opt_budget = head_max_len - sum(len(o) for o in opt_ids)")
    print("    if opt_budget < 16:  per = max(4, (head_max_len - 16) // n);  opt_ids = [o[:per] ...]")
    print("    With head_max_len=%d that is per = max(4, %d // n) -- and note the undocumented floor"
          % (head_max_len, head_max_len - 16))
    print("    of 4. A floor means head_max_len is NOT a hard cap at all; the cliff section below")
    print("    measures how far past it the head actually goes.")
    print("  Nothing is logged. The question still answers, with a plausible-looking confidence,")
    print("  having read %d tokens per option instead of %d." % (narrow["tokens_per_option"], wide["tokens_per_option"]))

    print("\n  The cliff -- one more option turns a silent loss into a hard error:")
    cliff, probe = {}, {}
    for n in (CLIFF_OK, CLIFF_FAIL):
        crit = {"team_%03d" % i: None for i in range(n)}
        question = {"big": {"type": "choice", "instructions": INS, "criteria": crit}}
        pq = agent._to_internal(question["big"])
        pids, pmarkers = build_sequence(agent.tok, "Customer disputes an invoice.", pq, max_len, head_max_len)
        probe[n] = {"options": n, "markers_kept": len(pmarkers), "head_start": pmarkers[0],
                    "tokens_per_option": pmarkers[1] - pmarkers[0], "last_marker": pmarkers[-1],
                    "seq_len": len(pids)}
        probe[n]["nth_marker_would_land_at"] = (
            probe[n]["head_start"] + probe[n]["tokens_per_option"] * (n - 1))
        try:
            out = agent.predict("Customer disputes an invoice.", question)
            ans = out["answers"]["big"]
            cliff[n] = {"raised": None, "choice": ans["choice"], "confidence": ans["confidence"]}
            print("    n=%-3d  %s  choice=%s  confidence=%.4f  input_tokens=%d"
                  % (n, C.c("OK   ", "green"), ans["choice"], ans["confidence"], out["usage"]["input_tokens"]))
        except ValueError as exc:
            cliff[n] = {"raised": "ValueError", "message": str(exc)}
            print("    n=%-3d  %s  ValueError: %s" % (n, C.c("RAISE", "red"), exc))

    fail_probe, ok_probe = probe[CLIFF_FAIL], probe[CLIFF_OK]
    print("\n  The message names head_max_len=%d, but that is not the limit that bound. Calling" % head_max_len)
    print("  build_sequence directly on the same two questions, and reading the markers it returns:")
    for n in (CLIFF_OK, CLIFF_FAIL):
        p = probe[n]
        print("    n=%-3d  markers kept %3d/%3d   first marker at %d, %d tokens each, last at %d, seq_len %d"
              % (n, p["markers_kept"], n, p["head_start"], p["tokens_per_option"],
                 p["last_marker"], p["seq_len"]))
    print("  Markers are dropped by the `m < max_len` filter at common.py:86, so the real ceiling is")
    print("  max_len=%d: marker %d would land at %d + %d*%d = %d >= %d, so it is filtered out, and"
          % (max_len, CLIFF_FAIL, fail_probe["head_start"], fail_probe["tokens_per_option"],
             CLIFF_FAIL - 1, fail_probe["nth_marker_would_land_at"], max_len))
    print("  agent.py then raises because len(markers)=%d != len(options)=%d. head_max_len=%d never"
          % (fail_probe["markers_kept"], CLIFF_FAIL, head_max_len))
    ok_span = (min(ok_probe["last_marker"] + ok_probe["tokens_per_option"], ok_probe["seq_len"])
               - ok_probe["head_start"])
    print("  bound anything: even the case that SUCCEEDS spans %d tokens of options (marker %d to the"
          % (ok_span, ok_probe["head_start"]))
    print("  end of the sequence at %d) against that nominal budget of %d."
          % (ok_probe["seq_len"], head_max_len))
    print("  The exact number moves with your instruction length and how your labels tokenise -- and a")
    print("  checkpoint with a different max_len moves it too (not measured here; only english is")
    print("  loaded). What does not move is that it is a cliff, and that everything below it is silent.")

    shrunk = narrow["tokens_per_option"] < wide["tokens_per_option"]
    gone = not narrow["desc_words_left"]
    print("\n  %s  tokens per option fell %d -> %d between %d and %d options, and the"
          % (C.c("FOOTGUN REPRODUCED", "yellow"), wide["tokens_per_option"],
             narrow["tokens_per_option"], OPTION_COUNTS[0], NARROW_N))
    print("  description is entirely absent from option[0] at %d -- the model is choosing between labels."
          % NARROW_N)
    print("  %s  keep choice questions at <= 6 options with descriptions under ~30 tokens."
          % C.c("WORKAROUND", "green"))
    print("  For high-cardinality routing use laya.shortlist_choice / predict_shortlist -- and read its")
    print("  docstring first, because the stock embed_fn is a weak retriever.")

    return {"by_option_count": measured, "cliff": cliff, "marker_probe": probe,
            "ok": bool(shrunk and gone and cliff[CLIFF_OK]["raised"] is None
                       and cliff[CLIFF_FAIL]["raised"] == "ValueError"
                       and probe[CLIFF_FAIL]["markers_kept"] < CLIFF_FAIL
                       and probe[CLIFF_OK]["markers_kept"] == CLIFF_OK)}


# =========================================================================== block 4


def block_four(agent) -> dict:
    """action.act_probability is a constant 1.0 that looks like a signal."""
    import torch

    from laya.common import QTYPES, build_sequence, collate_items

    C.section("BLOCK 4 / 4  -  every answer's `action` field is a hardcoded-looking 1.0")
    print("  Sweeping %d very different states x %d question primitives = %d answers, collecting"
          % (len(SWEEP_STATES), len(SWEEP_QUESTIONS), len(SWEEP_STATES) * len(SWEEP_QUESTIONS)))
    print("  every answers[qid]['action']['act_probability'] into a set.\n")

    values, n_answers = set(), 0
    for state in SWEEP_STATES:
        out = agent.predict(state, SWEEP_QUESTIONS)
        for ans in out["answers"].values():
            values.add(ans["action"]["act_probability"])
            n_answers += 1

    print("  distinct act_probability over %d answers: %s"
          % (n_answers, C.c(str(sorted(values)), "yellow")))
    print("  (states included an empty string, 3,000 identical characters, keyboard mash, an emoji,")
    print("  French, shouting, and a one-word 'refund' -- nothing moved it.)")

    print("\n  Why, straight from the model. agent.py:320 is `act = torch.softmax(act.float(), -1)`")
    print("  and agent.py:335 publishes `round(float(act[r, 0]), 4)`. Those are the input logits:")
    q = agent._to_internal(SWEEP_QUESTIONS["team"])
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    logit_rows = []
    for state in SWEEP_STATES[:3]:
        seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        batch = collate_items([[{"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]}]],
                              agent.tok.pad_token_id)
        with torch.no_grad():
            _, act = agent.model(
                batch["input_ids"].to(agent.device),
                batch["attention_mask"].to(agent.device),
                batch["marker_pos"].to(agent.device),
                batch["marker_mask"].to(agent.device),
                batch["qtype"].to(agent.device),
            )
        C.sync(str(agent.device))
        pair = act.float().cpu()[0].tolist()
        logit_rows.append({"state": state, "act_logits": [round(v, 1) for v in pair],
                           "gap": round(pair[0] - pair[1], 1)})
        print("    %-32s raw act logits [%9.1f, %9.1f]   gap %8.1f"
              % (short_repr(state, 32), pair[0], pair[1], pair[0] - pair[1]))

    saturates = float(torch.softmax(torch.tensor([17.0, 0.0]), 0)[0]) == 1.0
    print("\n    float(torch.softmax(torch.tensor([17.0, 0.0]), 0)[0]) == 1.0  ->  %s" % saturates)
    print("  A gap of 17 already saturates a float32 softmax to exactly 1.0. The gaps above are in the")
    print("  thousands, and agent.py rounds to 4 decimal places on top, so on every input measured here")
    print("  the published value cannot be anything but 1.0.")

    docs = {f: read_upstream(f) for f in ("README.md", "BENCHMARKS.md")}
    mentions = [f for f, text in docs.items() if text and "act_probability" in text]
    print("\n  What it is NOT: the act head really has %d outputs (that is the width of the tensor"
          % len(logit_rows[0]["act_logits"]))
    print("  printed above) and cfg['act_costs'] is %s, so this looks exactly like a trained"
          % agent.cfg.get("act_costs"))
    print("  escalate/answer head you could gate on. Grepping the upstream docs just now,")
    print("  'act_probability' is mentioned in: %s -- so nothing tells you not to."
          % (", ".join(mentions) if mentions else "none of %s" % ", ".join(sorted(docs))))

    print("\n  %s  %d answers, %d distinct value: %s."
          % (C.c("FOOTGUN REPRODUCED", "yellow"), n_answers, len(values), sorted(values)))
    print("  %s  ignore `action` entirely and gate on `confidence` instead (see make s3)."
          % C.c("WORKAROUND", "green"))
    print("  %s this is an empirical result over %d answers plus a mechanism that explains it,"
          % (C.c("honestly:", "dim"), n_answers))
    print("  not a proof that no input anywhere produces a small enough gap to move the value.")

    return {"n_answers": n_answers, "distinct_act_probability": sorted(values),
            "act_logits": logit_rows, "softmax_saturates_at_17": saturates,
            "act_costs": agent.cfg.get("act_costs"), "act_head_width": len(logit_rows[0]["act_logits"]),
            "act_probability_mentioned_in_docs": mentions,
            "ok": bool(values == {1.0} and n_answers >= 30)}


# =========================================================================== main


def main() -> None:
    C.header("06  Four things that fail silently",
             "email cleaning, right-side truncation, option shrinkage, and a dead action field")
    C.require_cached("english")

    one = block_one()

    print()
    agent = C.load_agent("english")
    two = block_two(agent)
    three = block_three(agent)
    four = block_four(agent)

    C.section("SUMMARY  -  four silent failures, four guards")
    rows = [
        ["1  clean_email_body()", "deletes a forwarded body, keeps %.0f%%" % one["cases"]["forwarded thread"]["pct_kept"],
         "check output length vs. input", "yes" if one["ok"] else "NO"],
        ["2  right-side truncation", "newest line dropped, score %.2f -> %.2f"
         % (two["layouts"]["UPDATE then boilerplate (newest first)"]["score"],
            two["layouts"]["boilerplate then UPDATE (oldest first)"]["score"]),
         "put decisive content first", "yes" if two["ok"] else "NO"],
        ["3  option shrinkage", "descriptions gone by %d options" % NARROW_N,
         "<= 6 options, or shortlist_choice", "yes" if three["ok"] else "NO"],
        ["4  action.act_probability", "constant %s over %d answers"
         % (four["distinct_act_probability"], four["n_answers"]),
         "gate on confidence instead", "yes" if four["ok"] else "NO"],
    ]
    C.table(rows, ["FOOTGUN", "WHAT YOU MEASURED HERE", "GUARD TO ADD", "REPRODUCED"])
    print("\n  None of the four raised, warned or logged anything on the way. Only footgun 3 ever")
    print("  raises, and only once you cross its cliff -- everything below the cliff is silent.")

    payload = {"env": C.env_summary(), "block1_email_cleaner": one, "block2_truncation": two,
               "block3_options": three, "block4_act_head": four}
    C.save_artifact("06_silent_footguns", payload)

    C.require(
        one["ok"] and two["ok"] and three["ok"] and four["ok"],
        "all four silent footguns reproduced: the forwarded body is cut to <25%% and the mid-body "
        "request dropped; UPDATE is absent from the built sequence but present with truncate_left "
        "and costs %.2f urgency points; option descriptions vanish by %d options and %d options "
        "raise where %d succeed; act_probability is %s over %d answers"
        % (two["score_drop"], NARROW_N, CLIFF_FAIL, CLIFF_OK, four["distinct_act_probability"],
           four["n_answers"]),
    )


if __name__ == "__main__":
    main()
