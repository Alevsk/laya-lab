"""Prove the fast loader is equivalent to stock `laya.load()`.

`scenarios/_common.py` skips the throwaway random initialisation that
`AutoModel.from_config` performs before `load_state_dict(..., strict=True)`
overwrites every parameter. That init costs ~30 s per checkpoint and its result
is never read.

This script loads each checkpoint BOTH ways and asserts the predictions are
identical, so the optimisation is a claim you can re-check rather than trust.

    uv run python tools/verify_loader.py
"""
import time
import warnings
warnings.filterwarnings("ignore")


import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenarios"))
from _common import fast_init, resolve_device  # noqa: E402


def fast_load(repo="convaiinnovations/laya", subfolder=None, device=None):
    """Exactly what scenarios/_common.load_agent does."""
    import laya

    with fast_init():
        return laya.load(repo, subfolder=subfolder, device=device)


if __name__ == "__main__":
    import laya
    DEV = resolve_device()
    failures = 0
    STATE = {"from": "ops@acme.io", "subject": "prod down",
             "body": "The checkout API has been returning 500s for 20 minutes. Revenue is stopped. Please escalate now."}
    QS = laya.triage_questions()

    for sub, name in ((None, "english"), ("multilingual", "multilingual"), ("typed-decisions", "typed-decisions")):
        t = time.time()
        fast = fast_load(subfolder=sub, device=DEV)
        t_fast = time.time() - t

        t = time.time()
        stock = laya.load("convaiinnovations/laya", subfolder=sub, device=DEV)
        t_stock = time.time() - t

        rf = fast.predict(STATE, QS)["answers"]
        rs = stock.predict(STATE, QS)["answers"]
        bad = []
        for qid in QS:
            k = rs[qid]["type"]
            a, b = rs[qid][k], rf[qid][k]
            ok = (a == b) if k == "choice" else abs(float(a) - float(b)) < 1e-9
            if not ok:
                bad.append(f"{qid}: {a} != {b}")
        failures += len(bad)
        print("%-16s fast=%5.2fs  stock=%5.2fs  speedup=%4.1fx  identical=%s %s"
              % (name, t_fast, t_stock, t_stock / max(t_fast, 1e-9), not bad, bad or ""))
        del fast, stock

    print()
    if failures:
        print("\033[31mFAIL\033[0m fast loader diverged from stock in %d place(s)" % failures)
        raise SystemExit(1)
    print("\033[32mPASS\033[0m fast loader is output-identical to stock laya.load() on all three checkpoints")
