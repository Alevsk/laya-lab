#!/usr/bin/env python
"""Does Laya actually run on this Mac's GPU, correctly and faster?

Answers three questions for every checkpoint, on the machine you are sitting at:

  1. Does it load and run on the Metal (MPS) backend at all?
  2. Are the answers IDENTICAL to the CPU answers?  A GPU that is fast and wrong
     is worse than no GPU, and MPS does have op-level gaps, so this is checked
     rather than assumed.
  3. How much faster is it, measured with the device synchronised?

    uv run python tools/gpu_check.py

Exits non-zero if the GPU produces different answers from the CPU.
"""
from __future__ import annotations

import os
import platform
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenarios"))

import _common as C  # noqa: E402

# One question of each primitive, so every head is exercised on both devices.
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments, invoices, refunds",
                     "technical": "bugs, outages, errors",
                     "sales": "pricing and new contracts"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["no rush", "this week", "today", "drop everything"],
    },
    "refund_requested": {"type": "noul", "instructions": "Is a refund being asked for?"},
}

STATE = {
    "from": "ops@acme.io",
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today "
            "or we will move to another provider.",
}


def values(answers: dict) -> dict:
    """The answer payload, flattened to comparable scalars."""
    out = {}
    for qid, a in answers.items():
        out[qid] = a[a["type"]]
        out[qid + ".confidence"] = a["confidence"]
        if "probabilities" in a:
            out[qid + ".probs"] = tuple(a["probabilities"].values())
    return out


def main() -> int:
    import torch

    gpu = None
    if torch.cuda.is_available():
        gpu = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        gpu = "mps"

    C.header("Laya GPU compatibility check",
             f"{platform.platform(terse=True)} / torch {torch.__version__}")

    chip = os.popen("sysctl -n machdep.cpu.brand_string 2>/dev/null").read().strip()
    cores = os.popen("system_profiler SPDisplaysDataType 2>/dev/null "
                     "| awk '/Total Number of Cores/{print $NF; exit}'").read().strip()
    C.table(
        [["chip", chip or "unknown"],
         ["GPU cores", cores or "n/a"],
         ["backend", gpu or "none — CPU only"],
         ["torch.backends.mps.is_built", str(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())],
         ["torch.backends.mps.is_available", str(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())]],
        ["", ""])

    if gpu is None:
        C.verdict(False, "No GPU backend available; Laya will run on CPU only.")
        return 1

    C.require_cached(*C.CHECKPOINTS)

    rows, mismatches = [], []
    for name in C.CHECKPOINTS:
        C.section(f"{name}: {gpu} vs cpu")

        agent_gpu = C.load_agent(name, device=gpu, quiet=True)
        _, t_gpu = C.timed(lambda ag=agent_gpu: ag.predict(STATE, QUESTIONS), device=gpu)
        v_gpu = values(agent_gpu.predict(STATE, QUESTIONS)["answers"])
        del agent_gpu

        agent_cpu = C.load_agent(name, device="cpu", quiet=True)
        _, t_cpu = C.timed(lambda ag=agent_cpu: ag.predict(STATE, QUESTIONS), repeat=3, warmup=1, device="cpu")
        v_cpu = values(agent_cpu.predict(STATE, QUESTIONS)["answers"])
        del agent_cpu

        diff = [k for k in v_cpu if v_cpu[k] != v_gpu[k]]
        mismatches += [(name, k) for k in diff]
        rows.append([name, f"{t_gpu['p50']:.1f} ms", f"{t_cpu['p50']:.1f} ms",
                     f"{t_cpu['p50'] / t_gpu['p50']:.1f}x",
                     C.c("identical", "green") if not diff else C.c(f"{len(diff)} DIFFER", "red")])
        print(f"  {name}: {gpu} {t_gpu['p50']:.1f} ms vs cpu {t_cpu['p50']:.1f} ms; "
              f"{len(v_cpu)} compared values, {len(diff)} different")

    C.section("summary")
    C.table(rows, ["checkpoint", gpu, "cpu", "speed-up", "answers"], align="lrrrl")

    C.save_artifact("gpu_check", {
        "env": C.env_summary(), "chip": chip, "gpu_cores": cores, "backend": gpu,
        "rows": [dict(zip(["checkpoint", "gpu_p50", "cpu_p50", "speedup", "agreement"], r)) for r in rows],
        "mismatches": mismatches,
    })

    C.require(not mismatches,
              f"all three checkpoints run on {gpu} and return answers identical to CPU, "
              f"{min(float(r[3][:-1]) for r in rows):.1f}-{max(float(r[3][:-1]) for r in rows):.1f}x faster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
