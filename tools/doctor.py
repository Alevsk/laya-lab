#!/usr/bin/env python
"""Health check for the Laya lab: interpreter, deps, device, checkpoints, inference.

    uv run python tools/doctor.py

Exits non-zero if anything required is missing, so `make doctor` is a real gate.
"""
from __future__ import annotations

import os
import platform
import sys
import time
import warnings

warnings.filterwarnings("ignore")

OK, WARN, FAIL = "ok", "warn", "FAIL"
_rows: list[tuple[str, str, str]] = []
_failed = False


def check(name: str, status: str, detail: str = "") -> None:
    global _failed
    if status == FAIL:
        _failed = True
    _rows.append((name, status, detail))


def main() -> int:
    check("platform", OK, f"{platform.system()} {platform.release()} / {platform.machine()}")
    check("python", OK if sys.version_info[:2] == (3, 12) else WARN, platform.python_version())

    try:
        import torch
        check("torch", OK, torch.__version__)
    except Exception as e:  # pragma: no cover
        check("torch", FAIL, repr(e))
        torch = None

    for mod in ("transformers", "safetensors", "huggingface_hub", "numpy"):
        try:
            m = __import__(mod)
            check(mod, OK, getattr(m, "__version__", "?"))
        except Exception as e:
            check(mod, FAIL, repr(e))

    try:
        import laya
        check("laya", OK, f"{laya.__version__} from {os.path.dirname(laya.__file__)}")
    except Exception as e:
        check("laya", FAIL, repr(e))
        return _render()

    # --- device -----------------------------------------------------------
    if torch is not None:
        if torch.cuda.is_available():
            check("device", OK, f"cuda ({torch.cuda.get_device_name(0)})")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            check("device", OK, "mps (Apple GPU) — roughly 12x faster than cpu here")
        else:
            check("device", WARN, "cpu only — expect ~550 ms per 5-question call")

    # --- checkpoints in cache --------------------------------------------
    from huggingface_hub import snapshot_download

    for name, sub in (("english", ""), ("multilingual", "multilingual"), ("typed-decisions", "typed-decisions")):
        prefix = f"{sub}/" if sub else ""
        try:
            p = snapshot_download(
                "convaiinnovations/laya",
                allow_patterns=[prefix + f for f in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")],
                local_files_only=True,
            )
            w = os.path.join(p, sub, "model.safetensors") if sub else os.path.join(p, "model.safetensors")
            size = os.path.getsize(w) / 1e6 if os.path.exists(w) else 0
            check(f"ckpt:{name}", OK if size else FAIL, f"{size:.0f} MB cached" if size else "weights missing")
        except Exception:
            check(f"ckpt:{name}", FAIL, "not cached — run `make warm`")

    # --- routing works without any model ---------------------------------
    try:
        from laya import Router
        t0 = time.time()
        r = Router().route({"body": "मुझसे दो बार शुल्क लिया गया"})
        dt = (time.time() - t0) * 1000
        check("router.route", OK if r.model == "multilingual" else FAIL, f"hindi -> {r.model} in {dt:.2f} ms")
    except Exception as e:
        check("router.route", FAIL, repr(e))

    # --- one real forward pass -------------------------------------------
    if not _failed:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scenarios"))
            from _common import load_agent, resolve_device  # type: ignore
            t0 = time.time()
            agent = load_agent(device=resolve_device())
            t_load = time.time() - t0
            t0 = time.time()
            res = agent.predict({"msg": "I was charged twice, refund me"},
                                {"refund": {"type": "noul", "instructions": "Is a refund requested?"}})
            t_infer = (time.time() - t0) * 1000
            v = res["answers"]["refund"]["noul"]
            check("inference", OK if v > 0.5 else WARN,
                  f"load {t_load:.1f}s, predict {t_infer:.0f} ms, P(refund)={v:.3f}")
        except Exception as e:
            check("inference", FAIL, repr(e))

    return _render()


def _render() -> int:
    w = max(len(n) for n, _, _ in _rows)
    print("\nLaya lab doctor\n" + "-" * (w + 40))
    for name, status, detail in _rows:
        mark = {"ok": "\033[32m ok \033[0m", "warn": "\033[33mwarn\033[0m", "FAIL": "\033[31mFAIL\033[0m"}[status]
        print(f"  [{mark}] {name:<{w}}  {detail}")
    print("-" * (w + 40))
    if _failed:
        print("\033[31mSomething is broken.\033[0m Run `make setup && make warm`, then re-run `make doctor`.")
        return 1
    print("\033[32mAll good.\033[0m Try `make scenarios`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
