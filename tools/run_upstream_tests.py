#!/usr/bin/env python
"""Run the upstream Laya test scripts the way its CI does, and summarise.

The files in upstream/tests/ are standalone scripts (they call sys.exit() at
module level), not pytest modules -- `pytest upstream/tests` raises
INTERNALERROR. This runner executes each one as a subprocess and reports.

    uv run python tools/run_upstream_tests.py           # all
    uv run python tools/run_upstream_tests.py router    # substring filter
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "upstream" / "tests"

# Scripts needing something this lab cannot provide, and why. Populated at runtime.
SKIP: dict[str, str] = {}

LOCAL_MODELS = Path(os.path.expanduser("~/laya_models"))


def _ensure_local_models() -> None:
    """test_local_e2e.py needs the checkpoints as plain directories; symlink them."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from link_local_models import link

        link(LOCAL_MODELS)
    except Exception as e:
        SKIP["test_local_e2e.py"] = f"could not link local checkpoints ({e}); run `make warm`"


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    skip_e2e = "--skip-e2e" in argv
    pattern = args[0] if args else ""
    scripts = sorted(p for p in TESTS.glob("test_*.py") if pattern in p.name)
    if skip_e2e:
        SKIP["test_local_e2e.py"] = "skipped by --skip-e2e (loads real weights)"
    if not scripts:
        print(f"no upstream tests matching {pattern!r} in {TESTS}", file=sys.stderr)
        return 2

    if not skip_e2e and any(p.name == "test_local_e2e.py" for p in scripts):
        _ensure_local_models()

    env = dict(os.environ)
    env.setdefault("LAYA_DEVICE", os.environ.get("LAYA_DEVICE", "cpu"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    results, failed = [], 0
    for script in scripts:
        if script.name in SKIP:
            results.append((script.name, "skip", 0.0, SKIP[script.name]))
            continue
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=1800)
        dt = time.time() - t0
        ok = proc.returncode == 0
        failed += not ok
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        results.append((script.name, "pass" if ok else "FAIL", dt,
                        tail[-1][:88] if tail else f"exit {proc.returncode}"))
        if not ok:
            print(f"\n----- {script.name} failed -----")
            print("\n".join((proc.stdout or "").splitlines()[-25:]))
            print("\n".join((proc.stderr or "").splitlines()[-25:]))

    w = max(len(n) for n, *_ in results)
    print(f"\nUpstream test suite ({TESTS.relative_to(ROOT)})\n" + "-" * (w + 60))
    for name, status, dt, note in results:
        mark = {"pass": "\033[32mpass\033[0m", "FAIL": "\033[31mFAIL\033[0m", "skip": "\033[33mskip\033[0m"}[status]
        print(f"  [{mark}] {name:<{w}}  {dt:6.1f}s  {note}")
    print("-" * (w + 60))
    n_run = sum(1 for _, s, _, _ in results if s != "skip")
    print(f"  {n_run - failed}/{n_run} passed, {sum(1 for _, s, _, _ in results if s == 'skip')} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
