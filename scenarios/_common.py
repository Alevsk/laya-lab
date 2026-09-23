"""Shared helpers for the Laya scenarios.

Nothing here changes what Laya computes. `load_agent`/`load_router` are ordinary
`laya.load` / `laya.Router` calls with one optimisation applied (see `fast_init`),
and every other helper is presentation or timing.
"""
from __future__ import annotations

import contextlib
import os
import statistics
import sys
import time
import warnings
from typing import Any
from collections.abc import Callable, Iterable, Sequence

warnings.filterwarnings("ignore", category=RuntimeWarning, module="laya.agent")

REPO = "convaiinnovations/laya"
SUBFOLDERS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
CHECKPOINTS = SUBFOLDERS  # alias: friendly name -> laya `subfolder` argument
RUNTIME_PATTERNS = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
WIDTH = 110

# ---------------------------------------------------------------- device


def resolve_device(prefer: str | None = None) -> str:
    """Best available device, honouring $LAYA_DEVICE then CUDA > MPS > CPU.

    MPS is used by default on Apple Silicon: it was verified bit-identical to CPU
    on all three checkpoints and is roughly 8x faster.
    """
    import torch

    choice = prefer or os.environ.get("LAYA_DEVICE")
    if choice:
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------- fast load


def _materialize_meta_buffers(model) -> list[str]:
    """Rebuild any buffer still on the meta device after an assign-load.

    A model built under `torch.device("meta")` allocates nothing. `load_state_dict`
    with `assign=True` then installs every *persistent* tensor straight from the
    checkpoint. What it cannot fill are **non-persistent** buffers -- tensors the
    module computes in `__init__` and deliberately keeps out of the state dict.
    ModernBERT has four: the rotary `inv_freq` tables.

    Those buffers are derived purely from the config and hold no learned state, so
    re-running the owning module's constructor on CPU reproduces them exactly.
    Returns the names of anything still on meta afterwards (must be empty).
    """
    owners = {}
    for mod_name, module in model.named_modules():
        for _, buf in module.named_buffers(recurse=False):
            if buf is not None and buf.is_meta:
                owners.setdefault(mod_name, module)

    for module in owners.values():
        cfg = getattr(module, "config", None)
        if cfg is None:
            continue
        try:
            rebuilt = type(module)(cfg)
        except Exception:
            continue
        for buf_name, buf in list(module.named_buffers(recurse=False)):
            if buf is not None and buf.is_meta and hasattr(rebuilt, buf_name):
                setattr(module, buf_name, getattr(rebuilt, buf_name).to("cpu"))

    return [f"{n}.{bn}" if n else bn
            for n, m in model.named_modules()
            for bn, b in m.named_buffers(recurse=False)
            if b is not None and b.is_meta] + \
           [n for n, prm in model.named_parameters() if prm.is_meta]


@contextlib.contextmanager
def fast_init():
    """Build checkpoints on the meta device instead of really allocating them.

    `laya.common.build_model` calls `AutoModel.from_config`, which allocates and
    randomly initialises the whole encoder (421M parameters for ModernBERT-large)
    a moment before `Agent.__init__` overwrites every one of them with
    `load_state_dict(..., strict=True)`. Measured here: 20-32 s of work whose
    result is never read.

    Building under `torch.device("meta")` costs 0.01 s and allocates nothing; the
    checkpoint tensors are then installed directly with `assign=True`. Because the
    load stays strict, a missing tensor still raises rather than silently leaving
    an uninitialised one behind.

    The build is abandoned and the stock path used instead if anything is left on
    the meta device afterwards, so a checkpoint this does not fully understand can
    never produce a half-initialised model. `make verify-loader` checks the outputs
    against stock `laya.load()` on all three checkpoints.

    Set LAYA_NO_FAST_INIT=1 to always use the stock path.
    """
    if os.environ.get("LAYA_NO_FAST_INIT"):
        yield
        return

    import torch

    import laya.agent as A
    import laya.common as C

    real = C.build_model

    def patched(cfg, encoder_dir=None):
        with torch.device("meta"):
            model = real(cfg, encoder_dir=encoder_dir)
        stock_load = model.load_state_dict

        def load_state_dict(state_dict, strict=True, assign=False):
            result = stock_load(state_dict, strict=strict, assign=True)
            leftover = _materialize_meta_buffers(model)
            if leftover:
                raise _MetaFallback(leftover)
            # `assign=True` installs the checkpoint tensors verbatim, so the model
            # inherits their storage dtype (these checkpoints ship fp16). The stock
            # path instead copies them *into* the fp32 tensors `from_config` built,
            # upcasting on the way. Match that, or fp16 weights meet fp32 activations
            # and MPS aborts inside MPSNDArrayMatrixMultiplication.
            model.float()
            return result

        model.load_state_dict = load_state_dict
        return model

    C.build_model, A.build_model = patched, patched
    try:
        yield
    finally:
        C.build_model, A.build_model = real, real


class _MetaFallback(RuntimeError):
    """Raised when meta-building leaves tensors unmaterialised; triggers stock load."""


def load_agent(checkpoint: str = "english", device: str | None = None, quiet: bool = False):
    """Load one Laya checkpoint by friendly name ('english'|'multilingual'|'typed-decisions')."""
    import laya

    if checkpoint not in SUBFOLDERS:
        raise ValueError(f"unknown checkpoint {checkpoint!r}; choose from {list(SUBFOLDERS)}")
    device = device or resolve_device()
    t0 = time.time()
    try:
        with fast_init():
            agent = laya.load(REPO, subfolder=SUBFOLDERS[checkpoint], device=device)
    except Exception as e:
        print(f"  fast load unavailable ({type(e).__name__}); using the stock path", flush=True)
        agent = laya.load(REPO, subfolder=SUBFOLDERS[checkpoint], device=device)
    if not quiet:
        print(f"  loaded {checkpoint} on {agent.device} in {time.time() - t0:.1f}s", flush=True)
    return agent


def load_router(names: Sequence[str] = ("english", "multilingual"), device: str | None = None,
                quiet: bool = False, **kw):
    """A `laya.Router` with `names` preloaded, built through `fast_init`."""
    from laya import Router

    device = device or resolve_device()
    t0 = time.time()
    try:
        with fast_init():
            router = Router(device=device, **kw)
            router.preload(list(names))
    except Exception as e:
        print(f"  fast load unavailable ({type(e).__name__}); using the stock path", flush=True)
        router = Router(device=device, **kw)
        router.preload(list(names))
    if not quiet:
        print(f"  router ready on {device} with {router.loaded} in {time.time() - t0:.1f}s", flush=True)
    return router


# ---------------------------------------------------------------- cache


def is_cached(name: str) -> bool:
    """True when `name`'s runtime files are already in the local HF cache.

    `snapshot_download(local_files_only=True)` without `allow_patterns` raises
    even for a perfectly usable checkpoint, because laya only ever fetches these
    four patterns and never the repo's README or assets. The patterns are required.
    """
    from huggingface_hub import snapshot_download

    sub = SUBFOLDERS[name]
    prefix = f"{sub}/" if sub else ""
    try:
        snapshot_download(REPO, local_files_only=True,
                          allow_patterns=[prefix + n for n in RUNTIME_PATTERNS])
        return True
    except Exception:
        return False


def require_cached(*names: str) -> None:
    """Exit(2) with instructions when a needed checkpoint has not been downloaded."""
    missing = [n for n in names if not is_cached(n)]
    if missing:
        print(f"\n  {c('missing checkpoints:', 'red')} {', '.join(missing)}")
        print("  Run `make warm` once (downloads ~2.3 GB), then re-run this scenario.\n")
        sys.exit(2)


# ---------------------------------------------------------------- timing


def sync(device: str | None = None) -> None:
    """Block until queued GPU work is done.

    CUDA and MPS dispatch asynchronously, so a bare `time.perf_counter()` around
    `predict()` stops before the GPU has finished and reports a number that is far
    too good (measured 68 ms where the true figure was 129 ms). Every timing helper
    here synchronises first.
    """
    import torch

    device = device or resolve_device()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(fn: Callable[[], Any], repeat: int = 7, warmup: int = 3,
          device: str | None = None) -> tuple[Any, dict[str, float]]:
    """Run `fn` and return (last result, {p50, min, max, mean} in milliseconds).

    Warms up first (the first call on a new tensor shape pays kernel compilation)
    and synchronises the device around every sample.
    """
    for _ in range(warmup):
        fn()
    sync(device)
    samples, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1000)
    return out, {
        "p50": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.fmean(samples),
    }


# ---------------------------------------------------------------- output


_C = {"dim": "\033[2m", "bold": "\033[1m", "green": "\033[32m", "red": "\033[31m",
      "yellow": "\033[33m", "cyan": "\033[36m", "off": "\033[0m"}


def _supports_colour() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(text: str, style: str) -> str:
    return f"{_C[style]}{text}{_C['off']}" if _supports_colour() else text


def _visible_len(s: str) -> int:
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def header(title: str, subtitle: str = "") -> None:
    print()
    print(c("=" * 78, "dim"))
    print(c(f" {title}", "bold"))
    if subtitle:
        print(c(f" {subtitle}", "dim"))
    print(c("=" * 78, "dim"))


def section(title: str) -> None:
    print(f"\n{c('--', 'dim')} {c(title, 'cyan')}")


def table(rows: Iterable[Sequence[Any]], headers: Sequence[str], align: str = "") -> None:
    """Print an aligned table. `align` is one char per column: 'l' or 'r'."""
    rows = [[("" if v is None else str(v)) for v in r] for r in rows]
    cols = len(headers)
    align = (align + "l" * cols)[:cols]
    widths = [max(_visible_len(str(headers[i])), *(_visible_len(r[i]) for r in rows)) if rows
              else _visible_len(str(headers[i])) for i in range(cols)]

    def fmt(cells, style=None):
        parts = []
        for i, cell in enumerate(cells):
            pad = " " * (widths[i] - _visible_len(str(cell)))
            parts.append((pad + str(cell)) if align[i] == "r" else (str(cell) + pad))
        line = "  ".join(parts)
        return c(line, style) if style else line

    print("  " + fmt(headers, "bold"))
    print("  " + c("  ".join("-" * w for w in widths), "dim"))
    for r in rows:
        print("  " + fmt(r))


def verdict(ok: bool, message: str) -> None:
    mark = c(" PASS ", "green") if ok else c(" FAIL ", "red")
    print(f"\n[{mark}] {message}")


def require(condition: bool, message: str) -> None:
    """Assert a scenario's stable invariant; exits non-zero so make targets gate on it."""
    verdict(bool(condition), message)
    if not condition:
        sys.exit(1)


def bar(value: float, width: int = 24, lo: float = 0.0, hi: float = 1.0) -> str:
    n = int(round(width * max(0.0, min(1.0, (value - lo) / (hi - lo)))))
    return c("#" * n, "cyan") + c("." * (width - n), "dim")


# ---------------------------------------------------------------- artefacts


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def save_artifact(name: str, payload: Any) -> str:
    """Write a scenario's raw results to out/<name>.json and return the path.

    Scenarios print a readable summary; this keeps the full numbers so a run can
    be re-read, diffed between machines, or pasted into a report.
    """
    import json

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  {c('artefact', 'dim')} {os.path.relpath(path, os.path.dirname(OUT_DIR))}")
    return path


def env_summary() -> dict[str, str]:
    """Machine/library facts worth stamping onto every artefact."""
    import platform

    import torch

    import laya

    return {
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "laya": laya.__version__,
        "device": resolve_device(),
    }
