#!/usr/bin/env python
"""Pre-download the Laya checkpoints so scenarios never pay a cold network cost.

Only the runtime files of each requested checkpoint are fetched (the same
allow_patterns laya.Agent uses), so this does not pull the repo's assets or
eval images.

    uv run python tools/warm_cache.py                 # all three
    uv run python tools/warm_cache.py english         # just one
"""
from __future__ import annotations

import sys
import time

REPO = "convaiinnovations/laya"

# name -> subfolder in the bundle repo ("" means repo root)
CHECKPOINTS = {
    "english": "",
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}

RUNTIME_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


def warm(name: str) -> tuple[str, float, str]:
    from huggingface_hub import snapshot_download

    sub = CHECKPOINTS[name]
    prefix = f"{sub}/" if sub else ""
    t0 = time.time()
    path = snapshot_download(
        REPO,
        allow_patterns=[prefix + f for f in RUNTIME_FILES],
    )
    return path, time.time() - t0, prefix or "<root>"


def main(argv: list[str]) -> int:
    names = argv[1:] or list(CHECKPOINTS)
    unknown = [n for n in names if n not in CHECKPOINTS]
    if unknown:
        print(f"unknown checkpoint(s): {unknown}; choose from {list(CHECKPOINTS)}", file=sys.stderr)
        return 2

    print(f"Warming {len(names)} checkpoint(s) from {REPO}\n")
    for name in names:
        path, secs, prefix = warm(name)
        print(f"  {name:<16} ok  {secs:6.1f}s  patterns={prefix}")
    print(f"\nHF cache: {path}")
    print("Checkpoints are cached. Scenarios will now load without network access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
