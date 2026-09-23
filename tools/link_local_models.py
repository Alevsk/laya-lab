#!/usr/bin/env python
"""Expose the cached checkpoints as local model directories.

`upstream/tests/test_local_e2e.py` -- the only upstream suite that exercises real
weights across all three checkpoints -- wants a directory laid out as

    <root>/laya                    <root>/laya-multilingual    <root>/laya-typed-decisions

Rather than copy 2.3 GB, this symlinks each name at the Hugging Face snapshot
already in the cache. Default root is ~/laya_models, matching that test's default.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = "convaiinnovations/laya"
RUNTIME = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
LAYOUT = {"laya": "", "laya-multilingual": "multilingual", "laya-typed-decisions": "typed-decisions"}


def link(root: Path) -> Path:
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(REPO, allow_patterns=list(RUNTIME), local_files_only=True))
    root.mkdir(parents=True, exist_ok=True)
    for name, sub in LAYOUT.items():
        target = snap / sub if sub else snap
        dest = root / name
        if dest.is_symlink() or dest.exists():
            if dest.is_symlink():
                dest.unlink()
            else:
                print(f"  {name}: real directory already present, leaving it alone")
                continue
        dest.symlink_to(target, target_is_directory=True)
        missing = [f for f in ("rl_agent_config.json", "model.safetensors", "tokenizer", "encoder")
                   if not (dest / f).exists()]
        print(f"  {name:<22} -> {target}" + (f"   MISSING {missing}" if missing else ""))
        if missing:
            raise SystemExit(f"checkpoint {name} incomplete: {missing}; run `make warm`")
    return root


if __name__ == "__main__":
    root = Path(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/laya_models"))
    print(f"Linking cached checkpoints into {root}")
    link(root)
    print("done — `make test` can now run the full end-to-end suite")
