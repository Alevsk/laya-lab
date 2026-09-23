"""Load a Laya checkpoint in ~1-2 s instead of 20-40 s.

Identical to the loader proven in the root repo (`scenarios/_common.py`, `make verify-loader`):
build the encoder on the meta device so the throwaway random initialisation never happens,
install the checkpoint tensors with `assign=True`, rebuild the non-persistent rotary buffers,
and upcast to fp32 to match the stock path. Falls back to stock `laya.load` on any surprise.
"""
from __future__ import annotations

import contextlib
import os


def _materialize_meta_buffers(model) -> list[str]:
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
            for bn, b in m.named_buffers(recurse=False) if b is not None and b.is_meta] + \
           [n for n, p in model.named_parameters() if p.is_meta]


class _MetaFallback(RuntimeError):
    pass


@contextlib.contextmanager
def fast_init():
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
            model.float()
            return result

        model.load_state_dict = load_state_dict
        return model

    C.build_model, A.build_model = patched, patched
    try:
        yield
    finally:
        C.build_model, A.build_model = real, real


def load_agent(checkpoint: str = "english", device: str | None = None):
    """`checkpoint` is 'english' | 'multilingual' | 'typed-decisions', or a local directory in
    laya's layout (rl_agent_config.json, model.safetensors, tokenizer/, encoder/) - e.g. one
    written by server/finetune.py."""
    import laya

    if os.path.isdir(checkpoint):
        repo, sub = checkpoint, None
    else:
        repo = "convaiinnovations/laya"
        sub = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}[checkpoint]
    try:
        with fast_init():
            return laya.load(repo, subfolder=sub, device=device)
    except Exception:
        return laya.load(repo, subfolder=sub, device=device)
