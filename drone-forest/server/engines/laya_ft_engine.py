"""The fine-tuned Laya engine: the same LayaEngine, pointed at a checkpoint trained by
server/finetune.py on this simulator's own telemetry with the heuristic as teacher.

Nothing else differs. Same framing, same shuffle, same four questions, same argmax - so a
difference between "laya" and "laya-ft" in the arena is the fine-tune and only the fine-tune.
LAYA_FT_CHECKPOINT points at the checkpoint directory (default data/checkpoints/laya-drone-ft).
"""
from __future__ import annotations

import os
from dataclasses import replace

from .base import DecisionEngine, register
from .laya_engine import LayaEngine, LayaEngineConfig

ENGINE_NAME = "laya-ft"
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  "data", "checkpoints", "laya-drone-ft")


class LayaFtEngine(LayaEngine):
    name = ENGINE_NAME

    def __init__(self) -> None:
        path = os.environ.get("LAYA_FT_CHECKPOINT", DEFAULT_CHECKPOINT)
        super().__init__(replace(LayaEngineConfig.from_env(), checkpoint=path))

    def warmup(self) -> None:
        if not os.path.isdir(self.config.checkpoint):
            raise FileNotFoundError(
                f"no fine-tuned checkpoint at {self.config.checkpoint}; run `make finetune` first")
        super().warmup()

    def describe(self) -> dict:
        d = super().describe()
        d.update({"engine": self.name, "fine_tuned": True, "trained_on": "this simulator's telemetry, heuristic teacher"})
        return d


@register(ENGINE_NAME)
def create_laya_ft_engine() -> DecisionEngine:
    return LayaFtEngine()
