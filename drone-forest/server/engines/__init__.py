"""Decision engines, registered on import.

`base` is the contract. Each concrete engine lives in its own module and calls
`@register("name")` at import time. `load_builtin_engines()` imports the built-ins so the
service can list them; a missing or broken engine module (Laya not installed, say) is reported
rather than fatal, so the service stays usable with whatever engines *can* load.
"""
from __future__ import annotations

import importlib

from .base import ENGINES, DecisionEngine, available, create, register  # noqa: F401

BUILTIN_MODULES = ("random_engine", "heuristic_engine", "laya_engine", "laya_ft_engine")
IMPORT_ERRORS: dict[str, str] = {}


def load_builtin_engines() -> list[str]:
    """Import every built-in engine module; return the names that registered."""
    for mod in BUILTIN_MODULES:
        try:
            importlib.import_module(f"{__name__}.{mod}")
        except Exception as e:  # keep the service alive without this engine
            IMPORT_ERRORS[mod] = f"{type(e).__name__}: {e}"
    return available()
