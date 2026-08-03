"""Config compatibility helpers for pinned-but-drifting training deps (TRL).

TRL removes/renames config fields between releases (e.g. GRPOConfig's
`max_prompt_length` was removed in 0.29); passing a stale key raises
TypeError at trainer startup. Filter kwargs against the fields the installed
version actually accepts so drift degrades loudly instead of crashing.
"""
from __future__ import annotations

import dataclasses
import inspect
from typing import Any


def filter_config_kwargs(config_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs restricted to the fields config_cls accepts, warning loudly.

    Uses dataclasses.fields when config_cls is a dataclass (GRPOConfig,
    SFTConfig, TrainingArguments all are), else inspect.signature. Dropped
    keys are printed so the run log shows exactly which knobs were ignored.
    """
    accepted: set[str]
    if dataclasses.is_dataclass(config_cls):
        accepted = {f.name for f in dataclasses.fields(config_cls) if f.init}
    else:
        try:
            params = inspect.signature(config_cls.__init__).parameters.values()
        except (TypeError, ValueError):
            return dict(kwargs)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return dict(kwargs)
        accepted = {p.name for p in params if p.name != "self"}

    kept = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        print(
            f"WARNING: {config_cls.__name__} in the installed version does not "
            f"accept {dropped} — dropping them. The behavior these keys "
            "controlled is NOT applied; check the TRL release notes.",
            flush=True,
        )
    return kept
