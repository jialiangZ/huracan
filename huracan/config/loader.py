"""YAML loading into typed ``HuracanConfig`` objects.

Keeps compatibility with the original flat YAML layout: user files are
deep-merged over dataclass defaults, YAML-1.1 float strings like ``5e-4``
are coerced back to numbers, and ``key=value`` dot-path overrides are
supported for the CLI.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

from .schema import HuracanConfig

_FLOAT_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _coerce(node):
    """YAML 1.1 parses '5e-4' as a string; convert numeric strings back."""
    if isinstance(node, dict):
        return {k: _coerce(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce(v) for v in node]
    if isinstance(node, str) and _FLOAT_RE.match(node):
        return float(node)
    return node


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _defaults_dict() -> dict:
    import dataclasses

    return dataclasses.asdict(HuracanConfig())


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    """Apply ``dot.path=value`` overrides (values parsed as YAML)."""
    d = copy.deepcopy(d)
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"override must look like key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        node = d
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into '{k}': '{kk}' is not a mapping")
        node[keys[-1]] = yaml.safe_load(v)
    return d


def load_config(path: str | Path, overrides: list[str] | None = None) -> HuracanConfig:
    """Load a YAML config file into a validated ``HuracanConfig``."""
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    user = _coerce(user)
    merged = _merge(_defaults_dict(), user)
    if overrides:
        merged = apply_overrides(merged, overrides)
    return HuracanConfig.from_dict(merged)


def var_names(cfg: HuracanConfig) -> list[str]:
    """Backward-compatible helper: ``var_names(cfg)``."""
    return cfg.var_names()
