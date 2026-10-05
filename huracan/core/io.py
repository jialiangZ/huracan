"""Checkpoint I/O -- the single owner of the on-disk phase checkpoint format.

Both the training layer (phase chaining) and the inference layer (model
loading) go through this module, so the format has exactly one definition:

    {"phase": str, "step": int, "models": {"fcst": state_dict, ...}}
"""

from __future__ import annotations

from pathlib import Path

import torch


def phase_path(workdir: str | Path, phase: str) -> Path:
    return Path(workdir) / f"ckpt_{phase}.pt"


def save_phase(workdir: str | Path, phase: str, step: int, state_dicts: dict) -> Path:
    """Atomically write one phase checkpoint (one file per phase, updated
    in place during training -- same convention as the original trainer)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ck = {"phase": phase, "step": step, "models": state_dicts}
    tmp = phase_path(workdir, phase).with_suffix(".tmp")
    torch.save(ck, tmp)
    final = phase_path(workdir, phase)
    tmp.replace(final)
    return final


def load_phase(workdir: str | Path, phase: str) -> dict:
    path = phase_path(workdir, phase)
    if not path.exists():
        raise FileNotFoundError(f"missing phase checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)
