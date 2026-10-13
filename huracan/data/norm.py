"""Per-variable normalization shared by states, forcings and observations.

``cfg`` arguments are typed ``HuracanConfig`` objects.  States are stored on
disk as raw physical values (fp16 memmap); the dataset normalizes on the
fly.  Observations are normalized with the *same* stats as the matching
state variable and missing pixels are set to exactly zero after
normalization (paper Sec. 2.1 item 5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Normalizer:
    def __init__(self, names: list[str], mean=None, std=None):
        self.names = list(names)
        self.mean = (
            np.zeros(len(names), dtype=np.float32)
            if mean is None
            else np.asarray(mean, np.float32)
        )
        self.std = (
            np.ones(len(names), dtype=np.float32)
            if std is None
            else np.asarray(std, np.float32)
        )

    @property
    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def decode(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "names": self.names,
                    "mean": self.mean.tolist(),
                    "std": self.std.tolist(),
                },
                f,
                indent=1,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        with open(path) as f:
            d = json.load(f)
        return cls(d["names"], d["mean"], d["std"])

    @classmethod
    def from_array(
        cls, names: list[str], arr: np.ndarray, sample_step: int = 8
    ) -> "Normalizer":
        a = arr[::sample_step].astype(np.float32)
        return cls(names, a.mean(axis=(0, 1)), a.std(axis=(0, 1)) + 1e-6)


def obs_norm_mapping(cfg, state_norm: Normalizer) -> Normalizer:
    """Build the observation normalizer reusing state stats (same variable =>
    same stats; e.g. obs of 850 hPa temperature uses t850 stats).

    ``cfg`` is a typed ``HuracanConfig``."""
    names: list[str] = []
    means, stds = [], []
    si = state_norm.index
    v = cfg.data.variables
    for lev in v.levels:
        for u in v.upper:
            names.append(f"{u}{lev}")
            means.append(state_norm.mean[si[f"{u}{lev}"]])
            stds.append(state_norm.std[si[f"{u}{lev}"]])
    for s in cfg.data.obs.surface_vars:
        names.append(s)
        means.append(state_norm.mean[si[s]])
        stds.append(state_norm.std[si[s]])
    assert len(names) == cfg.n_obs_channels(), (len(names), cfg.n_obs_channels())
    return Normalizer(names, means, stds)
