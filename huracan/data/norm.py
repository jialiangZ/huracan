"""Per-variable normalization shared by states, forcings and observations.

``cfg`` arguments are typed ``HuracanConfig`` objects.  States are stored on
disk as raw physical values (fp32 via the WB2 path); the dataset normalizes
on the fly.  Observations are normalized with the *same* stats as the
matching state variable; validity is tracked by explicit masks/counts
(P0), and missing pixels are exact zeros after normalization (paper Sec.
2.1 item 5).
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
        cls,
        names: list[str],
        arr: np.ndarray,
        sample_step: int = 8,
        frames: np.ndarray | slice | None = None,
    ) -> "Normalizer":
        """Fit normalization stats.

        ``frames`` (P0 fix): explicit frame indices/slice to fit on -- used
        to restrict the fit to the *training* period only so validation and
        test data can never leak into the normalizer.  Statistics are
        streamed in chunks (the array is usually a multi-GB memmap) and
        every stored frame contributes, so all four synoptic hours are
        always covered (``sample_step`` subsampling on a 6-hourly axis
        always lands on the same time of day).

        ``sample_step`` is retained for backward compatibility only.
        """
        if frames is None:
            frames = np.arange(0, arr.shape[0], sample_step)
        idx = np.atleast_1d(np.asarray(frames))
        if arr.ndim != 3:
            raise ValueError(f"expected (T, npix, C) array, got shape {arr.shape}")
        C = arr.shape[2]
        s = np.zeros(C, dtype=np.float64)
        s2 = np.zeros(C, dtype=np.float64)
        count = 0  # number of (frame, pixel) samples per channel
        chunk = 256
        for a in range(0, len(idx), chunk):
            block = np.asarray(arr[idx[a : a + chunk]], dtype=np.float64)
            s += block.sum(axis=(0, 1))
            s2 += np.square(block).sum(axis=(0, 1))
            count += block.shape[0] * block.shape[1]
        count = max(count, 1)
        mean = s / count
        var = np.maximum(s2 / count - np.square(mean), 0.0)
        return cls(names, mean, np.sqrt(var) + 1e-6)


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
