"""Torch dataset serving contiguous frame blocks for rollout training.

Each item is a block of ``2 + k*opc`` consecutive frames (k = number of
autoregressive model calls, opc = ``model.outputs_per_call``; k=0 -> the
single supervised step's leading pair) containing normalized states,
forcings and (optionally) observations.

``window()`` is the public API for inference: it returns the same tensor
bundle for an explicit frame range, so downstream code never touches the
dataset's internal memmaps or normalizer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config.schema import HuracanConfig
from .norm import Normalizer


class SphereDataset(Dataset):
    def __init__(
        self,
        cfg: HuracanConfig,
        split: str,
        rollout: int = 0,
        data_dir: str | Path | None = None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        # frames consumed by an unroll of ``rollout`` calls (leading pair
        # + one frame per emitted output); single source of truth is
        # huracan.time.CallSemantics
        self.frame_window = cfg.call_semantics().window_frames(rollout)
        root = Path(data_dir) if data_dir else Path(cfg.data.root) / cfg.data.source
        self.norm = Normalizer.load(root / "norm_state.json")

        period = getattr(cfg.data, f"{split}_period")
        self.state = np.load(root / "state.npy", mmap_mode="r")
        self.forcing = np.load(root / "forcing.npy", mmap_mode="r")
        self.obs_enabled = cfg.data.obs.enabled and (root / "obs.npy").exists()
        if cfg.data.obs.enabled and not self.obs_enabled:
            raise FileNotFoundError(
                f"{root / 'obs.npy'} missing: run scripts/prepare_obs.py "
                f"(or scripts/gen synthetic) before obs-enabled training"
            )
        self.obs = (
            np.load(root / "obs.npy", mmap_mode="r") if self.obs_enabled else None
        )

        t0_lo = max(1, period[0])
        t0_hi = min(period[1], self.state.shape[0] - self.frame_window)
        if t0_hi <= t0_lo:
            raise ValueError(
                f"empty {split} period: [{t0_lo}, {t0_hi}) "
                f"(T={self.state.shape[0]}, rollout={rollout})"
            )
        self.valid_t0 = np.arange(t0_lo, t0_hi)

        self.mean = torch.from_numpy(self.norm.mean)
        self.std = torch.from_numpy(self.norm.std)

    # -- public window API (training __getitem__ and inference share this) --
    def window(self, t0: int, frames: int) -> dict:
        """Normalized tensor bundle covering ``state[t0-1 : t0-1+frames]``.

        Returns {"state", "forcing", "obs"} tensors of (frames, npix, C);
        obs has zero channels when observations are disabled.
        """
        sl = slice(t0 - 1, t0 - 1 + frames)
        state = self._norm(self.state[sl])
        forc = torch.from_numpy(np.asarray(self.forcing[sl], dtype=np.float32))
        if self.obs_enabled:
            obs = torch.from_numpy(np.asarray(self.obs[sl], dtype=np.float32))
        else:
            obs = torch.zeros(state.shape[0], state.shape[1], 0)
        return {"state": state, "forcing": forc, "obs": obs}

    @property
    def first_t0(self) -> int:
        return int(self.valid_t0[0])

    def __len__(self):
        return len(self.valid_t0)

    def _norm(self, x: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)) - self.mean
        ) / self.std

    def __getitem__(self, i: int):
        t0 = int(self.valid_t0[i])
        out = self.window(t0, self.frame_window)
        out["t0"] = t0
        return out


def make_loader(
    cfg: HuracanConfig,
    split: str,
    rollout: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    data_dir=None,
):
    ds = SphereDataset(cfg, split, rollout, data_dir)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
