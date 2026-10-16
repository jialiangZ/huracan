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
        stride: int = 1,
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
                f"before obs-enabled training"
            )
        self.obs = (
            np.load(root / "obs.npy", mmap_mode="r") if self.obs_enabled else None
        )
        self.obs_mask = (
            np.load(root / "obs_mask.npy", mmap_mode="r")
            if self.obs_enabled and (root / "obs_mask.npy").exists()
            else None
        )

        # P0 fix: the *complete* window [t0-1, t0-1+frame_window) must stay
        # inside the split (and the file).  The old code capped only t0, so
        # the last training windows read past the split boundary into
        # validation frames (single-output L=8 calls: window [4378, 4388)
        # with a train end index of 4380).  Window w belongs to the split
        # iff period[0] <= t0-1 and t0-1+frame_window <= period[1].
        t0_lo = max(period[0] + 1, 1)
        end = min(period[1], self.state.shape[0]) - self.frame_window + 2
        t0_hi = end  # arange exclusive: max t0 = end-1 -> window end = period end
        if t0_hi <= t0_lo:
            raise ValueError(
                f"empty {split} period: [{t0_lo}, {t0_hi}) "
                f"(T={self.state.shape[0]}, window={self.frame_window})"
            )
        self.valid_t0 = np.arange(t0_lo, t0_hi)
        if stride > 1:
            self.valid_t0 = self.valid_t0[::stride]
        self.split_period = tuple(period)

        self.mean = torch.from_numpy(self.norm.mean)
        self.std = torch.from_numpy(self.norm.std)

    # -- public window API (training __getitem__ and inference share this) --
    def window(self, t0: int, frames: int, include_mask: bool = False) -> dict:
        """Normalized tensor bundle covering ``state[t0-1 : t0-1+frames]``.

        Returns {"state", "forcing", "obs"} tensors of (frames, npix, C);
        obs has zero channels when observations are disabled.  With
        ``include_mask`` (obs bundle with an obs_mask.npy sidecar, P0), the
        dict also carries "obs_mask" (frames, npix, C) bool -- validity from
        the report counts, not from value != 0."""
        sl = slice(t0 - 1, t0 - 1 + frames)
        state = self._norm(self.state[sl])
        forc = torch.from_numpy(np.asarray(self.forcing[sl], dtype=np.float32))
        out = {"state": state, "forcing": forc}
        if self.obs_enabled:
            out["obs"] = torch.from_numpy(np.asarray(self.obs[sl], dtype=np.float32))
            if include_mask and self.obs_mask is not None:
                packed = self.obs_mask[sl]  # (frames, npix, ceil(C/8)) uint8
                C = self.obs.shape[-1]
                m = np.unpackbits(packed, axis=-1, count=C)
                out["obs_mask"] = torch.from_numpy(m.astype(bool))
        else:
            out["obs"] = torch.zeros(state.shape[0], state.shape[1], 0)
        return out

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
    stride: int = 1,
):
    ds = SphereDataset(cfg, split, rollout, data_dir, stride=stride)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
