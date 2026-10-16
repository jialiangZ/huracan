"""LightningDataModule wrapping the framework-free ``SphereDataset``.

The only job of this class is adaptation: rollout length and batch size
come from the phase config; normalization and block slicing stay in
``huracan.data.dataset`` (which has no Lightning dependency).
"""

from __future__ import annotations

from pathlib import Path

import lightning as L

from ..config.schema import HuracanConfig
from ..data.dataset import make_loader


class HuracanDataModule(L.LightningDataModule):
    def __init__(
        self, cfg: HuracanConfig, phase: str, data_dir: str | Path | None = None
    ):
        super().__init__()
        self.cfg = cfg
        self.phase = phase
        pc = cfg.train.phases[phase]
        self.rollout = pc.n_calls()
        self.batch_size = pc.batch_size or cfg.train.batch_size
        self.val_batch_size = cfg.train.val_batch_size or self.batch_size
        self.val_stride = cfg.train.val_stride
        self.num_workers = cfg.train.num_workers
        self.data_dir = data_dir

    def train_dataloader(self):
        return make_loader(
            self.cfg,
            "train",
            self.rollout,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            data_dir=self.data_dir,
        )

    def val_dataloader(self):
        """P0: held-out validation stream.  Deterministic stride subsampling
        (``train.val_stride``) spans the whole validation period across all
        seasons without loading every window."""
        return make_loader(
            self.cfg,
            "val",
            self.rollout,
            self.val_batch_size,
            shuffle=False,
            num_workers=2,
            data_dir=self.data_dir,
            stride=self.val_stride,
        )
