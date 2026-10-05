"""Phase orchestration: fit one phase of the paper schedule (Table 3).

``fit_phase`` is the single entry point used by the CLI: it seeds, builds
the LightningModule + DataModule + callbacks, runs ``trainer.fit`` until
the phase's step budget is exhausted and returns the loss history.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger

from ..config.schema import HuracanConfig
from .callbacks import PhaseInitCallback, PhaseRunCallback
from .datamodule import HuracanDataModule
from .module import HuracanLitModule

PHASE_ORDER = ["deterministic", "stochastic", "fcst_ar", "assim", "joint"]
ALL_PHASES = PHASE_ORDER + ["hybrid"]  # hybrid needs ECMWF data; run explicitly


def _parse_device(device: str):
    d = device.lower()
    if d == "cpu":
        return "cpu", 1
    if d.startswith("cuda"):
        if ":" in d:
            return "gpu", [int(d.split(":")[1])]
        return "gpu", 1
    return "auto", "auto"


def fit_phase(
    cfg: HuracanConfig,
    phase: str,
    *,
    device: str | None = None,
    workdir: str | Path | None = None,
    data_dir: str | Path | None = None,
    ckpt_every: int = 2000,
    log_every: int = 50,
) -> dict:
    """Train one phase; returns {"losses": [...]}."""
    if phase not in ALL_PHASES:
        raise ValueError(f"unknown phase {phase!r}; valid: {ALL_PHASES}")
    workdir = Path(workdir or cfg.output_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed + zlib.crc32(phase.encode()) % 1000)
    np.random.seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    dev = device or cfg.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"
    accelerator, devices = _parse_device(dev)

    module = HuracanLitModule(cfg, phase)
    dm = HuracanDataModule(cfg, phase, data_dir=data_dir)
    run_cb = PhaseRunCallback(cfg, phase, workdir, ckpt_every=ckpt_every)
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=10_000_000,  # PhaseRunCallback stops at pc.steps
        enable_checkpointing=False,
        callbacks=[PhaseInitCallback(cfg, phase, workdir), run_cb],
        logger=CSVLogger(save_dir=str(workdir), name="logs", version=phase),
        num_sanity_val_steps=0,
        log_every_n_steps=log_every,
        default_root_dir=str(workdir),
    )
    trainer.fit(module, datamodule=dm)
    return {"losses": run_cb.losses}
