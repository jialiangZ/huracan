"""Phase-specific callbacks: weight chaining, step budget, checkpointing.

``PhaseInitCallback``  -- on_train_start, load the previous phase's weights
                          (obs encoder columns stay randomly initialized,
                          paper Sec 2.3).
``PhaseRunCallback``   -- stop at exactly ``pc.steps`` optimizer iterations,
                          save ``ckpt_{phase}.pt`` periodically and at the
                          end, write ``losses_{phase}.json``; with a
                          validation stream it also maintains
                          ``ckpt_{phase}_best.pt`` by held-out val loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import numpy as np

from ..config.schema import HuracanConfig
from ..core import io
from ..core.model import load_with_new_columns


class PhaseInitCallback(L.Callback):
    def __init__(self, cfg: HuracanConfig, phase: str, workdir: Path):
        super().__init__()
        self.cfg = cfg
        self.phase = phase
        self.workdir = Path(workdir)

    def on_train_start(self, trainer, pl_module):
        pc = self.cfg.train.phases[self.phase]
        prev = pc.init
        if not prev:
            return
        path = io.phase_path(self.workdir, prev)
        ck = io.load_phase(self.workdir, prev)
        src = ck["models"].get("fcst", None) or ck["models"]["assim"]

        if self.phase in ("assim", "joint"):
            # assim encoder gains obs columns; copy overlapping columns only
            load_with_new_columns(pl_module.assim, src)
        elif self.phase in ("stochastic", "fcst_ar", "hybrid"):
            pl_module.fcst.load_state_dict(src)
        if self.phase == "joint":
            fprev = pc.fcst_init or "fcst_ar"
            fpath = io.phase_path(self.workdir, fprev)
            ckf = io.load_phase(self.workdir, fprev)
            pl_module.fcst.load_state_dict(ckf["models"]["fcst"])
        print(f"[{self.phase}] initialized from {path.name}", flush=True)


class PhaseRunCallback(L.Callback):
    def __init__(
        self,
        cfg: HuracanConfig,
        phase: str,
        workdir: Path,
        ckpt_every: int = 2000,
    ):
        super().__init__()
        self.pc = cfg.train.phases[phase]
        self.phase = phase
        self.workdir = Path(workdir)
        self.ckpt_every = ckpt_every
        self.losses: list[float] = []
        self.best_val = float("inf")
        self.val_history: list[dict] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.losses = pl_module._losses
        step = pl_module._step
        done = step >= self.pc.steps
        if done:
            trainer.should_stop = True
        if step % self.ckpt_every == 0 or done:
            io.save_phase(self.workdir, self.phase, step, pl_module.state_dicts())

    def on_validation_end(self, trainer, pl_module):
        """P0: keep the best checkpoint by held-out validation loss
        (``ckpt_{phase}_best.pt``); the last-step checkpoint keeps its
        original name and role.  Test data never touches this path."""
        if not trainer.sanity_checking:
            pl_module._val_losses = getattr(pl_module, "_val_losses", [])
            val = (
                float(trainer.callback_metrics.get("val_loss", float("inf")))
                if not pl_module._val_losses
                else float(np.mean(pl_module._val_losses))
            )
            pl_module._val_losses.clear()
            self.val_history.append({"step": pl_module._step, "val_loss": val})
            print(
                f"  [val] step {pl_module._step} val_loss {val:.5f}", flush=True
            )
            if val < self.best_val:
                self.best_val = val
                io.save_phase(
                    self.workdir,
                    f"{self.phase}_best",
                    pl_module._step,
                    pl_module.state_dicts(),
                )

    def on_train_end(self, trainer, pl_module):
        io.save_phase(
            self.workdir, self.phase, pl_module._step, pl_module.state_dicts()
        )
        self.losses = pl_module._losses
        with open(self.workdir / f"losses_{self.phase}.json", "w") as f:
            json.dump(self.losses, f)
        if self.val_history:
            with open(self.workdir / f"val_{self.phase}.json", "w") as f:
                json.dump(
                    {"best": self.best_val, "history": self.val_history}, f, indent=1
                )
