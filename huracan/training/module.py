"""HuracanLitModule: the Lightning adapter around the framework-free core.

Uniformly uses *manual optimization* so that single-optimizer phases and the
dual-optimizer joint phase (assim + fcst trained together, paper Sec 2.3)
share one code path.  The unroll itself lives in ``core.rollout``; this
module only owns optimization, scheduling, logging and the step budget.
"""

from __future__ import annotations

import lightning as L
import torch

from ..config.schema import HuracanConfig
from ..core import losses as L2
from ..core.model import build_models
from ..core.rollout import unroll
from ..core.sht import SHTBackend
from .schedule import lr_factor

#: which models each phase trains (mirrors the original trainer exactly)
ACTIVE_MODELS: dict[str, tuple[str, ...]] = {
    "deterministic": ("fcst",),
    "stochastic": ("fcst",),
    "fcst_ar": ("fcst",),
    "assim": ("assim",),
    "joint": ("assim", "fcst"),
    "hybrid": ("fcst",),
}


class HuracanLitModule(L.LightningModule):
    def __init__(
        self,
        cfg: HuracanConfig,
        phase: str,
        sht_factory=None,
    ):
        super().__init__()
        if phase not in ACTIVE_MODELS:
            raise ValueError(f"unknown phase {phase!r}")
        self.cfg = cfg
        self.phase = phase
        self.pc = cfg.train.phases[phase]
        self.save_hyperparameters(ignore=["sht_factory"])

        fcst, assim = build_models(cfg, sht_factory)
        self.fcst = fcst
        self.assim = assim
        self._active = ACTIVE_MODELS[phase]

        # P0: area weights for the Gauss grid loss (None -> plain mean;
        # the planar square gets uniform weights)
        from ..core.grid import grid_from_cfg

        g = grid_from_cfg(cfg.grid)
        w = torch.as_tensor(g.pixel_weights, dtype=torch.float32)
        if cfg.grid.type == "planar":
            w = None  # uniform-area square
        else:
            w = w / w.mean()  # keep the loss scale comparable to unweighted
        self.pixel_w = w

        self.automatic_optimization = False
        self._step = 0
        self._warm = int(self.pc.steps * cfg.train.warmup_frac)
        self._losses: list[float] = []
        self._val_losses: list[float] = []

    # -- helpers -----------------------------------------------------------
    def models_dict(self) -> dict:
        return {"fcst": self.fcst, "assim": self.assim}

    def state_dicts(self) -> dict:
        """State dicts of the models this phase trains (checkpoint format)."""
        return {n: self.models_dict()[n].state_dict() for n in self._active}

    def _compute_loss(self, batch):
        state, forc = batch["state"], batch["forcing"]
        obs = batch["obs"] if self.pc.obs else None
        w = self.pixel_w
        if w is not None:
            w = w.to(state.device)

        preds, targets = unroll(
            self.models_dict(),
            state,
            forc,
            obs,
            n_calls=self.pc.n_calls(),
            split_assim=self.pc.split_assim(),
            ensemble=self.cfg.train.ensemble,
            use_noise=self.pc.noise,
            ic_noise=self.pc.ic_noise,
            grid_hw=(self.cfg.grid.nlat, self.cfg.grid.nlon),
        )
        if self.pc.loss == "crps":
            loss = L2.crps_loss(preds, targets, w)
        else:
            loss = L2.l1_loss(preds.mean(dim=1), targets, w)
        return loss

    # -- Lightning hooks -----------------------------------------------------
    def configure_optimizers(self):
        # one AdamW per active model: joint trains assim and fcst with
        # separate optimizers (two param groups would couple their schedules)
        return [
            torch.optim.AdamW(
                self.models_dict()[n].parameters(),
                lr=self.pc.lr,
                weight_decay=0.01,
            )
            for n in self._active
        ]

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        for o in opts:
            o.zero_grad(set_to_none=True)

        loss = self._compute_loss(batch)

        self.manual_backward(loss)
        params = [
            p
            for n in self._active
            for p in self.models_dict()[n].parameters()
            if p.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(params, self.cfg.train.grad_clip)

        factor = lr_factor(self.pc.schedule, self._step, self.pc.steps, self._warm)
        for o in opts:
            for pg in o.param_groups:
                pg["lr"] = self.pc.lr * factor
        for o in opts:
            o.step()
        self._step += 1

        self._losses.append(float(loss.detach()))
        lr_now = self.pc.lr * factor
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("lr", lr_now, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """P0: held-out validation loss (same objective as the phase's
        training loss) on windows that never overlap the training period.
        Model selection uses this metric only; the test split stays
        untouched during development."""
        with torch.no_grad():
            loss = self._compute_loss(batch)
        self._val_losses.append(float(loss))
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=False)
        return loss
