"""Phased trainer implementing the Huracan training schedule (paper Sec. 2.3,
Table 3):

  deterministic : forecast model, L1, no noise          (base model)
  stochastic    : + noise, CRPS objective, M=2 ensemble (stochastic base)
  fcst_ar       : 6-call autoregressive rollout finetune
  assim         : assimilation model (new obs encoder columns), 8-call rollout
  joint         : 8 assimilation calls + 8 forecast calls, both models trained
  hybrid        : optional forecast finetune on ECMWF analysis (pilot: ERA5)

Checkpoints chain automatically through each phase's `init` field.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from ..data.dataset import make_loader
from ..model import build_models
from .losses import crps_loss, l1_loss


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_with_new_columns(model: torch.nn.Module, state: dict) -> None:
    """Copy weights from a narrower encoder into a wider one; weight tensors
    whose shapes differ keep their fresh random initialization, except the
    first Linear of the encoder where only the new input *columns* are new
    (paper Sec. 2.3: 'except for the new columns in the encoder')."""
    own = model.state_dict()
    for k, v in state.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v
        elif (
            own[k].dim() == 2 and own[k].shape[0] == v.shape[0] and "embed.weight" in k
        ):
            own[k][:, : v.shape[1]] = v
    model.load_state_dict(own)


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


class HuracanTrainer:
    def __init__(
        self,
        cfg: dict,
        phase: str,
        device: str | None = None,
        workdir: str | None = None,
    ):
        self.cfg = cfg
        self.phase = phase
        self.pc = cfg["train"]["phases"][phase]
        self.out = Path(workdir or cfg["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg["seed"] + hash(phase) % 1000)
        np.random.seed(cfg["seed"])

        fcst, assim = build_models(cfg)
        self.models = {"fcst": fcst.to(self.device), "assim": assim.to(self.device)}
        self._init_from_previous()

        params = []
        self.optimizers = {}
        active = self._active_models()
        for name in active:
            opt = torch.optim.AdamW(
                self.models[name].parameters(), lr=self.pc["lr"], weight_decay=0.01
            )
            self.optimizers[name] = opt
            params += list(self.models[name].parameters())
        self.total_steps = self.pc["steps"]

    # -- checkpoint chaining ------------------------------------------------
    def _ckpt_path(self, phase: str) -> Path:
        return self.out / f"ckpt_{phase}.pt"

    def _active_models(self) -> list[str]:
        if self.phase in ("deterministic", "stochastic", "fcst_ar", "hybrid"):
            return ["fcst"]
        if self.phase == "assim":
            return ["assim"]
        if self.phase == "joint":
            return ["assim", "fcst"]
        raise ValueError(self.phase)

    def _init_from_previous(self):
        prev = self.pc.get("init")
        if not prev:
            return
        path = self._ckpt_path(prev)
        assert path.exists(), f"missing previous checkpoint {path}"
        ck = torch.load(path, map_location=self.device, weights_only=False)
        src = ck["models"]["fcst"] if "fcst" in ck["models"] else ck["models"]["assim"]
        if self.phase in ("assim", "joint"):
            # assim encoder gains obs columns; copy overlapping columns only
            load_with_new_columns(self.models["assim"], src)
        elif self.phase in ("stochastic", "fcst_ar", "hybrid"):
            # fcst <- fcst checkpoint: shapes match exactly
            self.models["fcst"].load_state_dict(src)
        if self.phase == "joint":
            fprev = self.pc.get("fcst_init", "fcst_ar")
            fpath = self._ckpt_path(fprev)
            assert fpath.exists(), f"joint phase needs {fpath} (run fcst_ar)"
            ckf = torch.load(fpath, map_location=self.device, weights_only=False)
            self.models["fcst"].load_state_dict(ckf["models"]["fcst"])
        print(f"[{self.phase}] initialized from {path.name}")

    # -- main loop -----------------------------------------------------------
    def run(self, log_every: int = 50, ckpt_every: int = 2000):
        pc = self.pc
        use_noise = pc.get("noise", False)
        loss_name = pc.get("loss", "l1")
        if self.phase == "joint":
            n_calls = pc["rollout_assim"] + pc["rollout_fcst"]
            split_assim = pc["rollout_assim"]
        else:
            n_calls = max(1, pc.get("rollout", 1))
            split_assim = n_calls if self.phase == "assim" else 0
        obs_enabled = pc.get("obs", False)
        batch_size = pc.get("batch_size", self.cfg["train"]["batch_size"])

        loader = make_loader(
            self.cfg,
            "train",
            n_calls,
            batch_size,
            shuffle=True,
            num_workers=self.cfg["train"].get("num_workers", 2),
        )
        oname = self._active_models()[0]

        t_start = time.time()
        losses = []
        step = 0
        while step < self.total_steps:
            for batch in loader:
                if step >= self.total_steps:
                    break
                for opt in self.optimizers.values():
                    opt.zero_grad(set_to_none=True)

                preds, targets = self._forward_batch(
                    batch, n_calls, split_assim, obs_enabled, use_noise
                )

                if loss_name == "crps":
                    loss = crps_loss(preds, targets)
                else:
                    loss = l1_loss(preds.mean(dim=1), targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for opt in self.optimizers.values()
                        for pg in opt.param_groups
                        for p in pg["params"]
                    ],
                    self.cfg["train"]["grad_clip"],
                )
                # learning-rate schedule
                warm = int(self.total_steps * self.cfg["train"]["warmup_frac"])
                sched = cosine_with_warmup(step, self.total_steps, warm)
                if pc.get("schedule", "cosine") == "constant" and step >= warm:
                    sched = 1.0
                for opt in self.optimizers.values():
                    for pg in opt.param_groups:
                        pg["lr"] = pc["lr"] * sched
                for opt in self.optimizers.values():
                    opt.step()

                losses.append(float(loss.detach()))
                step += 1
                if step % log_every == 0 or step == 1:
                    rate = step / (time.time() - t_start)
                    print(
                        f"[{self.phase}] step {step}/{self.total_steps} "
                        f"loss {np.mean(losses[-log_every:]):.4f} "
                        f"lr {self.optimizers[oname].param_groups[0]['lr']:.2e} "
                        f"({rate:.2f} it/s)",
                        flush=True,
                    )
                if step % ckpt_every == 0 or step == self.total_steps:
                    self.save(step, losses)

        self.save(step, losses)
        return losses

    def _forward_batch(self, batch, n_calls, split_assim, obs_enabled, use_noise):
        """Run the unroll: calls < split_assim use the assimilation model
        (with observations), later calls use the forecast model."""
        state = batch["state"].to(self.device, non_blocking=True)
        forc = batch["forcing"].to(self.device, non_blocking=True)
        obs = batch["obs"].to(self.device, non_blocking=True) if obs_enabled else None
        B, nframes, npix, C = state.shape
        M = self.cfg["train"]["ensemble"] if use_noise else 1

        preds = []
        for m in range(M):
            x_prev, x_cur = state[:, 0], state[:, 1]
            outs = []
            for j in range(n_calls):
                assim_call = j < split_assim
                model = self.models["assim" if assim_call else "fcst"]
                z = (
                    torch.randn(
                        B, npix, model.noise.mlp[0].in_features, device=self.device
                    )
                    if use_noise
                    else None
                )
                kw = {}
                if assim_call and obs_enabled:
                    kw = dict(obs_tm1=obs[:, 2 * j], obs_t=obs[:, 2 * j + 1])
                x1, x2 = model.step(
                    x_prev, x_cur, forc[:, 2 * j], forc[:, 2 * j + 1], z=z, **kw
                )
                outs += [x1, x2]
                x_prev, x_cur = x1, x2
            preds.append(torch.stack(outs, dim=1))
        preds = torch.stack(preds, dim=1)
        return preds, state[:, 2:]

    def save(self, step: int, losses: list):
        ck = {
            "phase": self.phase,
            "step": step,
            "cfg": self.cfg,
            "models": {
                k: v.state_dict()
                for k, v in self.models.items()
                if k in self._active_models()
            },
            "losses": losses[-1000:],
        }
        tmp = self._ckpt_path(self.phase).with_suffix(".tmp")
        torch.save(ck, tmp)
        tmp.replace(self._ckpt_path(self.phase))
        with open(self.out / f"losses_{self.phase}.json", "w") as f:
            json.dump(losses, f)
