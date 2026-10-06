"""EDA cycling + end-to-end forecast inference (paper Sec. 3.1).

``EdaRunner`` owns trained models and runs the trajectory over an explicit
tensor window (obtained from ``SphereDataset.window``), so inference never
touches dataset internals.  ``run_eda`` keeps the original one-call entry
point: load checkpoints (joint > assim + fcst_ar fallback), build the
window, run, save ``eda_result.pt``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..config.schema import HuracanConfig
from ..core import io
from ..core.model import build_models
from ..core.rollout import spinup_and_forecast
from ..data.dataset import SphereDataset


class EdaRunner:
    """Runs the EDA experiment: spin-up cycling then forecast, per member."""

    def __init__(self, assim, fcst, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.assim = assim.to(self.device).eval()
        self.fcst = fcst.to(self.device).eval()

    @classmethod
    def from_checkpoints(
        cls,
        cfg: HuracanConfig,
        workdir: str | Path,
        device: str | None = None,
    ) -> "EdaRunner":
        """Build models and load the best available weights: the joint
        checkpoint if present, else assim + fcst_ar (original fallback)."""
        workdir = Path(workdir)
        f_model, a_model = build_models(cfg)

        if io.phase_path(workdir, "joint").exists():
            ck = io.load_phase(workdir, "joint")
            a_model.load_state_dict(ck["models"]["assim"])
            f_model.load_state_dict(ck["models"]["fcst"])
        else:
            ck = io.load_phase(workdir, "assim")
            a_model.load_state_dict(ck["models"]["assim"])
            fpath = io.phase_path(workdir, "fcst_ar")
            if fpath.exists():
                f_model.load_state_dict(
                    io.load_phase(workdir, "fcst_ar")["models"]["fcst"]
                )
        return cls(a_model, f_model, device=device)

    def run(self, window: dict, *, t0: int, members: int, spinup: int, lead: int):
        state = window["state"].to(self.device)
        forcing = window["forcing"].to(self.device)
        obs = window["obs"].to(self.device)
        if obs.shape[-1] == 0:
            obs = None

        spinup_end, preds = spinup_and_forecast(
            self.assim,
            self.fcst,
            state,
            forcing,
            obs,
            members=members,
            spinup=spinup,
            lead=lead,
        )
        truth = state[2 * spinup + 2 : 2 * spinup + 2 + 2 * lead].cpu()
        return {
            "preds": preds,  # (M, 2F, npix, C) normalized 6-hourly fields
            "truth": truth,  # (2F, npix, C)
            "spinup_end": spinup_end,  # (M, npix, C)
            "t0": t0,
            "members": members,
            "spinup_calls": spinup,   # AR calls, 12 h each
            "lead_calls": lead,
        }


def run_eda(
    cfg: HuracanConfig,
    workdir: str | Path,
    device: str | None = None,
    start_index: int | None = None,
    data_dir: str | Path | None = None,
) -> dict:
    """Full experiment entry point (keeps the original call signature).

    Spin-up and lead time are calendar-driven (guide Sec. 12.3): days from
    the config are converted to AR calls at 12 h per call, so "3 months"
    can never silently become a wrong loop count again."""
    workdir = Path(workdir)
    ecfg = cfg.eval
    S, F = ecfg.spinup_calls(), ecfg.lead_calls()

    runner = EdaRunner.from_checkpoints(cfg, workdir, device=device)
    ds = SphereDataset(cfg, "val", rollout=S + F, data_dir=data_dir)
    t0 = start_index if start_index is not None else ds.first_t0

    window = ds.window(t0, 2 * (S + F) + 2)
    window["obs"] = window["obs"][: 2 * S]  # observations cover spin-up only

    res = runner.run(
        window,
        t0=t0,
        members=ecfg.members,
        spinup=S,
        lead=F,
    )
    res["spinup_days"] = ecfg.spinup_days
    res["lead_days"] = ecfg.lead_days
    torch.save(res, workdir / "eda_result.pt")
    return res
