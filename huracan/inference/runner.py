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

    def run(
        self,
        window: dict,
        *,
        t0: int,
        members: int,
        spinup: int,
        lead: int,
        collect_spinup: bool = False,
        ic_noise: float = 0.0,
        grid_hw: tuple[int, int] | None = None,
    ):
        state = window["state"].to(self.device)
        forcing = window["forcing"].to(self.device)
        obs = window["obs"].to(self.device)
        if obs.shape[-1] == 0:
            obs = None

        opc = self.fcst.outputs_per_call
        out = spinup_and_forecast(
            self.assim,
            self.fcst,
            state,
            forcing,
            obs,
            members=members,
            spinup=spinup,
            lead=lead,
            collect_spinup=collect_spinup,
            ic_noise=ic_noise,
            grid_hw=grid_hw,
        )
        spinup_mean = out[2] if collect_spinup else None
        spinup_end, preds = out[0], out[1]
        s0 = opc * spinup + 2  # first forecast-truth frame
        truth = state[s0 : s0 + opc * lead].cpu()
        res = {
            "preds": preds,  # (M, opc*lead, npix, C) normalized 6-hourly fields
            "truth": truth,  # (opc*lead, npix, C)
            "spinup_end": spinup_end,  # (M, npix, C)
            "t0": t0,
            "members": members,
            "spinup_calls": spinup,  # AR calls, opc*6 h each
            "lead_calls": lead,
        }
        # P0: persistence baseline initialization = the ERA5 field at the
        # forecast start (frame s0-1, the analysis actually available when
        # the forecast is issued).  The old script persisted ``truth[0]`` --
        # the first *future* validation field, i.e. a baseline built with
        # information from the verification period.
        res["persistence_init"] = state[s0 - 1].cpu()
        if collect_spinup:
            res["spinup_mean"] = spinup_mean  # (opc*spinup, npix, C) ens mean
        return res


def run_eda(
    cfg: HuracanConfig,
    workdir: str | Path,
    device: str | None = None,
    start_index: int | None = None,
    data_dir: str | Path | None = None,
    split: str = "val",
    start_indices: list[int] | None = None,
    ic_noise: float = 0.0,
    collect_spinup: bool = False,
) -> dict | list[dict]:
    """EDA evaluation entry point.

    P0 changes:
    * ``split`` selects the evaluation period explicitly (was hard-wired
      to "val"); every start point's full window (spin-up + lead) is
      verified to lie inside the split -- no silent reads across the
      boundary.
    * ``start_indices`` evaluates several start points and returns a list
      of per-start results (single start keeps the original dict return).
    * ``ic_noise`` perturbs each member's initial pair with spectrally
      smooth noise (the OSSE regime; 0 keeps the original perfect start).

    Spin-up and lead time are calendar-driven via ``cfg.call_semantics()``:
    at outputs_per_call=1 one call advances 6 h (60 d spin-up = 240 calls).
    """
    workdir = Path(workdir)
    ecfg = cfg.eval
    sem = cfg.call_semantics()
    S, F = cfg.spinup_calls(), cfg.lead_calls()

    runner = EdaRunner.from_checkpoints(cfg, workdir, device=device)
    ds = SphereDataset(cfg, split, rollout=S + F, data_dir=data_dir)
    if start_indices is None:
        t0s = [start_index if start_index is not None else ds.first_t0]
    else:
        t0s = list(start_indices)
    for t0 in t0s:
        if t0 not in set(ds.valid_t0.tolist()):
            raise ValueError(
                f"start index {t0} outside the legal {split} window range "
                f"[{ds.valid_t0[0]}, {ds.valid_t0[-1]}] for a "
                f"{sem.window_frames(S + F)}-frame spin-up+lead window"
            )

    results = []
    for t0 in t0s:
        window = ds.window(t0, sem.window_frames(S + F))
        window["obs"] = window["obs"][: sem.obs_frames(S)]  # spin-up only
        res = runner.run(
            window,
            t0=t0,
            members=ecfg.members,
            spinup=S,
            lead=F,
            collect_spinup=collect_spinup,
            ic_noise=ic_noise,
            grid_hw=(cfg.grid.nlat, cfg.grid.nlon),
        )
        res["split"] = split
        res["spinup_days"] = ecfg.spinup_days
        res["lead_days"] = ecfg.lead_days
        results.append(res)
    if len(results) == 1:
        torch.save(results[0], workdir / "eda_result.pt")
        return results[0]
    for i, res in enumerate(results):
        torch.save(res, workdir / f"eda_result_{res['split']}_{i}.pt")
    return results
