#!/usr/bin/env python3
"""P1 deliverable: R0 baseline metrics on the validation year (2012).

Three controls per start time, same seeds/spin-up/lead (4090 plan §3):
  obs      -- normal assimilation (the R0 model)
  zero-obs -- obs channels zeroed (architecture identical, no obs info)
  fcst     -- the FORECAST model cycling for the whole spin-up (a different
              control: zero-obs on the assim net is NOT the fcst net)

Metrics: empirical CRPS (main), fair CRPS (training estimator, reported
separately), ESSR (RMS spread/RMSE + mean-std variant), RMSE(ens mean),
persistence baseline = ERA5 field at forecast initialization.

Checkpoints: --ckpt best|load (default best = selected by held-out
validation loss; test data never used for selection).

Usage:
  python scripts/eval_r0.py --config configs/bsef_r0.yaml \
      --workdir runs/bsef/r0 --out runs/bsef/r0_eval \
      --starts 2012-01-15 2012-04-15 2012-07-15 2012-10-15
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from huracan.config import load_config
from huracan.core import io
from huracan.core.grid import grid_from_cfg
from huracan.core.metrics import (
    crps_ensemble,
    crps_fair,
    essr,
    rmse_ensemble_mean,
)
from huracan.core.model import build_models
from huracan.core.rollout import spinup_and_forecast
from huracan.data.dataset import SphereDataset


@torch.no_grad()
def run_control(cfg, assim, fcst, ds, t0, control, device, kind):
    sem = cfg.call_semantics()
    S, F = cfg.spinup_calls(), cfg.lead_calls()
    window = ds.window(t0, sem.window_frames(S + F))
    state = window["state"].to(device)
    forcing = window["forcing"].to(device)
    obs_full = window["obs"].to(device)
    if control == "zero-obs":
        obs_use = torch.zeros_like(obs_full[: sem.obs_frames(S)])
        assim_in, fcst_in = assim, fcst
    elif control == "fcst":
        obs_use = None
        assim_in, fcst_in = fcst, fcst  # forecast model cycles, no obs
    else:
        obs_use = obs_full[: sem.obs_frames(S)]
        assim_in, fcst_in = assim, fcst
    out = spinup_and_forecast(
        assim_in, fcst_in, state, forcing, obs_use,
        members=cfg.eval.members, spinup=S, lead=F,
        collect_spinup=False,
    )
    s0 = 1 * S + 2  # opc=1
    truth = state[s0 : s0 + 1 * F].cpu()
    persist = state[s0 - 1].cpu()
    preds = out[1].numpy()
    truth = truth.numpy()
    persist = persist.numpy()
    w = grid_from_cfg(cfg.grid).pixel_weights
    return {
        "crps": crps_ensemble(preds, truth, w),
        "crps_fair": crps_fair(preds, truth, w),
        "essr": essr(preds, truth, w, mode="rms"),
        "essr_mean_std": essr(preds, truth, w, mode="mean_std"),
        "rmse": rmse_ensemble_mean(preds, truth, w),
        "crps_persist": crps_ensemble(
            np.repeat(np.repeat(persist[None], truth.shape[0], 0)[None],
                      preds.shape[0], 0), truth, w
        ),
        "rmse_persist": rmse_ensemble_mean(
            np.repeat(persist[None], truth.shape[0], 0)[None], truth, w
        ),
        "spinup_end": out[0].numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", default="runs/bsef/r0")
    ap.add_argument("--out", default="runs/bsef/r0_eval")
    ap.add_argument("--starts", nargs="+",
                    default=["2012-01-15", "2012-04-15", "2012-07-15",
                             "2012-10-15"])
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg.device
    wd = Path(args.workdir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sfx = "_best" if args.ckpt == "best" else ""
    fcst, assim = build_models(cfg)
    assim.load_state_dict(io.load_phase(wd, f"assim{sfx}")["models"]["assim"])
    fcst.load_state_dict(io.load_phase(wd, f"fcst_ar{sfx}")["models"]["fcst"])
    assim, fcst = assim.to(device).eval(), fcst.to(device).eval()

    S, F = cfg.spinup_calls(), cfg.lead_calls()
    ds = SphereDataset(cfg, "val", rollout=S + F)
    times = [dt.datetime.fromisoformat(s)
             for s in json.load(open(Path(cfg.data.root) / cfg.data.source
                                     / "times.json"))]
    names = cfg.var_names()

    results = {}
    for control in ("obs", "zero-obs", "fcst"):
        agg = {}
        for start in args.starts:
            sd = dt.date.fromisoformat(start)
            t0 = next(i for i, t in enumerate(times)
                      if t.date() == sd and t.hour == 0)
            assert t0 in set(ds.valid_t0.tolist()), f"{start} t0={t0} illegal"
            r = run_control(cfg, assim, fcst, ds, t0, control, device,
                            args.ckpt)
            for k in ("crps", "crps_fair", "essr", "essr_mean_std", "rmse",
                      "crps_persist", "rmse_persist"):
                agg.setdefault(k, []).append(r[k])
            agg.setdefault("start_t0", []).append(t0)
            print(f"  [{control}] {start} t0={t0} done", flush=True)
        results[control] = {
            "start_t0": agg["start_t0"],
            **{
                m: np.mean(np.stack(agg[m]), 0)
                for m in ("crps", "crps_fair", "essr", "essr_mean_std",
                          "rmse", "crps_persist", "rmse_persist")
            },
        }
        cr = results[control]["crps"].mean()
        crp = results[control]["crps_persist"].mean()
        print(f"{control}: mean empirical CRPS {cr:.4f} vs persistence "
              f"{crp:.4f} (skill {1 - cr/crp:+.3f})", flush=True)

    with open(out / "r0_eval.json", "wb") as f:
        np.savez_compressed(
            f,
            names=np.array(names),
            controls=np.array(list(results.keys())),
            start_t0=np.array(results["obs"]["start_t0"]),
            **{f"{c}_{m}": results[c][m]
               for c in results
               for m in ("crps", "crps_fair", "essr", "essr_mean_std", "rmse",
                         "crps_persist", "rmse_persist")},
        )
    print(f"\nR0 evaluation -> {out / 'r0_eval.json'}")


if __name__ == "__main__":
    main()