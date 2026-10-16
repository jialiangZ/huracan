#!/usr/bin/env python3
"""R0 baseline metrics on the validation year (2012) -- v2 (2026-10-16).

Changes vs v1 (external review F6):
  * THREE persistence references, clearly separated by information budget:
      - sys_persist   : each member's OWN release state (spin-up end)
                       persisted through the forecast window -> an honest
                       same-information ensemble baseline;
      - sys_mean_pers : the deterministic variant (ensemble-mean state);
      - era5_persist  : the ERA5 analysis at release time -- an
                       ERA5-initialized reference with MORE information
                       than the system owns; reported separately, never
                       as the primary skill denominator.
  * dual lead-time convention: ``lead_from_release`` (6h at the first
    scored frame) and ``lead_from_cutoff`` (12h there; the last absorbed
    observation is one frame before the release state).
  * per-start metrics are saved (not only 4-window means), with cutoff /
    release frame indices and checkpoint hashes; output is a real .npz.

Usage:
  python scripts/eval_r0.py --config configs/bsef_r0.yaml \
      --workdir runs/bsef/r0 --out runs/bsef/r0_eval_v2 \
      --starts 2012-01-15 2012-04-15 2012-07-15 2012-10-15
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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

METRICS = ("crps", "crps_fair", "essr", "essr_mean_std", "rmse")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


@torch.no_grad()
def run_control(cfg, assim, fcst, ds, t0, control, device):
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
    spin_end, preds = spinup_and_forecast(
        assim_in, fcst_in, state, forcing, obs_use,
        members=cfg.eval.members, spinup=S, lead=F,
        collect_spinup=False,
    )
    release = 1 * S + 1   # assim output frame produced from the last obs
    s0 = release + 1      # first forecast frame (scored window starts)
    truth = state[s0 : s0 + F].cpu().numpy()          # (F, N, C)
    preds = preds.numpy()                              # (M, F, N, C)
    spin_end = spin_end.numpy()                        # (M, N, C) @ release
    era5_rel = state[release].cpu().numpy()            # (N, C) ERA5 @ release
    M = preds.shape[0]
    sys_persist = np.repeat(spin_end[:, None], F, axis=1)      # (M,F,N,C)
    era5_persist = np.repeat(era5_rel[None], F, axis=0)[None]  # (1,F,N,C)
    sys_mean = np.repeat(spin_end.mean(0)[None], F, axis=0)[None]
    w = grid_from_cfg(cfg.grid).pixel_weights
    out = {
        "release_frame": release,
        "cutoff_frame": release - 1,
        "lead_from_release_h": np.arange(1, F + 1) * 6,
        "lead_from_cutoff_h": np.arange(2, F + 2) * 6,
    }
    # the release state IS the system's 6h-from-cutoff prediction (it was
    # produced by the last assim call absorbing obs <= cutoff): score it
    # against truth at release -- the v9 "same-cutoff 6h" headline field.
    truth_c6 = state[release : release + 1].cpu().numpy()      # (1,N,C)
    ens_c6 = spin_end[:, None]                                  # (M,1,N,C)
    out["cutoff6h_crps"] = np.atleast_1d(crps_ensemble(ens_c6, truth_c6, w))[0]
    out["cutoff6h_crps_fair"] = np.atleast_1d(
        crps_fair(ens_c6, truth_c6, w))[0]
    out["cutoff6h_rmse"] = np.atleast_1d(
        rmse_ensemble_mean(ens_c6, truth_c6, w))[0]
    for name, ens in (
        ("", preds),
        ("sys_persist_", sys_persist),
        ("era5_persist_", era5_persist),
    ):
        out[f"{name}crps"] = crps_ensemble(ens, truth, w)
        out[f"{name}crps_fair"] = crps_fair(ens, truth, w)
        out[f"{name}rmse"] = rmse_ensemble_mean(ens, truth, w)
    out["sys_mean_persist_rmse"] = rmse_ensemble_mean(sys_mean, truth, w)
    out["essr"] = essr(preds, truth, w, mode="rms")
    out["essr_mean_std"] = essr(preds, truth, w, mode="mean_std")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", default="runs/bsef/r0")
    ap.add_argument("--out", default="runs/bsef/r0_eval_v2")
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
    ap_, fp_ = wd / f"ckpt_assim{sfx}.pt", wd / f"ckpt_fcst_ar{sfx}.pt"
    assim.load_state_dict(io.load_phase(wd, f"assim{sfx}")["models"]["assim"])
    fcst.load_state_dict(io.load_phase(wd, f"fcst_ar{sfx}")["models"]["fcst"])
    assim, fcst = assim.to(device).eval(), fcst.to(device).eval()

    S, F = cfg.spinup_calls(), cfg.lead_calls()
    ds = SphereDataset(cfg, "val", rollout=S + F)
    times = [dt.datetime.fromisoformat(s)
             for s in json.load(open(Path(cfg.data.root) / cfg.data.source
                                     / "times.json"))]
    names = cfg.var_names()

    results, per_start = {}, {}
    keys = [f"{p}{m}" for p in ("", "sys_persist_", "era5_persist_")
            for m in ("crps", "crps_fair", "rmse")] + \
           ["sys_mean_persist_rmse", "essr", "essr_mean_std",
            "cutoff6h_crps", "cutoff6h_crps_fair", "cutoff6h_rmse"]
    for control in ("obs", "zero-obs", "fcst"):
        agg = {k: [] for k in keys}
        for start in args.starts:
            sd = dt.date.fromisoformat(start)
            t0 = next(i for i, t in enumerate(times)
                      if t.date() == sd and t.hour == 0)
            assert t0 in set(ds.valid_t0.tolist()), f"{start} t0={t0} illegal"
            r = run_control(cfg, assim, fcst, ds, t0, control, device)
            for k in keys:
                agg[k].append(np.asarray(r[k]))
                per_start[(control, start, k)] = np.asarray(r[k])
            print(f"  [{control}] {start} t0={t0} done", flush=True)
        results[control] = {k: np.mean(np.stack(v), 0) for k, v in agg.items()}
        cr = results[control]["crps"].mean()
        for ref in ("sys_persist_crps", "era5_persist_crps"):
            print(f"{control}: CRPS {cr:.4f} vs {ref} "
                  f"{results[control][ref].mean():.4f} "
                  f"(skill {1 - cr / results[control][ref].mean():+.3f})",
                  flush=True)

    starts_arr = [s.replace("-", "") for s in args.starts]
    flat = {
        f"ps_{c}_{sd}_{k}": v
        for (c, s, k), v in per_start.items()
        for sd in [s.replace("-", "")]
    }
    np.savez_compressed(
        out / "r0_eval.npz",
        names=np.array(names), controls=np.array(list(results.keys())),
        starts=np.array(starts_arr),
        ckpt_assim_sha=_sha(ap_), ckpt_fcst_ar_sha=_sha(fp_),
        lead_from_release_h=np.arange(1, F + 1) * 6,
        lead_from_cutoff_h=np.arange(2, F + 2) * 6,
        **{f"{c}_{k}": results[c][k] for c in results for k in keys},
        **flat,
    )
    print(f"\nR0 evaluation -> {out / 'r0_eval.npz'}")


if __name__ == "__main__":
    main()
