#!/usr/bin/env python3
"""Run EDA cycling + end-to-end forecasts, then verify with CRPS/ESSR and
produce paper-style scorecards (Figures 3-6).

Usage:
  python scripts/run_eda.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from huracan.config import load_config
from huracan.core.grid import grid_from_cfg
from huracan.core.metrics import (
    crps_ensemble,
    crps_fair,
    essr,
    rmse_ensemble_mean,
)
from huracan.eval.scorecard import scorecard
from huracan.inference.runner import run_eda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--start-index", type=int, default=None)
    ap.add_argument("--start-indices", type=int, nargs="+", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--ic-noise", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.workdir:
        cfg.output_dir = args.workdir
    out = Path(cfg.output_dir)

    res_list = run_eda(
        cfg,
        out,
        device=args.device,
        start_index=args.start_index,
        start_indices=args.start_indices,
        split=args.split,
        ic_noise=args.ic_noise,
        collect_spinup=True,
    )
    if not isinstance(res_list, list):
        res_list = [res_list]

    grid = grid_from_cfg(cfg.grid)
    w = grid.pixel_weights  # lat-lon grid is not equal-area -> weight

    all_crps, all_crps_fair, all_ess, all_ess_ms, all_rmse = [], [], [], [], []
    for res in res_list:
        preds = res["preds"].numpy()  # (M, F, npix, C) normalized 6-hourly
        truth = res["truth"].numpy()
        # P0: persistence = the ERA5 analysis at forecast initialization
        # (actually available at issue time), NOT the first future field
        persist = np.repeat(res["persistence_init"].numpy()[None], truth.shape[0], axis=0)
        all_crps.append(crps_ensemble(preds, truth, w))
        all_crps_fair.append(crps_fair(preds, truth, w))
        all_ess.append(essr(preds, truth, w, mode="rms"))
        all_ess_ms.append(essr(preds, truth, w, mode="mean_std"))
        all_rmse.append(rmse_ensemble_mean(preds, truth, w))

    def _agg(v):
        return np.mean(np.stack(v), axis=0)  # mean over start points

    crps, crps_f = _agg(all_crps), _agg(all_crps_fair)
    ess, ess_ms, rmse = _agg(all_ess), _agg(all_ess_ms), _agg(all_rmse)
    names = cfg.var_names()[: preds.shape[-1]]

    np.savez(
        out / "metrics.npz",
        crps=crps,
        crps_fair=crps_f,
        essr=ess,
        essr_mean_std=ess_ms,
        rmse=rmse,
        names=np.array(names),
        start_t0=np.array([r["t0"] for r in res_list]),
        split=res_list[0].get("split", "val"),
    )
    res = res_list[0]
    print(
        f"\nEDA: {res['members']} members, split={args.split}, "
        f"{len(res_list)} start(s), spin-up {res['spinup_days']} d "
        f"({res['spinup_calls']} AR calls), lead {res['lead_days']} d"
    )
    print("CRPS empirical (normalized units; main metric), key variables by lead day:")
    key = [n for n in ["z500", "t850", "2t"] if n in names]
    for n in key:
        j = names.index(n)
        daily = crps[:, j].reshape(-1, 4).mean(axis=1)
        print(f"  {n:>5}: " + " ".join(f"{d:.3f}" for d in daily))
    print("ESSR (RMS spread / RMSE) by lead day:")
    for n in key:
        j = names.index(n)
        daily = ess[:, j].reshape(-1, 4).mean(axis=1)
        print(f"  {n:>5}: " + " ".join(f"{d:.2f}" for d in daily))

    # scorecard vs the corrected persistence baseline
    persist_crps = []
    for res in res_list:
        preds = res["preds"].numpy()
        truth = res["truth"].numpy()
        persist = np.repeat(
            res["persistence_init"].numpy()[None], truth.shape[0], axis=0
        )
        persist_crps.append(
            crps_ensemble(np.repeat(persist[None], preds.shape[0], axis=0), truth, w)
        )
    crps_base = _agg(persist_crps)
    scorecard(
        crps,
        crps_base,
        names,
        title="Huracan vs persistence (init-field)",
        out_path=str(out / "scorecard_persistence.png"),
    )

    # initial-condition spread visualization (paper Figure 6 style)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spin = res["spinup_end"].numpy()
    j = names.index("z500") if "z500" in names else 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    if grid is None:
        # cyclic-chain datasets: ensemble stats along the ring
        k = np.arange(spin.shape[1])
        axes[0].plot(k, spin.mean(0)[:, j], "-o", ms=3)
        axes[0].set_title("spin-up ensemble mean (normalized)")
        axes[1].plot(k, spin.std(0)[:, j], "-o", ms=3)
        axes[1].set_title("spin-up ensemble std")
        for ax in axes:
            ax.set_xlabel("site k")
    else:
        colat, lon = grid.colat, grid.lon_flat
        for ax, stat, lab in [
            (axes[0], spin.mean(0), "ensemble mean"),
            (axes[1], spin.std(0), "ensemble std"),
        ]:
            sc = ax.scatter(
                np.rad2deg(lon),
                90 - np.rad2deg(colat),
                c=stat[:, j],
                s=2,
                cmap="viridis",
            )
            ax.set_title(f"{lab} z500 (normalized)")
            fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "spinup_ensemble.png", dpi=180)
    print(f"\nSaved metrics.npz, scorecard and ensemble plots to {out}")


if __name__ == "__main__":
    main()
