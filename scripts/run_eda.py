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
from huracan.core import grid as G
from huracan.core.metrics import crps_ensemble, essr, rmse_ensemble_mean
from huracan.eval.scorecard import scorecard
from huracan.inference.runner import run_eda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--start-index", type=int, default=None)
    ap.add_argument(
        "--baseline",
        default="persistence",
        help="baseline for the scorecard: 'persistence'",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.workdir:
        cfg.output_dir = args.workdir
    out = Path(cfg.output_dir)

    res = run_eda(cfg, out, device=args.device, start_index=args.start_index)
    preds = res["preds"].numpy()  # (M, 2F, npix, C) normalized
    truth = res["truth"].numpy()

    names = cfg.var_names()[: preds.shape[-1]]
    crps = crps_ensemble(preds, truth)  # (2F, C)
    ess = essr(preds, truth)
    rmse = rmse_ensemble_mean(preds, truth)

    np.savez(out / "metrics.npz", crps=crps, essr=ess, rmse=rmse, names=np.array(names))
    print("\nCRPS (normalized units), key variables by lead day:")
    key = [n for n in ["z500", "t850", "2t"] if n in names]
    for n in key:
        j = names.index(n)
        daily = crps[:, j].reshape(-1, 4).mean(axis=1)
        print(f"  {n:>5}: " + " ".join(f"{d:.3f}" for d in daily))
    print("ESSR at day 1 / 5 / 10:")
    for n in key:
        j = names.index(n)
        daily = ess[:, j].reshape(-1, 4).mean(axis=1)
        i5 = min(4, len(daily) - 1)
        print(
            f"  {n:>5}: {daily[0]:.2f} / {daily[i5]:.2f} / {daily[-1]:.2f}"
        )

    # scorecard vs persistence baseline (observed last state persists)
    persist = np.repeat(truth[0][None], truth.shape[0], axis=0)
    crps_base = crps_ensemble(np.repeat(persist[None], preds.shape[0], axis=0), truth)
    scorecard(
        crps,
        crps_base,
        names,
        title="Huracan vs persistence",
        out_path=str(out / "scorecard_persistence.png"),
    )

    # initial-condition spread visualization (paper Figure 6 style)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nside = cfg.grid.nside
    spin = res["spinup_end"].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    for ax, stat, lab in [
        (axes[0], spin.mean(0), "ensemble mean"),
        (axes[1], spin.std(0), "ensemble std"),
    ]:
        j = names.index("z500") if "z500" in names else 0
        sc = ax.scatter(
            np.rad2deg(G.pixel_centers(nside)[1]),
            90 - np.rad2deg(G.pixel_centers(nside)[0]),
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
