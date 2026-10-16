#!/usr/bin/env python3
"""P2 diagnostics on the cached background ensembles (4090 plan Sec. 4.3).

D1 spread-error relation: per-variable correlation between member spread
    and |ensemble-mean error| against ERA5, binned spread-vs-mean-error,
    stratified by season / latitude band.  Ensemble variance should track
    error if the ensemble is a usable error-structure estimate.
D2 low-rank structure: with a FIXED shared local random projection
    (2 frames x 72 ch -> 32 dims, 6x12 windows with longitude wrap), the
    rank-4 leading covariance subspace of the member departures explains
    what fraction of the actual ensemble-mean error, compared with (a)
    random rank-4 subspaces (expected r/32) and (b) the top-4 variance
    coordinates ("diagonal only").  Truth is used ONLY for this
    post-hoc projection (upper-bound diagnostic).
D3 observation impact: background/analysis error curves for obs variants
    (full vs thinned vs station-holdout), same starts/seeds/cycles.

Usage:
  python scripts/diagnose_background.py --cache runs/bsef/bg_cache \
      --config configs/bsef_r0.yaml --out runs/bsef/bg_diag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def era5_truth(cfg, data_dir, t0, frames_needed):
    from huracan.data.dataset import SphereDataset

    ds = SphereDataset(cfg, "val", rollout=frames_needed - 2, data_dir=data_dir)
    return ds.window(t0, frames_needed)["state"].numpy()  # (F, N, C) normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="runs/bsef/bg_cache")
    ap.add_argument("--out", default="runs/bsef/bg_diag")
    ap.add_argument("--proj-dim", type=int, default=32)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--window", type=int, nargs=2, default=[6, 12])
    ap.add_argument("--n-lat-bands", type=int, default=6)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from huracan.config import load_config, var_names
    from huracan.core.grid import grid_from_cfg

    cfg = load_config(args.config)
    grid = grid_from_cfg(cfg.grid)
    names = var_names(cfg)
    nlat, nlon = grid.nlat, grid.nlon
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl, ww = args.window  # 6 x 12
    n_wlat, n_wlon = nlat // wl, nlon // ww

    # ---------------- D1 + D2 accumulation ----------------
    per_var = {n: {"err": [], "spr": []} for n in names}
    proj_expl, proj_rand, proj_diag = [], [], []
    rng = np.random.default_rng(7)
    # shared random projection: local (2 frames x C ch) block -> proj_dim
    rng = np.random.default_rng(7)
    Rfull = rng.standard_normal((2 * len(names), args.proj_dim))
    Q, _ = np.linalg.qr(Rfull)  # orthonormal columns (144 x 32)

    variants = {}
    for wdir in sorted(Path(args.cache).iterdir()):
        if not wdir.is_dir():
            continue
        meta = json.load(open(wdir / "meta.json"))
        states = np.load(wdir / "states.npy")  # (K, J+2, N, C) f16, flat
        K, F, N, C = states.shape
        t0, pre = meta["base_frame"], meta["spinup_boundary_call"]
        truth = era5_truth(cfg, meta["data_dir"], t0, F)  # (F, N, C)
        variant = meta.get("variant", "full")
        vrec = variants.setdefault(
            variant, {"bg_err": [], "an_err": [], "calls": []}
        )

        # sample j: bg = states[:, j:j+2]; analysis output states[:, j+2]
        # exists only for j <= F-3 (the last stored frame has no target
        # beyond it), hence the F-2 upper bound.
        for j in range(pre, F - 2):
            bg2 = states[:, j : j + 2].astype(np.float32)  # (K,2,N,C)
            err = bg2.mean(0) - truth[j : j + 2]  # (2,N,C)
            spr = bg2.std(0, ddof=1)
            # D1 accumulation (both frames, all pixels)
            for c, n in enumerate(names):
                per_var[n]["err"].append(np.abs(err[:, :, c]).ravel())
                per_var[n]["spr"].append(spr[:, :, c].ravel())
            # D3: bg error (t frame) and analysis error (post-call output)
            e_bg = float(np.sqrt(((bg2[:, 1].mean(0) - truth[j + 1]) ** 2).mean()))
            an = states[:, j + 2].astype(np.float32)  # absorbed window obs
            e_an = float(np.sqrt(((an.mean(0) - truth[j + 2]) ** 2).mean()))
            vrec["bg_err"].append(e_bg)
            vrec["an_err"].append(e_an)
            vrec["calls"].append(j)

            # D2: local windows on the departure fields (grid view; pixels
            # are stored row-major (lat, lon) so reshape is loss-free)
            dep5 = (bg2 - bg2.mean(0, keepdims=True)).reshape(
                K, 2, nlat, nlon, C
            )
            err5 = err.reshape(2, nlat, nlon, C)
            for wi in range(n_wlat):
                for wj in range(n_wlon):
                    sl_lat = slice(wi * wl, (wi + 1) * wl)
                    sl_lon = slice(wj * ww, (wj + 1) * ww)
                    # per-pixel vectors (2C,) within the local window
                    d = dep5[:, :, sl_lat, sl_lon, :].reshape(K, -1, 2 * C)
                    e = err5[:, sl_lat, sl_lon, :].reshape(-1, 2 * C)
                    dz = d @ Q  # (K, P, 32)
                    ez = e @ Q  # (P, 32)
                    cov = np.cov(dz.reshape(-1, args.proj_dim).T)
                    evals, evecs = np.linalg.eigh(cov)
                    U4 = evecs[:, -args.rank:]  # (32, r)
                    ebar = ez.mean(0)
                    nrm = np.linalg.norm(ebar) + 1e-9
                    expl = np.linalg.norm(U4.T @ ebar) ** 2 / nrm**2
                    # random rank-r orthonormal baseline (avg of 4 draws)
                    rb = 0.0
                    for _ in range(4):
                        Qr, _ = np.linalg.qr(
                            rng.standard_normal((args.proj_dim, args.rank))
                        )
                        rb += np.linalg.norm(Qr.T @ ebar) ** 2 / nrm**2
                    rb /= 4
                    # top-|r| variance coordinates
                    top = np.argsort(evals)[-args.rank:]
                    db = np.linalg.norm(ebar[top]) ** 2 / nrm**2
                    proj_expl.append(expl)
                    proj_rand.append(rb)
                    proj_diag.append(db)

        print(f"[{wdir.name}] processed {F - 2 - pre} samples", flush=True)

    # ---------------- reports ----------------
    report = {"D2": {
        "mode_explained_mean": float(np.mean(proj_expl)),
        "mode_explained_median": float(np.median(proj_expl)),
        "random_explained_mean": float(np.mean(proj_rand)),
        "diag_explained_mean": float(np.mean(proj_diag)),
        "rank": args.rank, "proj_dim": args.proj_dim,
        "n_windows": len(proj_expl),
    }}
    print("\nD2 low-rank structure (post-hoc, truth-projected):")
    print(f"  mode subspace explains  {report['D2']['mode_explained_mean']:.3f} "
          f"(median {report['D2']['mode_explained_median']:.3f})")
    print(f"  random rank-{args.rank} baseline {report['D2']['random_explained_mean']:.3f}"
          f"  (theory {args.rank/args.proj_dim:.3f})")
    print(f"  top-variance coords     {report['D2']['diag_explained_mean']:.3f}")

    # D1: per-variable correlation + binned plot
    d1 = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    key_vars = [n for n in ["z500", "t850", "u850", "2t", "msl"] if n in names]
    for n in names:
        e = np.concatenate(per_var[n]["err"])
        s = np.concatenate(per_var[n]["spr"])
        corr = float(np.corrcoef(s, e)[0, 1])
        d1[n] = {"corr_spread_abserr": corr, "rmse": float(np.sqrt((e**2).mean())),
                 "mean_spread": float(s.mean())}
    report["D1"] = d1
    print("\nD1 spread-error correlation (key vars):")
    for n in key_vars:
        print(f"  {n:>5}: corr {d1[n]['corr_spread_abserr']:.3f}  "
              f"rmse {d1[n]['rmse']:.3f}  spread {d1[n]['mean_spread']:.3f}")
    # binned curve for one key var
    if key_vars:
        n = "z500" if "z500" in names else key_vars[0]
        e = np.concatenate(per_var[n]["err"])
        s = np.concatenate(per_var[n]["spr"])
        qs = np.quantile(s, np.linspace(0, 1, 11))
        ix = np.clip(np.searchsorted(qs, s, side="right") - 1, 0, 9)
        mean_e = [float(e[ix == b].mean()) for b in range(10)]
        axes[0].plot(qs[:-1], mean_e, "o-")
        axes[0].set_xlabel("ensemble spread (normalized)")
        axes[0].set_ylabel("mean |error|")
        axes[0].set_title(f"spread-error relation: {n}")
    axes[1].bar(
        [v for v in key_vars],
        [d1[v]["corr_spread_abserr"] for v in key_vars],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("corr(spread, |error|)")
    fig.tight_layout()
    fig.savefig(out / "d1_spread_error.png", dpi=150)

    # D3: variant impact curves
    if len(variants) > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        d3 = {}
        for v, rec in variants.items():
            c = np.array(rec["calls"])
            bg = np.array(rec["bg_err"])
            an = np.array(rec["an_err"])
            ax.plot(c, an, label=f"{v}: analysis err")
            ax.plot(c, bg, "--", alpha=0.5, label=f"{v}: background err")
            d3[v] = {
                "bg_err_mean": float(bg.mean()),
                "an_err_mean": float(an.mean()),
                "obs_benefit": float((bg - an).mean()),
            }
        ax.set_xlabel("cycle call (6h each)")
        ax.set_ylabel("RMSE vs ERA5 (normalized, all channels)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "d3_obs_impact.png", dpi=150)
        report["D3"] = d3
        print("\nD3 observation impact (mean over calls):")
        for v, r in d3.items():
            print(f"  {v:>10}: bg {r['bg_err_mean']:.4f} -> an "
                  f"{r['an_err_mean']:.4f}  (benefit {r['obs_benefit']:+.4f})")

    with open(out / "bg_diag_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nreport -> {out / 'bg_diag_report.json'}")


if __name__ == "__main__":
    main()
