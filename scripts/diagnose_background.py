#!/usr/bin/env python3
"""P2 diagnostics on cached background ensembles -- v2 (2026-10-16).

Redesign notes (external review F3/F4/F5):
  D1: spread-error correlation is AREA-WEIGHTED (Gauss grid) and
      stratified by latitude band (``--n-lat-bands`` is used now).
  D2: PRIMARY statistic follows the v9 design -- each MEMBER's whole
      local two-frame field (wl x ww pixels x 2 frames x C channels,
      same-pixel frame pairing) is flattened and passed through ONE
      shared random orthonormal projection -> x_k in R^d.  The K x d
      departure matrix gives a covariance of rank <= K-1; its rank-r
      subspace is compared with the ensemble-MEAN error vector (same
      flatten/projection).  Baselines: random rank-r (theory r/d) and
      the CORRECT top-marginal-variance coordinates
      (argsort(diag(cov))[-r:], never eigenvalue-sorted indices).
      A secondary per-pixel POOLED variant (fixed same-pixel layout,
      explicitly named) is reported separately, never mixed with the
      member-level numbers.
  D3: observation impact uses SAME-BACKGROUND controls validated against
      the SAME target (truth at t+6h):
        fusion gain   = err(fcst_pred) - err(assim_pred)  (obs fusion)
        current gain  = err(zeroobs)   - err(assim_pred)  (current window)
      The old bg(t) vs an(t+6h) difference mixed forecast-error growth
      into the number; it is retained only as the cycle-drift trajectory.

Usage:
  python scripts/diagnose_background.py --config configs/bsef_r0.yaml \
      --cache runs/bsef/bg_cache_v2 --out runs/bsef/bg_diag_v2
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


def wcorr(x, y, w):
    """Weighted Pearson correlation."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
    sy = np.sqrt(np.average((y - my) ** 2, weights=w))
    return float(cov / (sx * sy + 1e-12))


def wrmse_scalar(x, y, w):
    """Area-weighted RMSE of x,y (N, C) averaged over channels."""
    w = w / w.sum()
    return float(np.sqrt((w[:, None] * (x - y) ** 2).sum(0)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="runs/bsef/bg_cache_v2")
    ap.add_argument("--out", default="runs/bsef/bg_diag_v2")
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
    W = grid.pixel_weights.astype(np.float64)          # (N,)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl, ww = args.window
    n_wlat, n_wlon = nlat // wl, nlon // ww
    P = wl * ww
    C = len(names)
    D_member = P * 2 * C

    lat_of_pixel = np.repeat(np.arange(nlat), nlon)
    bands = np.minimum(lat_of_pixel // (nlat // args.n_lat_bands),
                       args.n_lat_bands - 1)
    band_sel = [bands == b for b in range(args.n_lat_bands)]

    rng = np.random.default_rng(7)
    Q_member, _ = np.linalg.qr(rng.standard_normal((D_member, args.proj_dim)))
    Q_pixel, _ = np.linalg.qr(rng.standard_normal((2 * C, args.proj_dim)))

    # accumulators
    var_e = {n: [] for n in names}   # global: (err, spr, w) appended per sample
    var_s = {n: [] for n in names}
    band_e = {n: [[] for _ in range(args.n_lat_bands)] for n in names}
    band_s = {n: [[] for _ in range(args.n_lat_bands)] for n in names}
    memb_expl, memb_rand, memb_diag = [], [], []
    pool_expl, pool_rand, pool_diag = [], [], []
    variants = {}

    for wdir in sorted(Path(args.cache).iterdir()):
        if not wdir.is_dir():
            continue
        meta = json.load(open(wdir / "meta.json"))
        states = np.load(wdir / "states.npy")          # (K, J+2, N, C) f16
        K, F, N, _ = states.shape
        t0, pre = meta["base_frame"], meta["spinup_boundary_call"]
        truth = era5_truth(cfg, meta["data_dir"], t0, F)
        variant = meta.get("variant", "full")
        has_ctrl = (wdir / "fcst_pred.npy").exists()
        fc_pred = np.load(wdir / "fcst_pred.npy") if has_ctrl else None
        zo_pred = np.load(wdir / "zeroobs_pred.npy") if has_ctrl else None
        vrec = variants.setdefault(
            variant, {"bg_err": [], "an_err": [], "fc_err": [],
                      "zo_err": [], "fusion": [], "current": [], "calls": []})

        for j in range(pre, F - 2):
            bg2 = states[:, j : j + 2].astype(np.float32)   # (K,2,N,C)
            err = bg2.mean(0) - truth[j : j + 2]            # (2,N,C)
            spr = bg2.std(0, ddof=1)

            # ---- D1 accumulation (values + matching weights) ----
            w2 = np.tile(W, 2)  # (2N,) matches |err| flattening order
            for c, n in enumerate(names):
                e = np.abs(err[:, :, c]).ravel()
                s = spr[:, :, c].ravel()
                var_e[n].append(e)
                var_s[n].append(s)
                for b in range(args.n_lat_bands):
                    sel = np.tile(band_sel[b], 2)
                    band_e[n][b].append(e[sel])
                    band_s[n][b].append(s[sel])

            # ---- D3 same-background controls (same target t+6h) ----
            an = states[:, j + 2].astype(np.float32)
            vrec["bg_err"].append(
                wrmse_scalar(bg2[:, 1].mean(0), truth[j + 1], W))
            vrec["an_err"].append(
                wrmse_scalar(an.mean(0), truth[j + 2], W))
            vrec["calls"].append(j)
            if has_ctrl:
                e_fc = wrmse_scalar(
                    fc_pred[:, j].astype(np.float32).mean(0), truth[j + 2], W)
                e_zo = wrmse_scalar(
                    zo_pred[:, j].astype(np.float32).mean(0), truth[j + 2], W)
                vrec["fc_err"].append(e_fc)
                vrec["zo_err"].append(e_zo)
                vrec["fusion"].append(e_fc - e_an_of(vrec))
                vrec["current"].append(e_zo - e_an_of(vrec))

            # ---- D2 over local windows ----
            dep = (bg2 - bg2.mean(0, keepdims=True)).reshape(
                K, 2, nlat, nlon, C)
            err5 = err.reshape(2, nlat, nlon, C)
            for wi in range(n_wlat):
                for wj in range(n_wlon):
                    sa, sb = slice(wi * wl, (wi + 1) * wl), \
                        slice(wj * ww, (wj + 1) * ww)
                    d_pix = dep[:, :, sa, sb, :].transpose(0, 2, 3, 1, 4)
                    e_pix = err5[:, sa, sb, :].transpose(1, 2, 0, 3)
                    # (a) member-level: whole local field -> R^d, rank<=K-1
                    V = d_pix.reshape(K, -1) @ Q_member       # (K, d)
                    eV = e_pix.reshape(-1) @ Q_member         # (d,)
                    X = V - V.mean(0, keepdims=True)
                    cov = X.T @ X / max(K - 1, 1)
                    _, evecs = np.linalg.eigh(cov)
                    U = evecs[:, -args.rank:]
                    nrm = np.linalg.norm(eV) + 1e-12
                    memb_expl.append(
                        float(np.linalg.norm(U.T @ eV) ** 2 / nrm ** 2))
                    top = np.argsort(np.diag(cov))[-args.rank:]
                    memb_diag.append(
                        float(np.linalg.norm(eV[top]) ** 2 / nrm ** 2))
                    rb = 0.0
                    for _ in range(4):
                        Qr, _ = np.linalg.qr(rng.standard_normal(
                            (args.proj_dim, args.rank)))
                        rb += float(np.linalg.norm(Qr.T @ eV) ** 2 / nrm ** 2)
                    memb_rand.append(rb / 4)
                    # (b) pooled per-pixel variant (secondary)
                    dz = d_pix.reshape(K, P, 2 * C) @ Q_pixel   # (K,P,d)
                    ez = e_pix.reshape(P, 2 * C) @ Q_pixel      # (P,d)
                    covp = np.cov(dz.reshape(-1, args.proj_dim).T)
                    _, evcp = np.linalg.eigh(covp)
                    Up = evcp[:, -args.rank:]
                    ebar = ez.mean(0)
                    nrmp = np.linalg.norm(ebar) + 1e-12
                    pool_expl.append(
                        float(np.linalg.norm(Up.T @ ebar) ** 2 / nrmp ** 2))
                    topp = np.argsort(np.diag(covp))[-args.rank:]
                    pool_diag.append(
                        float(np.linalg.norm(ebar[topp]) ** 2 / nrmp ** 2))
                    rbp = 0.0
                    for _ in range(4):
                        Qr, _ = np.linalg.qr(rng.standard_normal(
                            (args.proj_dim, args.rank)))
                        rbp += float(np.linalg.norm(Qr.T @ ebar) ** 2
                                     / nrmp ** 2)
                    pool_rand.append(rbp / 4)

        print(f"[{wdir.name}] processed {F - 2 - pre} samples", flush=True)

    # ================= reports =================
    d2 = {
        "member_level": {
            "mode_explained_mean": float(np.mean(memb_expl)),
            "mode_explained_median": float(np.median(memb_expl)),
            "random_explained_mean": float(np.mean(memb_rand)),
            "diag_explained_mean": float(np.mean(memb_diag)),
            "note": "per-member whole local field, shared projection, "
                    "rank <= K-1 (v9 Sec.6 design)",
        },
        "pooled_per_pixel": {
            "mode_explained_mean": float(np.mean(pool_expl)),
            "mode_explained_median": float(np.median(pool_expl)),
            "random_explained_mean": float(np.mean(pool_rand)),
            "diag_explained_mean": float(np.mean(pool_diag)),
            "note": "secondary: covariance pooled over K x pixels; "
                    "explains the WINDOW-MEAN error; kept separate",
        },
        "rank": args.rank, "proj_dim": args.proj_dim,
        "n_windows": len(memb_expl),
        "theory_random": args.rank / args.proj_dim,
    }
    print("\nD2 low-rank structure (post-hoc, truth-projected):")
    for k in ("member_level", "pooled_per_pixel"):
        r = d2[k]
        print(f"  [{k}] mode {r['mode_explained_mean']:.3f} "
              f"(median {r['mode_explained_median']:.3f})  "
              f"random {r['random_explained_mean']:.3f}  "
              f"top-var coords {r['diag_explained_mean']:.3f}")

    d1 = {}
    for n in names:
        e = np.concatenate(var_e[n])
        s = np.concatenate(var_s[n])
        wflat = np.concatenate([np.tile(W, 2) for _ in var_e[n]])
        entry = {
            "corr_spread_abserr_w": wcorr(s, e, wflat),
            "rmse_w": float(np.sqrt(np.average(e ** 2, weights=wflat))),
            "mean_spread_w": float(np.average(s, weights=wflat)),
            "bands": {},
        }
        for b in range(args.n_lat_bands):
            eb = np.concatenate(band_e[n][b])
            sb = np.concatenate(band_s[n][b])
            wb = np.concatenate(
                [np.tile(W[band_sel[b]], 2) for _ in band_e[n][b]])
            entry["bands"][f"lat{b * nlat // args.n_lat_bands}"
                           f"-{(b + 1) * nlat // args.n_lat_bands}"] = {
                "corr_spread_abserr_w": wcorr(sb, eb, wb)}
        d1[n] = entry
    key_vars = [n for n in ["z500", "t850", "u850", "2t", "msl"] if n in names]
    print("\nD1 spread-error correlation (area-weighted; bands N->S):")
    for n in key_vars:
        bs = " ".join(f"{v['corr_spread_abserr_w']:.2f}"
                      for v in d1[n]["bands"].values())
        print(f"  {n:>5}: global {d1[n]['corr_spread_abserr_w']:.3f} "
              f"| bands [{bs}]")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(key_vars, [d1[n]["corr_spread_abserr_w"] for n in key_vars])
    ax.set_ylim(0, 1)
    ax.set_title("area-weighted corr(spread, |error|)")
    fig.tight_layout()
    fig.savefig(out / "d1_spread_error.png", dpi=150)
    plt.close(fig)

    d3 = {}
    if variants:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for v, rec in variants.items():
            c = np.array(rec["calls"])
            row = {
                "bg_err_mean": float(np.mean(rec["bg_err"])),
                "an_err_mean": float(np.mean(rec["an_err"])),
            }
            if rec["fusion"]:
                row.update({
                    "fc_err_mean": float(np.mean(rec["fc_err"])),
                    "zeroobs_err_mean": float(np.mean(rec["zo_err"])),
                    "fusion_gain_mean": float(np.mean(rec["fusion"])),
                    "current_obs_gain_mean": float(np.mean(rec["current"])),
                })
                ax.plot(c, rec["fc_err"], ":", alpha=0.6, label=f"{v}: fcst")
                ax.plot(c, rec["zo_err"], "-.", alpha=0.6, label=f"{v}: zeroobs")
            ax.plot(c, rec["an_err"], label=f"{v}: assim")
            ax.plot(c, rec["bg_err"], "--", alpha=0.4, label=f"{v}: bg (drift)")
            d3[v] = row
        ax.set_xlabel("cycle call (6h each); errors vs the SAME target t+6h "
                      "(bg curve vs t is the drift context)")
        ax.set_ylabel("area-weighted RMSE (normalized, all channels)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "d3_obs_impact.png", dpi=150)
        plt.close(fig)
        print("\nD3 observation impact (same background, same target):")
        for v, r in d3.items():
            if "fusion_gain_mean" in r:
                print(f"  {v:>10}: fcst {r['fc_err_mean']:.4f} -> assim "
                      f"{r['an_err_mean']:.4f} (fusion gain "
                      f"{r['fusion_gain_mean']:+.4f}); zeroobs "
                      f"{r['zeroobs_err_mean']:.4f} (current-obs gain "
                      f"{r['current_obs_gain_mean']:+.4f})")

    report = {"D1": d1, "D2": d2, "D3": d3}
    with open(out / "bg_diag_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nreport -> {out / 'bg_diag_report.json'}")


def e_an_of(vrec):
    return vrec["an_err"][-1]


if __name__ == "__main__":
    main()
