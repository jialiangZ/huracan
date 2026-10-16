#!/usr/bin/env python3
"""P2: cache background ensembles from the corrected R0 baseline.

Causal protocol (RESEARCH_PLAN v9 Sec. 3, 4090 plan Sec. 4.1):
with the single-output convention (one call = 6 h), call j consumes the
member pair (x_j, x_{j+1}) and obs frames (t-6h, t); the member states
ENTERING the call are the background B_t -- they have absorbed all
measurements up to (t-6h, t-6h] but NOT the current window (t-6h, t],
which lives in obs frame t (the ``obs_t`` slot; ``obs_tm1`` is history
context).  The cache therefore stores, per window:

    states.npy   (K, J+2, N, C) f16  member trajectories INCLUDING the two
                 ERA5 initial frames (normalized); sample j's background is
                 states[:, j:j+2], its new-evidence obs frame is
                 era5 obs[base + j], its target is ERA5[base + j + 1].
    meta.json    frame indices, spin-up flag boundary, seeds, ckpt hashes,
                 obs-variant provenance.

Members are cycled INDEPENDENTLY with the assim model (R0 has no
cross-member interaction), exactly like the EDA spin-up loop; the initial
pair is perturbed with spectrally smooth noise (ic_noise, normalized
units).  The first ``pre_calls`` calls are spin-up transient and flagged
(diagnostics may exclude them).

Observation variants (diagnostic 3): ``--obs-variant`` re-runs the same
cycle against a modified obs tensor (thinned pixels / held-out stations)
to measure observation impact; the variant tensor is built beforehand by
``scripts/make_obs_variant.py``.

Usage:
  python scripts/cache_background.py --config configs/bsef_r0.yaml \
      --workdir runs/bsef/r0 --out runs/bsef/bg_cache \
      --starts 2012-01-15 2012-04-15 2012-07-15 2012-10-15 \
      --days 21 --pre-days 7 --members 8
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
from huracan.core.model import build_models
from huracan.core.rollout import member_noise, smooth_noise
from huracan.data.dataset import SphereDataset


def _sha256(path: Path, limit_bytes=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit_bytes and h.digest_size and f.tell() >= limit_bytes:
                break
    return h.hexdigest()[:16]


@torch.no_grad()
def cycle_window(
    assim, ds, t0, calls, members, ic_noise, grid_hw, device, seed_base
):
    """Run ``members`` independent assim cycles for calls j=0..calls-1.

    Returns states (K, calls+2, N, C) float16 cpu."""
    sem_cfg = assim.outputs_per_call
    window = ds.window(t0, 2 + calls)
    state = window["state"].to(device)          # (2+calls, N, C) ERA5 truth
    forcing = window["forcing"].to(device)
    obs = window["obs"].to(device)              # (2+calls, N, C_obs)
    N, C = state.shape[1], state.shape[2]
    out = torch.empty(members, calls + 2, N, C, dtype=torch.float16)
    for m in range(members):
        torch.manual_seed(seed_base + m)
        x_prev = state[0:1].clone()
        x_cur = state[1:2].clone()
        if ic_noise > 0:
            p1 = smooth_noise(1, C, grid_hw, device)
            p2 = smooth_noise(1, C, grid_hw, device)
            x_prev = x_prev + ic_noise * p1
            x_cur = x_cur + ic_noise * p2
        traj = [x_prev.to(torch.float16).cpu(), x_cur.to(torch.float16).cpu()]
        for j in range(calls):
            z = member_noise(assim, 1, N, device)
            step_out = assim.step(
                x_prev,
                x_cur,
                forcing[j : j + 1],
                forcing[j + 1 : j + 2],
                obs_tm1=obs[j : j + 1],
                obs_t=obs[j + 1 : j + 2],
                z=z,
            )
            x_prev, x_cur = x_cur, step_out[0]
            traj.append(x_cur.to(torch.float16).cpu())
        out[m] = torch.cat(traj, dim=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True, help="R0 run dir (ckpts)")
    ap.add_argument("--out", default="runs/bsef/bg_cache")
    ap.add_argument("--starts", nargs="+",
                    default=["2012-01-15", "2012-04-15", "2012-07-15", "2012-10-15"])
    ap.add_argument("--days", type=float, default=21.0,
                    help="collection length per window (after pre-days)")
    ap.add_argument("--pre-days", type=float, default=7.0,
                    help="spin-up transient inside the window (flagged)")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--ic-noise", type=float, default=0.5)
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--data-dir", default=None,
                    help="obs-variant data dir (diagnostic 3); default = config")
    ap.add_argument("--variant-name", default="full")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"],
                    help="select checkpoints by held-out val loss (default)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg.device
    sem = cfg.call_semantics()
    calls = int(round(args.days * 24 / sem.advance_hours))
    pre = int(round(args.pre_days * 24 / sem.advance_hours))
    assert calls > pre > 0

    f_model, a_model = build_models(cfg)
    sfx = "_best" if args.ckpt == "best" else ""
    ck = io.load_phase(args.workdir, f"assim{sfx}")
    a_model.load_state_dict(ck["models"]["assim"])
    fpath = io.phase_path(Path(args.workdir), f"fcst_ar{sfx}")
    if fpath.exists():
        f_model.load_state_dict(io.load_phase(args.workdir, f"fcst_ar{sfx}")["models"]["fcst"])
    assim = a_model.to(device).eval()

    data_dir = args.data_dir
    ds = SphereDataset(cfg, "val", rollout=calls, data_dir=data_dir)
    times = [dt.datetime.fromisoformat(s)
             for s in json.load(open(Path(data_dir or Path(cfg.data.root) / cfg.data.source)
                                     / "times.json"))]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    grid_hw = (cfg.grid.nlat, cfg.grid.nlon)

    # provenance hashes (small files only; state.npy is content-addressed
    # by the assemble report instead of a 25 GB hash)
    src = Path(data_dir) if data_dir else Path(cfg.data.root) / cfg.data.source
    meta_common = {
        "variant": args.variant_name,
        "data_dir": str(src),
        "members": args.members,
        "ic_noise": args.ic_noise,
        "calls": calls,
        "pre_calls": pre,
        "outputs_per_call": sem.outputs_per_call,
        "ckpt_kind": args.ckpt,
        "ckpt": {
            "assim": _sha256(io.phase_path(args.workdir, f"assim{sfx}")),
            "fcst_ar": _sha256(fpath) if fpath.exists() else None,
        },
        "norm_meta": json.load(open(src / "norm_meta.json"))
        if (src / "norm_meta.json").exists() else None,
        "assemble_report": json.load(open(src / "assemble_report.json"))
        if (src / "assemble_report.json").exists() and (src / "assemble_report.json").is_file()
        else None,
    }

    for start in args.starts:
        sd = dt.date.fromisoformat(start)
        t0 = next(i for i, t in enumerate(times)
                  if t.date() == sd and t.hour == 0)
        # window legality inside the val split (dataset enforces it via
        # valid_t0; assert for a clear message)
        assert t0 in set(ds.valid_t0.tolist()), (
            f"start {start} (t0={t0}) needs frames [{t0-1}, {t0-1+2+calls}) "
            f"inside the val split"
        )
        wname = f"{start}_{args.variant_name}"
        wdir = out_root / wname
        wdir.mkdir(parents=True, exist_ok=True)
        states = cycle_window(
            assim, ds, t0, calls, args.members, args.ic_noise, grid_hw,
            device, args.seed_base
        )
        np.save(wdir / "states.npy", states.numpy())
        meta = dict(meta_common)
        meta.update({
            "start_date": start,
            "t0_frame": t0,
            "base_frame": t0,  # sample j: bg states[:, j:j+2], obs frame
            # t0+j (new evidence), target frame t0+j+1, forcing frames
            # t0-1+j .. t0+j (see window())
            "spinup_boundary_call": pre,
        })
        with open(wdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[{wname}] states {tuple(states.shape)} saved "
              f"(t0={t0}, {calls} calls, first {pre} flagged spin-up)",
              flush=True)


if __name__ == "__main__":
    main()
