#!/usr/bin/env python3
"""P2: cache background ensembles from the corrected R0 baseline -- v2.

v2 (2026-10-16, external review F5/F4 follow-ups): besides the member
trajectories, each call now stores TWO same-background side branches so
D3 can measure observation impact against the RIGHT controls:

    fcst_pred.npy    (K, calls, N, C) f16 -- the FORECAST model's 6h
                     prediction from exactly the same background pair and
                     the same member noise z as the assim cycle used;
    zeroobs_pred.npy (K, calls, N, C) f16 -- the assim model's prediction
                     from the same background with the CURRENT obs window
                     zeroed (history slot obs_tm1 unchanged).

Both are validated against the SAME target (truth at t+6h), so
fcst_err - an_err isolates obs fusion and zeroobs_err - an_err isolates
the current window's marginal evidence -- no mixed lead times (the old
D3 compared errors 6 h apart and was renamed cycle-level drift).

meta.json now records seed_base and a config fingerprint.

Causal protocol (RESEARCH_PLAN v9 Sec. 3, 4090 plan Sec. 4.1):
with the single-output convention (one call = 6 h), call j consumes the
member pair (x_j, x_{j+1}) and obs frames (t-6h, t); the member states
ENTERING the call are the background B_t -- they have absorbed all
measurements up to (t-6h, t-6h] but NOT the current window (t-6h, t],
which lives in obs frame t (the ``obs_t`` slot; ``obs_tm1`` is history
context).  Members are cycled INDEPENDENTLY (R0 has no cross-member
interaction); the initial pair is perturbed with spectrally smooth noise
(ic_noise, normalized units).  The first ``pre_calls`` calls are spin-up
transient and flagged.

Usage:
  python scripts/cache_background.py --config configs/bsef_r0.yaml \
      --workdir runs/bsef/r0b --out runs/bsef/bg_cache_v2 \
      --starts 2012-01-15 2012-04-15 2012-07-15 2012-10-15 \
      --days 21 --pre-days 7 --members 8 --ic-noise 0.5
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


@torch.no_grad()
def cycle_window(assim, fcst, ds, t0, calls, members, ic_noise, grid_hw,
                 device, seed_base, with_controls):
    """Run ``members`` independent assim cycles for calls j=0..calls-1.

    Returns (states (K, calls+2, N, C) f16, fcst_pred, zeroobs_pred) with
    the two side branches (K, calls, N, C) f16 or None each."""
    window = ds.window(t0, 2 + calls)
    state = window["state"].to(device)          # (2+calls, N, C) ERA5 truth
    forcing = window["forcing"].to(device)
    obs = window["obs"].to(device)              # (2+calls, N, C_obs)
    zero_obs = torch.zeros_like(obs[1:2])
    N, C = state.shape[1], state.shape[2]
    out = torch.empty(members, calls + 2, N, C, dtype=torch.float16)
    fc = torch.empty(members, calls, N, C, dtype=torch.float16) if with_controls else None
    zo = torch.empty(members, calls, N, C, dtype=torch.float16) if with_controls else None
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
                x_prev, x_cur,
                forcing[j : j + 1], forcing[j + 1 : j + 2],
                obs_tm1=obs[j : j + 1], obs_t=obs[j + 1 : j + 2], z=z,
            )
            if with_controls:
                # same background, same noise, same forcing windows:
                # (a) forecast model without obs, (b) assim with the
                # current obs window zeroed (history slot kept)
                fc_out = fcst.step(
                    x_prev, x_cur,
                    forcing[j : j + 1], forcing[j + 1 : j + 2], z=z,
                )
                zo_out = assim.step(
                    x_prev, x_cur,
                    forcing[j : j + 1], forcing[j + 1 : j + 2],
                    obs_tm1=obs[j : j + 1], obs_t=zero_obs, z=z,
                )
                fc[m, j] = fc_out[0].to(torch.float16).cpu()
                zo[m, j] = zo_out[0].to(torch.float16).cpu()
            x_prev, x_cur = x_cur, step_out[0]
            traj.append(x_cur.to(torch.float16).cpu())
        out[m] = torch.cat(traj, dim=0)
    return out, fc, zo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True, help="R0 run dir (ckpts)")
    ap.add_argument("--out", default="runs/bsef/bg_cache_v2")
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
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--no-controls", action="store_true",
                    help="skip the same-bg side branches (faster)")
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
    a_model.load_state_dict(
        io.load_phase(args.workdir, f"assim{sfx}")["models"]["assim"])
    fpath = io.phase_path(Path(args.workdir), f"fcst_ar{sfx}")
    if fpath.exists():
        f_model.load_state_dict(
            io.load_phase(args.workdir, f"fcst_ar{sfx}")["models"]["fcst"])
    assim = a_model.to(device).eval()
    fcst = f_model.to(device).eval()

    data_dir = args.data_dir
    ds = SphereDataset(cfg, "val", rollout=calls, data_dir=data_dir)
    times = [dt.datetime.fromisoformat(s)
             for s in json.load(open(Path(data_dir or Path(cfg.data.root) / cfg.data.source)
                                     / "times.json"))]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    grid_hw = (cfg.grid.nlat, cfg.grid.nlon)

    src = Path(data_dir) if data_dir else Path(cfg.data.root) / cfg.data.source
    cfg_bytes = open(args.config, "rb").read()
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
            "assim": _sha256(io.phase_path(Path(args.workdir), f"assim{sfx}")),
            "fcst_ar": _sha256(fpath) if fpath.exists() else None,
        },
        "seed_base": args.seed_base,
        "seed_rule": "per member: torch.manual_seed(seed_base + m); "
                     "side branches reuse the SAME z draw as the assim call",
        "config_sha256_16": hashlib.sha256(cfg_bytes).hexdigest()[:16],
        "with_controls": not args.no_controls,
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
        assert t0 in set(ds.valid_t0.tolist()), (
            f"start {start} (t0={t0}) needs frames [{t0-1}, {t0-1+2+calls}) "
            f"inside the val split"
        )
        wname = f"{start}_{args.variant_name}"
        wdir = out_root / wname
        wdir.mkdir(parents=True, exist_ok=True)
        states, fc, zo = cycle_window(
            assim, fcst, ds, t0, calls, args.members, args.ic_noise, grid_hw,
            device, args.seed_base, with_controls=not args.no_controls,
        )
        np.save(wdir / "states.npy", states.numpy())
        if fc is not None:
            np.save(wdir / "fcst_pred.npy", fc.numpy())
            np.save(wdir / "zeroobs_pred.npy", zo.numpy())
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
        extra = " + fcst/zeroobs controls" if fc is not None else ""
        print(f"[{wname}] states {tuple(states.shape)} saved "
              f"(t0={t0}, {calls} calls, first {pre} flagged spin-up{extra})",
              flush=True)


if __name__ == "__main__":
    main()
