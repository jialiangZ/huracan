"""Ensemble data assimilation (EDA) cycling and end-to-end forecast inference.

Reproduces the paper's experiment (Sec. 3.1): M independent EDA members are
initialized from the same analysis state and cycle the assimilation model for
a spin-up period (paper: >= 3 months at 4 cycles/day; scaled down for the
pilot), then issue forecasts by switching to the forecast model for
`lead_steps` calls (20 calls = 10 days at 2 x 6h per call).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import n_obs_channels
from .data.dataset import SphereDataset


@torch.no_grad()
def run_eda(
    cfg: dict,
    workdir: str | Path,
    device: str | None = None,
    start_index: int | None = None,
    data_dir: str | Path | None = None,
):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(workdir)

    fcst = (
        torch.load(out / "ckpt_fcst_ar.pt", map_location=dev, weights_only=False)[
            "models"
        ]["fcst"]
        if (out / "ckpt_fcst_ar.pt").exists()
        else None
    )
    joint = (
        torch.load(out / "ckpt_joint.pt", map_location=dev, weights_only=False)[
            "models"
        ]
        if (out / "ckpt_joint.pt").exists()
        else None
    )

    from .model import build_models

    f_model, a_model = build_models(cfg)
    if joint is not None:
        a_model.load_state_dict(joint["assim"])
        f_model.load_state_dict(joint["fcst"])
    else:
        a_model.load_state_dict(
            torch.load(out / "ckpt_assim.pt", map_location=dev, weights_only=False)[
                "models"
            ]["assim"]
        )
        if fcst is not None:
            f_model.load_state_dict(fcst)
    a_model, f_model = a_model.to(dev).eval(), f_model.to(dev).eval()

    ecfg = cfg["eval"]
    M = ecfg["members"]
    S, F = ecfg["spinup_steps"], ecfg["lead_steps"]

    ds = SphereDataset(cfg, "val", rollout=S + F, data_dir=data_dir)
    t0 = start_index if start_index is not None else int(ds.valid_t0[0])
    obs_enabled = ds.obs_enabled

    state = torch.from_numpy(
        np.asarray(ds.state[t0 - 1 : t0 + 2 * (S + F) + 1], dtype=np.float32)
    ).to(dev)
    forc = torch.from_numpy(
        np.asarray(ds.forcing[t0 - 1 : t0 + 2 * (S + F) + 1], dtype=np.float32)
    ).to(dev)
    obs = (
        torch.from_numpy(np.asarray(ds.obs[t0 - 1 : t0 + 2 * S], dtype=np.float32)).to(
            dev
        )
        if obs_enabled
        else None
    )

    # normalize states with the dataset normalizer
    mean = ds.mean.to(dev)
    std = ds.std.to(dev)
    state = (state - mean) / std

    noise_dim = a_model.noise.mlp[0].in_features
    npix = state.shape[1]

    members, spinup_end = [], []
    for m in range(M):
        torch.manual_seed(1000 + m)
        x_prev, x_cur = state[0:1].clone(), state[1:2].clone()
        for j in range(S):
            z = torch.randn(1, npix, noise_dim, device=dev)
            kw = (
                dict(obs_tm1=obs[2 * j : 2 * j + 1], obs_t=obs[2 * j + 1 : 2 * j + 2])
                if obs_enabled
                else {}
            )
            x1, x2 = a_model.step(
                x_prev,
                x_cur,
                forc[2 * j : 2 * j + 1],
                forc[2 * j + 1 : 2 * j + 2],
                z=z,
                **kw,
            )
            x_prev, x_cur = x1, x2
        spinup_end.append(x_cur.cpu())
        preds = []
        for j in range(S, S + F):
            z = torch.randn(1, npix, noise_dim, device=dev)
            x1, x2 = f_model.step(
                x_prev, x_cur, forc[2 * j : 2 * j + 1], forc[2 * j + 1 : 2 * j + 2], z=z
            )
            preds += [x1, x2]
            x_prev, x_cur = x1, x2
        members.append(torch.cat(preds, dim=0).cpu())  # (2F, npix, C)
        print(f"member {m + 1}/{M} done", flush=True)

    preds = torch.stack(members, dim=0)  # (M, 2F, npix, C)
    truth = state[2 * S + 2 : 2 * S + 2 + 2 * F].cpu()  # forecast window
    spinup = torch.cat(spinup_end, dim=0)  # (M, npix, C)

    res = {
        "preds": preds,
        "truth": truth,
        "spinup_end": spinup,
        "t0": t0,
        "members": M,
        "spinup_steps": S,
        "lead_steps": F,
    }
    torch.save(res, out / "eda_result.pt")
    return res
