"""Synthetic on-sphere dynamics for end-to-end pipeline smoke testing.

Generates a learnable multi-scale system: solid-body advection (nonlinear in
pixel space) composed with per-channel spectral wave dispersion and slow
decay/random injection.  Output layout matches the ERA5 pipeline exactly
(same files, same normalizers), so every downstream stage (training phases,
EDA cycling, CRPS/ESSR evaluation) runs identically for both sources.

"Observations" mimic in-situ data: noisy values of all channels at a random
subset of pixels per 6h frame (station-like sparse coverage).
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..config.schema import HuracanConfig
from ..core.grid import grid_from_cfg, insolation, rotation_about_axis, unit_vectors
from ..core.sht import build_real_basis
from .norm import Normalizer, obs_norm_mapping
from .synthetic_sht import RealBasisOps


def generate_synthetic(cfg: HuracanConfig, out_dir: str | Path, seed: int = 0):
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = cfg.data.synthetic
    grid = grid_from_cfg(cfg.grid)
    npix = grid.npix
    C = s.channels
    T = s.steps
    rng = np.random.default_rng(seed)

    # --- real harmonic basis for wave stepping (exact quadrature grid) ---
    lmax = grid.lmax
    colat, lon = grid.colat, grid.lon_flat
    B, (lidx, midx, kind) = build_real_basis(lmax, colat, lon)
    ops = RealBasisOps(B, lidx, midx, kind)

    # --- advection machinery (solid-body rotation about tilted axis) ----
    axis = unit_vectors(np.deg2rad(90 - 23.4), np.deg2rad(0.0))
    v = unit_vectors(colat, lon)
    tree = cKDTree(v)
    dt_hours = 6.0
    omega = s.rotation_speed * 2 * np.pi / 24.0  # rad per hour
    adv_angle = omega * dt_hours
    speeds = np.asarray(s.wave_speeds, dtype=np.float64)
    if speeds.size < C:  # cycle speeds over channels
        speeds = np.tile(speeds, int(np.ceil(C / speeds.size)))[:C]
    speeds = speeds[:C]

    state = np.zeros((T, npix, C), dtype=np.float32)
    x = ops.random_field(rng, C)
    x = 10.0 * x / np.abs(x).mean()

    for t in range(T):
        state[t] = x
        # 1) advect: sample the field at back-rotated positions
        back = rotation_about_axis(v, axis, -adv_angle)
        _, idx = tree.query(back, workers=-1)
        x = x[idx]
        # 2) wave dispersion per channel
        x = ops.wave_step(x.astype(np.float64), speeds, dt_hours)
        # 3) decay + stochastic injection keeps the system statistically steady
        x = x * s.decay_per_step + 0.02 * ops.random_field(rng, C, power=-2.0)

    # --- observations: sparse noisy samples of observed channels --------
    names = cfg.var_names()[:C]
    surf_obs = cfg.data.obs.surface_vars
    obs_ch = [
        names.index(f"{u}{lev}")
        for lev in cfg.data.variables.levels
        for u in cfg.data.variables.upper
    ] + [names.index(sv) for sv in surf_obs]
    obs_raw = state[:, :, obs_ch]
    obs_mask = rng.random((T, npix)) < s.obs_fraction
    obs_raw = obs_raw + s.obs_noise * obs_raw.std(axis=(0, 1)) * rng.standard_normal(
        obs_raw.shape
    )
    obs_raw = np.where(obs_mask[..., None], obs_raw, 0.0).astype(np.float32)

    # --- forcings + calendar --------------------------------------------
    start = dt.datetime(2010, 1, 1)
    times = [(start + dt.timedelta(hours=6 * i)) for i in range(T)]
    insol = np.stack(
        [
            insolation(
                colat,
                lon,
                t_.hour + t_.minute / 60,
                t_.timetuple().tm_yday,
            )
            for t_ in times
        ]
    ).astype(np.float32)
    zsfc = ops.random_field(rng, 1, power=-1.0) * 3000.0
    lsm = (ops.random_field(rng, 1, power=-1.0) > 0).astype(np.float32)
    forcing = np.zeros((T, npix, 3), dtype=np.float32)
    forcing[:, :, 0] = insol
    forcing[:, :, 1] = zsfc[:, 0][None, :]
    forcing[:, :, 2] = lsm[:, 0][None, :]

    # --- normalize + save ------------------------------------------------
    norm = Normalizer.from_array(names, state)
    norm.save(out / "norm_state.json")
    obs_cfg_norm = obs_norm_mapping(cfg, norm)
    obs_cfg_norm.save(out / "norm_obs.json")

    np.save(out / "state.npy", state.astype(np.float16))
    obs_n = obs_cfg_norm.encode(obs_raw)
    obs_n[~obs_mask] = 0.0
    np.save(out / "obs.npy", obs_n.astype(np.float16))
    np.save(
        out / "forcing.npy",
        (forcing / np.array([500.0, 10000.0, 1.0], dtype=np.float32)).astype(
            np.float16
        ),
    )
    with open(out / "times.json", "w") as f:
        json.dump([t_.isoformat() for t_ in times], f)
    print(
        f"synthetic data written to {out}  T={T} npix={npix} C={C} "
        f"({time.time() - t0:.1f}s)"
    )
    return out
