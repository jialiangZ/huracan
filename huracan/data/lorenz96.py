"""Lorenz-96 nature runs for the AI4DA cross-domain benchmark (DATA_PLAN
Sec. 3).

One-level main line (L96-1L): K=40, F=8, RK4 dt=0.05 -- the lingua franca
of the DA/ML literature (Brajard et al. 2020; KalmanNet; latent-space
EnKF studies).  Two-level ablation (L96-2L): Wilks-2005 parameters aligned
with Chattopadhyay et al. 2020; the STORED STATE IS X ONLY -- the fast Y
variables play the role of unresolved subgrid dynamics that the learned
assimilation has to cope with without an explicit model of them.

Frame semantics (see core/rollout.py): one frame = one dt = one
observation cycle.  There are no hours: experiments express lengths in AR
calls (``eval.spinup_calls_override`` / ``lead_calls_override``); one AR
call = 2 frames = 0.1 time units for the 1L default dt.

Dataset layout: THREE INDEPENDENT trajectories (fresh initial conditions,
shared warmup discard) concatenated into one ``state.npy`` as
[train | val | test].  ``generate_lorenz96`` returns and prints the
suggested ``train/val/test_period`` values INCLUDING a boundary buffer so
that no training window can splice two trajectories; copy those numbers
into the config verbatim.

No grid object is involved: "pixels" are the K variables in cyclic order
(npix = K), so the standard SphereDataset window API works unchanged for
rollout training.

Config contract (see configs/l96.yaml / l96_2l.yaml):
  data.source       = "lorenz96" (1L) or "lorenz96_2l" (2L)  [dir name]
  variables.upper   = ["x"], levels = [500] (placeholder), surface = []
  obs.enabled       = true, obs.surface_vars = []
  forcing           = ["f"]  (single constant channel, stored as F/F_ref=1)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config.schema import HuracanConfig
from .norm import Normalizer, obs_norm_mapping

_BOUNDARY_BUFFER = 32  # suggested period margin (>= 2*max_rollout + 2 = 18)


# ---------------------------------------------------------------------------
# dynamics (numpy; a differentiable torch twin joins with the EnKF baseline)
# ---------------------------------------------------------------------------


def rhs_1l(x: np.ndarray, F: float) -> np.ndarray:
    """dx_k/dt = (x_{k+1} - x_{k-2}) x_{k-1} - x_k + F, cyclic in k."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def _rk4(x, dt, f):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def run_1l(x0, n_steps, F, dt, rng=None, record=True):
    """Integrate the one-level system; returns (n_steps, K) states."""
    out = np.empty((n_steps, x0.size), dtype=np.float64)
    x = x0.astype(np.float64).copy()
    for i in range(n_steps):
        x = _rk4(x, dt, lambda s: rhs_1l(s, F))
        if record:
            out[i] = x
            if not np.isfinite(x).all() or np.abs(x).max() > 100.0:
                raise RuntimeError(
                    f"L96-1L blew up at step {i} (max |x| "
                    f"{np.abs(x).max():.1f}); check F/dt"
                )
    return out


def rhs_2l(x, y, F, h, b, c):
    """Two-scale Lorenz-96 (Lorenz 1995 / Wilks 2005), cyclic k (K) and
    j (J per k):

      dX_k = -X_{k-1}(X_{k-2} - X_{k+1}) - X_k + F - (hc/b) sum_j Y_{j,k}
      dY_j = -cb Y_{j+1}(Y_{j+2} - Y_{j-1}) - c Y_j + (hc/b) X_k
    """
    sumy = y.sum(axis=1)
    dx = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    dx -= (h * c / b) * sumy
    dy = (
        -c
        * b
        * np.roll(y, -1, axis=1)
        * (np.roll(y, -2, axis=1) - np.roll(y, 1, axis=1))
        - c * y
        + (h * c / b) * x[:, None]
    )
    return dx, dy


def run_2l(x0, y0, n_frames, F, h, b, c, dt, substeps):
    """Coupled RK4 at dt/substeps, recording X every substeps-th step.

    Operator splitting is avoided on purpose (the coupled system is
    cheap enough); dt/substeps must resolve the fast Y timescale, hence
    ``substeps`` >= 10 for the default c=b=10.

    Returns (X trace (n_frames, K), final Y (K, J)).
    """
    dt_s = dt / substeps
    out = np.empty((n_frames, x0.size), dtype=np.float64)
    x = x0.astype(np.float64).copy()
    y = y0.astype(np.float64).copy()
    for i in range(n_frames):
        for _ in range(substeps):
            k1x, k1y = rhs_2l(x, y, F, h, b, c)
            k2x, k2y = rhs_2l(x + 0.5 * dt_s * k1x, y + 0.5 * dt_s * k1y, F, h, b, c)
            k3x, k3y = rhs_2l(x + 0.5 * dt_s * k2x, y + 0.5 * dt_s * k2y, F, h, b, c)
            k4x, k4y = rhs_2l(x + dt_s * k3x, y + dt_s * k3y, F, h, b, c)
            x = x + dt_s / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            y = y + dt_s / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
        out[i] = x
        if not np.isfinite(y).all() or np.abs(y).max() > 500.0:
            raise RuntimeError(
                f"L96-2L blew up at frame {i} (max |Y| "
                f"{np.abs(y).max():.1f}); increase fast_substeps"
            )
    return out, y


# ---------------------------------------------------------------------------
# dataset generation (layout identical to generate_turbulence)
# ---------------------------------------------------------------------------


def _validate(cfg: HuracanConfig) -> None:
    names = cfg.var_names()
    if len(names) != 1:
        raise ValueError(
            f"lorenz96 expects exactly 1 prognostic channel "
            f"(variables.upper=['x'], levels=[<one placeholder>], "
            f"surface=[]); got {len(names)}: {names[:5]}..."
        )
    if len(cfg.data.forcing) != 1:
        raise ValueError("lorenz96 expects exactly 1 forcing channel ['f']")
    if cfg.data.obs.surface_vars:
        raise ValueError(
            "lorenz96 obs are the state channel itself; set data.obs.surface_vars = []"
        )


def generate_lorenz96(cfg: HuracanConfig, out_dir: str | Path, seed: int = 0) -> dict:
    """Run the nature run and write the standard dataset files.

    Writes state.npy / obs.npy / forcing.npy / times.json /
    norm_state.json / norm_obs.json with the exact layout of
    ``generate_turbulence`` (see data/dataset.py).  Returns the split
    boundaries plus suggested period config values.
    """
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lcfg = cfg.data.lorenz96
    _validate(cfg)
    names = cfg.var_names()
    F = lcfg.forcing_f
    rng = np.random.default_rng(seed)

    lengths = [lcfg.train_steps, lcfg.val_steps, lcfg.test_steps]
    segments = []
    for seg_i, n_steps in enumerate(lengths):
        t0 = time.time()
        if lcfg.two_scale:
            K = lcfg.n_slow
            x0 = rng.standard_normal(K)
            y0 = 0.1 * rng.standard_normal((K, lcfg.n_fast_per_slow))
            args = (
                F,
                lcfg.coupling_h,
                lcfg.coupling_b,
                lcfg.coupling_c,
                lcfg.dt,
                lcfg.fast_substeps,
            )
            warm, y = run_2l(x0, y0, lcfg.warmup_steps, *args)
            seg, _ = run_2l(warm[-1], y, n_steps, *args)
        else:
            K = lcfg.n_vars
            x = rng.standard_normal(K)
            for _ in range(lcfg.warmup_steps):
                x = _rk4(x, lcfg.dt, lambda s: rhs_1l(s, F))
            seg = run_1l(x, n_steps, F, lcfg.dt)
        segments.append(seg)
        print(
            f"  trajectory {seg_i} ({['train', 'val', 'test'][seg_i]}): "
            f"{n_steps} frames, std {seg.std():.2f} "
            f"({time.time() - t0:.1f}s)"
        )

    state = np.concatenate(segments, axis=0)[:, :, None]  # (T, K, 1)
    T = state.shape[0]
    boundaries = np.cumsum([0] + lengths)

    # normalizer from the TRAIN segment only (attractor climatology)
    norm = Normalizer.from_array(names, state[: boundaries[1]])
    norm.save(out / "norm_state.json")
    obs_norm_mapping(cfg, norm).save(out / "norm_obs.json")

    np.save(out / "state.npy", state.astype(np.float16))

    # forcing: single constant channel, normalized to 1 at the nominal F
    forcing = np.full(
        (T, state.shape[1], 1), F / max(lcfg.forcing_f, 1e-12), dtype=np.float16
    )
    np.save(out / "forcing.npy", forcing)
    with open(out / "times.json", "w") as f:
        json.dump([round(i * lcfg.dt, 6) for i in range(T)], f)

    write_observations(cfg, out, state, rng)

    buf = _BOUNDARY_BUFFER
    # start val/test at boundary+1: the window API ingests frame t0-1 as
    # the second input of the two-timestep convention, so the first t0 of
    # a new segment must not reach back into the previous trajectory
    periods = {
        "train_period": [1, int(boundaries[1]) - buf],
        "val_period": [int(boundaries[1]) + 1, int(boundaries[2]) - buf],
        "test_period": [int(boundaries[2]) + 1, int(boundaries[3]) - buf],
    }
    print(
        f"lorenz96{'_2l' if lcfg.two_scale else ''} data written to {out}  "
        f"T={T} npix={state.shape[1]} "
        f"({time.time() - t_start:.0f}s)"
    )
    print("  suggested config periods (window buffer included):")
    for k, v in periods.items():
        print(f"    data.{k}: {v}")
    return {"boundaries": boundaries.tolist(), "periods": periods}


# ---------------------------------------------------------------------------
# observations: every k-th variable, absolute-sigma noise, zero = missing
# ---------------------------------------------------------------------------


def write_observations(cfg: HuracanConfig, out: Path, state: np.ndarray, rng) -> Path:
    """obs.npy (T, npix, 1): variables with index % obs_every == 0 report
    truth + N(0, obs_sigma) noise, normalized with the state stats;
    everything else is exactly zero (the missing-data convention)."""
    lcfg = cfg.data.lorenz96
    T, npix, _ = state.shape
    sites = np.arange(0, npix, max(1, lcfg.obs_every))
    norm = Normalizer.load(out / "norm_obs.json")

    raw = state[:, sites, 0].astype(np.float32)
    noise = lcfg.obs_sigma * rng.standard_normal(raw.shape).astype(np.float32)
    obs = np.zeros((T, npix, 1), dtype=np.float32)
    obs[:, sites, 0] = ((raw + noise) - norm.mean[0]) / norm.std[0]
    np.save(out / "obs.npy", obs.astype(np.float16))
    print(f"  obs: {len(sites)}/{npix} variables report, sigma={lcfg.obs_sigma}")
    return out


def make_obs_variant(
    cfg: HuracanConfig,
    src_root: str | Path,
    dst_root: str | Path,
    *,
    obs_every: int,
    obs_sigma: float,
    seed: int = 0,
) -> Path:
    """Clone a lorenz96 dataset with a different observation network
    (density = 1/obs_every, noise = obs_sigma).  Symlinks the heavy files
    and regenerates only obs.npy -- sweeping without re-running nature."""
    import copy as _copy

    src, dst = Path(src_root), Path(dst_root)
    dst.mkdir(parents=True, exist_ok=True)
    for f in (
        "state.npy",
        "forcing.npy",
        "times.json",
        "norm_state.json",
        "norm_obs.json",
    ):
        link = dst / f
        if not link.exists():
            link.symlink_to(src.resolve() / f)
    vcfg = _copy.deepcopy(cfg)
    vcfg.data.lorenz96.obs_every = obs_every
    vcfg.data.lorenz96.obs_sigma = obs_sigma
    state = np.load(dst / "state.npy", mmap_mode="r")
    write_observations(vcfg, dst, state, np.random.default_rng(seed + 17))
    return dst
