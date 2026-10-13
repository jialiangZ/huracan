#!/usr/bin/env python3
"""Physics + pipeline tests for the idealized-turbulence OSSE data source.

Physics tests verify the spectral machinery of
``huracan/data/turbulence.py`` against analytic solutions:

  * Laplacian eigenvalues of a single harmonic
  * zonal wind of a solid-body rotation streamfunction (u = Omega a sin t)
  * solid rotation is a steady state of the full vorticity equation
    (J(psi, zeta + f) = 0)
  * energy conservation of the unforced inviscid alias-free RK4 scheme

Pipeline test: tiny end-to-end run (data generation -> deterministic +
assim training steps -> EDA cycling -> finite metrics) on CPU.

Usage:  .venv/bin/python tests/test_turbulence.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from huracan.config.schema import HuracanConfig, TurbCfg
from huracan.core.grid import gauss_grid
from huracan.core.metrics import crps_ensemble
from huracan.core.rollout import spinup_and_forecast
from huracan.core.model import build_models
from huracan.data.turbulence import (
    EARTH_OMEGA,
    EARTH_RADIUS,
    SpectralSphere,
    TurbulentAtmosphere,
    generate_turbulence,
)
from huracan.inference.runner import EdaRunner
from huracan.data.dataset import SphereDataset
from huracan.training.phases import fit_phase


def tiny_cfg(T: int = 48) -> HuracanConfig:
    cfg = HuracanConfig()
    cfg.run_name = "turb_test"
    cfg.output_dir = tempfile.mkdtemp(prefix="turb_test_")
    cfg.device = "cpu"
    cfg.grid.nlat, cfg.grid.nlon, cfg.grid.lmax = 16, 32, 15
    cfg.model.hidden, cfg.model.blocks = 32, 2
    cfg.model.k_filters = 4
    cfg.model.heads = 2
    cfg.model.swin_window_lat, cfg.model.swin_window_lon = 4, 8
    cfg.model.noise_dim, cfg.model.noise_embed_dim = 4, 16
    cfg.data.source = "turbulence"
    cfg.data.root = tempfile.mkdtemp(prefix="turb_data_")
    cfg.data.variables.levels = [300, 500, 850]
    cfg.data.obs.enabled = True
    cfg.data.train_period = [0, 30]
    cfg.data.val_period = [30, T]
    cfg.data.synthetic.channels = 22
    cfg.data.synthetic.steps = T
    cfg.data.synthetic.obs_fraction = 0.15
    cfg.data.synthetic.obs_noise = 0.3
    cfg.data.synthetic.turbulence = TurbCfg(
        n_layers=3,
        wind_rms=12.0,
        forcing_lmin=2,
        forcing_lmax=4,
        dt_seconds=1800.0,
        warmup_frames=2,
    )
    cfg.train.batch_size, cfg.train.ensemble = 2, 2
    cfg.train.num_workers = 0
    cfg.train.phases["deterministic"].steps = 25
    cfg.train.phases["assim"].steps = 25
    cfg.train.phases["assim"].rollout = 1
    cfg.train.phases["assim"].init = "deterministic"  # skip stochastic chain
    cfg.eval.members = 2
    cfg.eval.spinup_days = 1.0  # 2 AR calls
    cfg.eval.lead_days = 1.0  # 2 calls
    return cfg


def test_laplacian():
    """lap(cos t) = -2 cos t / a^2 (l=1 harmonic eigenvalue)."""
    grid = gauss_grid(16, 32, 15)
    s = SpectralSphere(grid)
    u = np.zeros((s.B.shape[1], 1))
    i10 = int(np.argmax(s.ell == 1))  # first l=1 column is m=0
    u[i10, 0] = np.sqrt(4 * np.pi / 3)  # psi = cos(colat)
    lap = s.synth(s.laplacian(u))
    expect = -2.0 / s.a**2 * np.cos(grid.colat)[:, None]
    err = np.abs(lap - expect).max()
    assert err < 1e-10, f"laplacian eigenvalue error {err:.2e}"
    print(f"  laplacian eigenvalues ok (max err {err:.1e})")


def test_solid_rotation_wind():
    """psi = Omega a^2 cos(t)  ->  u_east = Omega a sin(t), v_north = 0."""
    grid = gauss_grid(16, 32, 15)
    s = SpectralSphere(grid)
    # cos(colat) = sqrt(4 pi / 3) * g_10
    u = np.zeros((s.B.shape[1], 1))
    i10 = int(np.argmax(s.ell == 1))  # first l=1 column is m=0
    u[i10, 0] = EARTH_OMEGA * s.a**2 * np.sqrt(4 * np.pi / 3)
    u_e, v_n = s.wind(u)
    err_u = np.abs(u_e[:, 0] - EARTH_OMEGA * s.a * s.sin_t)
    err_v = np.abs(v_n[:, 0])
    assert err_u.max() < 1e-6 * EARTH_OMEGA * s.a, err_u.max()
    assert err_v.max() < 1e-9, err_v.max()
    print(
        f"  solid rotation wind ok (u err {err_u.max():.1e} m/s, "
        f"v err {err_v.max():.1e} m/s)"
    )


def test_solid_rotation_steady():
    """J(psi, zeta+f) = 0 for solid rotation: rhs (no forcing/drag/hyp)
    must vanish, including the beta term."""
    grid = gauss_grid(16, 32, 15)
    s = SpectralSphere(grid)
    u = np.zeros((s.B.shape[1], 1))
    i10 = int(np.argmax(s.ell == 1))
    u[i10, 0] = EARTH_OMEGA * s.a**2 * np.sqrt(4 * np.pi / 3)
    zeta = s.laplacian(u)  # relative vorticity = -2 Om cos t
    rhs = -s.jacobian(u, zeta) + s.beta_op(u)
    assert np.abs(rhs).max() < 1e-10, np.abs(rhs).max()
    print(f"  solid rotation steady state ok (|rhs| {np.abs(rhs).max():.1e})")


def test_energy_conservation():
    """Free inviscid evolution conserves kinetic energy (alias-free
    quadratic products + RK4)."""
    grid = gauss_grid(24, 48, 23)
    tcfg = TurbCfg(
        lmax_dyn=11,
        n_layers=2,
        wind_rms=30.0,
        drag_days=1e9,
        hyper_days=1e9,
        dt_seconds=1800.0,
    )
    atm = TurbulentAtmosphere.__new__(TurbulentAtmosphere)
    atm.ops = SpectralSphere(grid, tcfg.lmax_dyn)
    rng = np.random.default_rng(1)
    # hand-build an unforced, inviscid atmosphere
    s = atm.ops
    atm.n_layers = 2
    atm.forcing = np.zeros((s.B.shape[1], 2))
    atm.drag = 0.0
    atm.hyp = 0.0
    atm.tau_c = 0.0
    atm.t_eq = np.zeros((s.B.shape[1], 2))
    atm.tau_T = 0.0
    atm.q_src = np.zeros((s.B.shape[1], 2))
    atm.tau_q = 0.0
    band = (s.lidx >= 2) & (s.lidx <= 8)
    atm.zeta = np.where(band[:, None], rng.standard_normal((s.B.shape[1], 2)), 0.0)
    atm.zeta *= 30.0 * 5.0 / s.a / np.sqrt((atm.zeta**2).sum() / (4 * np.pi))
    atm.ttr = np.zeros_like(atm.zeta)
    atm.qtr = np.zeros_like(atm.zeta)

    e0 = s.energy(atm.psi()).sum()
    for _ in range(60):
        atm.step(1800.0)
    e1 = s.energy(atm.psi()).sum()
    drift = abs(e1 - e0) / e0
    # RK4 time-truncation accumulates ~dt^4; spectral (alias-free) part is
    # exact, so this stays at round-off-driven levels
    assert drift < 1e-7, f"energy drift {drift:.2e}"
    print(f"  energy conserved over 60 steps: drift {drift:.1e}")


def test_turbulence_data_and_pipeline():
    cfg = tiny_cfg()
    t0 = time.time()
    generate_turbulence(cfg, Path(cfg.data.root) / "turbulence", seed=0)

    root = Path(cfg.data.root) / "turbulence"
    state = np.load(root / "state.npy")
    obs = np.load(root / "obs.npy")
    assert state.shape == (cfg.data.synthetic.steps, 16 * 32, 22)
    assert np.isfinite(state.astype(np.float32)).all()
    assert np.abs(state.astype(np.float32)).max() < 65504  # fp16 safe
    assert (obs != 0).any() and np.isfinite(obs.astype(np.float32)).all()
    # chaotic dynamics: wind channels must evolve frame to frame
    j = [i for i, n in enumerate(cfg.var_names()) if n.startswith("u")][0]
    d = np.abs(
        state[10:14, :, j].astype(np.float32) - state[11:15, :, j].astype(np.float32)
    )
    scale = np.abs(state[:, :, j].astype(np.float32)).std()
    assert d.mean() > 0.005 * scale, "wind frozen between frames?"
    print(
        f"  data ok ({time.time() - t0:.0f}s): state {state.shape}, "
        f"|x|max {np.abs(state.astype(np.float32)).max():.0f}, "
        f"frame-diff {d.mean():.1f}"
    )

    # quick training: deterministic + assim, then EDA cycling with obs
    for phase in ["deterministic", "assim"]:
        res = fit_phase(cfg, phase, device="cpu")
        assert np.isfinite(res["losses"]).all(), f"non-finite loss in {phase}"
        print(
            f"  phase {phase}: {len(res['losses'])} steps, "
            f"loss {np.mean(res['losses'][:5]):.4f} -> "
            f"{np.mean(res['losses'][-5:]):.4f}"
        )

    runner = EdaRunner.from_checkpoints(cfg, cfg.output_dir, device="cpu")
    S, F = cfg.spinup_calls(), cfg.lead_calls()
    sem = cfg.call_semantics()
    ds = SphereDataset(cfg, "val", rollout=S + F)
    window = ds.window(ds.first_t0, sem.window_frames(S + F))
    spin_end, preds, spin_mean = spinup_and_forecast(
        runner.assim,
        runner.fcst,
        window["state"],
        window["forcing"],
        window["obs"][: sem.obs_frames(S)],
        members=cfg.eval.members,
        spinup=S,
        lead=F,
        collect_spinup=True,
    )
    assert spin_mean.shape == (sem.frames_for_calls(S), 16 * 32, 22)
    truth = window["state"][
        sem.frames_for_calls(S) + 2 : sem.frames_for_calls(S)
        + 2
        + sem.frames_for_calls(F)
    ].numpy()
    crps = crps_ensemble(preds.numpy(), truth)
    assert np.isfinite(crps).all() and (crps > 0).all()
    print(f"  EDA ok: preds {tuple(preds.shape)}, CRPS mean {crps.mean():.3f}")
    print(
        f"  spinup mean trajectory {tuple(spin_mean.shape)} finite: "
        f"{bool(np.isfinite(spin_mean.numpy()).all())}"
    )


def main():
    print("== spectral physics ==")
    test_laplacian()
    test_solid_rotation_wind()
    test_solid_rotation_steady()
    test_energy_conservation()
    print("== turbulence data + pipeline ==")
    test_turbulence_data_and_pipeline()
    print("\nALL TURBULENCE TESTS PASSED")


if __name__ == "__main__":
    main()
