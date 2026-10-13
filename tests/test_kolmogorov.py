#!/usr/bin/env python3
"""Physics + pipeline tests for the Kolmogorov-flow OSSE data source.

Physics:
  * exact laminar steady state psi* = A/(nu kf^4) sin(kf y) is preserved
    by the pseudo-spectral RK4 solver (forcing balances viscosity)
  * free inviscid evolution conserves kinetic energy (2/3-dealiased
    quadratic term + RK4)

Trunk:
  * PlanarSHT round-trip is the identity and Parseval holds
  * HuracanSFNO with the planar backend forward/backward-passes and its
    untrained residual output stays near persistence

Pipeline: tiny end-to-end run (data generation -> deterministic + assim
training steps -> EDA cycling with obs -> finite metrics) on CPU.

Usage:  .venv/bin/python tests/test_kolmogorov.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from huracan.config.schema import HuracanConfig, KolmoCfg
from huracan.core.grid import planar_grid
from huracan.core.model import build_models
from huracan.core.planar_fft import PlanarSHT
from huracan.core.metrics import crps_ensemble
from huracan.core.rollout import spinup_and_forecast
from huracan.data.kolmogorov import (
    KolmogorovFlow,
    generate_kolmogorov,
    kolmogorov_steady_state,
)
from huracan.data.dataset import SphereDataset
from huracan.inference.runner import EdaRunner
from huracan.training.phases import fit_phase


def tiny_cfg(T: int = 48, N: int = 32) -> HuracanConfig:
    cfg = HuracanConfig()
    cfg.run_name = "kolmo_test"
    cfg.output_dir = tempfile.mkdtemp(prefix="kolmo_test_")
    cfg.device = "cpu"
    cfg.grid.type = "planar"
    cfg.grid.nlat, cfg.grid.nlon, cfg.grid.lmax = N, N, N - 1
    cfg.model.hidden, cfg.model.blocks = 32, 2
    cfg.model.k_filters = 4
    cfg.model.heads = 2
    cfg.model.swin_window_lat, cfg.model.swin_window_lon = 8, 16
    cfg.model.noise_dim, cfg.model.noise_embed_dim = 4, 16
    cfg.data.source = "kolmogorov"
    cfg.data.root = tempfile.mkdtemp(prefix="kolmo_data_")
    cfg.data.variables.upper = ["w", "psi", "u", "v"]
    cfg.data.variables.levels = [500]
    cfg.data.variables.surface = []
    cfg.data.forcing = []
    cfg.data.obs.enabled = True
    cfg.data.obs.surface_vars = []
    cfg.data.train_period = [0, 30]
    cfg.data.val_period = [30, T]
    cfg.data.synthetic.channels = 4
    cfg.data.synthetic.steps = T
    cfg.data.synthetic.obs_fraction = 0.15
    cfg.data.synthetic.obs_noise = 0.3
    cfg.data.synthetic.kolmogorov = KolmoCfg(
        nu=5e-3,
        kf=4,
        u_rms=1.0,
        dt=0.025,
        dt_frame=0.25,
        warmup_frames=2,
    )
    cfg.train.batch_size, cfg.train.ensemble = 2, 2
    cfg.train.num_workers = 0
    cfg.train.phases["deterministic"].steps = 20
    cfg.train.phases["assim"].steps = 20
    cfg.train.phases["assim"].rollout = 1
    cfg.train.phases["assim"].init = "deterministic"
    cfg.eval.members = 2
    cfg.eval.spinup_days = 1.0
    cfg.eval.lead_days = 1.0
    return cfg


def test_steady_state():
    """The exact laminar steady state must be preserved by the solver."""
    N, nu, kf, A = 32, 1e-2, 4, 1.0
    sim = KolmogorovFlow(N, nu, kf, np.random.default_rng(0))
    psi_star, omega_star = kolmogorov_steady_state(N, nu, kf, A)
    sim.forcing_amp = A
    sim.omega_hat = np.fft.rfft2(omega_star, norm="ortho")
    e0 = np.abs(sim.omega_hat - np.fft.rfft2(omega_star, norm="ortho")).max()
    assert e0 == 0.0
    for _ in range(100):
        sim.step(0.02)
    w, psi, u, v = sim.state()
    err = np.abs(psi - psi_star).max() / np.abs(psi_star).max()
    assert err < 1e-8, f"steady state drift {err:.2e}"
    print(f"  laminar steady state preserved (drift {err:.1e})")


def test_energy_conservation():
    """Free inviscid evolution conserves kinetic energy.

    Energy is computed on the rfft half-spectrum with Hermitian weights
    (kx > 0 columns count twice); the plain half-sum is not the true
    quadratic form and drifts even for an exactly conservative scheme."""
    N = 32
    sim = KolmogorovFlow(N, 0.0, 4, np.random.default_rng(1))
    sim.forcing_amp = 0.0
    hw = np.ones(sim.k2.shape)
    hw[:, 1:-1] = 2.0  # kx>0 double, except the Nyquist column

    def energy(sim):
        ph = sim.psi_hat(sim.omega_hat)
        return 0.5 * float((hw * sim.k2 * np.abs(ph) ** 2).sum())

    e0 = energy(sim)
    for _ in range(60):
        sim.step(0.02)
    e1 = energy(sim)
    drift = abs(e1 - e0) / e0
    assert drift < 1e-8, f"energy drift {drift:.2e}"
    print(f"  energy conserved over 60 steps: drift {drift:.1e}")


def test_planar_sht():
    torch.manual_seed(0)
    grid = planar_grid(32, 32)
    sht = PlanarSHT(grid)
    x = torch.randn(2, grid.npix, 5)
    c = sht(x)
    assert c.shape == (2, 32, 17, 5) and c.is_complex()
    x2 = sht.inverse(c)
    err = (x2 - x).abs().max().item()
    assert err < 1e-5, f"round-trip error {err:.2e}"
    # Parseval (ortho) on the rfft half-spectrum with Hermitian weights:
    # kx > 0 columns carry both signs of kx and count twice, except the
    # Nyquist column (last), which exists once
    hw = torch.ones(1, c.shape[1], c.shape[2], 1)
    hw[:, :, 1:-1] = 2.0
    p_err = (
        abs((x**2).sum().item() - (hw * c.abs() ** 2).sum().item())
        / (x**2).sum().item()
    )
    assert p_err < 1e-4, f"Parseval violation {p_err:.2e}"
    print(f"  PlanarSHT round-trip {err:.1e}, Parseval {p_err:.1e}")


def test_planar_trunk():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    fcst, assim = build_models(cfg)
    assert isinstance(fcst.sht, PlanarSHT)
    B, npix = 2, cfg.grid.nlat * cfg.grid.nlon
    n_state = cfg.n_prognostic()
    x = torch.randn(B, npix, n_state)
    f = torch.zeros(B, npix, 0)
    n_obs = cfg.n_obs_channels()
    o = torch.randn(B, npix, n_obs)
    z = torch.randn(B, npix, fcst.noise.mlp[0].in_features)
    x1, x2 = fcst.step(x, x, f, f, z=z)
    xa1, _ = assim.step(x, x, f, f, o, o, z=z)
    assert x1.shape == (B, npix, n_state)
    (x1.sum() + xa1.sum()).backward()
    # near-persistence untrained output (residual init)
    assert (x1 - x).abs().max() < 0.1
    n_par = sum(p.numel() for p in fcst.parameters())
    print(
        f"  planar trunk ok: {n_par / 1e3:.0f}K params, "
        f"untrained |delta| {(x1 - x).abs().max():.3f}"
    )


def test_pipeline():
    cfg = tiny_cfg()
    t0 = time.time()
    generate_kolmogorov(cfg, Path(cfg.data.root) / "kolmogorov", seed=0)
    root = Path(cfg.data.root) / "kolmogorov"
    state = np.load(root / "state.npy")
    obs = np.load(root / "obs.npy")
    assert state.shape == (cfg.data.synthetic.steps, 32 * 32, 4)
    assert np.isfinite(state.astype(np.float32)).all()
    assert (obs != 0).any()

    for phase in ["deterministic", "assim"]:
        res = fit_phase(cfg, phase, device="cpu")
        assert np.isfinite(res["losses"]).all()
        print(f"  phase {phase}: {len(res['losses'])} steps ok")

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
        ic_noise=0.5,
        grid_hw=(cfg.grid.nlat, cfg.grid.nlon),
    )
    truth = window["state"][sem.frames_for_calls(S) + 2 :].numpy()
    crps = crps_ensemble(preds.numpy(), truth)
    assert np.isfinite(crps).all() and (crps > 0).all()
    assert np.isfinite(spin_mean.numpy()).all()
    print(
        f"  pipeline ok: preds {tuple(preds.shape)}, CRPS {crps.mean():.3f} "
        f"({time.time() - t0:.0f}s)"
    )


def main():
    print("== kolmogorov physics ==")
    test_steady_state()
    test_energy_conservation()
    print("== planar FNO trunk ==")
    test_planar_sht()
    test_planar_trunk()
    print("== kolmogorov data + pipeline ==")
    test_pipeline()
    print("\nALL KOLMOGOROV TESTS PASSED")


if __name__ == "__main__":
    main()
