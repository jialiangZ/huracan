#!/usr/bin/env python3
"""Physics + pipeline tests for the Lorenz-96 data source (DATA_PLAN Sec.3).

Physics tests (Huracan-agnostic, pure integrator):
  * attractor climatology: F=8 variance in the literature range, bounded
  * largest Lyapunov exponent ~0.9 (error-growth fit, loose bounds)
  * two-scale variant stays finite/bounded at the default substeps

Pipeline tests: generate tiny datasets, verify the standard file layout
(state/obs/forcing/norms/times), the missing-data obs convention, window
buffering at trajectory boundaries, and that SphereDataset serves the
data unchanged.  Also exercises the parameterized frame semantics
(calls_for_frames + eval spinup/lead call overrides).

Usage:  .venv/bin/python tests/test_lorenz96.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from huracan.config.schema import HuracanConfig
from huracan.data.dataset import SphereDataset
from huracan.data.lorenz96 import generate_lorenz96, make_obs_variant, run_1l
from huracan.time import CallSemantics


def l96_cfg(train=400, val=200, test=200, warmup=500, two_scale=False):
    cfg = HuracanConfig()
    cfg.output_dir = tempfile.mkdtemp(prefix="l96_run_")
    cfg.device = "cpu"
    cfg.data.source = "lorenz96_2l" if two_scale else "lorenz96"
    cfg.data.root = tempfile.mkdtemp(prefix="l96_data_")
    cfg.data.variables.upper = ["x"]
    cfg.data.variables.levels = [500]  # placeholder level name only
    cfg.data.variables.surface = []
    cfg.data.forcing = ["f"]
    cfg.data.obs.enabled = True
    cfg.data.obs.surface_vars = []
    cfg.grid.nlat, cfg.grid.nlon, cfg.grid.lmax = (
        1,
        (36 if two_scale else 40),
        (35 if two_scale else 39),
    )
    cfg.model.backbone = "mlp1d"
    cfg.model.hidden, cfg.model.blocks = 32, 2
    cfg.model.noise_dim, cfg.model.noise_embed_dim = 4, 16
    cfg.model.conv_kernel = 7
    l = cfg.data.lorenz96
    l.train_steps, l.val_steps, l.test_steps = train, val, test
    l.warmup_steps = warmup
    l.two_scale = two_scale
    return cfg


def test_climatology():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(40)
    traj = run_1l(x, 8000, F=8.0, dt=0.05)
    traj = traj[1000:]  # discard transient
    var = traj.var()
    assert np.isfinite(traj).all()
    assert np.abs(traj).max() < 30.0, "attractor excursion too large"
    assert 4.0 < var < 16.0, f"climatological variance {var:.1f} out of range"
    print(f"  climatology ok: var {var:.2f}, max |x| {np.abs(traj).max():.1f}")


def test_lyapunov():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(40)
    traj = run_1l(x, 1200, F=8.0, dt=0.05)  # transient -> attractor IC
    x0, xp = traj[-1].copy(), traj[-1].copy()
    xp[0] += 1e-6
    a = run_1l(x0, 1200, F=8.0, dt=0.05)
    b = run_1l(xp, 1200, F=8.0, dt=0.05)
    err = np.linalg.norm(a - b, axis=1)
    t = np.arange(err.size) * 0.05
    m = (err > 1e-4) & (err < 3.0)  # growth regime before saturation
    lam = np.polyfit(t[m], np.log(err[m]), 1)[0]
    assert 0.5 < lam < 1.5, f"lambda_1 = {lam:.2f} out of range"
    print(f"  lyapunov ok: lambda_1 = {lam:.2f} (literature ~0.9)")


def _check_dataset(cfg, res, two_scale: bool):
    root = Path(cfg.data.root) / cfg.data.source
    state = np.load(root / "state.npy")
    obs = np.load(root / "obs.npy")
    K = cfg.data.lorenz96.n_slow if two_scale else cfg.data.lorenz96.n_vars
    T = sum(
        [
            cfg.data.lorenz96.train_steps,
            cfg.data.lorenz96.val_steps,
            cfg.data.lorenz96.test_steps,
        ]
    )
    assert state.shape == (T, K, 1), state.shape
    assert obs.shape == (T, K, 1), obs.shape
    assert np.isfinite(state.astype(np.float32)).all()
    s32 = state.astype(np.float32)
    assert np.abs(s32).max() < 40.0, "state not on the attractor"
    lcfg = cfg.data.lorenz96
    sites = np.arange(0, K, lcfg.obs_every)
    hidden = np.setdiff1d(np.arange(K), sites)
    # missing-data convention: unobserved variables are EXACTLY zero
    assert np.abs(obs[:, hidden, 0]).max() == 0.0
    # observed values: normalized truth+noise -> std slightly above 1
    vals = obs[:, sites, 0].astype(np.float32)
    assert np.abs(vals).max() > 0.5
    assert vals.std() > 0.9, f"obs std {vals.std():.2f} -- obs missing noise?"
    with open(root / "times.json") as f:
        times = json.load(f)
    assert len(times) == T and times[1] == lcfg.dt

    # windows must never splice two trajectories: with the suggested
    # periods every training window stays inside the train segment
    for split in ("train", "val", "test"):
        setattr(cfg.data, f"{split}_period", res["periods"][f"{split}_period"])
    b = res["boundaries"]
    for split, lo, hi in (
        ("train", 0, b[1]),
        ("val", b[1], b[2]),
        ("test", b[2], b[3]),
    ):
        ds = SphereDataset(cfg, split, rollout=8)
        for t0 in (ds.first_t0, int(ds.valid_t0[-1])):
            w = ds.window(t0, 18)
            assert w["state"].shape == (18, K, 1)
            assert w["obs"].shape[-1] == 1
            idx = np.arange(t0 - 1, t0 - 1 + 18)
            assert idx.min() >= lo and idx.max() < hi, (
                f"{split} window spliced trajectories: {idx.min()}..{idx.max()}"
                f" vs segment [{lo},{hi})"
            )
    return root


def test_generate_1l_and_dataset():
    t0 = time.time()
    cfg = l96_cfg()
    res = generate_lorenz96(cfg, Path(cfg.data.root) / cfg.data.source, seed=0)
    root = _check_dataset(cfg, res, two_scale=False)
    print(
        f"  1L dataset + windows ok ({time.time() - t0:.1f}s): "
        f"T={sum(res['boundaries'][1:])}, periods {res['periods']}"
    )

    # obs sweeps reuse the nature run (only obs.npy regenerated)
    var_root = make_obs_variant(
        cfg,
        root,
        Path(cfg.data.root) / "lorenz96_sparse",
        obs_every=4,
        obs_sigma=0.5,
    )
    obs2 = np.load(var_root / "obs.npy")
    assert obs2.shape == (800, 40, 1)
    assert np.abs(obs2[:, 1::2, 0]).max() == 0.0  # odd never reports
    assert np.abs(obs2[:, 2, 0]).max() == 0.0  # k=2 outside every-4th net
    assert np.abs(obs2[:, 4, 0]).max() > 0.0  # k=4 reports
    print("  obs variant (every 4th, sigma 0.5) ok")


def test_generate_2l():
    t0 = time.time()
    cfg = l96_cfg(train=200, val=100, test=100, warmup=300, two_scale=True)
    res = generate_lorenz96(cfg, Path(cfg.data.root) / cfg.data.source, seed=0)
    _check_dataset(cfg, res, two_scale=True)
    print(
        f"  2L dataset ok ({time.time() - t0:.1f}s): state X-only, "
        f"Y = unresolved subgrid"
    )


def test_frame_semantics():
    sem = CallSemantics()  # paper default: 6 h frames, 2 outputs per call
    # atmosphere paper contract unchanged
    assert sem.calls_for_hours(72) == 6 and sem.outputs_for_hours(96) == 16
    # dataset-agnostic frame counting
    assert sem.calls_for_frames(18) == 9 and sem.calls_for_frames(2) == 1
    assert sem.window_frames(8) == 18 and sem.frames_for_calls(8) == 16
    try:
        sem.calls_for_frames(17)
        raise AssertionError("17 frames must not be divisible")
    except ValueError:
        pass
    cfg = HuracanConfig()
    assert cfg.spinup_calls() == 120  # 60 d x 2 calls/day
    cfg.eval.spinup_calls_override = 25
    cfg.eval.lead_calls_override = 50
    assert cfg.spinup_calls() == 25 and cfg.lead_calls() == 50
    print("  frame semantics ok: hours API intact, frames API + call overrides active")


def test_l96_pipeline():
    """Tiny end-to-end: generate -> deterministic+assim training on the
    mlp1d backbone -> EDA cycling with observations -> finite CRPS."""
    import torch

    from huracan.core.metrics import crps_ensemble
    from huracan.core.rollout import spinup_and_forecast
    from huracan.inference.runner import EdaRunner
    from huracan.training.phases import fit_phase

    cfg = l96_cfg()
    res = generate_lorenz96(cfg, Path(cfg.data.root) / cfg.data.source, seed=0)
    for split in ("train", "val", "test"):
        setattr(cfg.data, f"{split}_period", res["periods"][f"{split}_period"])

    cfg.train.batch_size, cfg.train.ensemble, cfg.train.num_workers = 4, 2, 0
    cfg.train.phases["deterministic"].steps = 25
    cfg.train.phases["assim"].steps = 25
    cfg.train.phases["assim"].rollout = 1
    cfg.train.phases["assim"].init = "deterministic"  # skip stochastic chain
    cfg.eval.members = 2
    cfg.eval.spinup_calls_override = 2
    cfg.eval.lead_calls_override = 2

    for phase in ("deterministic", "assim"):
        r = fit_phase(cfg, phase, device="cpu")
        assert np.isfinite(r["losses"]).all(), f"non-finite loss in {phase}"
        print(
            f"  phase {phase}: loss "
            f"{np.mean(r['losses'][:5]):.4f} -> "
            f"{np.mean(r['losses'][-5:]):.4f}"
        )

    runner = EdaRunner.from_checkpoints(cfg, cfg.output_dir, device="cpu")
    S, F = cfg.spinup_calls(), cfg.lead_calls()
    sem = cfg.call_semantics()
    ds = SphereDataset(cfg, "val", rollout=S + F)
    window = ds.window(ds.first_t0, sem.window_frames(S + F))
    assert window["state"].shape[1] == 40 and window["obs"].shape[-1] == 1
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
    truth = window["state"][
        sem.frames_for_calls(S)
        + 2 : sem.frames_for_calls(S)
        + 2
        + sem.frames_for_calls(F)
    ].numpy()
    crps = crps_ensemble(preds.numpy(), truth)
    assert np.isfinite(crps).all() and (crps > 0).all()
    assert np.isfinite(spin_mean.numpy()).all()
    print(f"  EDA ok: preds {tuple(preds.shape)}, CRPS mean {crps.mean():.3f}")


def main():
    print("== lorenz96 physics ==")
    test_climatology()
    test_lyapunov()
    print("== lorenz96 data + pipeline ==")
    test_generate_1l_and_dataset()
    test_generate_2l()
    test_l96_pipeline()
    print("== frame semantics ==")
    test_frame_semantics()
    print("\nALL LORENZ96 TESTS PASSED")


if __name__ == "__main__":
    main()
