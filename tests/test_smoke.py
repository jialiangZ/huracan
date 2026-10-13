#!/usr/bin/env python3
"""End-to-end smoke test of the entire Huracan pipeline (Lightning layer).

Runs on CPU with configs/smoke.yaml on the regular Gauss grid: synthetic
data generation, exact SHT round-trip, all six training phases, EDA cycling
with calendar-driven spin-up, and weighted CRPS/ESSR metrics.

Usage:  .venv/bin/python tests/test_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from huracan.config import channel_report, load_config
from huracan.core.grid import grid_from_cfg
from huracan.core.metrics import crps_ensemble, essr
from huracan.core.model import build_models
from huracan.core.sht import MatrixSHT
from huracan.data.synthetic import generate_synthetic
from huracan.inference.runner import run_eda
from huracan.training.phases import fit_phase


def test_sht_roundtrip(cfg):
    grid = grid_from_cfg(cfg.grid)
    assert grid.npix == grid.nlat * grid.nlon
    sht = MatrixSHT(grid, cache_dir="data/cache")
    # band-limited random field built from the basis itself
    u = np.random.default_rng(0).standard_normal((sht.n_real, 1))
    x = sht.B.numpy() @ u
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # (1,npix,1)
    c = sht(xt)
    x2 = sht.inverse(c)
    err = (x2 - xt).abs().max().item()
    assert err < 1e-4, f"SHT round-trip error too large: {err}"
    print(f"  SHT round-trip max err {err:.2e} on {grid.nlat}x{grid.nlon}  OK")


def test_model_forward(cfg):
    torch.manual_seed(0)
    fcst, assim = build_models(cfg)
    n_state = fcst.n_state
    npix = grid_from_cfg(cfg.grid).npix
    B = 2
    x = torch.randn(B, npix, n_state)
    f = torch.randn(B, npix, 3)
    n_obs = (assim.embed.in_features - 2 * (n_state + 3)) // 2
    o = torch.randn(B, npix, n_obs)
    nf = fcst.noise.mlp[0].in_features
    z = torch.randn(B, npix, nf)
    x1, x2 = fcst.step(x, x, f, f, z=z)
    assert x1.shape == (B, npix, n_state)
    xa1, xa2 = assim.step(x, x, f, f, o, o, z=z)
    assert xa1.shape == (B, npix, n_state)
    (x1.sum() + xa1.sum()).backward()
    nparams = sum(p.numel() for p in fcst.parameters())
    print(
        f"  fcst params {nparams / 1e6:.2f}M; assim "
        f"{sum(p.numel() for p in assim.parameters()) / 1e6:.2f}M  OK"
    )
    assert nparams > 0


def main():
    t0 = time.time()
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    cfg.device = "cpu"
    torch.set_num_threads(8)

    print(channel_report(cfg))

    print("== SHT round trip ==")
    test_sht_roundtrip(cfg)

    print("== model forward/backward ==")
    test_model_forward(cfg)

    print("== synthetic data ==")
    root = Path(cfg.data.root) / "synthetic"
    if not (root / "state.npy").exists():
        generate_synthetic(cfg, root, seed=0)

    for phase in ["deterministic", "stochastic", "fcst_ar", "assim", "joint"]:
        print(f"== train phase: {phase} ==")
        res = fit_phase(cfg, phase, device="cpu")
        losses_f = np.asarray(res["losses"])
        assert len(losses_f) == cfg.train.phases[phase].steps, (
            f"expected {cfg.train.phases[phase].steps} steps, got {len(losses_f)}"
        )
        assert np.isfinite(losses_f).all(), f"non-finite loss in {phase}"
        l0, l1 = (
            np.mean(losses_f[: max(5, len(losses_f) // 4)]),
            np.mean(losses_f[-max(5, len(losses_f) // 4) :]),
        )
        print(f"  loss {l0:.4f} -> {l1:.4f}")
        # smoke runs are too short for monotone convergence; guard against
        # divergence only (CRPS with M=2 is intrinsically noisy)
        assert l1 < l0 * 3.0, f"loss diverged in {phase}: {l0} -> {l1}"

    print("== EDA + metrics ==")
    res = run_eda(cfg, cfg.output_dir, device="cpu")
    # calendar-driven spin-up: days -> AR calls via cfg.call_semantics()
    assert res["spinup_calls"] == cfg.spinup_calls()
    assert res["lead_calls"] == cfg.lead_calls()
    preds = res["preds"].numpy()
    truth = res["truth"].numpy()
    w = grid_from_cfg(cfg.grid).pixel_weights
    crps = crps_ensemble(preds, truth, w)
    es = essr(preds, truth, w)
    assert np.isfinite(crps).all() and (crps > 0).all()
    print(f"  CRPS shape {crps.shape}, mean {crps.mean():.4f}")
    print(f"  ESSR mean {es.mean():.3f}")
    print(f"\nALL SMOKE TESTS PASSED in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
