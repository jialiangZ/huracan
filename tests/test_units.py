#!/usr/bin/env python3
"""Unit tests for the refactored layers: typed config + provenance, time
semantics, SHT exactness on the Gauss grid, lat-lon Swin periodicity,
rollout parity, channel accounting, checkpoint io.

Usage:  .venv/bin/python tests/test_units.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from huracan.config import channel_report, load_config, provenance_check
from huracan.config.schema import ConfigError
from huracan.core import io
from huracan.core.grid import gauss_grid, grid_from_cfg
from huracan.core.model import build_models
from huracan.core.rollout import (
    AR_ADVANCE_HOURS,
    OUTPUT_INTERVAL_HOURS,
    OUTPUTS_PER_CALL,
    calls_for,
    outputs_for,
    unroll,
)
from huracan.core.sht import MatrixSHT, build_real_basis
from huracan.core.swin import SwinLatLonBlock

ROOT = Path(__file__).resolve().parents[1]


def make_tiny_cfg():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg.device = "cpu"
    return cfg


# ---------------------------------------------------------------------------
# config + provenance + channel accounting
# ---------------------------------------------------------------------------


def test_config_typed_and_coerced():
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    # YAML 1.1 would parse '5e-4' as a string; loader must coerce to float
    assert cfg.train.phases["deterministic"].lr == 5e-4
    assert isinstance(cfg.train.phases["deterministic"].lr, float)
    assert (cfg.grid.nlat, cfg.grid.nlon, cfg.grid.lmax) == (48, 96, 47)
    assert cfg.n_prognostic() == 5 * 13 + 7
    assert cfg.n_obs_channels() == 5 * 13 + 5
    assert cfg.model.prediction == "residual"
    print("  typed config + float coercion  OK")


def test_config_overrides():
    cfg = load_config(
        ROOT / "configs" / "smoke.yaml",
        overrides=["train.phases.deterministic.steps=99", "grid.nlon=24"],
    )
    assert cfg.train.phases["deterministic"].steps == 99
    assert cfg.grid.nlon == 24
    print("  dot-path overrides  OK")


def test_config_unknown_key_rejected():
    d = tempfile.mkdtemp()
    p = Path(d) / "bad.yaml"
    p.write_text("model:\n  hidde: 3\n")
    try:
        load_config(p)
        raise AssertionError("expected ConfigError for unknown key")
    except ConfigError as e:
        assert "hidde" in str(e)
    # invalid prediction mode must be rejected
    p.write_text("model:\n  prediction: extrapolate\n")
    try:
        load_config(p)
        raise AssertionError("expected ValueError for bad prediction mode")
    except ConfigError:
        raise
    except ValueError:
        pass
    print("  unknown key / bad value rejected  OK")


def test_channel_report_and_provenance():
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    rep = channel_report(cfg)
    assert "72" in rep and "269" in rep and "334" in rep
    assert provenance_check(cfg) == [], "stale provenance paths"
    # paper constraints file must stay loadable
    pc = load_config(ROOT / "configs" / "paper_constraints.yaml")
    assert pc.grid.lmax == 179 and pc.eval.members == 48
    assert pc.train.phases["deterministic"].steps == 347000
    print("  channel accounting + provenance registry  OK")


# ---------------------------------------------------------------------------
# time semantics (guide Sec. 1.2 / 12.3 -- the P0 correction)
# ---------------------------------------------------------------------------


def test_time_semantics():
    assert (OUTPUT_INTERVAL_HOURS, OUTPUTS_PER_CALL, AR_ADVANCE_HOURS) == (6, 2, 12)
    assert calls_for(72) == 6     # 6-step rollout = 3 days  [PAPER Sec 2.3]
    assert calls_for(96) == 8     # 8-step rollout = 4 days  [PAPER Sec 2.3]
    assert calls_for(240) == 20   # 10-day forecast
    assert outputs_for(240) == 40 # 40 six-hourly fields
    cfg = load_config(ROOT / "configs" / "paper_constraints.yaml")
    assert cfg.eval.spinup_calls() == 184  # 92 days x 2 calls/day
    try:
        calls_for(6)  # not a multiple of the 12 h advance
        raise AssertionError("expected ValueError for 6h")
    except ValueError:
        pass
    print("  time semantics: 1 AR call = 12 h, two 6 h outputs  OK")


# ---------------------------------------------------------------------------
# Gauss grid + SHT exactness
# ---------------------------------------------------------------------------


def test_grid_and_sht_exactness():
    grid = gauss_grid(16, 32, 15)  # exact: nlat = lmax+1, nlon = 2*lmax+2
    assert grid.npix == 512
    sht = MatrixSHT(grid, cache_dir="data/cache")
    # band-limited random field built from the basis itself
    rng = np.random.default_rng(0)
    u = rng.standard_normal((sht.n_real, 1))
    x = sht.B.numpy() @ u
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # (1,npix,1)
    c = sht(xt)
    x2 = sht.inverse(c)
    err = (x2 - xt).abs().max().item()
    assert err < 1e-4, f"SHT round-trip error too large: {err}"
    assert torch.is_complex(c) and c.shape[1] == grid.lmax + 1
    # quadrature forward of a basis column ~= the unit coefficient vector
    u7 = (sht.Fwd @ sht.B[:, 7]).abs().max().item()
    assert abs(u7 - 1.0) < 1e-5, "quadrature is not orthonormal"
    # invalid grids must be rejected
    for bad in [(8, 32, 15), (16, 16, 15)]:
        try:
            gauss_grid(*bad)
            raise AssertionError(f"gauss_grid{bad} should be invalid")
        except ValueError:
            pass
    print(f"  Gauss grid + exact SHT (roundtrip err {err:.1e})  OK")


# ---------------------------------------------------------------------------
# lat-lon Swin: shapes + exact longitude periodicity
# ---------------------------------------------------------------------------


def test_swin_lon_periodicity():
    torch.manual_seed(0)
    nlat, nlon, C = 16, 32, 16
    blk = SwinLatLonBlock(C, nlat, nlon, (4, 8), heads=2, mlp_ratio=2.0).eval()
    x = torch.randn(2, nlat * nlon, C)
    with torch.no_grad():
        out = blk(x)
        rolled = torch.roll(x.view(2, nlat, nlon, C), 8, dims=2).view(2, nlat * nlon, C)
        out_r = blk(rolled)
        expect = torch.roll(out.view(2, nlat, nlon, C), 8, dims=2).view(2, nlat * nlon, C)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    err = (out_r - expect).abs().max().item()
    assert err < 1e-5, f"Swin not longitude-equivariant: {err}"
    # window must divide grid
    try:
        SwinLatLonBlock(C, 16, 32, (5, 8), heads=2)
        raise AssertionError("odd window should be rejected")
    except ValueError:
        pass
    print(f"  Swin lat-lon shapes + exact lon wrap (err {err:.1e})  OK")


# ---------------------------------------------------------------------------
# rollout parity: unroll() must reproduce the original trainer semantics
# ---------------------------------------------------------------------------


def reference_unroll(models, state, forcing, obs, n_calls, split_assim, ensemble,
                     use_noise):
    """The _forward_batch loop of the original hand-rolled trainer,
    transplanted verbatim as a reference."""
    B, nframes, npix, C = state.shape
    M = ensemble if use_noise else 1
    preds = []
    for m in range(M):
        x_prev, x_cur = state[:, 0], state[:, 1]
        outs = []
        for j in range(n_calls):
            assim_call = j < split_assim
            model = models["assim" if assim_call else "fcst"]
            z = (
                torch.randn(B, npix, model.noise.mlp[0].in_features)
                if use_noise
                else None
            )
            kw = {}
            if assim_call and obs is not None:
                kw = dict(obs_tm1=obs[:, 2 * j], obs_t=obs[:, 2 * j + 1])
            x1, x2 = model.step(
                x_prev, x_cur, forcing[:, 2 * j], forcing[:, 2 * j + 1], z=z, **kw
            )
            outs += [x1, x2]
            x_prev, x_cur = x1, x2
        preds.append(torch.stack(outs, dim=1))
    preds = torch.stack(preds, dim=1)
    return preds, state[:, 2:]


def test_rollout_parity():
    cfg = make_tiny_cfg()
    torch.manual_seed(0)
    fcst, assim = build_models(cfg)
    for m in (fcst, assim):
        for p in m.parameters():
            torch.nn.init.normal_(p, std=0.02)
    models = {"assim": assim, "fcst": fcst}

    grid = grid_from_cfg(cfg.grid)
    B, npix, C = 2, grid.npix, fcst.n_state
    n_calls, split = 3, 2
    state = torch.randn(B, 2 * n_calls + 2, npix, C)
    forcing = torch.randn(B, 2 * n_calls + 2, npix, 3)
    n_obs = cfg.n_obs_channels()
    obs = torch.randn(B, 2 * n_calls + 2, npix, n_obs) if n_obs else None

    torch.manual_seed(123)
    p_new, t_new = unroll(
        models, state, forcing, obs,
        n_calls=n_calls, split_assim=split, ensemble=2, use_noise=True,
    )
    torch.manual_seed(123)
    p_ref, t_ref = reference_unroll(
        models, state, forcing, obs, n_calls, split, 2, True
    )
    assert torch.equal(p_new, p_ref), "rollout diverged from reference semantics"
    assert torch.equal(t_new, t_ref)
    assert p_new.shape == (B, 2, 2 * n_calls, npix, C)
    print("  rollout parity with original trainer semantics  OK")


# ---------------------------------------------------------------------------
# prediction parameterization flag ([REPO] residual vs direct)
# ---------------------------------------------------------------------------


def test_prediction_flag():
    cfg = make_tiny_cfg()
    grid = grid_from_cfg(cfg.grid)
    n_state, n_forc = cfg.n_prognostic(), cfg.n_forcing()
    x = torch.randn(1, grid.npix, n_state)
    f = torch.randn(1, grid.npix, n_forc)

    torch.manual_seed(0)
    fcst_res, _ = build_models(cfg)

    cfg.model.prediction = "direct"
    torch.manual_seed(0)
    fcst_dir, _ = build_models(cfg)

    s1r, s2r = fcst_res.step(x, x, f, f)
    s1d, _ = fcst_dir.step(x, x, f, f)
    # untrained residual model == persistence; direct model != persistence
    assert torch.allclose(s1r, x, atol=1e-6)
    assert not torch.allclose(s1d, x, atol=1e-3)
    print("  prediction flag: residual=persistence-init, direct=raw  OK")


# ---------------------------------------------------------------------------
# checkpoint io
# ---------------------------------------------------------------------------


def test_io_roundtrip():
    cfg = make_tiny_cfg()
    fcst, _ = build_models(cfg)
    with tempfile.TemporaryDirectory() as d:
        io.save_phase(d, "deterministic", 7, {"fcst": fcst.state_dict()})
        ck = io.load_phase(d, "deterministic")
        assert ck["phase"] == "deterministic" and ck["step"] == 7
        fcst2, _ = build_models(cfg)
        fcst2.load_state_dict(ck["models"]["fcst"])
        x = torch.randn(1, grid_from_cfg(cfg.grid).npix, fcst.n_in)
        assert torch.allclose(fcst.eval()(x), fcst2.eval()(x))
    print("  checkpoint io roundtrip  OK")


if __name__ == "__main__":
    test_config_typed_and_coerced()
    test_config_overrides()
    test_config_unknown_key_rejected()
    test_channel_report_and_provenance()
    test_time_semantics()
    test_grid_and_sht_exactness()
    test_swin_lon_periodicity()
    test_rollout_parity()
    test_prediction_flag()
    test_io_roundtrip()
    print("\nALL UNIT TESTS PASSED")
