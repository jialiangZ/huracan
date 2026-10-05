#!/usr/bin/env python3
"""Unit tests for the refactored layers: typed config, rollout parity with
the original trainer semantics, SHT injection/sharing, checkpoint io.

Usage:  .venv/bin/python tests/test_units.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from huracan.config import load_config
from huracan.config.schema import ConfigError
from huracan.core import io
from huracan.core.model import build_models
from huracan.core.rollout import unroll

ROOT = Path(__file__).resolve().parents[1]

def make_tiny_cfg():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg.device = "cpu"
    return cfg


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_typed_and_coerced():
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    # YAML 1.1 would parse '5e-4' as a string; loader must coerce to float
    assert cfg.train.phases["deterministic"].lr == 5e-4
    assert isinstance(cfg.train.phases["deterministic"].lr, float)
    assert cfg.grid.nside == 16
    assert cfg.n_prognostic() == 5 * 13 + 7
    assert cfg.n_obs_channels() == 5 * 13 + 5
    assert cfg.data.obs.surface_vars == ["2t", "2d", "10u", "10v", "msl"]
    print("  typed config + float coercion  OK")


def test_config_overrides():
    cfg = load_config(
        ROOT / "configs" / "smoke.yaml",
        overrides=["train.phases.deterministic.steps=99", "grid.nside=8"],
    )
    assert cfg.train.phases["deterministic"].steps == 99
    assert cfg.grid.nside == 8
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
    print("  unknown key rejected  OK")


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

    B, npix, C = 2, 12 * 8 * 8, fcst.n_state
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
# SHT sharing / injection
# ---------------------------------------------------------------------------


def test_sht_shared_and_injectable():
    cfg = make_tiny_cfg()
    from huracan.core import grid as G
    from huracan.core.sht import MatrixSHT

    colat, lon = G.pixel_centers(cfg.grid.nside)
    shared = MatrixSHT(
        cfg.grid.nside, cfg.grid.lmax,
        np.asarray(colat), np.asarray(lon), cache_dir="data/cache",
    )

    # injection: the models must use exactly the instance the factory returns
    fcst, assim = build_models(cfg, sht_factory=lambda: shared)
    assert fcst.sht is shared and assim.sht is shared
    # exactly one copy of the basis in the checkpoint, none under blocks
    keys = list(fcst.state_dict().keys())
    assert "sht.B" in keys
    assert not any(k.startswith("blocks.") and ".sht." in k for k in keys)
    print("  SHT injected + shared, single checkpoint copy  OK")


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
        x = torch.randn(1, 12 * 8 * 8, fcst.n_in)
        assert torch.allclose(fcst.eval()(x), fcst2.eval()(x))
    print("  checkpoint io roundtrip  OK")


if __name__ == "__main__":
    test_config_typed_and_coerced()
    test_config_overrides()
    test_config_unknown_key_rejected()
    test_rollout_parity()
    test_sht_shared_and_injectable()
    test_io_roundtrip()
    print("\nALL UNIT TESTS PASSED")
