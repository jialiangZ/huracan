#!/usr/bin/env python3
"""P0 regression tests (RESEARCH_PLAN v9 / 4090 plan Sec. 2).

Every fix from the audit gets a pinned test here so it can never silently
regress:

  R1  ISD/IGRA wind sentinels are checked on the RAW integer before any
      unit conversion (missing -9999 -> no u/v; the old code produced
      +/-999.9 m/s components for ~5% of real ISD rows).
  R2  physical zero wind (calm report) is a *valid* observation and gets
      normalized like any other value (validity from counts, not != 0).
  R3  IGRA unknown launch hour (99) and missing lat/lon are skipped.
  R4  snap_to_frame is a forward window (t-6h, t]: 22:00/23:00 reports land
      on the NEXT day's 00:00 frame with a preserved offset; nothing inside
      the window is dropped for being near midnight.
  R5  build_obs_tensor writes the validity bundle (mask/counts/offsets) and
      a genuine zero report survives normalization as a non-zero normalized
      value flagged valid.
  R6  SphereDataset windows never cross the split boundary (the full
      [t0-1, t0-1+frame_window) block stays inside the split).
  R7  Normalizer.from_array(frames=...) fits only the requested frames;
      changing other frames cannot move the fit (no val/test leak).
  R8  metrics: empirical vs fair CRPS hand checks ([0,2] vs y=1 -> 0.5 / 0),
      ESSR rms/mean_std distinction, M=1 degenerate rule.
  R9  losses accept pixel weights and match a manual weighted mean.
  R10 bsef_r0 split indices are EXACT calendar-year boundaries of the real
      times.json (skipped when data/era5 is absent).
  R11 single-output frame arithmetic: 8 calls need 10 frames; 240-call
      spin-up consumes 241 obs frames.
  R12 run() exposes persistence_init = the ERA5 field at forecast start
      (never a future verification field).

Usage:  .venv/bin/python tests/test_p0_regressions.py
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from huracan.config import load_config
from huracan.core.grid import grid_from_cfg
from huracan.core.metrics import crps_ensemble, crps_fair, essr
from huracan.core.losses import crps_loss, l1_loss
from huracan.data.dataset import SphereDataset
from huracan.data.norm import Normalizer
from huracan.data.obs import (
    build_obs_tensor,
    parse_igra_zip,
    parse_isd_gz,
    snap_to_frame,
)
from huracan.time import CallSemantics

ROOT = Path(__file__).resolve().parents[1]


def _isd_line(y, mo, d, h, temp, dew, slp, wdir, wspd, width=6):
    def f(v):
        return str(v).rjust(width)

    return (f"{y} {mo:02d} {d:02d} {h:02d} " + f(temp) + f(dew) + f(slp)
            + f(wdir) + f(wspd) + " -9999 -9999 -9999\n")


def test_isd_sentinels_and_physical_zero():  # R1 + R2
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x-2010.gz"
        # row 1: wspd missing, dir valid  -> must NOT produce wind components
        # row 2: calm report dir=0 spd=0  -> 10u = 10v = 0.0 (valid zeros)
        # row 3: temp missing             -> no 2t
        rows = (
            _isd_line(2010, 1, 1, 0, 101, 50, 10130, 130, -9999)
            + _isd_line(2010, 1, 1, 1, 102, 51, 10131, 0, 0)
            + _isd_line(2010, 1, 1, 2, -9999, 52, 10132, 200, 35)
        )
        with gzip.open(p, "wt") as f:
            f.write(rows)
        out = list(parse_isd_gz(p))
    assert len(out) == 3, out
    t0, v0 = out[0]
    assert "10u" not in v0 and "10v" not in v0, "missing wspd leaked wind"
    assert "2t" in v0  # valid temp kept
    _, v1 = out[1]
    assert v1["10u"] == 0.0 and v1["10v"] == 0.0, "calm report lost"
    _, v2 = out[2]
    assert "2t" not in v2 and v2["10u"] != 0.0 or v2["10u"] == 0.0  # dir 200 -> u != 0
    assert abs(v2["10u"]) > 0.1, "valid wind corrupted"
    print("  R1/R2 ISD sentinel + physical-zero handling      OK")


def _igra_head(hour: int) -> str:
    buf = [" "] * 80

    def put(start, text):
        buf[start : start + len(text)] = list(text)

    put(0, "#" + "X" * 11)
    put(13, "2010")
    put(18, "01")
    put(21, "01")
    put(24, f"{hour:02d}")
    put(27, "0000")
    put(31, "2".rjust(5))
    put(37, "ncdc-gts")
    put(55, "367000")  # lat 36.7 (x1e4)
    put(63, "672000")  # lon 67.2 (x1e4)
    return "".join(buf).rstrip() + "\n"


def _igra_level(p_pa, gph, temp_x10, rh, wdir, wspd_x10) -> str:
    buf = [" "] * 60

    def put(start, text):
        buf[start : start + len(text)] = list(text)

    put(9, str(p_pa).rjust(6))
    put(16, str(gph).rjust(5))
    put(22, str(temp_x10).rjust(5))
    put(28, str(rh).rjust(5))
    put(40, str(wdir).rjust(5))
    put(46, str(wspd_x10).rjust(5))
    return "".join(buf).rstrip() + "\n"


def test_igra_sentinels_and_hour99():  # R1 + R3
    # level 1: wind speed missing (-9999), dir valid -> no u/v
    # level 2: wind dir missing, speed valid -> no u/v
    lev_missing = _igra_level(50000, 5500, -70, 50, 270, -9999)
    lev_missing2 = _igra_level(50000, 5500, -70, 50, -9999, 123)
    with tempfile.TemporaryDirectory() as d:
        z = Path(d) / "XXXXXXXXXXX.zip"
        with zipfile.ZipFile(z, "w") as zf:
            # hour 99 (unknown -> skip) and hour 00 (keep), 2 levels each
            zf.writestr(
                "data.txt",
                _igra_head(99) + lev_missing + lev_missing2
                + _igra_head(0) + lev_missing + lev_missing2,
            )
        profs = list(parse_igra_zip(z, [500]))
    assert len(profs) == 1, f"hour-99 launch must be skipped, got {len(profs)}"
    prof = profs[0][3]
    assert 500 in prof and "t" in prof[500], "valid T kept"
    assert "u" not in prof[500] and "v" not in prof[500], (
        "missing wspd or wdir leaked wind into the profile"
    )
    assert profs[0][0] == dt.datetime(2010, 1, 1, 0)
    print("  R1/R3 IGRA sentinel + hour-99 skip              OK")


def test_snap_to_frame_forward_window():  # R4
    base = dt.datetime(2010, 1, 1)
    time_idx = {
        base + dt.timedelta(hours=h): i
        for i, h in enumerate(range(0, 24 * 3, 6))
    }
    cases = [
        (dt.datetime(2010, 1, 1, 0, 0), 0, 0.0),    # exactly on frame
        (dt.datetime(2010, 1, 1, 0, 30), 1, -5.5),  # (0,6] window
        (dt.datetime(2010, 1, 1, 3, 0), 1, -3.0),
        (dt.datetime(2010, 1, 1, 6, 0), 1, 0.0),
        (dt.datetime(2010, 1, 1, 18, 0), 3, 0.0),
        (dt.datetime(2010, 1, 1, 22, 0), 4, -2.0),  # next day 00:00
        (dt.datetime(2010, 1, 1, 23, 59), 4, -1 / 60),
        (dt.datetime(2010, 1, 2, 0, 0), 4, 0.0),
    ]
    for t, want_idx, want_off in cases:
        idx, off = snap_to_frame(t, time_idx)
        assert idx == want_idx, (t, idx, want_idx)
        assert abs(off - want_off) < 1e-6, (t, off, want_off)
        assert -6.0 < off <= 0.0
    # outside the frame index range -> None
    idx, off = snap_to_frame(dt.datetime(2010, 1, 30, 1, 0), time_idx)
    assert idx is None and off is None
    print("  R4 forward-window snap + offsets                 OK")


def test_build_obs_tensor_bundle():  # R5
    cfg = load_config(ROOT / "configs" / "osse_smoke.yaml")
    grid = grid_from_cfg(cfg.grid)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "isd").mkdir()
        (d / "igra").mkdir()
        times = [
            (dt.datetime(2010, 1, 1) + dt.timedelta(hours=6 * i)).isoformat()
            for i in range(8)
        ]
        json.dump(times, open(d / "times.json", "w"))
        # one ISD station at grid point nearest to (0.1, 0.1) -> colat/lon
        station_lat, station_lon = 0.0, 0.0
        csv = "sid,lat,lon\nTEST01,0.0,0.0\n"
        open(d / "isd_stations.csv", "w").write(csv)
        rows = "".join(
            [
                _isd_line(2010, 1, 1, 0, 150, 100, 10150, 90, 30),   # u=-3
                _isd_line(2010, 1, 1, 6, 150, 100, 10150, 0, 0),     # calm!
            ]
        )
        with gzip.open(d / "isd" / "TEST01-2010.gz", "wt") as f:
            f.write(rows)
        names = cfg.var_names()
        # means=1, stds=1: a physical zero normalizes to -1 (inside the
        # +/-8 clip), distinguishable from the exact-0 missing encoding
        norm = Normalizer(names, np.ones(len(names)), np.ones(len(names)))
        out = build_obs_tensor(cfg, d, d / "times.json", norm, d / "obs.npy",
                               workers=2)
        assert Path(out).exists()
        obs = np.load(d / "obs.npy")
        mask = np.unpackbits(
            np.load(d / "obs_mask.npy"), axis=-1, count=obs.shape[-1]
        ).astype(bool)
        counts = np.load(d / "obs_counts.npy")
        offs = np.load(d / "obs_offsets.npy")
        audit = json.load(open(d / "obs_audit.json"))
        # two frames carry reports; the calm report must be valid and
        # normalized (physical 0 m/s -> (0-mean)/std != 0 in general)
        frames = np.unique(np.nonzero(mask)[0])
        assert list(frames) == [0, 1]
        u_ch = cfg.n_obs_channels() - 5 + 2  # surface block: 2t,2d,10u,10v,msl
        px = np.nonzero(mask[0, :, u_ch])[0]
        assert len(px) == 1
        assert obs[0, px[0], u_ch] != 0.0  # -3 m/s normalized
        # calm frame: 10u valid, value = (0 - 1)/1 = -1, NOT the missing 0
        assert mask[1, px[0], u_ch], "calm (zero) report must be VALID"
        assert abs(obs[1, px[0], u_ch] - (-1.0)) < 1e-2
        assert counts[0, px[0]] >= 3 and counts[2, px[0]] == 0
        assert offs[0, px[0], u_ch] == 0  # hourly report at 00:00, offset 0
        assert offs[1, px[0], u_ch] == 0
        assert audit["n_reports_total"] >= 6
    print("  R5 obs bundle (mask/counts/offsets/audit)        OK")


def test_dataset_window_stays_in_split():  # R6
    cfg = load_config(ROOT / "configs" / "osse_smoke.yaml")
    cfg.model.outputs_per_call = 1
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        T, npix, C = 40, grid_from_cfg(cfg.grid).npix, cfg.n_prognostic()
        rng = np.random.default_rng(0)
        np.save(d / "state.npy", rng.standard_normal((T, npix, C)).astype(np.float32))
        np.save(d / "forcing.npy", rng.standard_normal((T, npix, 3)).astype(np.float16))
        np.save(d / "obs.npy", rng.standard_normal((T, npix, cfg.n_obs_channels())).astype(np.float16))
        json.dump(
            [(dt.datetime(2010, 1, 1) + dt.timedelta(hours=6 * i)).isoformat()
             for i in range(T)],
            open(d / "times.json", "w"),
        )
        Normalizer(
            cfg.var_names(), np.zeros(C), np.ones(C)
        ).save(d / "norm_state.json")
        cfg.data.train_period = [0, 20]
        cfg.data.val_period = [20, 40]
        fw = cfg.call_semantics().window_frames(8)  # 2 + 8 = 10 frames
        ds = SphereDataset(cfg, "train", rollout=8, data_dir=d)
        for t0 in ds.valid_t0:
            assert t0 - 1 >= 0 and t0 - 1 + fw <= 20, (t0, "crosses train end")
        assert ds.valid_t0[-1] - 1 + fw == 20, "must use the full split, no off-by-one"
        dsv = SphereDataset(cfg, "val", rollout=8, data_dir=d)
        for t0 in dsv.valid_t0:
            assert t0 - 1 >= 20 and t0 - 1 + fw <= 40, (t0, "crosses val end")
        ds_strided = SphereDataset(cfg, "val", rollout=8, data_dir=d, stride=4)
        assert len(ds_strided.valid_t0) == int(np.ceil(len(dsv.valid_t0) / 4))
        # single-output L=8 needs 10 frames (audit example: t0=4379 with a
        # 4380 boundary must now be illegal)
        assert fw == 10
        assert 20 - fw + 1 not in set()  # tautology guard for readability
        assert ds.valid_t0[-1] == 20 - fw + 1
    print("  R6 dataset windows never cross the split         OK")


def test_normalizer_frames_arg():  # R7
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((24, 8, 3)).astype(np.float32) * [1, 10, 100]
    arr[12:] += 5.0  # "validation" part differs
    names = ["a", "b", "c"]
    n_train = Normalizer.from_array(names, arr, frames=np.arange(0, 12))
    manual_mean = arr[:12].mean(axis=(0, 1))
    assert np.allclose(n_train.mean, manual_mean, rtol=1e-5)
    assert np.allclose(
        n_train.std, arr[:12].std(axis=(0, 1)) + 1e-6, rtol=1e-4
    )
    arr2 = arr.copy()
    arr2[12:] *= 100  # mutate everything after the fit window
    n2 = Normalizer.from_array(names, arr2, frames=np.arange(0, 12))
    assert np.allclose(n_train.mean, n2.mean), "val frames leaked into the fit"
    # backward-compat path still works
    _ = Normalizer.from_array(names, arr[:2])
    print("  R7 normalizer train-only fit                     OK")


def test_metrics_hand_checks():  # R8
    preds = np.array([[[[0.0]]], [[[2.0]]]])  # members [0,2]
    truth = np.array([[[1.0]]])
    e = crps_ensemble(preds, truth)[0, 0]
    f = crps_fair(preds, truth)[0, 0]
    assert abs(e - 0.5) < 1e-9, e
    assert abs(f - 0.0) < 1e-9, f
    # M=1 degenerate: both reduce to E|X-y|
    p1 = np.array([[[[3.0]]]])
    assert abs(crps_ensemble(p1, truth)[0, 0] - 2.0) < 1e-9
    assert abs(crps_fair(p1, truth)[0, 0] - 2.0) < 1e-9
    # ESSR modes: rms vs mean of local std (non-constant spread per pixel)
    pr = np.array([[[0.5, 2.0]], [[-0.5, -2.0]]])  # stds ~ [.707, 2.83]
    tr = np.array([[0.3, -0.3]])
    r_rms = essr(pr, tr, mode="rms")[0]
    r_ms = essr(pr, tr, mode="mean_std")[0]
    assert r_rms > r_ms  # sqrt(mean(s^2)) >= mean(s), strict when s varies
    print("  R8 metrics empirical/fair/ESSR modes            OK")


def test_weighted_losses():  # R9
    # NB: B, M, D are all distinct so accidental broadcasting cannot pass
    B, M, D, P, C = 3, 2, 4, 5, 2
    tgt = torch.randn(B, D, P, C)
    w = torch.rand(P) + 0.1
    pred = torch.randn(B, D, P, C)
    got = l1_loss(pred, tgt, w)
    flat = (pred - tgt).abs().mean(-1)  # (B,D,P)
    want = ((flat * w).sum(-1) / w.sum()).mean()
    assert torch.allclose(got, want, rtol=1e-5)
    ens = torch.randn(B, M, D, P, C)
    got_c = crps_loss(ens, tgt, w)
    t1 = (ens - tgt.unsqueeze(1)).abs().mean(dim=1)  # E_m |X - y|
    diff = (ens.unsqueeze(2) - ens.unsqueeze(1)).abs()
    off = diff.sum(dim=(1, 2)) / (M * (M - 1))
    base = t1 - 0.5 * off  # (B,D,P,C)
    want_c = ((base * w[..., None]).sum(-1) / w.sum()).mean()
    assert torch.allclose(got_c, want_c, rtol=1e-5)
    # unweighted path unchanged vs plain mean
    assert torch.allclose(l1_loss(pred, tgt), (pred - tgt).abs().mean())
    assert torch.allclose(
        crps_loss(ens, tgt), (t1 - 0.5 * off).mean()
    )
    print("  R9 weighted losses                               OK")


def test_bsef_split_exact_years():  # R10
    times_path = ROOT / "data" / "era5" / "times.json"
    if not times_path.exists():
        print("  R10 skipped (no data/era5/times.json on this machine)")
        return
    times = [dt.datetime.fromisoformat(s) for s in json.load(open(times_path))]
    # continuity: strictly 6-hourly steps, no gaps/duplicates
    deltas = {(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)}
    assert deltas == {6 * 3600}, deltas
    cfg = load_config(ROOT / "configs" / "bsef_r0.yaml")
    for name, (lo, hi), want in [
        ("train", cfg.data.train_period, ("2010-01-01T00:00:00", "2011-12-31T18:00:00")),
        ("val", cfg.data.val_period, ("2012-01-01T00:00:00", "2012-12-31T18:00:00")),
        ("test", cfg.data.test_period, ("2013-01-01T00:00:00", "2013-12-31T18:00:00")),
    ]:
        assert times[lo].isoformat() == want[0], (name, times[lo])
        assert times[hi - 1].isoformat() == want[1], (name, times[hi - 1])
    print("  R10 bsef splits are exact calendar boundaries     OK")


def test_single_output_arithmetic():  # R11
    sem = CallSemantics(outputs_per_call=1)
    assert sem.window_frames(8) == 10
    assert sem.obs_frames(240) == 241
    assert sem.calls_for_hours(24) == 4
    assert sem.advance_hours == 6.0
    print("  R11 single-output frame arithmetic               OK")


def test_persistence_init_field():  # R12
    cfg = load_config(ROOT / "configs" / "osse_smoke.yaml")
    cfg.model.outputs_per_call = 1
    from huracan.core.model import build_models

    assim, fcst = build_models(cfg)[1], build_models(cfg)[0]
    runner_mod = __import__(
        "huracan.inference.runner", fromlist=["EdaRunner"]
    ).EdaRunner
    r = runner_mod(assim, fcst, device="cpu")
    S, F, M = 2, 3, 2
    fw = cfg.call_semantics().window_frames(S + F)
    T, npix, C = fw, grid_from_cfg(cfg.grid).npix, cfg.n_prognostic()
    state = torch.randn(T, npix, C)
    window = {
        "state": state,
        "forcing": torch.randn(T, npix, 3),
        "obs": torch.randn(cfg.call_semantics().obs_frames(S), npix,
                           cfg.n_obs_channels()),
    }
    res = r.run(window, t0=1, members=M, spinup=S, lead=F)
    s0 = 1 * S + 2
    assert torch.equal(res["persistence_init"], state[s0 - 1])
    assert res["persistence_init"].shape == (npix, C)
    print("  R12 persistence init = forecast-start field      OK")


if __name__ == "__main__":
    test_isd_sentinels_and_physical_zero()
    test_igra_sentinels_and_hour99()
    test_snap_to_frame_forward_window()
    test_build_obs_tensor_bundle()
    test_dataset_window_stays_in_split()
    test_normalizer_frames_arg()
    test_metrics_hand_checks()
    test_weighted_losses()
    test_bsef_split_exact_years()
    test_single_output_arithmetic()
    test_persistence_init_field()
    print("\nALL P0 REGRESSION TESTS PASSED")
