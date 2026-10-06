#!/usr/bin/env python3
"""Convert downloaded ERA5 GRIB into the pipeline's HEALPix .npy format.

- bilinear interpolation from the regular 1-degree lat-lon grid to NESTED
  HEALPix pixels (precomputed stencil, lon wraparound, no per-call trees)
- variable stacking into (T, npix, C) fp16
- analytic insolation forcing + static surface geopotential / land-sea mask
- per-variable normalization statistics + 6-hourly UTC calendar

Usage:
  python scripts/prepare_era5.py --config configs/pilot.yaml --years 2010 2011 2012 2013
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from huracan.config import load_config, var_names
from huracan.core.grid import grid_from_cfg
from huracan.data.interp import LatLonBilinear
from huracan.data.norm import Normalizer


def stack_year(path_pl, path_sl, interp, names_idx, levels, npix):
    """One year of GRIB -> (times, arr[T, npix, C])."""
    pl = xr.open_dataset(path_pl, engine="cfgrib")
    sl = xr.open_dataset(path_sl, engine="cfgrib")
    times = pd.to_datetime(sl.valid_time.values)
    T = len(times)
    arr = np.zeros((T, npix, len(names_idx)), dtype=np.float32)

    for short, cf in {"z": "z", "t": "t", "u": "u", "v": "v", "q": "q"}.items():
        da = pl[cf]
        levs = da.isobaricInhPa.values
        for lev in levels:
            j = int(np.argmin(np.abs(levs - lev)))
            assert abs(levs[j] - lev) < 1e-6, f"missing level {lev}"
            sub = da.isel(isobaricInhPa=j)  # lazy (time, lat, lon)
            tgt = names_idx[f"{short}{lev}"]
            for t in range(T):
                arr[t, :, tgt] = interp(sub.isel(valid_time=t).values)
        print(f"    {short} done", flush=True)

    for short, cf in {
        "2t": "t2m",
        "2d": "d2m",
        "10u": "u10",
        "10v": "v10",
        "tcc": "tcc",
        "tcwv": "tcwv",
        "msl": "msl",
    }.items():
        sub = sl[cf]
        tgt = names_idx[short]
        for t in range(T):
            arr[t, :, tgt] = interp(sub.isel(valid_time=t).values)
        print(f"    {short} done", flush=True)

    pl.close()
    sl.close()
    return times, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rawdir", default="data/raw")
    ap.add_argument("--years", type=int, nargs="+", default=[2010, 2011, 2012, 2013])
    args = ap.parse_args()

    cfg = load_config(args.config)
    grid = grid_from_cfg(cfg.grid)
    colat, lon = grid.colat, grid.lon_flat
    outdir = Path(cfg.data.root) / "era5"
    outdir.mkdir(parents=True, exist_ok=True)
    raw = Path(args.rawdir)

    names = var_names(cfg)
    names_idx = {n: i for i, n in enumerate(names)}
    levels = cfg.data.variables.levels

    # grid geometry from the first file
    probe = xr.open_dataset(raw / f"era5_pl_{args.years[0]}.grib", engine="cfgrib")
    interp = LatLonBilinear(probe.latitude.values, probe.longitude.values, colat, lon)
    probe.close()
    npix = grid.npix

    all_times, all_arr = [], []
    for year in args.years:
        print(f"== {year}", flush=True)
        times, arr = stack_year(
            raw / f"era5_pl_{year}.grib",
            raw / f"era5_sl_{year}.grib",
            interp,
            names_idx,
            levels,
            npix,
        )
        all_times.append(times)
        all_arr.append(arr)

    times_all = pd.DatetimeIndex(np.concatenate(all_times))
    state = np.concatenate(all_arr, axis=0)
    order = np.argsort(times_all.values)
    times_all, state = times_all[order], state[order]
    dts = [t.to_pydatetime() for t in times_all]

    # forcing
    insol = np.stack(
        [
            G.insolation(colat, lon, d.hour + d.minute / 60, d.timetuple().tm_yday)
            for d in dts
        ]
    )
    st = xr.open_dataset(raw / "era5_static.grib", engine="cfgrib")
    zsfc = interp(st["z"].isel(valid_time=0).values)
    lsm = np.clip(interp(st["lsm"].isel(valid_time=0).values), 0, 1)
    st.close()
    forcing = np.zeros((len(dts), npix, 3), dtype=np.float32)
    forcing[:, :, 0] = insol
    forcing[:, :, 1] = zsfc[None]
    forcing[:, :, 2] = lsm[None]

    norm = Normalizer.from_array(names, state)
    norm.save(outdir / "norm_state.json")
    np.save(outdir / "state.npy", state.astype(np.float16))
    np.save(
        outdir / "forcing.npy",
        (forcing / np.array([500.0, 1e5, 1.0], dtype=np.float32)).astype(np.float16),
    )
    with open(outdir / "times.json", "w") as f:
        json.dump([d.isoformat() for d in dts], f)
    print(f"saved {state.shape} states to {outdir}")


if __name__ == "__main__":
    main()
