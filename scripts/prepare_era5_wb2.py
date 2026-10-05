#!/usr/bin/env python3
"""Build the ERA5 HEALPix training data from the public WeatherBench2 zarr
(GCS anonymous access) -- replaces the slow CDS regridded requests.

WeatherBench2 (Rasp et al.) serves ERA5 6-hourly at 1.0 deg
(360x181, equiangular with poles) with exactly the 13 pressure levels used
by Huracan, all required surface variables and statics, 1959-2023.

Usage:
  python scripts/prepare_era5_wb2.py --config configs/pilot.yaml \
      --years 2010 2011 2012 2013 [--workers 6]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

WB2_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr"
)

UPPER = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
}
SINGLE = {
    "2t": "2m_temperature",
    "2d": "2m_dewpoint_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "tcc": "total_cloud_cover",
    "tcwv": "total_column_water_vapour",
    "msl": "mean_sea_level_pressure",
}

_W = {}


def _init_worker(colat, lon, npix):
    import fsspec
    import xarray as xr
    from huracan.data.interp import LatLonBilinear

    ds = xr.open_zarr(fsspec.get_mapper(WB2_URL, token="anon"), consolidated=True)
    interp = LatLonBilinear(ds.latitude.values, ds.longitude.values, colat, lon)
    _W.update(ds=ds, interp=interp, npix=npix)


def _fetch_one(job):
    """job: (name, wb2_var, year, level_or_None, out_path)."""
    name, var, year, level, out_path = job
    if Path(out_path).exists():
        return f"{name} {year}: cached"
    for attempt in range(3):
        try:
            ds, interp = _W["ds"], _W["interp"]
            t0 = time.time()
            da = ds[var].sel(time=slice(f"{year}-01-01", f"{year}-12-31 18:00"))
            if level is not None:
                da = da.sel(level=level)
            # WB2 zarrs store dims as (time, longitude, latitude)
            da = da.transpose("time", "latitude", "longitude")
            vals = np.asarray(da.values, dtype=np.float32)  # (T, lat, lon)
            T = vals.shape[0]
            out = np.empty((T, _W["npix"]), dtype=np.float32)
            for t in range(T):
                out[t] = interp(vals[t])
            # float32 (NOT fp16): geopotential (~6e5) overflows fp16
            np.save(out_path, np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))
            return f"{name} {year}: {T} steps {time.time() - t0:.0f}s"
        except Exception as e:  # network retries
            if attempt == 2:
                return f"{name} {year}: FAILED {str(e)[:80]}"
            time.sleep(10 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2010, 2011, 2012, 2013])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tmpdir", default="data/wb2_chunks")
    args = ap.parse_args()

    from huracan.config import load_config, var_names
    from huracan.core import grid as G
    from huracan.data.norm import Normalizer

    cfg = load_config(args.config)
    nside = cfg["grid"]["nside"]
    colat, lon = G.pixel_centers(nside)
    colat, lon = np.asarray(colat), np.asarray(lon)
    npix = G.npix(nside)
    levels = cfg["data"]["variables"]["levels"]
    names = var_names(cfg)
    name_idx = {n: i for i, n in enumerate(names)}

    tmp = Path(args.tmpdir)
    tmp.mkdir(parents=True, exist_ok=True)
    outdir = Path(cfg["data"]["root"]) / "era5"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- parallel fetch ------------------------------------
    jobs = []
    for short, wb2var in UPPER.items():
        for lev in levels:
            for y in args.years:
                jobs.append(
                    (
                        f"{short}{lev}",
                        wb2var,
                        y,
                        lev,
                        str(tmp / f"{short}{lev}_{y}.npy"),
                    )
                )
    for short, wb2var in SINGLE.items():
        for y in args.years:
            jobs.append((short, wb2var, y, None, str(tmp / f"{short}_{y}.npy")))

    t_start = time.time()
    # spawn (not fork): gcsfs/aiohttp async state deadlocks in forked children
    with mp.get_context("spawn").Pool(
        args.workers, initializer=_init_worker, initargs=(colat, lon, npix)
    ) as pool:
        for i, msg in enumerate(pool.imap_unordered(_fetch_one, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {msg}", flush=True)
    print(f"all chunks fetched in {(time.time() - t_start) / 60:.1f} min")

    # ---------------- assemble ------------------------------------------
    import fsspec
    import xarray as xr

    ds = xr.open_zarr(fsspec.get_mapper(WB2_URL, token="anon"), consolidated=True)
    from huracan.data.interp import LatLonBilinear

    interp = LatLonBilinear(ds.latitude.values, ds.longitude.values, colat, lon)
    times = pd.to_datetime(
        ds["time"]
        .sel(time=slice(f"{args.years[0]}-01-01", f"{args.years[-1]}-12-31 18:00"))
        .values
    )
    T = len(times)
    assert T > 0
    state = np.zeros((T, npix, len(names)), dtype=np.float32)
    for name, i in name_idx.items():
        for yi, y in enumerate(args.years):
            chunk = np.load(tmp / f"{name}_{y}.npy")
            sl = slice(yi * chunk.shape[0], (yi + 1) * chunk.shape[0])
            state[sl, :, i] = chunk
        print(f"  assembled {name}", flush=True)

    dts = [t.to_pydatetime() for t in times]
    insol = np.stack(
        [
            G.insolation(colat, lon, d.hour + d.minute / 60, d.timetuple().tm_yday)
            for d in dts
        ]
    )
    zsfc = interp(
        np.asarray(
            ds["geopotential_at_surface"].transpose("latitude", "longitude").values,
            dtype=np.float32,
        )
    )
    lsm = np.clip(
        interp(
            np.asarray(
                ds["land_sea_mask"].transpose("latitude", "longitude").values,
                dtype=np.float32,
            )
        ),
        0,
        1,
    )
    forcing = np.zeros((T, npix, 3), dtype=np.float32)
    forcing[:, :, 0] = insol
    forcing[:, :, 1] = zsfc[None]
    forcing[:, :, 2] = lsm[None]

    norm = Normalizer.from_array(names, state)
    norm.save(outdir / "norm_state.json")
    np.save(outdir / "state.npy", state)  # fp32
    np.save(
        outdir / "forcing.npy",
        (forcing / np.array([500.0, 1e5, 1.0], dtype=np.float32)).astype(np.float16),
    )
    with open(outdir / "times.json", "w") as f:
        json.dump([d.isoformat() for d in dts], f)
    print(f"saved {state.shape} states to {outdir}")


if __name__ == "__main__":
    main()
