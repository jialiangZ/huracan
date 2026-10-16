#!/usr/bin/env python3
"""Refetch the 2m dewpoint temperature from the WB2 *native-resolution*
zarr (1h, 0.25 deg) -- every regridded 6h zarr in the bucket ships an
all-NaN ``2m_dewpoint_temperature`` (P0 data audit finding), which the
original fetch silently turned into all-zero chunks via ``nan_to_num``.

Writes valid ``2d_{year}.npy`` chunks (48x96 Gauss grid, float32) into the
wb2 chunk dir, replacing the zero ones.  Fails loudly on NaN or out-of-
physical-range values -- no silent zeroing, ever.

Usage:
  python scripts/fix_2d_from_wb2_native.py --years 2010 2011 2012 2013 \
      --config configs/bsef_r0.yaml --workers 7
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

NATIVE_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
)


def _fetch_year(args):
    year, colat, lon, npix, out_path = args
    import fsspec
    import xarray as xr

    from huracan.data.interp import LatLonBilinear

    ds = xr.open_zarr(fsspec.get_mapper(NATIVE_URL, token="anon"), consolidated=True)
    interp = LatLonBilinear(ds.latitude.values, ds.longitude.values, colat, lon)
    stamps = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 18:00", freq="6h")
    avail = pd.DatetimeIndex(ds.time.values)
    have = avail[(avail >= stamps[0]) & (avail <= stamps[-1])]
    want = [t for t in stamps if t in set(have)]
    missing = [t.isoformat() for t in stamps if t not in set(have)]
    if missing:
        print(f"2d {year}: {len(missing)} 6h stamps absent from source "
              f"(dataset end-of-record): {missing[:2]}...", flush=True)
    out = np.empty((len(want), npix), dtype=np.float32)
    t0 = time.time()
    for i, t in enumerate(want):
        frame = np.asarray(
            ds["2m_dewpoint_temperature"].sel(time=t).values, dtype=np.float32
        )
        if not np.isfinite(frame).all():
            raise RuntimeError(
                f"2d {year} {t}: source contains non-finite values -- "
                f"refusing to write (no silent zeroing)"
            )
        out[i] = interp(frame)
    lo, hi = 150.0, 340.0  # physical dewpoint window [K], generous
    bad = (out < lo) | (out > hi)
    if bad.any():
        raise RuntimeError(f"2d {year}: {int(bad.sum())} values outside "
                           f"[{lo},{hi}] K after interpolation")
    np.save(out_path, out)
    return f"2d {year}: {len(want)} frames, mean {out.mean():.1f} K, " \
           f"{time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", type=int, nargs="+",
                    default=list(range(2010, 2024)))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--tmpdir", default="data/wb2_chunks")
    args = ap.parse_args()

    from huracan.config import load_config
    from huracan.core.grid import grid_from_cfg

    cfg = load_config(args.config)
    grid = grid_from_cfg(cfg.grid)
    colat, lon = grid.colat, grid.lon_flat

    tmp = Path(args.tmpdir)
    tmp.mkdir(parents=True, exist_ok=True)
    jobs = [
        (y, colat, lon, grid.npix, str(tmp / f"2d_{y}.npy")) for y in args.years
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(_fetch_year, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
