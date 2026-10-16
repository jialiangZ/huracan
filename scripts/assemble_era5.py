#!/usr/bin/env python3
"""Reassemble state.npy from the local wb2 chunk cache with the P0 fixes:

  * cumulative per-year offsets (leap years; the old ``yi * len`` placement
    misaligned every year after 2012 and overwrote earlier years), and
  * strict validation: no all-zero frames, no non-finite values, exact
    year boundaries against the time axis.

Writes to a NEW directory (the original data/era5 stays untouched):
  {out}/state.npy        (T, npix, C) float32
  {out}/forcing.npy      symlink to the source (unchanged by the bug)
  {out}/times.json       symlink
  {out}/assemble_report.json

Usage:
  python scripts/assemble_era5.py --config configs/bsef_r0.yaml \
      --source-dir data/era5 --tmpdir data/wb2_chunks --out-dir data/era5_v2 \
      --years 2010 ... 2023
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-dir", default="data/era5")
    ap.add_argument("--tmpdir", default="data/wb2_chunks")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from huracan.config import load_config, var_names

    cfg = load_config(args.config)
    src = Path(args.source_dir)
    tmp = Path(args.tmpdir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    times = [dt.datetime.fromisoformat(s) for s in json.load(open(src / "times.json"))]
    T = len(times)
    years = sorted({t.year for t in times})
    names = var_names(cfg)

    state = np.lib.format.open_memmap(
        out / "state.npy", mode="w+", dtype=np.float32,
        shape=(T, 4608 if cfg.grid.nlat == 48 else cfg.grid.nlat * cfg.grid.nlon,
               len(names)),
    )
    report = {"years": years, "channels": {}, "T": T}
    zero_frames_total = 0
    for ci, name in enumerate(names):
        off = 0
        for y in years:
            chunk = np.load(tmp / f"{name}_{y}.npy")
            n = chunk.shape[0]
            state[off : off + n, :, ci] = chunk
            off += n
        assert off == T, (name, off, T)
        col = np.asarray(state[:, :, ci])  # materialize for validation
        nf = int((~np.isfinite(col)).sum())
        zf = int((col == 0).all(axis=1).sum())
        zero_frames_total += zf
        report["channels"][name] = {
            "min": float(col.min()), "max": float(col.max()),
            "nonfinite": nf, "all_zero_frames": zf,
        }
        if nf or zf:
            raise SystemExit(
                f"channel {name}: {nf} non-finite values, {zf} all-zero "
                f"frames -- chunk cache still corrupted, aborting"
            )
        print(f"  {name}: [{col.min():.1f}, {col.max():.1f}] OK", flush=True)
    # year boundary check against the time axis
    off = 0
    boundaries = {}
    for y in years:
        n = sum(
            1 for t in times if t.year == y
        )
        assert times[off].year == y and times[off].month == 1 and times[off].day == 1
        boundaries[y] = [off, off + n]
        off += n
    assert off == T
    report["year_boundaries"] = {str(k): v for k, v in boundaries.items()}
    state.flush()

    for f in ("forcing.npy", "times.json"):
        link = out / f
        if not link.exists():
            link.symlink_to(src.resolve() / f)
    report["source_dir"] = str(src.resolve())
    report["tmpdir"] = str(tmp.resolve())
    with open(out / "assemble_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"reassembly complete -> {out} (T={T}, {len(names)} channels, "
          f"{zero_frames_total} zero frames)")


if __name__ == "__main__":
    main()
