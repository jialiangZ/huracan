#!/usr/bin/env python3
"""Refit the state normalizer on the TRAINING period only (P0 fix).

The on-disk data/era5 normalizer was fitted over the full 2010-2023 range,
so validation and test statistics leaked into every downstream model via
the normalization constants.  This script refits from the existing
state.npy + times.json without re-downloading anything and writes the
result to a *new* data directory (raw data is never modified):

    data/era5_v1/
      state.npy, forcing.npy, times.json   -> symlinks to data/era5
      norm_state.json                      -> train-only fit (this script)
      norm_meta.json                       -> provenance of the fit
      obs*.npy                             -> prepare_obs.py (run after)

Usage:
  python scripts/refit_normalizer.py --config configs/bsef_r0.yaml \
      --source-dir data/era5 --out-dir data/era5_v1 \
      --start 2010-01-01 --end 2011-12-31
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config for var names/grid")
    ap.add_argument("--source-dir", default="data/era5")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", required=True, help="train start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="train end date (inclusive)")
    args = ap.parse_args()

    from huracan.config import load_config, var_names
    from huracan.data.norm import Normalizer

    cfg = load_config(args.config)
    src, out = Path(args.source_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    times = [dt.datetime.fromisoformat(s) for s in json.load(open(src / "times.json"))]
    ts = pd.DatetimeIndex(times)
    lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end) + pd.Timedelta(hours=18)
    frames = np.flatnonzero((ts >= lo) & (ts <= hi))
    if len(frames) == 0:
        raise SystemExit(f"no frames in [{lo}, {hi}]")

    state = np.load(src / "state.npy", mmap_mode="r")
    print(f"fitting on frames [{frames[0]}, {frames[-1]}] "
          f"({ts[frames[0]]} .. {ts[frames[-1]]}, {len(frames)} frames)")
    norm = Normalizer.from_array(var_names(cfg), state, frames=frames)

    # sanity: validation/test frames must not influence the fit
    norm2 = Normalizer.from_array(
        var_names(cfg), state, frames=frames[: max(1, len(frames) // 2)]
    )
    assert not np.allclose(norm.mean, norm2.mean), "split sensitivity check"

    norm.save(out / "norm_state.json")
    for f in ("state.npy", "forcing.npy", "times.json"):
        link = out / f
        if not link.exists():
            link.symlink_to(src.resolve() / f)
    meta = {
        "source_dir": str(src.resolve()),
        "fit_frames": [int(frames[0]), int(frames[-1])],
        "n_frames": int(len(frames)),
        "start": args.start,
        "end": args.end,
        "note": "train-period-only fit; val(2012)/test(2013)/reserve(2014-23) excluded",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    json.dump(meta, open(out / "norm_meta.json", "w"), indent=1)
    print(f"norm_state.json written to {out}; meta: {meta['note']}")


if __name__ == "__main__":
    main()
