#!/usr/bin/env python3
"""P2 diagnostic 3: build observation variants for impact analysis.

Two variant types of the SAME obs period (val 2012), written as new data
dirs that symlink everything except the obs bundle:

  thin      -- keep a random ``frac`` of obs-carrying (frame, pixel) slots
               per frame (values zeroed + mask cleared elsewhere); coverage
               reduced while the spatial distribution stays roughly matched.
  holdout   -- rebuild from raw IGRA/ISD with a seeded random ``frac`` of
               stations from each network excluded entirely (station-level
               removal, the scientifically meaningful thinning).

Both use the SAME normalizer as the parent data dir (train-only fit).

Usage:
  python scripts/make_obs_variant.py --config configs/bsef_r0.yaml \
      --parent data/era5_v1 --kind thin --frac 0.5 --out data/era5_v1_thin50
  python scripts/make_obs_variant.py --config configs/bsef_r0.yaml \
      --parent data/era5_v1 --kind holdout --frac 0.2 --out data/era5_v1_hold20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--kind", choices=["thin", "holdout"], required=True)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    from huracan.config import load_config
    from huracan.data.norm import Normalizer
    from huracan.data.obs import build_obs_tensor

    cfg = load_config(args.config)
    parent, out = Path(args.parent), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("state.npy", "forcing.npy", "times.json", "norm_state.json",
              "norm_meta.json", "assemble_report.json", "obs_offsets.npy"):
        src = parent / f
        if src.exists() and not (out / f).exists():
            (out / f).symlink_to(src.resolve())

    if args.kind == "thin":
        obs = np.load(parent / "obs.npy", mmap_mode="r")
        packed = np.load(parent / "obs_mask.npy", mmap_mode="r")
        counts = np.load(parent / "obs_counts.npy", mmap_mode="r")
        T = obs.shape[0]
        rng = np.random.default_rng(args.seed)
        n_lost = 0
        obs_o = np.lib.format.open_memmap(
            out / "obs.npy", mode="w+", dtype=np.float16, shape=obs.shape
        )
        pack_o = np.lib.format.open_memmap(
            out / "obs_mask.npy", mode="w+", dtype=np.uint8, shape=packed.shape
        )
        cnt_o = np.lib.format.open_memmap(
            out / "obs_counts.npy", mode="w+", dtype=np.uint8, shape=counts.shape
        )
        C = obs.shape[2]
        step = 64
        for a in range(0, T, step):
            b = min(a + step, T)
            m = np.unpackbits(
                np.asarray(packed[a:b]), axis=-1, count=C
            ).astype(bool)
            o = np.asarray(obs[a:b]).copy()
            drop = (m.any(axis=-1)) & (
                rng.random((b - a, m.shape[1])) >= args.frac
            )
            o[drop] = 0.0
            m[drop] = False
            n_lost += int(drop.sum())
            obs_o[a:b] = o.astype(np.float16)
            pack_o[a:b] = np.packbits(m, axis=-1)
            cnt_o[a:b] = m.sum(axis=-1).astype(np.uint8)
        obs_o.flush(); pack_o.flush(); cnt_o.flush()
        kind_note = f"kept frac {args.frac} of obs-carrying pixels/frame"
    else:
        # station holdout: rebuild the gridded tensor without a seeded
        # random subset of stations from each network
        import pandas as pd

        igra = sorted((Path("data/raw_obs/igra")).glob("*.zip"))
        isd = pd.read_csv("data/raw_obs/isd_stations.csv")["sid"].tolist()
        isd_files = sorted((Path("data/raw_obs/isd")).glob("*.gz"))
        isd_ids = sorted({g.name.rsplit("-", 1)[0] for g in isd_files})
        rng = np.random.default_rng(args.seed)
        ex_igra = set(rng.choice(
            [z.stem for z in igra],
            size=int(len(igra) * args.frac), replace=False))
        ex_isd = set(rng.choice(
            [s for s in isd_ids if s in set(isd)],
            size=int(len(isd_ids) * args.frac), replace=False))
        norm = Normalizer.load(parent / "norm_state.json")
        build_obs_tensor(
            cfg, "data/raw_obs", parent / "times.json", norm,
            out / "obs.npy", workers=args.workers,
            exclude_igra=ex_igra, exclude_isd=ex_isd,
        )
        kind_note = (f"excluded {len(ex_igra)}/{len(igra)} IGRA stations, "
                     f"{len(ex_isd)}/{len(isd_ids)} ISD stations "
                     f"(seed {args.seed})")

    meta = {
        "kind": args.kind, "frac": args.frac, "seed": args.seed,
        "parent": str(parent.resolve()), "note": kind_note,
    }
    json.dump(meta, open(out / "variant_meta.json", "w"), indent=1)
    print(f"variant written to {out}: {kind_note}")


if __name__ == "__main__":
    main()
