#!/usr/bin/env python3
"""Generate the Lorenz-96 nature-run dataset (DATA_PLAN Sec. 3).

Writes state/obs/forcing/times/norms into ``data/<source>`` (source comes
from the config: "lorenz96" one-level, "lorenz96_2l" two-scale ablation)
with the exact layout of the other generators, and prints the suggested
train/val/test periods -- already baked into configs/l96*.yaml.

Usage:
  python scripts/generate_lorenz96.py --config configs/l96.yaml [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huracan.config import load_config
from huracan.data.lorenz96 import generate_lorenz96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="default: <data.root>/<data.source>",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = (
        Path(args.out_dir) if args.out_dir else (Path(cfg.data.root) / cfg.data.source)
    )
    res = generate_lorenz96(cfg, out, seed=args.seed)
    print("\nboundaries:", res["boundaries"])
    print("config periods:", res["periods"])


if __name__ == "__main__":
    main()
