#!/usr/bin/env python3
"""Grid raw IGRA/ISD observations into the DA observation tensor.

Usage:
  python scripts/prepare_obs.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huracan.config import load_config
from huracan.data.norm import Normalizer
from huracan.data.obs import build_obs_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--obs-dir", default="data/raw_obs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.data.root) / cfg.data.source
    norm = Normalizer.load(root / "norm_state.json")
    build_obs_tensor(cfg, args.obs_dir, root / "times.json", norm, root / "obs.npy")


if __name__ == "__main__":
    main()
