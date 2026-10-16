#!/usr/bin/env python3
"""Rebuild the corrected obs bundle (IGRA RH /10 + -8888 QA fix, F2) into
data/era5_v1b, then the two diagnostic variants.  Must run as a FILE so
multiprocessing spawn can re-import __main__ (a `python -` heredoc
deadlocks the pool)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build():
    from huracan.config import load_config
    from huracan.data.norm import Normalizer
    from huracan.data.obs import build_obs_tensor

    cfg = load_config("configs/bsef_r0.yaml")
    norm = Normalizer.load("data/era5_v1b/norm_state.json")
    build_obs_tensor(
        cfg, "data/raw_obs", Path("data/era5_v1b/times.json"), norm,
        Path("data/era5_v1b/obs.npy"), workers=12,
    )


if __name__ == "__main__":  # REQUIRED: spawn workers re-import this module
    build()
