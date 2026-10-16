#!/usr/bin/env python3
"""P0 asset manifest: register code, environment, data, checkpoints, runs.

Writes runs/bsef/manifest.json; safe to re-run (idempotent update of the
mutable sections).  Raw data and old runs are NEVER modified -- only
hashed/registered.

Usage:
  python scripts/make_manifest.py [--skip-hashes]  # skip large-file hashing
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def git_info():
    def run(*a):
        return subprocess.run(a, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": sorted(
            l.split()[-1] for l in run("git", "status", "--porcelain").splitlines() if l
        ),
    }


def env_info():
    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 1
        )
        try:
            info["cuda"] = torch.version.cuda
        except Exception:
            pass
    return info


def hash_file(p: Path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def register_data(skip_hashes=False):
    """Register the ERA5 data dirs + obs raw archives."""
    out = {}
    for name in ("era5", "era5_v2", "era5_v1", "raw_obs",
                 "turbulence", "kolmogorov"):
        d = ROOT / "data" / name
        if not d.exists():
            continue
        rec = {"path": str(d)}
        files = {}
        for f in sorted(d.iterdir()):
            if f.is_file() and not f.is_symlink():
                rec_f = {"size_mb": round(f.stat().st_size / 2**20, 1)}
                if f.suffix in (".npy", ".json", ".pt") and not skip_hashes:
                    if f.stat().st_size < 3e9:  # hash up to ~3 GB per file
                        rec_f["sha256_16"] = hash_file(f)
                files[f.name] = rec_f
            elif f.is_symlink():
                files[f.name] = {"symlink_to": str(f.resolve())}
        rec["files"] = files
        # data-level facts
        if (d / "times.json").exists() and (d / "state.npy").exists():
            times = json.load(open(d / "times.json"))
            a = np.load(d / "state.npy", mmap_mode="r")
            rec["state"] = {
                "shape": list(a.shape), "dtype": str(a.dtype),
                "first": times[0], "last": times[-1], "n_frames": len(times),
            }
        out[name] = rec
    # raw obs counts
    igra = list((ROOT / "data/raw_obs/igra").glob("*.zip")) if (ROOT / "data/raw_obs/igra").exists() else []
    isd = list((ROOT / "data/raw_obs/isd").glob("*.gz")) if (ROOT / "data/raw_obs/isd").exists() else []
    if igra or isd:
        out["raw_obs"]["n_igra_zips"] = len(igra)
        out["raw_obs"]["n_isd_gz"] = len(isd)
    return out


def register_runs():
    """Register pre-existing run dirs (OSSE) without touching them."""
    out = {}
    rd = ROOT / "runs"
    if not rd.exists():
        return out
    for d in sorted(rd.iterdir()):
        if not d.is_dir() or d.name == "bsef":
            continue
        rec = {"path": str(d)}
        ckpts = sorted(f.name for f in d.glob("ckpt_*.pt"))
        rec["checkpoints"] = ckpts
        for f in ("osse_summary.md", "osse_results.json"):
            if (d / f).exists():
                rec[f] = (d / f).read_text()[:2000]
        loss_files = sorted(f.name for f in d.glob("losses_*.json"))
        if loss_files:
            rec["loss_files"] = loss_files
        out[d.name] = rec
    return out


def register_checkpoints(workdir: Path, skip_hashes=False):
    out = {}
    for f in sorted(workdir.glob("ckpt_*.pt")):
        rec = {"size_mb": round(f.stat().st_size / 2**20, 1)}
        if not skip_hashes:
            rec["sha256_16"] = hash_file(f)
        out[f.name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hashes", action="store_true")
    args = ap.parse_args()

    manifest_path = ROOT / "runs/bsef/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    m = {}
    if manifest_path.exists():
        m = json.load(open(manifest_path))
    m.update({
        "updated": dt.datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "env": env_info(),
        "protocol": {
            "outputs_per_call": 1,
            "call_advance_hours": 6.0,
            "splits": {
                "train": "2010-2011 frames [0, 2920)",
                "val": "2012 frames [2920, 4384) (leap)",
                "test": "2013 frames [4384, 5844)",
                "reserve": "2014-2023 frames [5844, 19032) never loaded",
            },
            "normalizer": "train-period-only fit (data/era5_v1/norm_meta.json)",
            "crps_eval": "empirical (main), fair reported separately",
            "essr": "RMS spread / RMSE (main), mean-std variant reported",
            "loss": "area-weighted (Gauss pixel weights), all methods alike",
        },
        "data": register_data(args.skip_hashes),
        "legacy_runs": register_runs(),
    })
    r0 = ROOT / "runs/bsef/r0"
    if r0.exists():
        m["r0_checkpoints"] = register_checkpoints(r0, args.skip_hashes)
    with open(manifest_path, "w") as f:
        json.dump(m, f, indent=1)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
