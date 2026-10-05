#!/usr/bin/env python3
"""Train Huracan through the paper's phased schedule.

Usage:
  python scripts/train.py --config configs/pilot.yaml --phase all
  python scripts/train.py --config configs/pilot.yaml --phase stochastic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from huracan.config import load_config
from huracan.train.trainer import HuracanTrainer

PHASE_ORDER = ["deterministic", "stochastic", "fcst_ar", "assim", "joint"]
ALL_PHASES = PHASE_ORDER + ["hybrid"]  # hybrid needs ECMWF data; run explicitly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", default="all", help="phase name, comma list, or 'all'")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="key=value config overrides (dot paths, yaml values)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    for kv in args.override:
        k, v = kv.split("=", 1)
        node = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node[kk]
        import yaml

        node[keys[-1]] = yaml.safe_load(v)
    if args.workdir:
        cfg["output_dir"] = args.workdir

    phases = ALL_PHASES if args.phase == "all" else args.phase.split(",")

    if cfg["data"]["source"] == "synthetic":
        from huracan.data.synthetic import generate_synthetic

        root = Path(cfg["data"]["root"]) / "synthetic"
        if not (root / "state.npy").exists():
            generate_synthetic(cfg, root, seed=cfg["seed"])

    torch.set_float32_matmul_precision("high")
    for ph in phases:
        print(f"\n===== phase: {ph} =====", flush=True)
        tr = HuracanTrainer(cfg, ph, device=args.device)
        tr.run()


if __name__ == "__main__":
    main()
