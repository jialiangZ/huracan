#!/usr/bin/env python3
"""Train Huracan through the paper's phased schedule (Table 3).

Usage:
  python scripts/train.py --config configs/pilot.yaml --phase all
  python scripts/train.py --config configs/pilot.yaml --phase stochastic
  python scripts/train.py --config configs/smoke.yaml --phase all \
      --override train.phases.deterministic.steps=20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huracan.config import channel_report, load_config
from huracan.training.phases import ALL_PHASES, fit_phase


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

    cfg = load_config(args.config, overrides=args.override or None)
    if args.workdir:
        cfg.output_dir = args.workdir

    print(channel_report(cfg))
    print("\n(provenance: PAPER vs REPO/INFERRED choices are registered in "
          "huracan/config/provenance.py)\n")

    phases = ALL_PHASES if args.phase == "all" else args.phase.split(",")

    if cfg.data.source == "synthetic":
        from huracan.data.synthetic import generate_synthetic

        root = Path(cfg.data.root) / "synthetic"
        if not (root / "state.npy").exists():
            generate_synthetic(cfg, root, seed=cfg.seed)

    for ph in phases:
        print(f"\n===== phase: {ph} =====", flush=True)
        fit_phase(cfg, ph, device=args.device, workdir=args.workdir)


if __name__ == "__main__":
    main()
