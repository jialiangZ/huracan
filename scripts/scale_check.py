#!/usr/bin/env python3
"""C-phase check: does the F1 loss-scale fix change training trajectories?

The old crps_loss bug was EXACTLY a factor C/npix = 72/4608 = 1/64 on the
loss (hence on every gradient), nothing else.  Adam is scale-invariant
asymptotically, but grad-clip thresholds and the AdamW epsilon are not.
This script runs two SHORT stochastic-phase trainings from the same
deterministic checkpoint, same seed, same data:

  run A : the fixed loss (current code)
  run B : the fixed loss divided by 64  (== the old bug's objective)

and compares B*64 against A on the validation curve.  If they coincide to
~1e-3 relative, the existing stochastic/fcst_ar checkpoints are usable as
trained baselines (the objective they optimized differed from the fixed
one only by the constant); otherwise those phases must be retrained.

Usage (GPU2):
  CUDA_VISIBLE_DEVICES=2 python scripts/scale_check.py \
      --config configs/bsef_r0.yaml --from-ckpt runs/bsef/r0 \
      --steps 400 --out runs/bsef/scalecheck
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNNER = r"""
import os, sys
sys.path.insert(0, ".")
import huracan.core.losses as L
_s = float(os.environ["HURACAN_CRPS_SCALE"])
_f = L.crps_loss
L.crps_loss = lambda *a, **k: _f(*a, **k) / _s
sys.argv = ["train.py"] + sys.argv[1:]
exec(compile(open("scripts/train.py").read(), "scripts/train.py", "exec"),
     {"__name__": "__main__", "__file__": "scripts/train.py"})
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bsef_r0.yaml")
    ap.add_argument("--from-ckpt", default="runs/bsef/r0")
    ap.add_argument("--phase", default="stochastic")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="runs/bsef/scalecheck")
    args = ap.parse_args()

    out = Path(args.out)
    runs = {}
    for tag, scale in (("a", 1.0), ("b", 64.0)):
        wd = out / tag
        wd.mkdir(parents=True, exist_ok=True)
        src = Path(args.from_ckpt) / "ckpt_deterministic.pt"
        dst = wd / "ckpt_deterministic.pt"
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
        env = dict(os.environ, HURACAN_CRPS_SCALE=str(scale))
        cmd = [
            sys.executable, "-c", RUNNER,
            "--config", args.config, "--phase", args.phase,
            "--workdir", str(wd),
            "--override",
            f"train.phases.{args.phase}.steps={args.steps}",
            "train.val_check_interval=50",
        ]
        print(f"[scale_check] run {tag} (loss / {scale:g}) -> {wd}", flush=True)
        subprocess.run(cmd, check=True, env=env, cwd=ROOT)
        runs[tag] = wd

    va = json.load(open(runs["a"] / f"val_{args.phase}.json"))["history"]
    vb = json.load(open(runs["b"] / f"val_{args.phase}.json"))["history"]
    common = sorted(set(h["step"] for h in va) & set(h["step"] for h in vb))
    da = {h["step"]: h["val_loss"] for h in va}
    db = {h["step"]: h["val_loss"] for h in vb}
    worst = 0.0
    rows = []
    for s in common:
        a, b64 = da[s], db[s] * 64.0
        rel = abs(b64 - a) / max(abs(a), 1e-12)
        worst = max(worst, rel)
        rows.append((s, a, b64, rel))
    print("\n step |   A (fixed)   |  B*64 (old)   | rel.diff")
    for s, a, b64, rel in rows:
        print(f"{s:5d} | {a:13.6f} | {b64:13.6f} | {rel:.2e}")
    verdict = "EQUIVALENT" if worst < 1e-3 else "DIVERGENT"
    print(f"\nmax relative deviation of B*64 vs A: {worst:.3e}  ->  {verdict}")
    json.dump(
        {"phase": args.phase, "steps": args.steps, "rows": rows,
         "max_rel_dev": worst, "verdict": verdict},
        open(out / "scale_check.json", "w"), indent=1,
    )
    print(f"report -> {out / 'scale_check.json'}")


if __name__ == "__main__":
    main()
