#!/usr/bin/env python3
"""Measure 4090 training cost of the R0 baseline phases (4090 plan §5.3).

Runs a fixed number of REAL optimizer steps per phase (same code path as
training, incl. data loading and the Lightning loop), then reports:
  ms/step, peak torch.cuda memory (allocated/reserved), device total.
The first N_WARMUP steps are excluded from the timing (cache warm-up).

Writes runs/bsef/bench_r0.json and prints a cost projection for the full
R0 schedule (deterministic 40k / stochastic 15k / fcst_ar 3k / assim 5k).

Usage:
  python scripts/bench_r0.py --config configs/bsef_r0.yaml --steps 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import lightning as L

N_WARMUP = 10


class _Bench(L.Callback):
    def __init__(self, steps: int):
        super().__init__()
        self.t = []
        self.mem = []
        self.steps = steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t.append(time.perf_counter() - self._t0)
        self.mem.append(torch.cuda.max_memory_allocated() / 2**30)
        if batch_idx + 1 >= self.steps:
            trainer.should_stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--phases", nargs="+",
                    default=["deterministic", "stochastic", "fcst_ar", "assim"])
    ap.add_argument("--workdir", default="runs/bsef/bench")
    args = ap.parse_args()

    from huracan.config import load_config
    from huracan.training.phases import fit_phase

    cfg = load_config(args.config)
    out = {"gpu": None, "phases": {}, "steps_measured": args.steps}
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_total_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30

    for phase in args.phases:
        cb = _Bench(args.steps)
        full_steps = cfg.train.phases[phase].steps
        cfg.train.phases[phase].steps = args.steps  # bench budget
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        fit_phase(
            cfg,
            phase,
            device="cuda",
            workdir=args.workdir,
            ckpt_every=10**9,
            log_every=10**6,
            extra_callbacks=[cb],
        )
        cfg.train.phases[phase].steps = full_steps  # restore
        t = sorted(cb.t[N_WARMUP:])
        med = t[len(t) // 2] * 1000
        rec = {
            "ms_per_step_median": round(med, 1),
            "ms_per_step_mean": round(sum(t) / len(t) * 1000, 1),
            "peak_mem_alloc_gib": round(max(cb.mem), 2),
            "peak_mem_reserved_gib": round(
                torch.cuda.max_memory_reserved() / 2**30, 2
            ) if torch.cuda.is_available() else None,
            "full_phase_steps": full_steps,
            "full_phase_hours_est": round(
                (sum(t) / len(t)) * full_steps / 3600, 2
            ),
        }
        out["phases"][phase] = rec
        print(f"[{phase}] {rec['ms_per_step_median']} ms/step, "
              f"peak {rec['peak_mem_alloc_gib']} GiB, "
              f"full {rec['full_phase_steps']} steps -> "
              f"{rec['full_phase_hours_est']} h", flush=True)

    total_h = sum(r["full_phase_hours_est"] for r in out["phases"].values())
    out["r0_total_hours_est"] = round(total_h, 2)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.workdir) / "bench_r0.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nR0 total estimate: {total_h:.1f} h "
          f"(bench_r0.json written to {args.workdir})")


if __name__ == "__main__":
    main()
