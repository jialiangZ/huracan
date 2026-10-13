#!/usr/bin/env python3
"""OSSE-style controls on a trained L96 run (DATA_PLAN Sec. 6).

Three controls from the SAME perturbed initial state (same member seeds,
paired comparison):

  with-obs      assimilation model + real observations
  zero-obs      assimilation model + all-zero obs channels (architecture
                identical, no information -- the assimilation ablation)
  fcst-spinup   forecast model runs the spin-up (no assimilation mechanism)

Outputs: spin-up analysis-error curve (the direct assimilation-capability
evidence: with-obs error should FALL with cycles while controls cannot),
forecast RMSE/CRPS vs lead, all against climatology / persistence /
observation-error baselines.  All errors in normalized units (1.0 = one
climatological std of the L96 attractor).

Usage:
  python scripts/run_l96_controls.py --config configs/l96.yaml
      [--starts 3] [--ic-noise 1.0] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from huracan.config import load_config
from huracan.core.metrics import crps_ensemble, rmse_ensemble_mean
from huracan.core.rollout import spinup_and_forecast
from huracan.data.dataset import SphereDataset
from huracan.inference.runner import EdaRunner


def run_control(runner, window, S, F, members, ic_noise, grid_hw, mode, dev, sem):
    state = window["state"].to(dev)
    forcing = window["forcing"].to(dev)
    obs = window["obs"].to(dev)
    if mode == "with-obs":
        assim, obs_in = runner.assim, obs[: sem.obs_frames(S)]
    elif mode == "zero-obs":
        assim, obs_in = (
            runner.assim,
            torch.zeros_like(obs[: sem.obs_frames(S)]),
        )
    else:  # fcst-spinup: forecast model does the cycling, obs ignored
        assim, obs_in = runner.fcst, None
    spin_end_t, preds, spin_mean = spinup_and_forecast(
        assim,
        runner.fcst,
        state,
        forcing,
        obs_in,
        members=members,
        spinup=S,
        lead=F,
        collect_spinup=True,
        ic_noise=ic_noise,
        grid_hw=grid_hw,
    )
    ns, nf = sem.frames_for_calls(S), sem.frames_for_calls(F)
    truth_spin = state[2 : ns + 2].cpu()
    truth = state[ns + 2 : ns + 2 + nf].cpu()
    return (
        (spin_mean - truth_spin).pow(2).mean(dim=(1, 2)).sqrt().numpy(),
        rmse_ensemble_mean(preds.numpy(), truth.numpy()),
        crps_ensemble(preds.numpy(), truth.numpy()),
        truth.numpy(),
        spin_end_t.numpy(),  # (M, npix, C): analysis ensemble at spin-up end
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--ic-noise", type=float, default=1.0)
    ap.add_argument(
        "--spinup-calls",
        type=int,
        default=None,
        help="override eval.spinup_calls_override",
    )
    ap.add_argument(
        "--lead-calls", type=int, default=None, help="override eval.lead_calls_override"
    )
    ap.add_argument("--tag", default="", help="output filename suffix (e.g. s15)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.spinup_calls:
        cfg.eval.spinup_calls_override = args.spinup_calls
    if args.lead_calls:
        cfg.eval.lead_calls_override = args.lead_calls
    out = Path(args.workdir or cfg.output_dir)
    dev = args.device or cfg.device
    runner = EdaRunner.from_checkpoints(cfg, out, device=dev)
    S, F = cfg.spinup_calls(), cfg.lead_calls()
    M = cfg.eval.members
    grid_hw = (cfg.grid.nlat, cfg.grid.nlon)
    sem = cfg.call_semantics()

    ds = SphereDataset(cfg, "val", rollout=S + F)
    t0s = np.linspace(ds.first_t0, int(ds.valid_t0[-1]), args.starts).astype(int)

    spin_err = {m: [] for m in ("with-obs", "zero-obs", "fcst-spinup")}
    fc_rmse, fc_crps, clim_rmse, persist_rmse = [], [], [], []
    for t0 in t0s:
        w = ds.window(int(t0), sem.window_frames(S + F))
        truth = None
        for mode in spin_err:
            se, rm, cr, truth, spin_end = run_control(
                runner, w, S, F, M, args.ic_noise, grid_hw, mode, dev, sem
            )
            spin_err[mode].append(se)
            if mode == "with-obs":
                fc_rmse.append(rm)
                fc_crps.append(cr)
                # FAIR persistence: each member persists its own analysis
                # state (same starting error as the forecast), NOT the truth
                persist = np.repeat(spin_end[:, None], truth.shape[0], axis=0)
                persist_rmse.append(rmse_ensemble_mean(persist, truth))
                clim_rmse.append(
                    np.sqrt((truth**2).mean(axis=(1, 2)))
                )  # climatology = 0 in normalized units

    def _frames(x):
        x = np.stack(x).mean(0)
        return x.mean(axis=-1) if x.ndim == 2 else x

    spin_err = {m: np.stack(v).mean(0) for m, v in spin_err.items()}
    fc_rmse, fc_crps = _frames(fc_rmse), _frames(fc_crps)
    persist_rmse = _frames(persist_rmse)
    clim_rmse = np.stack(clim_rmse).mean(0)

    # observation-error floor in normalized units
    norm = json.load(open(Path(cfg.data.root) / cfg.data.source / "norm_state.json"))
    lcfg = cfg.data.lorenz96
    obs_floor = lcfg.obs_sigma / norm["std"][0]

    # ---------------- report ----------------
    n_frames = spin_err["with-obs"].shape[0]
    k = max(n_frames // 10, 1)
    print(
        f"\nL96 controls: {M} members, {args.starts} starts, "
        f"ic_noise={args.ic_noise:.1f} (paired seeds), "
        f"spin-up {S} calls, lead {F} calls"
    )
    print("spin-up analysis RMSE (normalized; 1.0 = climate std):")
    print(f"  {'cycle':>6} {'with-obs':>9} {'zero-obs':>9} {'fcst':>9}")
    for i in list(range(0, n_frames, k)) + [n_frames - 1]:
        print(
            f"  {i + 1:>6} {spin_err['with-obs'][i]:>9.3f} "
            f"{spin_err['zero-obs'][i]:>9.3f} "
            f"{spin_err['fcst-spinup'][i]:>9.3f}"
        )
    print(f"  obs-error floor ~ {obs_floor:.3f}")

    print("\nforecast RMSE vs lead (1 call = 0.1 t.u.; persist = analysis):")
    leads = [1, 5, 10, 25, 50]
    print(f"  {'lead':>5} {'with-obs':>9} {'persist':>9} {'climate':>9} {'skill':>7}")
    for L in leads:
        i = min(2 * L - 1, fc_rmse.shape[0] - 1)
        skill = 1 - fc_rmse[i] / clim_rmse[i]
        print(
            f"  {L:>5} {fc_rmse[i]:>9.3f} {persist_rmse[i]:>9.3f} "
            f"{clim_rmse[i]:>9.3f} {skill:>7.0%}"
        )
    print(f"  CRPS @lead50: {fc_crps[-1]:.3f}")

    # ---------------- plot ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    for mode, c in zip(
        ("with-obs", "zero-obs", "fcst-spinup"), ("tab:blue", "tab:orange", "tab:red")
    ):
        ax.plot(np.arange(1, n_frames + 1) / 2, spin_err[mode], c, label=mode)
    ax.axhline(obs_floor, color="gray", ls="--", lw=1, label="obs-error floor")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="climatology")
    ax.set_xlabel("assimilation cycle (AR calls)")
    ax.set_ylabel("analysis RMSE (normalized)")
    ax.set_title("spin-up: analysis error over cycles")
    ax.legend(fontsize=8)
    ax = axes[1]
    lead = np.arange(1, fc_rmse.shape[0] + 1) / 2
    ax.plot(lead, fc_rmse, "tab:blue", label="forecast (after with-obs spin-up)")
    ax.plot(lead, persist_rmse, "tab:green", label="persistence")
    ax.plot(lead, clim_rmse, "k:", label="climatology")
    ax.set_xlabel("lead (AR calls)")
    ax.set_ylabel("RMSE (normalized)")
    ax.set_title("forecast error vs lead time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = out / f"l96_controls{args.tag}.png"
    fig.savefig(png, dpi=160)

    summary = {
        "spinup_final": {m: float(v[-1]) for m, v in spin_err.items()},
        "spinup_first": {m: float(v[0]) for m, v in spin_err.items()},
        "fc_rmse_leads": {
            str(L): float(fc_rmse[min(2 * L - 1, fc_rmse.shape[0] - 1)]) for L in leads
        },
        "climate_leads": {
            str(L): float(clim_rmse[min(2 * L - 1, clim_rmse.shape[0] - 1)])
            for L in leads
        },
        "crps_final": float(fc_crps[-1]),
        "ic_noise": args.ic_noise,
        "members": M,
        "starts": int(t0s.size),
        "spinup_calls": S,
        "lead_calls": F,
    }
    with open(out / f"l96_controls{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nsaved {png} and {out / f'l96_controls{args.tag}.json'}")


if __name__ == "__main__":
    main()
