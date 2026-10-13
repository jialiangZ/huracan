#!/usr/bin/env python3
"""OSSE: verify Huracan's data-assimilation capability on idealized but
physically real dynamics (forced-dissipative barotropic turbulence on the
rotating sphere = 2D Navier-Stokes + beta + advected T/q tracers; see
huracan/data/turbulence.py).

Experiment design (classic observing-system simulation experiment):

  1. nature run  -- chaotic turbulent truth, held-out validation segment
  2. training    -- all 5 phases on the train segment (truth never seen)
  3. EDA treatment: assim model + real observations during spin-up cycling
  4. controls, same trained models, same seeds:
       zero-obs : assim model with all-zero observation channels
       fcst-only: forecast model runs the spin-up (no obs input at all)
  5. baselines: climatology (train mean), persistence, observation error
  6. metrics:
       analysis RMSE vs spin-up call  (treatment vs controls) -- the DA
           signature: with obs the error must *decay below the observation
           error floor* (filtering), without obs it must not
       forecast RMSE / CRPS / ESSR vs lead time

Outputs into the workdir: osse_results.json, osse_summary.md,
osse_curves.png (and per-variant training logs/checkpoints).

Usage:
  python scripts/run_osse.py --config configs/osse_smoke.yaml            # CPU
  python scripts/run_osse.py --config configs/osse.yaml --device cuda:0  # 4090
  python scripts/run_osse.py --config configs/osse_smoke.yaml \
      --sweep-frac 0.03 0.06 0.12        # observation-density sweep
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from huracan.config import load_config
from huracan.core.grid import grid_from_cfg
from huracan.core.metrics import crps_ensemble, essr
from huracan.core.rollout import spinup_and_forecast
from huracan.data.dataset import SphereDataset
from huracan.data.norm import Normalizer
from huracan.data.turbulence import generate_turbulence, make_obs_variant
from huracan.inference.runner import EdaRunner
from huracan.training.phases import fit_phase

DEFAULT_PHASES = ["deterministic", "stochastic", "fcst_ar", "assim", "joint"]
MODES = ["obs", "zero", "fcst"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def wrmse_ch(pred: np.ndarray, truth: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted RMSE.  pred (D, npix, C) (already member-averaged) or
    (npix, C) vs truth of matching shape -> (D, C) or (C,)."""
    err2 = (pred - truth) ** 2
    reduce = lambda e: np.sqrt((e * w[None, :, None]).sum(axis=-2) / w.sum())
    if err2.ndim == 3:
        return reduce(err2)
    return np.sqrt((err2 * w[:, None]).sum(axis=0) / w.sum())


def daily(x: np.ndarray) -> np.ndarray:
    """(2F, ...) 6-hourly -> (F, ...) daily means (4 fields per day)."""
    F = x.shape[0] // 2
    return x[: 2 * F].reshape(F, 2, *x.shape[1:]).mean(axis=1)


def key_channel(names, name):
    return names.index(name) if name in names else None


def ensure_data(cfg, regen: bool) -> Path:
    """Generate the nature run if missing; refuse to reuse a dataset whose
    shape belongs to a *different* config (grid/channels changed)."""
    root = Path(cfg.data.root) / cfg.data.source
    if regen or not (root / "state.npy").exists():
        if (root / "state.npy").exists():
            for f in ("state.npy", "obs.npy", "forcing.npy"):
                (root / f).unlink(missing_ok=True)
        if cfg.data.source == "kolmogorov":
            from huracan.data.kolmogorov import generate_kolmogorov

            generate_kolmogorov(cfg, root, seed=cfg.seed)
        else:
            generate_turbulence(cfg, root, seed=cfg.seed)
        return root
    state = np.load(root / "state.npy", mmap_mode="r")
    want = (
        cfg.data.synthetic.steps,
        grid_from_cfg(cfg.grid).npix,
        cfg.n_prognostic(),
    )
    if tuple(state.shape) != want:
        raise RuntimeError(
            f"{root}/state.npy has shape {tuple(state.shape)} but this "
            f"config needs {want} (T, npix, channels). Re-run with "
            f"--regen-data, or point data.root at a fresh directory."
        )
    return root


def run_mode(
    cfg, workdir: Path, device: str, mode: str, data_dir: Path, ic_noise: float = 0.0
):
    """One EDA trajectory.  Returns tensors + truth slices (normalized)."""
    runner = EdaRunner.from_checkpoints(cfg, workdir, device=device)
    S, F = cfg.spinup_calls(), cfg.lead_calls()
    M = cfg.eval.members
    sem = cfg.call_semantics()
    ds = SphereDataset(cfg, "val", rollout=S + F, data_dir=data_dir)
    window = ds.window(ds.first_t0, sem.window_frames(S + F))

    state = window["state"].to(runner.device)
    forcing = window["forcing"].to(runner.device)
    obs = window["obs"][: sem.obs_frames(S)].to(runner.device)
    if mode == "zero":
        obs = torch.zeros_like(obs)
    elif mode == "fcst":
        obs = None

    assim = runner.assim if mode in ("obs", "zero") else runner.fcst
    spin_end, preds, spin_mean = spinup_and_forecast(
        assim,
        runner.fcst,
        state,
        forcing,
        obs,
        members=M,
        spinup=S,
        lead=F,
        collect_spinup=True,
        ic_noise=ic_noise,
        grid_hw=(cfg.grid.nlat, cfg.grid.nlon),
    )
    st = window["state"]
    ns, nf = sem.frames_for_calls(S), sem.frames_for_calls(F)
    return {
        "spinup_mean": spin_mean.numpy(),  # (ns, npix, C) ens mean
        "spinup_end_mean": spin_end.mean(dim=0).numpy(),  # (npix, C)
        "preds": preds.numpy(),  # (M, nf, npix, C)
        "truth_analysis": st[2 : ns + 2].numpy(),
        "truth_analysis_final": st[ns + 1].numpy(),
        "truth_forecast": st[ns + 2 : ns + 2 + nf].numpy(),
        "t0": int(ds.first_t0),
    }


def climatology(cfg, data_dir: Path) -> np.ndarray:
    """Train-period per-pixel channel mean (normalized), (npix, C)."""
    root = Path(data_dir)
    norm = Normalizer.load(root / "norm_state.json")
    state = np.load(root / "state.npy", mmap_mode="r")
    a, b = cfg.data.train_period
    m = np.zeros(state.shape[1:], dtype=np.float64)
    n = 0
    for i in range(a, b, 32):
        m += state[i : min(i + 32, b)].mean(axis=0, dtype=np.float64)
        n += 1
    m /= n
    return ((m - norm.mean[None, :]) / norm.std[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--phases", default=",".join(DEFAULT_PHASES))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--regen-data", action="store_true")
    ap.add_argument("--sweep-frac", nargs="*", type=float, default=[])
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument(
        "--ic-noise",
        type=float,
        default=0.75,
        help="per-member initial-condition noise (sigma) for the spin-up; "
        "the OSSE recovery setting (0 = perfect IC)",
    )
    ap.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="key=value config overrides (dot paths, yaml values)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override or None)
    if args.members:
        cfg.eval.members = args.members
    workdir = Path(args.workdir or f"runs/{cfg.run_name}")
    workdir.mkdir(parents=True, exist_ok=True)
    phases = [p for p in args.phases.split(",") if p]
    device = args.device or cfg.device

    grid = grid_from_cfg(cfg.grid)
    w = grid.pixel_weights
    names = cfg.var_names()
    tcfg = cfg.data.synthetic.turbulence

    print(
        f"OSSE: grid {grid.nlat}x{grid.nlon} (spectral L_dyn="
        f"{tcfg.lmax_dyn or 'auto'}), {len(names)} channels, "
        f"obs fraction {cfg.data.synthetic.obs_fraction}, "
        f"obs noise {cfg.data.synthetic.obs_noise} sigma, "
        f"{cfg.eval.members} members, spin-up {cfg.eval.spinup_days} d, "
        f"lead {cfg.eval.lead_days} d"
    )

    base_root = ensure_data(cfg, args.regen_data)

    # -- variants: base observation config (+ optional density sweep) -----
    if args.sweep_frac:
        variants = []
        for frac in args.sweep_frac:
            tag = f"f{frac:g}"
            vroot = Path(cfg.data.root) / f"{cfg.data.source}_{tag}"
            if args.regen_data or not (vroot / "obs.npy").exists():
                make_obs_variant(
                    cfg,
                    base_root,
                    vroot,
                    obs_fraction=frac,
                    obs_noise=cfg.data.synthetic.obs_noise,
                    seed=cfg.seed,
                )
            variants.append((tag, frac, cfg.data.synthetic.obs_noise, vroot))
    else:
        variants = [
            (
                "base",
                cfg.data.synthetic.obs_fraction,
                cfg.data.synthetic.obs_noise,
                base_root,
            )
        ]

    results = {
        "config": vars(args) | {"config_file": args.config},
        "grid": [grid.nlat, grid.nlon],
        "names": names,
        "variants": [],
    }
    with open(base_root / "norm_state.json") as f:
        results["std"] = json.load(f)["std"]  # per-channel std (raw units)

    for tag, frac, noise, vroot in variants:
        vcfg = copy.deepcopy(cfg)
        vcfg.data.source = vroot.name
        vwork = workdir / f"osse_{tag}"
        vwork.mkdir(parents=True, exist_ok=True)
        print(f"\n===== variant {tag}: obs fraction {frac}, noise {noise} =====")

        if not args.skip_train:
            for ph in phases:
                print(f"  train {ph} ...", flush=True)
                t0 = time.time()
                res = fit_phase(vcfg, ph, device=device, workdir=vwork)
                lo = np.mean(res["losses"][: max(5, len(res["losses"]) // 5)])
                hi = np.mean(res["losses"][-max(5, len(res["losses"]) // 5) :])
                print(
                    f"    {ph}: {len(res['losses'])} steps, "
                    f"loss {lo:.4f} -> {hi:.4f} ({time.time() - t0:.0f}s)"
                )

        vres = {"tag": tag, "obs_fraction": frac, "obs_noise": noise, "modes": {}}
        clim = climatology(vcfg, vroot)
        truth_af = truth_f = None

        for mode in MODES:
            print(f"  EDA mode: {mode} ...", flush=True)
            t0 = time.time()
            out = run_mode(vcfg, vwork, device, mode, vroot, ic_noise=args.ic_noise)

            ana_curve = wrmse_ch(out["spinup_mean"], out["truth_analysis"], w)
            ana_final = wrmse_ch(
                out["spinup_end_mean"][None], out["truth_analysis_final"][None], w
            )[0]
            preds, truth = out["preds"], out["truth_forecast"]
            fc_rmse = wrmse_ch(preds.mean(axis=0), truth, w)
            fc_crps = crps_ensemble(preds, truth, w)
            fc_essr = essr(preds, truth, w)
            vres["modes"][mode] = {
                "analysis_curve": ana_curve.tolist(),  # (2S, C)
                "analysis_final": ana_final.tolist(),  # (C,)
                "forecast_rmse": fc_rmse.tolist(),  # (2F, C)
                "forecast_crps": fc_crps.tolist(),
                "forecast_essr": fc_essr.tolist(),
                "t0": out["t0"],
            }
            if mode == "obs":
                truth_af, truth_f = (
                    out["truth_analysis_final"],
                    out["truth_forecast"],
                )
            print(
                f"    analysis RMSE (chan mean) start "
                f"{ana_curve[0].mean():.3f} -> end {ana_curve[-1].mean():.3f}; "
                f"forecast d1 {daily(fc_rmse)[0].mean():.3f} "
                f"d{daily(fc_rmse).shape[0]} "
                f"{daily(fc_rmse)[-1].mean():.3f}  ({time.time() - t0:.0f}s)"
            )

        # baselines (obs-independent, from the treatment truth slices)
        clim_f = np.repeat(clim[None], truth_f.shape[0], axis=0)
        persist = np.repeat(truth_af[None], truth_f.shape[0], axis=0)
        vres["baselines"] = {
            "climatology_rmse": wrmse_ch(clim_f, truth_f, w).tolist(),
            "persistence_rmse": wrmse_ch(persist, truth_f, w).tolist(),
            "obs_error_floor": noise,
        }
        results["variants"].append(vres)

    # -- report ----------------------------------------------------------
    out_json = workdir / "osse_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f)
    summary = write_summary(results, workdir)
    plot_curves(results, workdir)
    print(summary)
    print(f"\nresults: {out_json}")
    print(f"summary: {workdir / 'osse_summary.md'}")
    print(f"figure : {workdir / 'osse_curves.png'}")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def write_summary(results, workdir: Path) -> str:
    names = results["names"]
    lines = [
        "# OSSE 同化能力验证报告 (assimilation capability)",
        "",
        f"* grid {results['grid'][0]}x{results['grid'][1]}, "
        f"{len(names)} channels, {results['variants'][0]['modes']['obs']['t0']} "
        f"start frame (held-out nature-run segment)",
    ]
    key_idx = {k: key_channel(names, k) for k in ("u850", "z500", "t850", "msl")}
    for v in results["variants"]:
        lines += [
            "",
            f"## variant `{v['tag']}`: obs fraction {v['obs_fraction']}, "
            f"obs noise {v['obs_noise']} sigma ({len(MODES)} EDA modes)",
            "",
            "### 分析误差 analysis RMSE (normalized, channel mean)",
            "",
            "| spin-up call | with obs | zero-obs | fcst-spinup |",
            "|---|---|---|---|",
        ]
        ac = {
            m: np.asarray(v["modes"][m]["analysis_curve"]).mean(axis=1) for m in MODES
        }
        S = len(ac["obs"])
        rows = sorted(set([0, S // 4, S // 2, 3 * S // 4, S - 1]))
        for i in rows:
            lines.append(
                f"| {i + 1} | {ac['obs'][i]:.3f} | {ac['zero'][i]:.3f} "
                f"| {ac['fcst'][i]:.3f} |"
            )
        fin = {m: np.mean(v["modes"][m]["analysis_final"]) for m in MODES}
        floor = v["baselines"]["obs_error_floor"]
        lines += [
            "",
            f"* **最终分析误差**: with obs **{fin['obs']:.3f}** vs zero-obs "
            f"{fin['zero']:.3f} vs fcst-spinup {fin['fcst']:.3f} "
            f"(观测误差下限 {floor:.2f})",
        ]
        if fin["obs"] < 0.85 * fin["zero"]:
            extra = (
                "，且低于观测误差下限（滤波生效）"
                if fin["obs"] < floor
                else f"（距观测误差下限 {floor:.2f} 还差 "
                f"{fin['obs'] / max(floor, 1e-9):.1f}x）"
            )
            lines.append(
                f"* ✅ 同化有效: 有观测的分析误差比无观测对照低 "
                f"{1 - fin['obs'] / max(fin['zero'], 1e-9):.0%}" + extra
            )
        else:
            lines.append("* ⚠️ 同化信号不明显: 需要更长训练/更多观测/检查配置")

        lines += [
            "",
            "### 预报 skill (normalized RMSE, channel mean)",
            "",
            "| lead day | with obs | zero-obs | fcst-spinup | clim | persist |",
            "|---|---|---|---|---|---|",
        ]
        rm = {
            m: daily(np.asarray(v["modes"][m]["forecast_rmse"]).mean(axis=1))
            for m in MODES
        }
        cl = daily(np.asarray(v["baselines"]["climatology_rmse"]).mean(axis=1))
        pe = daily(np.asarray(v["baselines"]["persistence_rmse"]).mean(axis=1))
        D = len(rm["obs"])
        for d in sorted(set([0, D // 2, D - 1])):
            lines.append(
                f"| {d + 1} | {rm['obs'][d]:.3f} | {rm['zero'][d]:.3f} "
                f"| {rm['fcst'][d]:.3f} | {cl[d]:.3f} | {pe[d]:.3f} |"
            )
        crps = daily(np.asarray(v["modes"]["obs"]["forecast_crps"]).mean(axis=1))
        essr = np.asarray(v["modes"]["obs"]["forecast_essr"]).mean()
        lines += [
            "",
            f"* CRPS (with obs) day1 {crps[0]:.3f} -> day{D} {crps[-1]:.3f}; "
            f"mean ESSR {essr:.2f} (1.0 = calibrated)",
        ]
        # key channels in physical units (normalized rmse * channel std)
        std = np.asarray(results.get("std", []))
        if std.size:
            fin_ch = {m: np.asarray(v["modes"][m]["analysis_final"]) for m in MODES}
            for nm, unit, mult in (
                ("u850", "m/s", 1.0),
                ("t850", "K", 1.0),
                ("z500", "m2/s2", 10.0),
                ("msl", "hPa", 1.0),
            ):
                k = key_idx.get(nm)
                if k is None:
                    continue
                lines.append(
                    f"* {nm} 最终分析误差: {fin_ch['obs'][k] * std[k] * mult:.2f} "
                    f"({unit}; 无观测对照 "
                    f"{fin_ch['zero'][k] * std[k] * mult:.2f})"
                )
    md = "\n".join(lines)
    (workdir / "osse_summary.md").write_text(md)
    return md


def plot_curves(results, workdir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = len(results["variants"]) > 1
    fig, axes = plt.subplots(
        2,
        2 if sweep else 2,
        figsize=(11, 7.5),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]
    colors = {m: c for m, c in zip(MODES, ["C0", "C1", "C3"])}

    v0 = results["variants"][0]
    floor = v0["baselines"]["obs_error_floor"]
    for m in MODES:
        y = np.asarray(v0["modes"][m]["analysis_curve"]).mean(axis=1)
        ax_a.plot(
            np.arange(1, len(y) + 1) / 2,
            y,
            color=colors[m],
            label={
                "obs": "with obs",
                "zero": "zero-obs ctrl",
                "fcst": "fcst-spinup ctrl",
            }[m],
        )
    ax_a.axhline(floor, color="k", ls="--", lw=1, label="obs error floor")
    ax_a.set_xlabel("spin-up day (12 h per AR call)")
    ax_a.set_ylabel("analysis RMSE (normalized)")
    ax_a.set_title("(a) analysis error during EDA cycling")
    ax_a.legend(fontsize=8)

    for m in MODES:
        y = daily(np.asarray(v0["modes"][m]["forecast_rmse"]).mean(axis=1))
        ax_b.plot(
            np.arange(1, len(y) + 1),
            y,
            color=colors[m],
            label={
                "obs": "with obs",
                "zero": "zero-obs ctrl",
                "fcst": "fcst-spinup ctrl",
            }[m],
        )
    cl = daily(np.asarray(v0["baselines"]["climatology_rmse"]).mean(axis=1))
    pe = daily(np.asarray(v0["baselines"]["persistence_rmse"]).mean(axis=1))
    ax_b.plot(np.arange(1, len(cl) + 1), cl, "k:", label="climatology")
    ax_b.plot(np.arange(1, len(pe) + 1), pe, "k--", lw=1, label="persistence")
    ax_b.set_xlabel("lead day")
    ax_b.set_ylabel("forecast RMSE (normalized)")
    ax_b.set_title("(b) forecast error vs lead time")
    ax_b.legend(fontsize=8)

    crps = daily(np.asarray(v0["modes"]["obs"]["forecast_crps"]).mean(axis=1))
    es = daily(np.asarray(v0["modes"]["obs"]["forecast_essr"]).mean(axis=1))
    ax_c.plot(np.arange(1, len(crps) + 1), crps, "C0", label="CRPS (with obs)")
    ax_c.set_xlabel("lead day")
    ax_c.set_ylabel("CRPS (normalized)")
    ax_c.set_title("(c) CRPS")
    ax2 = ax_c.twinx()
    ax2.plot(np.arange(1, len(es) + 1), es, "C2", label="ESSR")
    ax2.axhline(1.0, color="C2", ls=":", lw=1)
    ax2.set_ylabel("ESSR", color="C2")
    ax_c.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="lower right")

    if sweep:
        fr = [v["obs_fraction"] for v in results["variants"]]
        for m in MODES:
            y = [np.mean(v["modes"][m]["analysis_final"]) for v in results["variants"]]
            ax_d.plot(
                fr,
                y,
                "o-",
                color=colors[m],
                label={
                    "obs": "with obs",
                    "zero": "zero-obs ctrl",
                    "fcst": "fcst-spinup ctrl",
                }[m],
            )
        ax_d.axhline(floor, color="k", ls="--", lw=1, label="obs error floor")
        ax_d.set_xlabel("observation fraction (area-uniform stations)")
        ax_d.set_ylabel("final analysis RMSE")
        ax_d.set_title("(d) analysis error vs observation density")
        ax_d.legend(fontsize=8)
    else:
        ana = np.asarray(v0["modes"]["obs"]["analysis_final"])
        k = len(ana)
        ax_d.barh(np.arange(k)[::-1], ana, height=0.8)
        nz = np.nonzero(ana > 0.05)[0]
        ax_d.set_yticks(np.arange(k)[::-1][nz])
        ax_d.set_yticklabels([results["names"][i] for i in nz], fontsize=5)
        ax_d.axvline(floor, color="k", ls="--", lw=1)
        ax_d.set_xlabel("final analysis RMSE (with obs)")
        ax_d.set_title("(d) per-channel final analysis error")

    fig.tight_layout()
    fig.savefig(workdir / "osse_curves.png", dpi=160)


if __name__ == "__main__":
    main()
