#!/usr/bin/env python3
"""Demo: how each training phase loads weights and data.

Walks the full phase chain (deterministic -> stochastic -> fcst_ar -> assim
-> joint -> hybrid) the same way ``scripts/train.py --phase all`` would, but
instead of training it:

  1. builds the phase's LightningModule (fresh random models),
  2. runs ``PhaseInitCallback.on_train_start`` for real and verifies *how*
     weights arrive (exact copy vs ``load_with_new_columns``),
  3. writes a placeholder ``ckpt_{phase}.pt`` so the next phase can chain
     from it (a real run writes this file after training, via
     ``PhaseRunCallback``),
  4. prints the data block shapes and the training-time ``unroll`` shapes,
  5. finally demonstrates ``EdaRunner.from_checkpoints``: the joint branch
     and the assim+fcst_ar fallback.

CPU + configs/smoke.yaml, well under a minute:

    .venv/bin/python scripts/demo_phase_loading.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from huracan.config import load_config
from huracan.core import io
from huracan.core.rollout import unroll
from huracan.data.dataset import SphereDataset
from huracan.data.synthetic import generate_synthetic
from huracan.inference.runner import EdaRunner
from huracan.training.callbacks import PhaseInitCallback
from huracan.training.module import ACTIVE_MODELS, HuracanLitModule
from huracan.training.phases import ALL_PHASES

LINE = "=" * 72
THIN = "-" * 72


def hdr(i: int, n: int, phase: str) -> None:
    print(
        f"\n{LINE}\n[{i}/{n}] phase: {phase}   (trains: {'+'.join(ACTIVE_MODELS[phase])})\n{LINE}"
    )


def call_schedule(pc) -> str:
    n, s = pc.n_calls(), pc.split_assim()
    if s == 0:
        who = f"{n} call(s), all fcst"
    elif s == n:
        who = f"{n} call(s), all assim (fed obs each call)"
    else:
        who = f"{n} calls: first {s} assim (obs) then {n - s} fcst"
    return who


def fp(w: torch.Tensor) -> str:
    return f"{w.float().sum().item():+.4f}"


def sd_exact(model: torch.nn.Module, src: dict) -> bool:
    own = model.state_dict()
    if set(src) - set(own):
        return False
    return all(torch.equal(own[k], v) for k, v in src.items())


def demo_phase(cfg, phase: str, i: int, n: int, workdir: Path) -> None:
    hdr(i, n, phase)
    pc = cfg.train.phases[phase]

    if pc.rollout_assim is not None:
        rtxt = f"rollout_assim={pc.rollout_assim}, rollout_fcst={pc.rollout_fcst}"
    else:
        rtxt = f"rollout={pc.rollout}"
    print(
        f"  config : steps={pc.steps} lr={pc.lr:.1e} loss={pc.loss} "
        f"schedule={pc.schedule} noise={pc.noise} obs={pc.obs}"
    )
    print(
        f"           {rtxt} -> n_calls={pc.n_calls()}, split_assim={pc.split_assim()}"
    )
    print(
        f"           unroll schedule: {call_schedule(pc)}"
        f"   M={cfg.train.ensemble if pc.noise else 1} members"
    )

    # -- fresh module, exactly as fit_phase builds it ----------------------
    torch.manual_seed(cfg.seed + 1000 + i)
    module = HuracanLitModule(cfg, phase)
    fcst, assim = module.fcst, module.assim
    fcst_n_in = fcst.n_in
    print(
        f"  models : fcst embed Linear({fcst.n_in} -> {fcst.embed.out_features})   "
        f"assim embed Linear({assim.n_in} -> {assim.embed.out_features})   "
        f"(assim encoder +{assim.n_in - fcst.n_in} obs columns)"
    )

    # -- weight loading: the interesting part ------------------------------
    if pc.init is None:
        print(
            f"  init   : {pc.init!r} -> weights stay at fresh random init "
            f"(fcst embed sum {fp(fcst.embed.weight.data)})"
        )
    else:
        ck = io.load_phase(workdir, pc.init)  # the only torch.load path
        src = ck["models"]["fcst"] if "fcst" in ck["models"] else ck["models"]["assim"]
        print(
            f"  init   : {pc.init!r} -> {io.phase_path(workdir, pc.init).name} "
            f"[models in ckpt: {'+'.join(ck['models'])}]"
        )
        print(
            f"           loading through core/io.load_phase + PhaseInitCallback.on_train_start"
        )

        if phase in ("assim", "joint"):
            obs_cols_before = assim.embed.weight.data[:, fcst_n_in:].clone()
        fcst_before = fcst.embed.weight.data.clone()
        assim_before = assim.embed.weight.data.clone()

        PhaseInitCallback(cfg, phase, workdir).on_train_start(None, module)

        if phase in ("stochastic", "fcst_ar", "hybrid"):
            print(
                f"           fcst embed sum: {fp(fcst_before)} -> {fp(fcst.embed.weight.data)}  "
                f"exact copy from ckpt: {'OK' if sd_exact(fcst, src) else 'MISMATCH'}"
            )
        elif phase == "assim":
            inh = assim.embed.weight.data[:, : src["embed.weight"].shape[1]]
            print(
                f"           assim embed sum: {fp(assim_before)} -> {fp(assim.embed.weight.data)}"
            )
            print(
                f"           inherited cols [0:{src['embed.weight'].shape[1]}] match ckpt: "
                f"{'OK' if torch.equal(inh, src['embed.weight']) else 'MISMATCH'}"
            )
            print(
                f"           new obs cols [{fcst_n_in}:{assim.n_in}] untouched by load "
                f"(stay random): "
                f"{'OK' if torch.equal(assim.embed.weight.data[:, fcst_n_in:], obs_cols_before) else 'MISMATCH'}"
            )
            other = {k: v for k, v in src.items() if k != "embed.weight"}
            print(
                f"           all other tensors copied exactly: "
                f"{'OK' if all(torch.equal(assim.state_dict()[k], v) for k, v in other.items()) else 'MISMATCH'}"
            )
        elif phase == "joint":
            srca = ck["models"]["assim"]
            ckf = io.load_phase(workdir, pc.fcst_init)
            print(
                f"           assim <- ckpt_{pc.init}: exact copy: "
                f"{'OK' if sd_exact(assim, srca) else 'MISMATCH'}"
            )
            print(
                f"           fcst  <- ckpt_{pc.fcst_init} (fcst_init): exact copy: "
                f"{'OK' if sd_exact(fcst, ckf['models']['fcst']) else 'MISMATCH'}"
            )

    # -- placeholder checkpoint so the chain can continue ------------------
    io.save_phase(workdir, phase, step=0, state_dicts=module.state_dicts())
    print(
        f"  saved  : {io.phase_path(workdir, phase).name} "
        f"[models: {'+'.join(module.state_dicts())}]  (placeholder; a real run "
        f"saves this after training via PhaseRunCallback)"
    )

    # -- data shapes --------------------------------------------------------
    ds = SphereDataset(cfg, "train", rollout=pc.n_calls())
    b = ds[0]
    opc = cfg.model.outputs_per_call
    frames = cfg.call_semantics().window_frames(pc.n_calls())
    print(
        f"  data   : len(dataset)={len(ds)}  block = 2+{opc}*{pc.n_calls()} = "
        f"{frames} frames (t0-1 .. t0+{opc * pc.n_calls()})"
    )
    for k in ("state", "forcing", "obs"):
        print(f"           {k:7s} {tuple(b[k].shape)}")

    # -- training-time unroll shapes ----------------------------------------
    with torch.no_grad():
        preds, targets = unroll(
            module.models_dict(),
            b["state"].unsqueeze(0),
            b["forcing"].unsqueeze(0),
            b["obs"].unsqueeze(0) if pc.obs else None,
            n_calls=pc.n_calls(),
            split_assim=pc.split_assim(),
            ensemble=cfg.train.ensemble,
            use_noise=pc.noise,
        )
    m = cfg.train.ensemble if pc.noise else 1
    print(
        f"  unroll : preds {tuple(preds.shape)}   (B=1, M={m}, "
        f"{opc * pc.n_calls()} frames)"
    )
    print(f"           targets {tuple(targets.shape)}   (= state[:, 2:], ERA5 truth)")


def demo_inference_loading(cfg, workdir: Path) -> None:
    print(f"\n{LINE}\ninference loading: EdaRunner.from_checkpoints\n{LINE}")
    dev = "cpu"

    runner = EdaRunner.from_checkpoints(cfg, workdir, device=dev)
    print(f"  branch 1: ckpt_joint.pt present -> assim + fcst both from joint")
    print(
        f"            loaded: assim embed in_features={runner.assim.embed.in_features}, "
        f"fcst embed in_features={runner.fcst.embed.in_features}"
    )

    joint = io.phase_path(workdir, "joint")
    bak = joint.with_suffix(".bak")
    shutil.move(joint, bak)
    runner2 = EdaRunner.from_checkpoints(cfg, workdir, device=dev)
    print(
        f"  branch 2: ckpt_joint.pt removed -> fallback: assim <- ckpt_assim.pt, "
        f"fcst <- ckpt_fcst_ar.pt"
    )
    print(
        f"            loaded: assim embed in_features={runner2.assim.embed.in_features}, "
        f"fcst embed in_features={runner2.fcst.embed.in_features}"
    )
    shutil.move(bak, joint)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"),
    )
    ap.add_argument(
        "--workdir", default=None, help="demo workdir (default runs/demo_phase_loading)"
    )
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(0)
    cfg = load_config(args.config)
    workdir = Path(args.workdir or "runs/demo_phase_loading")
    workdir.mkdir(parents=True, exist_ok=True)

    if cfg.data.source == "synthetic":
        root = Path(cfg.data.root) / "synthetic"
        if not (root / "state.npy").exists():
            print("generating synthetic data ...")
            generate_synthetic(cfg, root, seed=cfg.seed)

    print(LINE)
    print("channel layout (shared by every phase)")
    print(THIN)
    _m = HuracanLitModule(cfg, "deterministic")  # builds both models at once
    fcst, assim = _m.fcst, _m.assim
    print(
        f"  n_state={fcst.n_state}  n_forcing={cfg.n_forcing()}  "
        f"n_obs={cfg.n_obs_channels()}"
    )
    print(f"  fcst.n_in  = 2*(state+forcing)             = {fcst.n_in}")
    print(f"  assim.n_in = 2*(state+forcing) + 2*n_obs    = {assim.n_in}")
    print(f"  params: fcst {sum(p.numel() for p in fcst.parameters()) / 1e6:.3f}M")

    for i, phase in enumerate(ALL_PHASES, 1):
        demo_phase(cfg, phase, i, len(ALL_PHASES), workdir)

    demo_inference_loading(cfg, workdir)

    print(f"\ncheckpoints now in {workdir}:")
    for p in sorted(workdir.glob("ckpt_*.pt")):
        print(f"  {p.name}")

    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
