"""Configuration loading (YAML -> nested dict with defaults)."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

_FLOAT_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _coerce(node):
    """YAML 1.1 parses '5e-4' as a string; convert numeric strings back."""
    if isinstance(node, dict):
        return {k: _coerce(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce(v) for v in node]
    if isinstance(node, str) and _FLOAT_RE.match(node):
        return float(node)
    return node


DEFAULTS = {
    "run_name": "run",
    "output_dir": "runs/run",
    "seed": 0,
    "device": "cuda",
    "grid": {
        "nside": 16,  # HEALPix nside; 16 -> 3072 px (~3.7 deg), 64 -> 1 deg
        "lmax": 32,  # spectral truncation; paper uses 179 at nside=64
    },
    "model": {
        "hidden": 192,
        "blocks": 6,
        "k_filters": 8,  # paper: 15 compressed spectral filters (Sec. A.2)
        "swin_window": 8,  # window size within each HEALPix face
        "heads": 3,
        "mlp_ratio": 4.0,
        "noise_dim": 8,  # per-pixel Gaussian noise channels
        "noise_embed_dim": 64,  # MLP-processed noise embedding dim (Sec. 2.2)
        "degree_freqs": 16,  # sinusoidal encoding dims for spectral degree
    },
    "data": {
        "source": "synthetic",  # 'synthetic' or 'era5'
        "root": "data",
        "variables": {
            "upper": ["z", "t", "u", "v", "q"],
            "levels": [
                50,
                100,
                150,
                200,
                250,
                300,
                400,
                500,
                600,
                700,
                850,
                925,
                1000,
            ],  # 13 levels (paper)
            "surface": ["2t", "2d", "10u", "10v", "tcc", "tcwv", "msl"],
        },
        "forcing": ["insol", "zsfc", "lsm"],
        "obs": {
            "enabled": False,
            # channels: upper-air z,t,u,v,q x levels (in-situ) + surface
            "surface_vars": ["2t", "2d", "10u", "10v", "msl"],
        },
        "train_period": [0, 3000],  # step index range (6-hourly) for training
        "val_period": [3000, 3200],
        "synthetic": {
            "channels": 6,
            "steps": 4000,
            "wave_speeds": [0.010, 0.006, 0.003, 0.0015, 0.0008, 0.0],
            "rotation_speed": 0.15,  # rad/day solid body advection
            "decay_per_step": 0.9995,
            "obs_fraction": 0.12,  # fraction of pixels with in-situ obs
            "obs_noise": 0.15,
        },
    },
    "train": {
        "batch_size": 8,
        "ensemble": 2,  # paper trains stochastic models with 2 members
        "grad_clip": 1.0,
        "warmup_frac": 0.02,
        "num_workers": 2,
        "phases": {
            # order and settings follow paper Table 3 (scaled down)
            "deterministic": {
                "steps": 40000,
                "lr": 5e-4,
                "loss": "l1",
                "rollout": 0,
                "noise": False,
                "init": None,
            },
            "stochastic": {
                "steps": 15000,
                "lr": 2e-4,
                "loss": "crps",
                "rollout": 0,
                "noise": True,
                "init": "deterministic",
            },
            "fcst_ar": {
                "steps": 3000,
                "lr": 3e-6,
                "loss": "crps",
                "rollout": 6,
                "noise": True,
                "init": "stochastic",
            },
            "assim": {
                "steps": 5000,
                "lr": 2e-4,
                "loss": "crps",
                "rollout": 8,
                "noise": True,
                "obs": True,
                "init": "stochastic",
            },
            "joint": {
                "steps": 2000,
                "lr": 3e-6,
                "loss": "crps",
                "rollout_assim": 8,
                "rollout_fcst": 8,
                "noise": True,
                "obs": True,
                "init": "assim",
            },
            "hybrid": {
                "steps": 1000,
                "lr": 3e-6,
                "loss": "crps",
                "rollout": 6,
                "noise": True,
                "init": "fcst_ar",
            },
        },
    },
    "eval": {
        "members": 8,  # paper: 48 EDA members
        "spinup_steps": 120,  # 12h model calls during spin-up (120 -> 60 days)
        "lead_steps": 20,  # forecast model calls (20 -> 10 days)
        "start_index": None,  # step index in val/test set for EDA init
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    return _coerce(cfg)


def var_names(cfg: dict) -> list[str]:
    """Ordered list of prognostic variable channel names (paper: 72)."""
    v = cfg["data"]["variables"]
    names = [f"{u}{lev}" for lev in v["levels"] for u in v["upper"]]
    names += list(v["surface"])
    return names


def n_prognostic(cfg: dict) -> int:
    return len(var_names(cfg))


def n_obs_channels(cfg: dict) -> int:
    v = cfg["data"]["variables"]
    o = cfg["data"]["obs"]
    if not o.get("enabled", False):
        return 0
    return len(v["upper"]) * len(v["levels"]) + len(o["surface_vars"])
