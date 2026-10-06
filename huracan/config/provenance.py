"""Provenance registry: which config values are paper facts, which are
inferences, which are repo choices (guide Sec. 26).

Every entry maps a config dot-path to (tag, source note).  ``*`` wildcards
inside ``train.phases.*`` apply to every phase.  ``check()`` verifies that
each registered path still resolves on a real config -- if you rename a
schema field, the provenance table must follow (enforced by unit tests).
"""

from __future__ import annotations

from .schema import HuracanConfig

PAPER = "PAPER"
INFERRED = "INFERRED"
REPO = "REPO"

PROVENANCE: dict[str, tuple[str, str]] = {
    # -- grid / spectral ----------------------------------------------------
    "grid.nlat": (INFERRED, "1 deg + L=179 constrain a 180x360-class regular grid; exact family/nodes OPEN (guide Sec. 7)"),
    "grid.nlon": (INFERRED, "same as grid.nlat; nlon >= 2*lmax+1 for exact SHT"),
    "grid.lmax": (PAPER, "Appendix A.2: '1-degree resolution implies L = 179' (pilot uses less)"),
    # -- model ---------------------------------------------------------------
    "model.hidden": (REPO, "not published"),
    "model.blocks": (REPO, "not published"),
    "model.k_filters": (PAPER, "A.2: k = 15 at paper scale (pilot uses 8)"),
    "model.swin_window_lat": (REPO, "Swin hyperparameters not published (guide M07-M10)"),
    "model.swin_window_lon": (REPO, "Swin hyperparameters not published"),
    "model.heads": (REPO, "not published"),
    "model.mlp_ratio": (REPO, "not published"),
    "model.noise_dim": (REPO, "noise dimension not published (guide S01-S04)"),
    "model.noise_embed_dim": (REPO, "not published"),
    "model.degree_freqs": (REPO, "sinusoidal degree encoding stated, its size not"),
    "model.prediction": (REPO, "residual vs direct target not specified (guide M13/M14)"),
    "model.grad_checkpoint": (REPO, "engineering choice"),
    # -- data ------------------------------------------------------------------
    "data.variables.upper": (PAPER, "Table 2: z,t,u,v,q"),
    "data.variables.levels": (REPO, "13 levels confirmed but the list is not published (guide V01)"),
    "data.variables.surface": (PAPER, "Table 2: 2t,2d,10u,10v,tcc,tcwv,msl"),
    "data.forcing": (PAPER, "Table 2 input-only: insolation + constants z,lsm"),
    "data.obs.enabled": (REPO, "pilot uses IGRA/ISD only; satellite channels are Level-2 work"),
    "data.obs.surface_vars": (REPO, "Table 2 in-situ surface = 6 vars incl. tcc; repo omits tcc"),
    "data.train_period": (REPO, "pilot subset of paper's 1979-2014 / 2010-2022"),
    "data.val_period": (REPO, "paper reserves 2023 for validation"),
    # -- training --------------------------------------------------------------
    "train.batch_size": (PAPER, "A.3: batch size 16"),
    "train.ensemble": (PAPER, "A.3: stochastic training ensemble size 2"),
    "train.grad_clip": (REPO, "not published"),
    "train.warmup_frac": (REPO, "not published"),
    "train.num_workers": (REPO, "engineering choice"),
    "train.phases.*.steps": (PAPER, "Table 3: 347k/149k/14k/46k/13k/2.7k"),
    "train.phases.*.lr": (PAPER, "Table 3 max LR per phase"),
    "train.phases.*.schedule": (PAPER, "Table 3: cosine / constant"),
    "train.phases.*.loss": (PAPER, "Sec 2.3: L1 deterministic, CRPS stochastic"),
    "train.phases.*.rollout": (PAPER, "Sec 2.3: 6-step=3d fcst, 8-step=4d assim"),
    "train.phases.*.rollout_assim": (PAPER, "Sec 2.3: joint = 8 assim + 8 fcst calls"),
    "train.phases.*.rollout_fcst": (PAPER, "Sec 2.3: joint = 8 assim + 8 fcst calls"),
    "train.phases.*.noise": (PAPER, "Sec 2.3: noise added from stochastic phase on"),
    "train.phases.*.init": (PAPER, "Sec 2.3: phase chaining; obs columns random"),
    # -- eval -------------------------------------------------------------------
    "eval.members": (PAPER, "Sec 3.1: 48 independent EDA members"),
    "eval.spinup_days": (PAPER, "Sec 3.1: at least 3 months of assimilation (>= ~92 d)"),
    "eval.lead_days": (PAPER, "Sec 3.1: 10 lead days"),
}


def resolve(cfg: HuracanConfig, path: str):
    """Walk a dot-path (with ``*`` wildcards over mapping sections) on a
    config; raises AttributeError for unknown fields."""
    def walk(node, parts):
        if not parts:
            return node
        head, rest = parts[0], parts[1:]
        if head == "*":
            if not isinstance(node, dict):
                raise AttributeError(path)
            return [walk(node[k], rest) for k in node]
        return walk(getattr(node, head), rest)

    return walk(cfg, path.split("."))


def check(cfg: HuracanConfig) -> list[str]:
    """Return the list of registered paths that no longer resolve (should
    stay empty; unit-tested)."""
    broken = []
    for path in PROVENANCE:
        try:
            resolve(cfg, path)
        except AttributeError:
            broken.append(path)
    return broken


def report() -> str:
    """Aligned provenance table for CLI output."""
    width = max(len(p) for p in PROVENANCE)
    lines = [f"{'config key':{width}}  provenance  source"]
    for path, (tag, note) in PROVENANCE.items():
        lines.append(f"{path:{width}}  {tag:<9}  {note}")
    return "\n".join(lines)
