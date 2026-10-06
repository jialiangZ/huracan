"""Typed configuration for the Huracan reproduction.

Modules receive only the dataclass section they need (e.g. the model takes
``ModelCfg``, never the whole config), so no module has to understand the
global schema.  YAML files keep their original layout; see ``loader.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import get_type_hints


class ConfigError(ValueError):
    """Raised for unknown keys, missing keys or bad values in a config."""


# ---------------------------------------------------------------------------
# leaf sections
# ---------------------------------------------------------------------------


@dataclass
class GridCfg:
    """Gauss-Legendre x equispaced regular grid [INFERRED].

    The paper fixes 1-degree output and L=179 (A.2) but never names the grid
    family (guide Sec. 7); nlat/nlon at paper scale would be 180/360."""

    nlat: int = 48
    nlon: int = 96
    lmax: int = 47


@dataclass
class ModelCfg:
    hidden: int = 192                 # [REPO] not published
    blocks: int = 6                   # [REPO] not published
    k_filters: int = 8                # paper A.2: k=15 at scale
    swin_window_lat: int = 6          # [REPO] not published; must divide nlat
    swin_window_lon: int = 12         # [REPO] not published; must divide nlon
    heads: int = 3                    # [REPO] not published
    mlp_ratio: float = 4.0            # [REPO] not published
    noise_dim: int = 8                # [REPO] not published
    noise_embed_dim: int = 64         # [REPO] not published
    degree_freqs: int = 16            # [REPO] not published
    grad_checkpoint: bool = True      # [REPO] engineering choice
    prediction: str = "residual"      # [REPO] residual|direct (guide Sec 8.6)

    def __post_init__(self):
        if self.prediction not in ("residual", "direct"):
            raise ValueError(
                f"model.prediction must be 'residual' or 'direct', "
                f"got {self.prediction!r}"
            )


@dataclass
class ObsCfg:
    enabled: bool = False
    surface_vars: list[str] = field(
        default_factory=lambda: ["2t", "2d", "10u", "10v", "msl"]
    )


@dataclass
class VariablesCfg:
    upper: list[str] = field(default_factory=lambda: ["z", "t", "u", "v", "q"])
    levels: list[int] = field(
        default_factory=lambda: [
            50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
        ]
    )
    surface: list[str] = field(
        default_factory=lambda: ["2t", "2d", "10u", "10v", "tcc", "tcwv", "msl"]
    )


@dataclass
class SyntheticCfg:
    channels: int = 6
    steps: int = 4000
    wave_speeds: list[float] = field(
        default_factory=lambda: [0.010, 0.006, 0.003, 0.0015, 0.0008, 0.0]
    )
    rotation_speed: float = 0.15
    decay_per_step: float = 0.9995
    obs_fraction: float = 0.12
    obs_noise: float = 0.15


@dataclass
class DataCfg:
    source: str = "synthetic"
    root: str = "data"
    variables: VariablesCfg = field(default_factory=VariablesCfg)
    forcing: list[str] = field(default_factory=lambda: ["insol", "zsfc", "lsm"])
    obs: ObsCfg = field(default_factory=ObsCfg)
    train_period: list[int] = field(default_factory=lambda: [0, 3000])
    val_period: list[int] = field(default_factory=lambda: [3000, 3200])
    synthetic: SyntheticCfg = field(default_factory=SyntheticCfg)


@dataclass
class PhaseCfg:
    """One training phase (paper Table 3).  ``init`` names the previous
    phase checkpoint to chain from; ``fcst_init`` (joint only) names the
    checkpoint supplying the forecast model weights."""

    steps: int = 1000
    lr: float = 1e-4
    loss: str = "l1"
    schedule: str = "cosine"
    rollout: int = 0
    rollout_assim: int | None = None
    rollout_fcst: int | None = None
    noise: bool = False
    obs: bool = False
    init: str | None = None
    fcst_init: str | None = None
    batch_size: int | None = None

    def n_calls(self) -> int:
        """Autoregressive model calls per training sample."""
        if self.rollout_assim is not None:
            return self.rollout_assim + (self.rollout_fcst or 0)
        return max(1, self.rollout)

    def split_assim(self) -> int:
        """Calls using the assimilation model (rest use the forecast model)."""
        if self.rollout_assim is not None:
            return self.rollout_assim
        return self.n_calls() if self.obs else 0


def _default_phases() -> dict[str, PhaseCfg]:
    return {
        "deterministic": PhaseCfg(
            steps=40000, lr=5e-4, loss="l1", rollout=0, noise=False, init=None
        ),
        "stochastic": PhaseCfg(
            steps=15000, lr=2e-4, loss="crps", rollout=0, noise=True,
            init="deterministic",
        ),
        "fcst_ar": PhaseCfg(
            steps=3000, lr=3e-6, loss="crps", rollout=6, noise=True,
            init="stochastic",
        ),
        "assim": PhaseCfg(
            steps=5000, lr=2e-4, loss="crps", rollout=8, noise=True, obs=True,
            init="stochastic",
        ),
        "joint": PhaseCfg(
            steps=2000, lr=3e-6, loss="crps", rollout_assim=8, rollout_fcst=8,
            noise=True, obs=True, init="assim", fcst_init="fcst_ar",
        ),
        "hybrid": PhaseCfg(
            steps=1000, lr=3e-6, loss="crps", rollout=6, noise=True,
            init="fcst_ar",
        ),
    }


@dataclass
class TrainCfg:
    batch_size: int = 8
    ensemble: int = 2
    grad_clip: float = 1.0
    warmup_frac: float = 0.02
    num_workers: int = 2
    phases: dict[str, PhaseCfg] = field(default_factory=_default_phases)


@dataclass
class EvalCfg:
    """EDA / forecast experiment sizing, in calendar units [PAPER Sec. 3.1].

    The spin-up length is deliberately expressed in *days* (calendar-driven,
    guide Sec. 12.3): 3 months from 2023-10-01 is ~92 days, i.e. 184
    autoregressive calls at 12 h per call -- never a hard-coded loop count."""

    members: int = 8                  # paper: 48
    spinup_days: float = 60.0         # paper: at least 3 months (~92 d)
    lead_days: float = 10.0           # paper: 10 lead days
    start_index: int | None = None

    # one AR call advances 12 h (core.rollout.AR_ADVANCE_HOURS); schema must
    # stay importable from core, so the constant is restated here
    def spinup_calls(self) -> int:
        return int(round(self.spinup_days * 24 / 12))

    def lead_calls(self) -> int:
        return int(round(self.lead_days * 24 / 12))


# ---------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------


@dataclass
class HuracanConfig:
    run_name: str = "run"
    output_dir: str = "runs/run"
    seed: int = 0
    device: str = "cuda"
    grid: GridCfg = field(default_factory=GridCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    # -- derived channel counts ------------------------------------------
    def var_names(self) -> list[str]:
        """Ordered prognostic channel names (paper: 72)."""
        v = self.data.variables
        names = [f"{u}{lev}" for lev in v.levels for u in v.upper]
        names += list(v.surface)
        return names

    def n_prognostic(self) -> int:
        return len(self.var_names())

    def n_forcing(self) -> int:
        return len(self.data.forcing)

    def n_obs_channels(self) -> int:
        if not self.data.obs.enabled:
            return 0
        v = self.data.variables
        return len(v.upper) * len(v.levels) + len(self.data.obs.surface_vars)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "HuracanConfig":
        return _build(cls, d, "config")


# ---------------------------------------------------------------------------
# dict -> dataclass builder (strict: unknown keys are rejected)
# ---------------------------------------------------------------------------


def _check_keys(d: dict, cls: type, path: str) -> None:
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - valid
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in '{path}'; "
            f"valid keys: {sorted(valid)}"
        )


def _missing(d: dict, cls: type, path: str) -> None:
    for f in dataclasses.fields(cls):
        if f.name not in d and (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ):
            raise ConfigError(f"missing required key '{path}.{f.name}'")


def _convert(value, typ, path: str):
    origin = getattr(typ, "__origin__", None)

    if origin is None and dataclasses.is_dataclass(typ):
        if not isinstance(value, dict):
            raise ConfigError(f"'{path}' must be a mapping for {typ.__name__}")
        _check_keys(value, typ, path)
        _missing(value, typ, path)
        hints = get_type_hints(typ)
        kwargs = {}
        for f in dataclasses.fields(typ):
            if f.name in value:
                kwargs[f.name] = _convert(
                    value[f.name], hints[f.name], f"{path}.{f.name}"
                )
        return typ(**kwargs)

    if origin is dict:  # e.g. dict[str, PhaseCfg]
        (k_type, v_type) = typ.__args__
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ConfigError(f"'{path}' keys must be strings")
            out[k] = _convert(v, v_type, f"{path}.{k}")
        return out

    if origin in (list, tuple):
        (e_type,) = typ.__args__
        return [_convert(v, e_type, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is type(None) or typ is type(None):  # pragma: no cover
        return value

    # Optional[X] -> typing.Union[X, None]
    if hasattr(typ, "__args__") and type(None) in typ.__args__:
        if value is None:
            return None
        inner = (
            typ.__args__[0]
            if isinstance(typ.__args__[1], type(None))
            else typ.__args__[1]
        )
        return _convert(value, inner, path)

    return _scalar(value, typ, path)


def _scalar(value, typ, path: str):
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{path}' must be a number, got {value!r}")
        return float(value)
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{path}' must be an integer, got {value!r}")
        return value
    if typ is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{path}' must be a string, got {value!r}")
        return value
    if typ is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{path}' must be a boolean, got {value!r}")
        return value
    raise ConfigError(f"'{path}': unsupported config type {typ}")  # pragma: no cover


def _build(cls: type, d: dict, path: str):
    if not isinstance(d, dict):
        raise ConfigError(f"'{path}' must be a mapping for {cls.__name__}")
    _check_keys(d, cls, path)
    _missing(d, cls, path)
    hints = get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in d:
            kwargs[f.name] = _convert(d[f.name], hints[f.name], f"{path}.{f.name}")
    return cls(**kwargs)
