"""Typed configuration package (schema + YAML loader)."""

from .loader import load_config, var_names
from .schema import (
    ConfigError,
    DataCfg,
    EvalCfg,
    GridCfg,
    HuracanConfig,
    ModelCfg,
    ObsCfg,
    PhaseCfg,
    SyntheticCfg,
    TrainCfg,
    VariablesCfg,
)

__all__ = [
    "load_config",
    "var_names",
    "ConfigError",
    "HuracanConfig",
    "GridCfg",
    "ModelCfg",
    "DataCfg",
    "VariablesCfg",
    "ObsCfg",
    "SyntheticCfg",
    "PhaseCfg",
    "TrainCfg",
    "EvalCfg",
]
