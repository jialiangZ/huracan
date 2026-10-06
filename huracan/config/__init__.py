"""Typed configuration package (schema + YAML loader + provenance)."""

from .loader import load_config, var_names
from .provenance import PAPER, INFERRED, PROVENANCE, REPO, check, report
from .report import channel_report
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
    "channel_report",
    "provenance_report",
    "PROVENANCE",
    "provenance_check",
    "PAPER",
    "INFERRED",
    "REPO",
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

provenance_report = report
provenance_check = check
