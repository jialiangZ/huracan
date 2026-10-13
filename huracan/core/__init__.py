"""Framework-free core: grid, SHT, neural modules, model, rollout, losses,
metrics.  Nothing in this package may import lightning or YAML -- it is the
reusable domain layer under the training/inference applications.
"""

from .grid import SphericalGrid, gauss_grid, grid_from_cfg
from .model import HuracanSFNO, build_models, load_with_new_columns
from .rollout import spinup_and_forecast, unroll
from .sht import MatrixSHT, SHTBackend
from .swin import SwinLatLonBlock

__all__ = [
    "SphericalGrid",
    "gauss_grid",
    "grid_from_cfg",
    "HuracanSFNO",
    "build_models",
    "load_with_new_columns",
    "unroll",
    "spinup_and_forecast",
    "MatrixSHT",
    "SHTBackend",
    "SwinLatLonBlock",
]
