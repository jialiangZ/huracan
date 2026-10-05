"""Lightning training layer.

This package (plus ``training.datamodule``) is the *only* place in the
codebase that imports lightning; the core/data packages stay
framework-free.
"""

from .module import HuracanLitModule
from .phases import ALL_PHASES, PHASE_ORDER, fit_phase

__all__ = ["HuracanLitModule", "fit_phase", "PHASE_ORDER", "ALL_PHASES"]
