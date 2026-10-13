"""Planar (2-D periodic) FFT backend for the Huracan SFNO trunk.

Satisfies the same ``SHTBackend`` protocol as ``MatrixSHT`` so the
unmodified ``HuracanSFNO`` (Swin mixer, compressed spectral filter,
noise embedding, two-timestep ``step``) runs on the periodic square --
this is the FNO trunk for the Kolmogorov-flow OSSE line.

Coefficient layout: ``forward`` maps (B, npix, C) real to
(B, L, M, C) complex with L = nlat (ky rows, unshifted FFT order),
M = nlon//2 + 1 (kx >= 0).  Both transforms use ``norm='ortho'``, so
coefficient magnitudes are comparable to the orthonormal spherical
basis (Parseval holds) and the filter/skip balance trained on the
sphere transfers.  The compressed spectral filter is applied to the
*full* mode set -- its parameters (k x C x C mixing + per-degree
weights) do not scale with the mode count, so no mode truncation is
needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .grid import SphericalGrid


class PlanarSHT(nn.Module):
    """rfft2/irfft2 pair behind the SHTBackend protocol."""

    def __init__(self, grid: SphericalGrid):
        super().__init__()
        if grid.npix != grid.nlat * grid.nlon:
            raise ValueError("planar backend expects a row-major square")
        self.grid = grid
        self.nlat, self.nlon = grid.nlat, grid.nlon
        self.L, self.M = grid.nlat, grid.nlon // 2 + 1
        self.npix = grid.npix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, npix, C) real -> c (B, L, M, C) complex."""
        B, N, C = x.shape
        if N != self.npix:
            raise ValueError(f"expected npix={self.npix}, got {N}")
        x4 = x.view(B, self.nlat, self.nlon, C)
        return torch.fft.rfft2(x4, dim=(1, 2), norm="ortho")

    def inverse(self, c: torch.Tensor) -> torch.Tensor:
        """c (B, L, M, C) complex -> x (B, npix, C) real."""
        B = c.shape[0]
        x4 = torch.fft.irfft2(c, s=(self.nlat, self.nlon), dim=(1, 2), norm="ortho")
        return x4.reshape(B, self.npix, c.shape[-1])
