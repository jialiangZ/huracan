"""Spherical harmonic transform (SHT) on the regular Gauss lat-lon grid.

Design
------
We use the *real* harmonic basis
    g_{l,0}   = N_lm P_l^m(cos t)                  (m = 0)
    g^c_{lm}  = sqrt(2) N_lm P_l^m(cos t) cos(m p) (m > 0)
    g^s_{lm}  = sqrt(2) N_lm P_l^m(cos t) sin(m p)
with columns orthonormal under int g g' dOmega = delta.  On the exact
Gauss-Legendre grid (nlat >= lmax+1, nlon >= 2*lmax+1, see core.grid) the
forward transform is the *quadrature* sum
    coeff_c = sum_i  w_i * B[i, c] * f[i]
which is exact -- no least-squares pseudo-inverse is involved.

Coefficients are stored in a complex (L, M) = (lmax+1, lmax+1) layout as
    c[l, 0] = alpha_{l0}
    c[l, m] = (alpha^c_{lm} - i alpha^s_{lm}) / sqrt(2)   (m > 0)
so that the inverse transform is  x = Re( sum_m w_m A_m c_m )  with
w_0 = 1, w_{m>0} = 2, and real-linear channel mixing of c preserves the
structure (exactly what the compressed spectral filter applies).

The dense-matrix backend is exact and fast for smoke/pilot scales
(npix <= ~5k).  At paper scale (180x360, lmax=179) the basis matrix alone
would be ~17 GB, so a torch-harmonics adapter must be injected through the
same ``SHTBackend`` protocol (see HuracanSFNO(sht_factory=...)).
"""

from __future__ import annotations

import math as _math
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from .grid import SphericalGrid


@runtime_checkable
class SHTBackend(Protocol):
    """Minimal interface every spherical-harmonic backend must provide.

    ``MatrixSHT`` below satisfies it; torch-harmonics can be dropped in
    behind the same two methods for paper-scale grids.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def inverse(self, c: torch.Tensor) -> torch.Tensor: ...


def _assoc_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """P_l^m(x) without Condon-Shortley phase, stable recurrence."""
    if m > l:
        return np.zeros_like(x)
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt(np.maximum(1.0 - x * x, 0.0))
        fact = 1.0
        for _ in range(m):
            pmm = -pmm * fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    pll = None
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pmmp1


def build_real_basis(lmax: int, colat: np.ndarray, lon: np.ndarray):
    """Real orthonormal basis matrix B (npix, n_real) with column layout per
    degree l: [m=0] then for m=1..l: [cos, sin].  Also returns index tables."""
    cols, lidx, midx, kind = [], [], [], []  # kind: 0=m0, 1=cos, 2=sin
    ct = np.cos(colat)
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            nlm = np.sqrt(
                (2 * l + 1)
                / (4 * np.pi)
                * float(_math.factorial(l - m))
                / float(_math.factorial(l + m))
            )
            p = _assoc_legendre(l, m, ct) * nlm
            if m == 0:
                cols.append(p.copy())
                lidx.append(l)
                midx.append(0)
                kind.append(0)
            else:
                s2 = np.sqrt(2.0)
                cols.append(s2 * p * np.cos(m * lon))
                lidx.append(l)
                midx.append(m)
                kind.append(1)
                cols.append(s2 * p * np.sin(m * lon))
                lidx.append(l)
                midx.append(m)
                kind.append(2)
    B = np.stack(cols, axis=1).astype(np.float64)
    return B, (np.array(lidx), np.array(midx), np.array(kind))


class MatrixSHT(nn.Module):
    """Exact real-basis SHT on the Gauss grid, complex (L, M) layout.

    forward:  x (B, npix, C) real  -> c (B, L, M, C) complex
    inverse:  c (B, L, M, C)       -> x (B, npix, C) real
    """

    def __init__(self, grid: SphericalGrid, cache_dir: str | Path = "data/cache"):
        super().__init__()
        grid.validate()
        self.grid = grid
        self.lmax = grid.lmax
        self.L, self.M = grid.lmax + 1, grid.lmax + 1

        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / f"sht_gauss_{grid.nlat}x{grid.nlon}_l{grid.lmax}.npz"
        if f.exists():
            z = np.load(f)
            B, Fwd, lidx, midx, kind = (
                z["B"],
                z["Fwd"],
                z["lidx"],
                z["midx"],
                z["kind"],
            )
        else:
            colat, lon = grid.colat, grid.lon_flat
            B, (lidx, midx, kind) = build_real_basis(grid.lmax, colat, lon)
            # exact quadrature: total weight per pixel = w_lat * dlon
            w = np.repeat(grid.w_lat, grid.nlon) * (2 * np.pi / grid.nlon)
            Fwd = (w[:, None] * B).T  # (n_real, npix)
            np.savez_compressed(f, B=B, Fwd=Fwd, lidx=lidx, midx=midx, kind=kind)

        # --- scatter/gather between real coefficient vector and complex map
        m0 = kind == 0
        cs = kind == 1
        sn = kind == 2
        lm_lin = lidx * self.M + midx  # slot in (L, M)
        self.n_real = B.shape[1]

        self.register_buffer("B", torch.from_numpy(B.astype(np.float32)))
        self.register_buffer("Fwd", torch.from_numpy(Fwd.astype(np.float32)))
        self.register_buffer("lin_m0", torch.from_numpy(lm_lin[m0]))
        self.register_buffer("real_m0", torch.from_numpy(np.where(m0)[0]))
        self.register_buffer("lin_cs", torch.from_numpy(lm_lin[cs]))
        self.register_buffer("real_cs", torch.from_numpy(np.where(cs)[0]))
        self.register_buffer("lin_sn", torch.from_numpy(lm_lin[sn]))
        self.register_buffer("real_sn", torch.from_numpy(np.where(sn)[0]))
        self.register_buffer("sqrt2", torch.tensor(2**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, npix, C) real -> c (B, L, M, C) complex."""
        B, N, C = x.shape
        u = torch.matmul(self.Fwd, x)  # (B, n_real, C)
        creal = x.new_zeros((B, self.L * self.M, C))
        cimag = x.new_zeros((B, self.L * self.M, C))
        creal.index_copy_(1, self.lin_m0, u.index_select(1, self.real_m0))
        cimag.index_copy_(1, self.lin_sn, -u.index_select(1, self.real_sn) / self.sqrt2)
        creal.index_copy_(1, self.lin_cs, u.index_select(1, self.real_cs) / self.sqrt2)
        c = torch.complex(creal, cimag).view(B, self.L, self.M, C)
        return c

    def inverse(self, c: torch.Tensor) -> torch.Tensor:
        """c (B, L, M, C) complex -> x (B, npix, C) real."""
        B = c.shape[0]
        C = c.shape[-1]
        cf = c.reshape(B, self.L * self.M, C)
        u = torch.zeros((B, self.n_real, C), device=c.device, dtype=torch.float32)
        u.index_copy_(1, self.real_m0, cf.index_select(1, self.lin_m0).real)
        u.index_copy_(
            1, self.real_cs, cf.index_select(1, self.lin_cs).real * self.sqrt2
        )
        u.index_copy_(
            1, self.real_sn, -cf.index_select(1, self.lin_sn).imag * self.sqrt2
        )
        return torch.matmul(self.B, u)
