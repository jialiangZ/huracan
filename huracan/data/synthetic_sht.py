"""Numpy helpers over the real harmonic basis (used by data-prep workers,
no torch dependency)."""

from __future__ import annotations

import numpy as np


class RealBasisOps:
    """Packed operations on the real spherical-harmonic basis B (npix, n_real).

    Column layout per degree l: [m=0], then for m=1..l: [cos, sin].
    """

    def __init__(self, B: np.ndarray, lidx, midx, kind):
        self.B = B
        self.B_pinv = np.linalg.pinv(B)
        self.n_real = B.shape[1]
        lidx = np.asarray(lidx)
        midx = np.asarray(midx)
        kind = np.asarray(kind)
        self.lcol = lidx.astype(np.float64)
        # cos/sin column pairs for m > 0
        cmap = {}
        for i in np.where(kind == 1)[0]:
            cmap[(int(lidx[i]), int(midx[i]))] = int(i)
        self.pair_l = []
        self.pair_c = []
        self.pair_s = []
        for i in np.where(kind == 2)[0]:
            key = (int(lidx[i]), int(midx[i]))
            if key in cmap:
                self.pair_l.append(lidx[i])
                self.pair_c.append(cmap[key])
                self.pair_s.append(int(i))
        self.pair_l = np.array(self.pair_l, dtype=np.float64)
        self.pair_c = np.array(self.pair_c, dtype=np.int64)
        self.pair_s = np.array(self.pair_s, dtype=np.int64)

    def random_field(self, rng, C, power=-1.5):
        u = rng.standard_normal((self.n_real, C))
        u *= (self.lcol[:, None] + 1.0) ** (power / 2)
        return (self.B @ u).astype(np.float32)

    def to_coeffs(self, x):
        return self.B_pinv @ x

    def to_field(self, u):
        return self.B @ u

    def wave_step(self, x, speeds, dt):
        """Rotate cos/sin harmonic pairs by theta_l = c_k * sqrt(l(l+1)) * dt
        (dispersion-like propagation per channel)."""
        u = self.to_coeffs(x)
        th = (
            self.pair_l[:, None] ** 0.5
            * (self.pair_l[:, None] + 1) ** 0.5
            * np.abs(speeds)[None, :]
            * dt
        )  # (npair, C)
        cs, sn = np.cos(th), np.sin(th)
        uc = u[self.pair_c] * cs - u[self.pair_s] * sn
        us = u[self.pair_c] * sn + u[self.pair_s] * cs
        u = u.copy()
        u[self.pair_c] = uc
        u[self.pair_s] = us
        return self.to_field(u)
