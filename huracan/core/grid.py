"""Regular spherical grid: Gauss-Legendre latitudes x equispaced longitudes.

Provenance (docs/HURACAN_REPRODUCTION_GUIDE_REVISED.md, Sec. 7): the paper
specifies 1-degree resolution and L=179 (Appendix A.2) but never names a
grid family -- HEALPix appears nowhere in the text, and the exact node
layout stays [OPEN].  This module implements the high-confidence inference:
a regular lat-lon grid compatible with standard SHT libraries.

Why Gauss-Legendre latitudes rather than equiangular:
  * nlat = lmax+1 Gauss nodes make the SHT quadrature *exact* for degree
    lmax (an equiangular grid would need nlat ~= 2*lmax for the same);
  * nlon >= 2*lmax+1 makes the longitude sum exact;
  * it matches the native layout of torch-harmonics, the planned backend at
    paper scale (180x360/lmax=179 is exactly of this form).

Pixel ordering is row-major over (lat, lon): north -> south, lon in [0, 2pi).
A flat (npix,) field therefore reshapes losslessly to (nlat, nlon) images,
which is what the lat-lon Swin windows consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.schema import GridCfg


@dataclass(frozen=True)
class SphericalGrid:
    nlat: int
    nlon: int
    lmax: int
    lat: np.ndarray  # (nlat,) radians, descending (north -> south)
    lon: np.ndarray  # (nlon,) radians in [0, 2*pi)
    w_lat: np.ndarray  # (nlat,) Gauss weights over sin(lat), sum == 2

    # -- geometry -----------------------------------------------------------
    @property
    def npix(self) -> int:
        return self.nlat * self.nlon

    @property
    def colat(self) -> np.ndarray:
        """(npix,) colatitudes, row-major over (lat, lon)."""
        return np.repeat(np.pi / 2 - self.lat, self.nlon)

    @property
    def lon_flat(self) -> np.ndarray:
        """(npix,) longitudes, row-major over (lat, lon)."""
        return np.tile(self.lon, self.nlat)

    @property
    def pixel_weights(self) -> np.ndarray:
        """(npix,) area weights normalized to mean 1.

        Unlike HEALPix, a lat-lon grid is not equal-area: pixels shrink
        cos(lat) towards the poles, so spatial averages must be weighted
        (used by core.metrics)."""
        w = np.repeat(self.w_lat, self.nlon) * (2 * np.pi / self.nlon)
        return w / w.mean()

    # -- validation -----------------------------------------------------------
    def validate(self) -> "SphericalGrid":
        if self.nlat < self.lmax + 1:
            raise ValueError(
                f"exact Gauss SHT needs nlat >= lmax+1 (got nlat={self.nlat}, "
                f"lmax={self.lmax})"
            )
        if self.nlon < 2 * self.lmax + 1:
            raise ValueError(
                f"exact longitude sum needs nlon >= 2*lmax+1 (got nlon={self.nlon}, "
                f"lmax={self.lmax})"
            )
        return self


def gauss_grid(nlat: int, nlon: int, lmax: int | None = None) -> SphericalGrid:
    """Gauss-Legendre x equispaced grid with exact SHT quadrature.

    lmax defaults to the largest exactly-representable degree:
    min(nlat - 1, (nlon - 1) // 2)."""
    if lmax is None:
        lmax = min(nlat - 1, (nlon - 1) // 2)
    x, w = np.polynomial.legendre.leggauss(nlat)  # x = sin(lat), ascending
    lat = np.arcsin(x[::-1])  # north -> south
    w_lat = w[::-1]
    lon = 2 * np.pi * np.arange(nlon) / nlon
    return SphericalGrid(int(nlat), int(nlon), int(lmax), lat, lon, w_lat).validate()


def grid_from_cfg(gcfg: GridCfg) -> SphericalGrid:
    return gauss_grid(gcfg.nlat, gcfg.nlon, gcfg.lmax)


# ---------------------------------------------------------------------------
# analytic forcings / geometry helpers (grid-agnostic, operate on colat/lon)
# ---------------------------------------------------------------------------


def solar_declination(day_of_year: np.ndarray) -> np.ndarray:
    return np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365.0)


def insolation(
    colat: np.ndarray,
    lon: np.ndarray,
    utc_hour: float,
    day_of_year: float,
    solar_constant: float = 1361.0,
) -> np.ndarray:
    """Top-of-atmosphere insolation [W/m^2] at every pixel (analytic forcing)."""
    lat = np.pi / 2 - colat
    decl = solar_declination(np.asarray(day_of_year))
    hour_angle = lon + np.deg2rad(15.0) * (utc_hour - 12.0)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(
        hour_angle
    )
    return solar_constant * np.clip(cos_zenith, 0.0, None)


def unit_vectors(colat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.sin(colat) * np.cos(lon), np.sin(colat) * np.sin(lon), np.cos(colat)],
        axis=-1,
    )


def rotation_about_axis(v: np.ndarray, axis: np.ndarray, angle: np.ndarray):
    """Rodrigues rotation of vectors v (..., 3) about unit axis by angle (scalar
    or broadcastable) -- used for solid-body advection of the synthetic data."""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    dot = np.einsum("...i,i->...", v, axis)[..., None] * axis
    cross = np.cross(np.broadcast_to(axis, v.shape), v)
    return v * c + cross * s + dot * (1 - c)
