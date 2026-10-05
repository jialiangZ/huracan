"""HEALPix grid utilities (NESTED ordering) and analytic forcings.

We use NESTED ordering so pixels reshape cleanly into 12 faces of
nside x nside images -- required by the face-wise Swin Transformer.
"""

from __future__ import annotations

import math

import numpy as np


def nside_is_valid(nside: int) -> bool:
    return nside >= 1 and (nside & (nside - 1)) == 0


def npix(nside: int) -> int:
    return 12 * nside * nside


def pixel_centers(nside: int):
    """NESTED-order pixel centers as (colat, lon) in radians, shape (npix,).

    Uses astropy-healpix. Results are cached to <cache_dir>/grid_n{n}.npz.
    """
    from astropy_healpix import HEALPix

    hp = HEALPix(nside=nside, order="nested")
    lon, lat = hp.healpix_to_lonlat(np.arange(npix(nside)))
    return (np.pi / 2 - np.asarray(lat.rad)), np.asarray(lon.rad)


def nest_deinterleave(nside: int):
    """Return index arrays (face, y, x) for each NESTED pixel.

    nested pixel p = face*nside^2 + interleave_bits(x, y)  (x fastest).
    """
    assert nside_is_valid(nside), "nside must be a power of 2"
    within = np.arange(nside * nside)
    x = np.zeros(within.size, dtype=np.int64)
    y = np.zeros(within.size, dtype=np.int64)
    for b in range(int(math.log2(nside))):
        x |= ((within >> (2 * b)) & 1) << b
        y |= ((within >> (2 * b + 1)) & 1) << b
    face = np.arange(12, dtype=np.int64).repeat(nside * nside)
    xx = np.tile(x, 12)
    yy = np.tile(y, 12)
    return face, yy, xx


def nest_to_face_perm(nside: int) -> np.ndarray:
    """Permutation such that  x_faces = x_nested[..., perm, :]  reshapes to
    (..., 12, nside, nside, C) with (face, y, x) row-major layout."""
    face, y, x = nest_deinterleave(nside)
    return face * nside * nside + y * nside + x


def face_to_nest_perm(nside: int) -> np.ndarray:
    return np.argsort(nest_to_face_perm(nside))


def pixel_area_weights(nside: int) -> np.ndarray:
    """HEALPix pixels are equal area -> uniform weights (ones)."""
    return np.ones(npix(nside), dtype=np.float64)


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
