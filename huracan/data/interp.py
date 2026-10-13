"""Shared bilinear interpolation stencil: regular lat-lon -> HEALPix."""

from __future__ import annotations

import numpy as np


class LatLonBilinear:
    """Precomputed bilinear stencil: regular lat-lon field -> HEALPix pixels."""

    def __init__(self, lats: np.ndarray, lons: np.ndarray, colat, lon):
        self.flip = False
        if lats[1] < lats[0]:
            lats = lats[::-1]
            self.flip = True
        lat_t = 90.0 - np.rad2deg(colat)  # target latitudes
        lon_t = np.mod(np.rad2deg(lon), 360.0)  # target longitudes [0,360)
        self.dlat = lats[1] - lats[0]
        self.dlon = lons[1] - lons[0]
        fy = (lat_t - lats[0]) / self.dlat
        fx = (lon_t - lons[0]) / self.dlon
        self.y0 = np.clip(np.floor(fy).astype(int), 0, len(lats) - 2)
        self.x0 = np.floor(fx).astype(int) % len(lons)
        self.x1 = (self.x0 + 1) % len(lons)
        self.wy = fy - self.y0
        self.wx = fx - np.floor(fx)
        self.shape = (len(lats), len(lons))

    def __call__(self, field: np.ndarray) -> np.ndarray:
        assert field.shape == self.shape, (field.shape, self.shape)
        if self.flip:
            field = field[::-1]
        f00 = field[self.y0, self.x0]
        f01 = field[self.y0, self.x1]
        f10 = field[self.y0 + 1, self.x0]
        f11 = field[self.y0 + 1, self.x1]
        wy, wx = self.wy, self.wx
        return (
            f00 * (1 - wy) * (1 - wx)
            + f01 * (1 - wy) * wx
            + f10 * wy * (1 - wx)
            + f11 * wy * wx
        ).astype(np.float32)
