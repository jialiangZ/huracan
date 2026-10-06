"""Verification metrics: empirical CRPS, spread-skill ratio (ESSR),
RMSE of the ensemble mean -- computed per lead time and variable channel.

Spatial averages accept optional per-pixel ``w`` (npix,) weights.  The
Gauss lat-lon grid is *not* equal-area, so pass ``grid.pixel_weights``;
with ``w=None`` a plain mean is used (HEALPix-style).  Which weighting the
paper used is [OPEN] (guide E03), but on a lat-lon grid an unweighted mean
is provably pole-biased, so the weighted form is the sane default.
"""

from __future__ import annotations

import numpy as np


def _spatial_mean(v: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    """v (D, npix, C) -> (D, C), optionally weighted over npix."""
    if w is None:
        return v.mean(axis=1)
    return (v * w[None, :, None]).sum(axis=1) / w.sum()


def crps_ensemble(
    preds: np.ndarray, truth: np.ndarray, w: np.ndarray | None = None
) -> np.ndarray:
    """preds (M, D, npix, C), truth (D, npix, C) -> CRPS (D, C).

    Empirical CRPS = E|X - y| - 0.5 E|X - X'| (over all member pairs).
    Implemented with a member-pair loop to avoid an M^2-sized temporary."""
    m1 = np.abs(preds - truth[None]).mean(axis=0)  # (D, npix, C)
    M = preds.shape[0]
    if M < 2:
        return _spatial_mean(m1, w)
    acc = np.zeros_like(m1)
    for i in range(M):
        for j in range(M):
            if i != j:
                acc += np.abs(preds[i] - preds[j])
    m2 = acc / (M * (M - 1))  # E_{m != m'} |x-x'|
    return _spatial_mean(m1 - 0.5 * m2, w)  # (D, C)


def essr(
    preds: np.ndarray, truth: np.ndarray, w: np.ndarray | None = None
) -> np.ndarray:
    """Ensemble spread-skill ratio -> (D, C).

    spread = weighted pixel std over members; skill = RMSE of ensemble mean.
    The paper gives no exact formula (guide E04/E05); this is the plain
    finite-ensemble estimator."""
    spread = _spatial_mean(preds.std(axis=0, ddof=1), w)  # (D, C)
    emean = preds.mean(axis=0)
    skill = np.sqrt(_spatial_mean((emean - truth) ** 2, w))  # (D, C)
    return spread / np.maximum(skill, 1e-8)


def rmse_ensemble_mean(
    preds: np.ndarray, truth: np.ndarray, w: np.ndarray | None = None
) -> np.ndarray:
    emean = preds.mean(axis=0)
    return np.sqrt(_spatial_mean((emean - truth) ** 2, w))  # (D, C)
