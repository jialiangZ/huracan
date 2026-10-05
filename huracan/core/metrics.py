"""Verification metrics: empirical CRPS, spread-skill ratio (ESSR),
RMSE of the ensemble mean -- computed per lead time and variable channel.

All averages are over HEALPix pixels (equal-area, so a plain mean is the
area-weighted mean).
"""

from __future__ import annotations

import numpy as np


def crps_ensemble(preds: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """preds (M, D, npix, C), truth (D, npix, C) -> CRPS (D, C).

    Empirical CRPS = E|X - y| - 0.5 E|X - X'| (over all member pairs).
    Implemented with a member-pair loop to avoid an M^2-sized temporary."""
    m1 = np.abs(preds - truth[None]).mean(axis=0)  # (D, npix, C)
    M = preds.shape[0]
    if M < 2:
        return m1.mean(axis=1)
    acc = np.zeros_like(m1)
    for i in range(M):
        for j in range(M):
            if i != j:
                acc += np.abs(preds[i] - preds[j])
    m2 = acc / (M * (M - 1))  # E_{m != m'} |x-x'|
    return (m1 - 0.5 * m2).mean(axis=1)  # (D, C)


def essr(preds: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Ensemble spread-skill ratio -> (D, C).

    spread = mean pixel std over members; skill = RMSE of ensemble mean."""
    spread = preds.std(axis=0, ddof=1).mean(axis=1)  # (D, npix, C)->(D,C)
    emean = preds.mean(axis=0)
    skill = np.sqrt(((emean - truth) ** 2).mean(axis=1))  # (D, C)
    return spread / np.maximum(skill, 1e-8)


def rmse_ensemble_mean(preds: np.ndarray, truth: np.ndarray) -> np.ndarray:
    emean = preds.mean(axis=0)
    return np.sqrt(((emean - truth) ** 2).mean(axis=1))  # (D, C)
