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

    **Empirical** CRPS estimator (P0 main metric):

        CRPS = E|X - y| - 1/(2M^2) sum_m sum_m' |x_m - x_m'|

    (the pair sum includes the diagonal, i.e. the fair-CRPS denominator
    M(M-1) is NOT used here -- that unbiased variant lives in
    :func:`crps_fair`).  Hand check: members [0, 2], y = 1 ->
    E|X-y| = 1, pair term = (0+2+2+0)/8 = 0.5 -> empirical CRPS = 0.5.

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
    m2 = (acc + 0.0) / (M * M)  # diagonal contributes 0; 1/M^2 normalization
    return _spatial_mean(m1 - 0.5 * m2, w)  # (D, C)


def crps_fair(
    preds: np.ndarray, truth: np.ndarray, w: np.ndarray | None = None
) -> np.ndarray:
    """Fair (unbiased-spread) CRPS: like :func:`crps_ensemble` but the pair
    term averages over ordered pairs m != m' (denominator M(M-1)).

    Hand check: members [0, 2], y = 1 -> pair term = 1 -> fair CRPS = 0.
    This is the estimator the *training* objective uses (losses.crps_loss);
    report it separately -- never under the empirical name."""
    m1 = np.abs(preds - truth[None]).mean(axis=0)
    M = preds.shape[0]
    if M < 2:
        return _spatial_mean(m1, w)
    acc = np.zeros_like(m1)
    for i in range(M):
        for j in range(M):
            if i != j:
                acc += np.abs(preds[i] - preds[j])
    m2 = acc / (M * (M - 1))
    return _spatial_mean(m1 - 0.5 * m2, w)


def essr(
    preds: np.ndarray,
    truth: np.ndarray,
    w: np.ndarray | None = None,
    mode: str = "rms",
) -> np.ndarray:
    """Ensemble spread-skill ratio -> (D, C).

    mode="rms" (P0 primary diagnostic): sqrt(mean(spread^2)) / RMSE(ens mean)
    -- the RMS-spread form used throughout the DA literature.

    mode="mean_std" (legacy, kept for continuity): mean(local std) /
    RMSE(ens mean).  Both are finite-ensemble quantities; a value of 1 does
    not by itself certify calibration."""
    spread_p = preds.std(axis=0, ddof=1)  # (D, npix, C)
    emean = preds.mean(axis=0)
    skill = np.sqrt(_spatial_mean((emean - truth) ** 2, w))  # (D, C)
    if mode == "rms":
        spread = np.sqrt(_spatial_mean(spread_p**2, w))
    elif mode == "mean_std":
        spread = _spatial_mean(spread_p, w)
    else:
        raise ValueError(f"unknown essr mode {mode!r} ('rms' | 'mean_std')")
    return spread / np.maximum(skill, 1e-8)


def rmse_ensemble_mean(
    preds: np.ndarray, truth: np.ndarray, w: np.ndarray | None = None
) -> np.ndarray:
    emean = preds.mean(axis=0)
    return np.sqrt(_spatial_mean((emean - truth) ** 2, w))  # (D, C)
