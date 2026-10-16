"""Training losses: L1 (deterministic phases) and the CRPS ensemble objective
(stochastic phases), following GenCast/AIFS-CRPS practice cited by the paper.

P0 common correction: spatial averages accept per-pixel weights ``w``
(npix,).  The Gauss lat-lon grid is NOT equal-area, so an unweighted mean
is pole-biased; training and evaluation now share the same weighting
convention (``grid.pixel_weights``; uniform for the planar backbone).
This is a *baseline correction* applied to all methods alike -- any gain
from it belongs to the fix, not to a method contribution.
"""

from __future__ import annotations

import torch


def l1_loss(
    pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor | None = None
) -> torch.Tensor:
    """pred (B, ...) / target (B, ...): weighted mean absolute error (the
    mean over non-pixel axes is plain)."""
    d = (pred - target).abs()
    if w is None:
        return d.mean()
    # weighted mean over the pixel axis (dim=2 for (B, D, npix, C)),
    # plain mean over the remaining axes
    d = (d * w[..., None]).sum(dim=2) / w.sum()
    return d.mean()


def crps_loss(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    w: torch.Tensor | None = None,
) -> torch.Tensor:
    """ensemble: (B, M, ...) members; target: (B, ...) .

    CRPS_est = E_m |x_m - y|  -  0.5 * E_{m != m'} |x_m - x_m'|
    (fair estimator; for the paper's ensemble size M=2 this reduces to
    mean|.| - 0.5 |x_1 - x_2|.)

    ``w`` (npix,) weights the pixel axis; with ``ensemble`` shaped
    (B, M, D, npix, C) the pixel axis is 3.
    """
    term1 = (ensemble - target.unsqueeze(1)).abs().mean(dim=1)  # (B, D, npix, C)
    M = ensemble.shape[1]
    diff = (ensemble.unsqueeze(2) - ensemble.unsqueeze(1)).abs()  # (B,M,M,...)
    if M < 2:
        base = term1
    else:
        offdiag = diff.sum(dim=(1, 2)) / (M * (M - 1))
        base = term1 - 0.5 * offdiag
    if w is None:
        return base.mean()
    # weighted mean over the pixel axis of ``base`` -- which is (B, D, npix,
    # C) after the member reduction, i.e. dim=-2.  (Do NOT index by
    # ``ensemble.dim()-2``: that would hit the channel axis of ``base``
    # and silently change the loss scale by C/npix.)
    base = (base * w[..., None]).sum(dim=-2) / w.sum()
    return base.mean()
