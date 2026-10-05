"""Training losses: L1 (deterministic phases) and the CRPS ensemble objective
(stochastic phases), following GenCast/AIFS-CRPS practice cited by the paper.

HEALPix pixels are equal-area, so plain means are area-weighted means.
"""

from __future__ import annotations

import torch


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def crps_loss(ensemble: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ensemble: (B, M, ...) members; target: (B, ...) .

    CRPS_est = E_m |x_m - y|  -  0.5 * E_{m != m'} |x_m - x_m'|
    (fair estimator; for the paper's ensemble size M=2 this reduces to
    mean|.| - 0.5 |x_1 - x_2|.)
    """
    m = ensemble.mean(dim=1)  # not used; kept for clarity
    term1 = (ensemble - target.unsqueeze(1)).abs().mean()
    M = ensemble.shape[1]
    if M < 2:
        return term1
    diff = (ensemble.unsqueeze(2) - ensemble.unsqueeze(1)).abs()  # (B,M,M,...)
    offdiag = diff.sum(dim=(1, 2)) / (M * (M - 1))
    return term1 - 0.5 * offdiag.mean()
