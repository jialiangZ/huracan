"""Scorecard plotting in the style of the paper (Figures 3-4): relative CRPS
skill score of system A vs baseline B per variable and lead day.

skill = (crps_B - crps_A) / crps_B * 100   (blue = A better)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def scorecard(
    cand: np.ndarray,
    base: np.ndarray,
    var_names: list[str],
    steps_per_day: int = 4,
    title: str = "Huracan vs baseline",
    out_path: str = "scorecard.png",
    vmax: float = 20.0,
):
    """cand, base: (D, C) CRPS arrays (candidate / baseline)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ss = (base - cand) / base * 100.0
    D, C = ss.shape
    days = D // steps_per_day
    ss_d = ss[: days * steps_per_day].reshape(days, steps_per_day, C).mean(axis=1)

    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * len(var_names), 1.0 + 0.4 * days))
    im = ax.imshow(ss_d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=90, fontsize=7)
    ax.set_yticks(range(days))
    ax.set_yticklabels([str(d + 1) for d in range(days)])
    ax.set_ylabel("lead day")
    for i in range(days):
        for j in range(len(var_names)):
            v = ss_d[i, j]
            if not np.isfinite(v):
                continue
            ax.text(
                j, i, f"{v:.0f}", ha="center", va="center", fontsize=5, color="black"
            )
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="% CRPS skill score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return ss_d
