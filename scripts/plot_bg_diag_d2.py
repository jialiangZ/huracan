#!/usr/bin/env python3
"""D2 background-error low-dimensionality bar chart (bg_diag D2, no plot
existed -- json only).  Reads runs/bsef/bg_diag/bg_diag_report.json.

Usage:
  python scripts/plot_bg_diag_d2.py [--report runs/bsef/bg_diag/bg_diag_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/bsef/bg_diag/bg_diag_report.json")
    args = ap.parse_args()

    d2 = json.load(open(args.report))["D2"]
    vals = {
        "mode\n(rank-4 subspace)": d2["mode_explained_mean"],
        "random\nbaseline": d2["random_explained_mean"],
        "diag\n(coordinates)": d2["diag_explained_mean"],
    }
    med = d2.get("mode_explained_median")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(
        vals.keys(), vals.values(),
        color=["tab:blue", "tab:gray", "tab:orange"], width=0.55,
    )
    for b, v in zip(bars, vals.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}",
                ha="center", fontsize=10)
    if med is not None:
        ax.axhline(med, color="tab:blue", ls="--", lw=1,
                   label=f"mode median {med:.1%}")
        ax.legend(fontsize=8)
    ax.set_ylabel("fraction of background-error variance explained")
    ax.set_ylim(0, max(vals.values()) * 1.25)
    ax.set_title(
        f"Background error is low-dimensional\n"
        f"(proj_dim={d2['proj_dim']}, rank={d2['rank']}, "
        f"n={d2['n_windows']} windows)",
        fontsize=10,
    )
    fig.tight_layout()
    out = Path(args.report).parent / "d2_lowdim.png"
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
