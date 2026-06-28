"""Generate the paper's figures from the committed experiment artifacts.

Reproducible: reads the version-controlled JSON summaries under experiments/ and
writes vector PDFs (and PNGs) into docs/paper/figures/. Run from the repo root:

    python docs/paper/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LR, RF = "#2a6f97", "#e07a5f"


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig_three_levels():
    """The central figure: realisable evasion at three measurement levels, from a
    single experiment (experiments/manifold_attack.json)."""
    d = json.loads((EXP / "manifold_attack.json").read_text())["results"]
    # level 1: feature-space success (ignore feasibility) -- from the free run
    # level 2: post-filter realisable -- free run successes that pass is_feasible
    # level 3: manifold-constrained realisable -- search confined to the manifold
    levels = ["Feature-space\n(no feasibility)", "Post-filter\nrealisable",
              "Manifold-constrained\nrealisable"]
    lr = [d["logreg"]["free_postfilter"]["feature_success_rate"],
          d["logreg"]["free_postfilter"]["realisable_rate"],
          d["logreg"]["manifold"]["realisable_rate"]]
    rf = [d["rf"]["free_postfilter"]["feature_success_rate"],
          d["rf"]["free_postfilter"]["realisable_rate"],
          d["rf"]["manifold"]["realisable_rate"]]

    x = np.arange(len(levels)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b1 = ax.bar(x - w / 2, lr, w, label="Logistic Regression", color=LR)
    b2 = ax.bar(x + w / 2, rf, w, label="Random Forest", color=RF)
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("evasion success rate"); ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, "three_levels")


if __name__ == "__main__":
    fig_three_levels()
