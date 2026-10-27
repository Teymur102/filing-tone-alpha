"""The three committed charts."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

LOW, MID, HIGH = "#c0392b", "#95a5a6", "#1f6fb2"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9.5})


def car_paths(panel: pd.DataFrame, paths: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, sig, title in ((axes[0], "finbert_tone", "FinBERT tone"), (axes[1], "lm_tone", "Loughran-McDonald tone")):
        d = paths.merge(panel[["accessionNumber", f"terc_{sig}"]].dropna(), on="accessionNumber")
        for terc, color, name in ((1, LOW, "Low tone"), (2, MID, "Middle"), (3, HIGH, "High tone")):
            g = d[d[f"terc_{sig}"] == terc].pivot(index="accessionNumber", columns="k", values="ar")
            g = g.loc[:, g.columns >= -1]
            cum = g.cumsum(axis=1) * 100
            m = cum.mean()
            se = cum.std(ddof=1) / np.sqrt(len(cum))
            ax.plot(m.index, m.values, color=color, lw=2, label=f"{name} (n={len(cum)})")
            ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.12, lw=0)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(1.5, color="k", lw=0.6, ls=":")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{title}: cumulative abnormal return by tercile")
        ax.set_xlabel("Trading days after earnings release (day 0)")
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("Mean CAR from day -1 (%), 95% band")
    fig.text(0.5, -0.02, "Terciles use breakpoints from the prior 365 days of releases only. Dotted line: drift window starts at day +2.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "car_paths_by_tone_tercile.png", bbox_inches="tight")
    plt.close(fig)


def agreement(panel: pd.DataFrame, corr: float, spear: float, agree: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.scatter(panel.lm_tone, panel.finbert_tone, s=6, alpha=0.35, color=HIGH, lw=0)
    ax.set_xlabel("Loughran-McDonald net tone  (pos - neg) / (pos + neg)")
    ax.set_ylabel("FinBERT tone  mean[P(pos) - P(neg)]")
    ax.set_title(f"Per-document tone: Pearson {corr:.2f}, Spearman {spear:.2f}")
    ax = axes[1]
    both = panel.dropna(subset=["terc_lm_tone", "terc_finbert_tone"])
    m = pd.crosstab(both.terc_finbert_tone, both.terc_lm_tone, normalize="index") * 100
    im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m.values[i, j]:.0f}%", ha="center", va="center",
                    color="white" if m.values[i, j] > 55 else "black")
    ax.set_xticks(range(3), ["Low", "Mid", "High"])
    ax.set_yticks(range(3), ["Low", "Mid", "High"])
    ax.set_xlabel("LM tercile")
    ax.set_ylabel("FinBERT tercile")
    ax.grid(False)
    ax.set_title(f"Same tercile {agree:.0%} of the time (row %)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "lm_vs_finbert_agreement.png", bbox_inches="tight")
    plt.close(fig)


def oos_equity(p: pd.DataFrame, signal: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for col, color, name in (("gross", HIGH, "Gross"), ("net", LOW, f"Net of {C.COST_BPS_PER_SIDE} bps/side costs")):
        ax.plot(p.index, (1 + p[col]).cumprod() - 1, color=color, lw=1.6, label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Out-of-sample long-short drift portfolio ({signal}), "
                 f"{p.index.min():%Y-%m} to {p.index.max():%Y-%m}")
    ax.set_ylabel("Cumulative return")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "oos_long_short.png", bbox_inches="tight")
    plt.close(fig)
