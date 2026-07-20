"""Smile and full implied vol surface plots."""

from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd

from vol_surface.surface.build import year_fraction


def plot_smiles(surface: pd.DataFrame, x: str = "moneyness", ax: plt.Axes | None = None) -> plt.Axes:
    """IV vs. strike/moneyness for a single expiry, overlaid across expiries
    to show the term structure of the skew."""
    ax = ax or plt.gca()
    for expiry, group in surface.sort_values(x).groupby("expiry"):
        label = pd.Timestamp(expiry).date().isoformat()
        ax.plot(group[x], group["iv"], marker="o", markersize=3, label=label)
    ax.set_xlabel(x)
    ax.set_ylabel("implied vol")
    ax.set_title("Implied vol smile by expiry")
    ax.legend(fontsize=8)
    return ax


def plot_surface_3d(
    surface: pd.DataFrame,
    as_of: dt.datetime | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """3D surface (moneyness x time-to-expiry x IV) via triangulation, since
    strikes differ per expiry and don't form a regular grid."""
    as_of = as_of or dt.datetime.now()
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    T = surface["expiry"].apply(lambda e: year_fraction(e, as_of))
    ax.plot_trisurf(surface["moneyness"], T, surface["iv"], cmap="viridis", linewidth=0.2, antialiased=True)
    ax.set_xlabel("moneyness (K/S)")
    ax.set_ylabel("T (years)")
    ax.set_zlabel("implied vol")
    ax.set_title("Implied vol surface")
    return ax


def plot_surface_heatmap(
    surface: pd.DataFrame,
    moneyness_bins: int = 20,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap fallback (expiry x binned moneyness -> mean IV) for when 3D
    rendering is more trouble than it's worth."""
    ax = ax or plt.gca()

    binned = surface.copy()
    binned["moneyness_bin"] = pd.cut(binned["moneyness"], moneyness_bins)
    pivot = binned.pivot_table(index="expiry", columns="moneyness_bin", values="iv", aggfunc="mean", observed=True)

    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{iv.mid:.2f}" for iv in pivot.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([pd.Timestamp(e).date().isoformat() for e in pivot.index], fontsize=6)
    ax.set_xlabel("moneyness (K/S)")
    ax.set_ylabel("expiry")
    ax.set_title("Implied vol surface (heatmap)")
    plt.colorbar(im, ax=ax, label="implied vol")
    return ax
