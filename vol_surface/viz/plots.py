"""Smile and full implied vol surface plots."""

from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vol_surface.surface.build import year_fraction
from vol_surface.surface.local_vol import LocalVolSurface
from vol_surface.surface.svi import SVIFitResult


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


def plot_svi_fit(
    surface: pd.DataFrame,
    fits: dict[pd.Timestamp, SVIFitResult],
    as_of: dt.datetime | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Market IV points per expiry overlaid with their fitted SVI smile
    (see `surface/svi.py`). Expiries whose fit didn't converge still show
    their market points, just without a curve."""
    ax = ax or plt.gca()
    as_of = as_of or dt.datetime.now()

    for i, (expiry, group) in enumerate(surface.sort_values("log_moneyness").groupby("expiry")):
        color = f"C{i % 10}"
        label = pd.Timestamp(expiry).date().isoformat()
        ax.scatter(group["log_moneyness"], group["iv"], s=14, color=color, label=label)

        fit = fits.get(expiry)
        if fit is not None and fit.ok:
            T = year_fraction(expiry, as_of)
            lo, hi = fit.k_range or (group["log_moneyness"].min(), group["log_moneyness"].max())
            k_grid = np.linspace(lo, hi, 200)
            ax.plot(k_grid, fit.params.implied_vol(k_grid, T), color=color, linewidth=1.5)

    ax.set_xlabel("log-moneyness (log(K/S))")
    ax.set_ylabel("implied vol")
    ax.set_title("Implied vol smile: market vs. SVI fit")
    ax.legend(fontsize=8)
    return ax


def plot_surface_3d(
    surface: pd.DataFrame,
    as_of: dt.datetime | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """3D surface (moneyness x time-to-expiry x IV) via triangulation of the
    raw market quotes, since strikes differ per expiry and don't form a
    regular grid. Noisy by construction -- see `plot_svi_surface_3d` for
    the smoothed, arbitrage-checked equivalent."""
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


def plot_svi_surface_3d(
    fits: dict[pd.Timestamp, SVIFitResult],
    as_of: dt.datetime | None = None,
    n_k: int = 400,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """3D surface (moneyness x time-to-expiry x IV) built from the fitted
    SVI curves rather than raw quotes: smooth, and free of the
    negative-total-variance arbitrage `fit_svi_slice` already excludes.

    Each expiry contributes only over the log-moneyness window it was
    actually fit on (`SVIFitResult.k_range`), matching `plot_svi_fit`;
    expiries whose fit didn't converge are skipped.

    `n_k` is how densely each expiry's curve is sampled before
    triangulation -- raising it smooths the surface *within* an expiry.
    It can't smooth *across* expiries: there are only as many T-slices as
    fitted expiries, so the surface is faceted between them regardless
    (real term-structure gaps, not a rendering artifact -- interpolating
    across them would be inventing data for expiries nobody quoted).
    """
    as_of = as_of or dt.datetime.now()
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    moneyness, T_years, iv = [], [], []
    for expiry, fit in fits.items():
        if not fit.ok:
            continue
        T = year_fraction(expiry, as_of)
        k_grid = np.linspace(*fit.k_range, n_k)
        moneyness.append(np.exp(k_grid))
        T_years.append(np.full(n_k, T))
        iv.append(fit.params.implied_vol(k_grid, T))

    if not moneyness:
        raise ValueError("no SVI fits converged; nothing to plot")

    ax.plot_trisurf(
        np.concatenate(moneyness), np.concatenate(T_years), np.concatenate(iv), cmap="viridis", linewidth=0.2, antialiased=True
    )
    ax.set_xlabel("moneyness (K/S)")
    ax.set_ylabel("T (years)")
    ax.set_zlabel("implied vol")
    ax.set_title("Implied vol surface (SVI fit)")
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


def plot_local_vol_surface_3d(
    local_vol: LocalVolSurface,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """3D Dupire local vol surface (forward moneyness x T x local vol).

    Drawn on the regular grid `build_local_vol_surface` returns, so this
    is `plot_surface` rather than the triangulation the market-quote plots
    need. Deliberately a different colormap from the implied vol panels:
    the two are different quantities on similar-looking axes, and reading
    one for the other is the easiest mistake to make here.

    Grid points where local vol does not exist are `NaN` and render as
    holes -- the honest depiction, since the surface genuinely implies no
    real instantaneous vol there.
    """
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    forward_moneyness, T = np.meshgrid(np.exp(local_vol.k), local_vol.T)
    ax.plot_surface(forward_moneyness, T, local_vol.local_vol, cmap="magma", linewidth=0.2, antialiased=True)
    ax.set_xlabel("forward moneyness (K/F)")
    ax.set_ylabel("T (years)")
    ax.set_zlabel("local vol")
    ax.set_title("Dupire local vol surface")
    return ax


def plot_mc_reprice(reprice: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Market IV vs. Monte-Carlo-repriced IV per strike, with MC error bars.

    The comparison is drawn in vol points rather than price because a cent
    of error means something very different on a one-week wing than on a
    five-month at-the-money. Error bars are the Monte Carlo standard error
    converted to vol via the local vega (`dSigma ~ dPrice / vega`), so a
    point sitting off the diagonal by more than its bar is a real repricing
    gap rather than simulation noise.
    """
    ax = ax or plt.gca()

    for i, (expiry, group) in enumerate(reprice.dropna(subset=["mc_iv"]).groupby("expiry")):
        ax.errorbar(
            group["market_iv"],
            group["mc_iv"],
            yerr=group["iv_std_error"],
            fmt="o",
            markersize=3,
            linewidth=0.8,
            color=f"C{i % 10}",
            label=pd.Timestamp(expiry).date().isoformat(),
        )

    lo = float(min(reprice["market_iv"].min(), reprice["mc_iv"].min()))
    hi = float(max(reprice["market_iv"].max(), reprice["mc_iv"].max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", label="exact repricing")

    ax.set_xlabel("market implied vol")
    ax.set_ylabel("Monte Carlo implied vol")
    ax.set_title("Local vol repricing check")
    ax.legend(fontsize=7)
    return ax
