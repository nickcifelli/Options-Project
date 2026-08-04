"""Smoke tests for viz/plots.py -- render without error and touch the
data in the expected shape (headless Agg backend, see conftest.py)."""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vol_surface.surface.svi import SVIFitResult, SVIParams
from vol_surface.viz.plots import plot_smiles, plot_surface_3d, plot_surface_heatmap, plot_svi_fit, plot_svi_surface_3d

AS_OF = dt.datetime(2026, 1, 1)


@pytest.fixture
def surface() -> pd.DataFrame:
    expiries = [dt.datetime(2026, 2, 1), dt.datetime(2026, 4, 1)]
    rows = []
    for expiry in expiries:
        for strike in (90.0, 100.0, 110.0):
            rows.append(
                {
                    "expiry": expiry,
                    "strike": strike,
                    "moneyness": strike / 100.0,
                    "log_moneyness": float(np.log(strike / 100.0)),
                    "iv": 0.2 + 0.001 * abs(strike - 100.0),
                    "option_type": "call",
                }
            )
    return pd.DataFrame(rows)


def test_plot_smiles_draws_one_line_per_expiry(surface):
    fig, ax = plt.subplots()
    result_ax = plot_smiles(surface, ax=ax)
    assert result_ax is ax
    assert len(ax.get_lines()) == surface["expiry"].nunique()
    plt.close(fig)


def test_plot_surface_3d_runs_without_error(surface):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    result_ax = plot_surface_3d(surface, as_of=AS_OF, ax=ax)
    assert result_ax is ax
    plt.close(fig)


def test_plot_surface_3d_creates_own_axes_when_none_given(surface):
    ax = plot_surface_3d(surface, as_of=AS_OF)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_surface_heatmap_runs_without_error(surface):
    fig, ax = plt.subplots()
    result_ax = plot_surface_heatmap(surface, moneyness_bins=3, ax=ax)
    assert result_ax is ax
    plt.close(fig)


def test_plot_svi_fit_draws_a_curve_for_converged_expiries(surface):
    params = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.2)
    fits = {expiry: SVIFitResult(params) for expiry in surface["expiry"].unique()}

    fig, ax = plt.subplots()
    result_ax = plot_svi_fit(surface, fits, as_of=AS_OF, ax=ax)

    assert result_ax is ax
    assert len(ax.collections) == surface["expiry"].nunique()  # one scatter per expiry
    assert len(ax.get_lines()) == surface["expiry"].nunique()  # one fitted curve per expiry
    plt.close(fig)


def test_plot_svi_fit_skips_curve_for_unconverged_expiry(surface):
    fits = {expiry: SVIFitResult(None, "did not converge") for expiry in surface["expiry"].unique()}

    fig, ax = plt.subplots()
    result_ax = plot_svi_fit(surface, fits, as_of=AS_OF, ax=ax)

    assert result_ax is ax
    assert len(ax.collections) == surface["expiry"].nunique()
    assert len(ax.get_lines()) == 0
    plt.close(fig)


def test_plot_svi_surface_3d_runs_without_error(surface):
    params = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.2)
    fits = {
        expiry: SVIFitResult(params, k_range=(float(group["log_moneyness"].min()), float(group["log_moneyness"].max())))
        for expiry, group in surface.groupby("expiry")
    }

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    result_ax = plot_svi_surface_3d(fits, as_of=AS_OF, ax=ax)

    assert result_ax is ax
    plt.close(fig)


def test_plot_svi_surface_3d_skips_unconverged_expiries():
    # Two converged expiries at distinct T (needed for a non-degenerate
    # triangulation -- a single T slice is a line, not a surface) plus one
    # that failed to fit; the failed one should just be left out.
    params = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.2)
    fits = {
        dt.datetime(2026, 2, 1): SVIFitResult(params, k_range=(-0.1, 0.1)),
        dt.datetime(2026, 3, 1): SVIFitResult(params, k_range=(-0.1, 0.1)),
        dt.datetime(2026, 4, 1): SVIFitResult(None, "did not converge"),
    }

    ax = plot_svi_surface_3d(fits, as_of=AS_OF)

    assert ax is not None
    plt.close(ax.figure)


def test_plot_svi_surface_3d_raises_when_nothing_converged():
    fits = {dt.datetime(2026, 2, 1): SVIFitResult(None, "did not converge")}

    with pytest.raises(ValueError, match="no SVI fits converged"):
        plot_svi_surface_3d(fits, as_of=AS_OF)
