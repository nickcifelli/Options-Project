"""Smoke tests for viz/plots.py -- render without error and touch the
data in the expected shape (headless Agg backend, see conftest.py)."""

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from vol_surface.viz.plots import plot_smiles, plot_surface_3d, plot_surface_heatmap

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
                    "log_moneyness": 0.0,
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
