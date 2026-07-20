"""Tests for surface/build.py using a synthetic chain (no network calls)."""

import datetime as dt

import pandas as pd
import pytest

from vol_surface.pricing.black_scholes import price as bs_price
from vol_surface.surface.build import build_surface, year_fraction

AS_OF = dt.datetime(2026, 1, 1)
R, Q = 0.04, 0.0


def _synthetic_row(S, K, expiry, sigma, option_type):
    T = year_fraction(expiry, AS_OF)
    mid = bs_price(S, K, T, R, sigma, option_type, Q)
    return {
        "spot": S,
        "strike": K,
        "expiry": expiry,
        "option_type": option_type,
        "mid": mid,
        "bid": mid - 0.01,
        "ask": mid + 0.01,
    }


def test_build_surface_recovers_known_ivs():
    expiry = dt.datetime(2026, 4, 1)
    chain = pd.DataFrame(
        [
            _synthetic_row(100.0, 90.0, expiry, 0.25, "call"),
            _synthetic_row(100.0, 100.0, expiry, 0.20, "call"),
            _synthetic_row(100.0, 110.0, expiry, 0.30, "put"),
        ]
    )
    surface = build_surface(chain, r=R, q=Q, as_of=AS_OF)

    assert len(surface) == 3
    assert set(surface.columns) == {"expiry", "strike", "moneyness", "log_moneyness", "iv", "option_type"}

    by_strike = surface.set_index("strike")
    assert by_strike.loc[90.0, "iv"] == pytest.approx(0.25, abs=1e-4)
    assert by_strike.loc[100.0, "iv"] == pytest.approx(0.20, abs=1e-4)
    assert by_strike.loc[110.0, "iv"] == pytest.approx(0.30, abs=1e-4)


def test_build_surface_computes_moneyness():
    expiry = dt.datetime(2026, 4, 1)
    chain = pd.DataFrame([_synthetic_row(100.0, 120.0, expiry, 0.2, "call")])
    surface = build_surface(chain, r=R, q=Q, as_of=AS_OF)
    assert surface.loc[0, "moneyness"] == 1.2


def test_build_surface_skips_expired_rows():
    past_expiry = dt.datetime(2025, 1, 1)
    chain = pd.DataFrame([_synthetic_row(100.0, 100.0, past_expiry, 0.2, "call")])
    surface = build_surface(chain, r=R, q=Q, as_of=AS_OF)
    assert surface.empty


def test_build_surface_skips_rows_that_fail_iv_solve():
    expiry = dt.datetime(2026, 4, 1)
    bad_row = _synthetic_row(100.0, 100.0, expiry, 0.2, "call")
    bad_row["mid"] = -5.0  # unsolvable price
    chain = pd.DataFrame([bad_row])
    surface = build_surface(chain, r=R, q=Q, as_of=AS_OF)
    assert surface.empty


def test_year_fraction_nonnegative_and_zero_in_past():
    future = year_fraction(dt.datetime(2026, 7, 1), AS_OF)
    assert future > 0
    past = year_fraction(dt.datetime(2025, 1, 1), AS_OF)
    assert past == 0.0
