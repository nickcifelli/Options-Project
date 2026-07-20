"""Tests for surface/parity.py using a synthetic chain (no network calls)."""

import datetime as dt

import pandas as pd
import pytest

from vol_surface.surface.parity import check_parity

AS_OF = dt.datetime(2026, 1, 1)
EXPIRY = dt.datetime(2026, 4, 1)
R, Q = 0.04, 0.0


def _leg(option_type, mid, spread=0.02):
    return {
        "expiry": EXPIRY,
        "strike": 100.0,
        "spot": 100.0,
        "option_type": option_type,
        "mid": mid,
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
    }


def _theoretical_diff(K=100.0, S=100.0, T=None):
    import numpy as np

    from vol_surface.surface.build import year_fraction

    T = T if T is not None else year_fraction(EXPIRY, AS_OF)
    return S * np.exp(-Q * T) - K * np.exp(-R * T)


def test_parity_not_flagged_when_within_spread():
    diff = _theoretical_diff()
    call_mid = 5.0
    put_mid = call_mid - diff
    chain = pd.DataFrame([_leg("call", call_mid), _leg("put", put_mid)])
    result = check_parity(chain, r=R, q=Q, as_of=AS_OF)

    assert len(result) == 1
    assert not result.loc[0, "flagged"]
    assert result.loc[0, "deviation"] == pytest.approx(0.0, abs=1e-8)


def test_parity_flagged_when_deviation_exceeds_spread():
    diff = _theoretical_diff()
    call_mid = 5.0
    put_mid = call_mid - diff + 1.0  # blow past parity by $1, spreads are $0.02 each
    chain = pd.DataFrame([_leg("call", call_mid), _leg("put", put_mid)])
    result = check_parity(chain, r=R, q=Q, as_of=AS_OF)

    assert len(result) == 1
    assert result.loc[0, "flagged"]


def test_parity_skips_strikes_without_both_legs():
    chain = pd.DataFrame([_leg("call", 5.0)])
    result = check_parity(chain, r=R, q=Q, as_of=AS_OF)
    assert result.empty


def test_parity_skips_expired_rows():
    past_call = _leg("call", 5.0)
    past_put = _leg("put", 4.0)
    past_call["expiry"] = dt.datetime(2025, 1, 1)
    past_put["expiry"] = dt.datetime(2025, 1, 1)
    chain = pd.DataFrame([past_call, past_put])
    result = check_parity(chain, r=R, q=Q, as_of=AS_OF)
    assert result.empty
