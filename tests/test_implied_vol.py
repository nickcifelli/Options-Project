"""Tests for the Brent's-method implied volatility solver."""

import pytest

from vol_surface.pricing.black_scholes import price as bs_price
from vol_surface.pricing.implied_vol import implied_vol

S, K, T, R, Q = 100.0, 105.0, 0.75, 0.03, 0.01


@pytest.mark.parametrize("option_type,true_sigma", [("call", 0.15), ("call", 0.45), ("put", 0.15), ("put", 0.45)])
def test_round_trip_recovers_known_sigma(option_type, true_sigma):
    market_price = bs_price(S, K, T, R, true_sigma, option_type, Q)
    result = implied_vol(market_price, S, K, T, R, option_type, Q)
    assert result.ok
    assert result.sigma == pytest.approx(true_sigma, abs=1e-6)


def test_atm_round_trip():
    market_price = bs_price(100.0, 100.0, 0.5, 0.02, 0.30, "call", 0.0)
    result = implied_vol(market_price, 100.0, 100.0, 0.5, 0.02, "call", 0.0)
    assert result.ok
    assert result.sigma == pytest.approx(0.30, abs=1e-6)


def test_price_below_no_arbitrage_lower_bound_fails_cleanly():
    # Deep ITM call (S=150, K=100): the no-arbitrage lower bound is well
    # above zero, so a near-zero quote is unexplainable by any vol.
    result = implied_vol(1.0, 150.0, 100.0, T, R, "call", Q)
    assert not result.ok
    assert "no-arbitrage" in result.reason


def test_price_above_no_arbitrage_upper_bound_fails_cleanly():
    result = implied_vol(S * 10, S, K, T, R, "call", Q)
    assert not result.ok
    assert "no-arbitrage" in result.reason


def test_expired_option_returns_none_with_reason():
    result = implied_vol(5.0, S, K, 0.0, R, "call", Q)
    assert not result.ok
    assert result.sigma is None
    assert result.reason is not None


def test_negative_price_returns_none_with_reason():
    result = implied_vol(-1.0, S, K, T, R, "call", Q)
    assert not result.ok
    assert result.reason is not None


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        implied_vol(5.0, S, K, T, R, "straddle", Q)
