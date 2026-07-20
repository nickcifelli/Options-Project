"""Tests for the closed-form Black-Scholes pricer and Greeks."""

import math

import pytest

from vol_surface.pricing.black_scholes import greeks, price

# Reference values from Hull, "Options, Futures, and Other Derivatives",
# cross-checked against standard online BS calculators.
# S=42, K=40, r=10%, sigma=20%, T=0.5y (no dividend): call ~= 4.76, put ~= 0.81
HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA = 42.0, 40.0, 0.5, 0.10, 0.20


def test_call_price_matches_reference():
    assert price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call") == pytest.approx(4.76, abs=0.01)


def test_put_price_matches_reference():
    assert price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put") == pytest.approx(0.81, abs=0.01)


def test_put_call_parity_holds():
    call = price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
    put = price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put")
    lhs = call - put
    rhs = HULL_S - HULL_K * math.exp(-HULL_R * HULL_T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_atm_call_with_zero_rate_and_dividend_is_symmetric_with_put():
    # With r=q=0 and S=K, call and put must be identical by parity.
    call = price(100.0, 100.0, 1.0, 0.0, 0.25, "call")
    put = price(100.0, 100.0, 1.0, 0.0, 0.25, "put")
    assert call == pytest.approx(put, abs=1e-8)


def test_t_zero_returns_intrinsic_value():
    assert price(105.0, 100.0, 0.0, 0.05, 0.2, "call") == pytest.approx(5.0)
    assert price(95.0, 100.0, 0.0, 0.05, 0.2, "call") == pytest.approx(0.0)
    assert price(95.0, 100.0, 0.0, 0.05, 0.2, "put") == pytest.approx(5.0)
    assert price(105.0, 100.0, 0.0, 0.05, 0.2, "put") == pytest.approx(0.0)


def test_sigma_zero_returns_discounted_forward_intrinsic():
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.0
    forward = S * math.exp((r - q) * T)
    expected_call = math.exp(-r * T) * max(forward - K, 0.0)
    assert price(S, K, T, r, 0.0, "call", q) == pytest.approx(expected_call)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        price(100.0, 100.0, 1.0, 0.05, 0.2, "straddle")


def test_negative_T_raises():
    with pytest.raises(ValueError):
        price(100.0, 100.0, -1.0, 0.05, 0.2, "call")


def test_call_delta_in_zero_one_range():
    g = greeks(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
    assert 0.0 < g.delta < 1.0


def test_put_call_delta_relationship():
    # delta_call - delta_put == e^{-qT} for the same parameters.
    q = 0.02
    call_g = greeks(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", q)
    put_g = greeks(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put", q)
    assert call_g.delta - put_g.delta == pytest.approx(math.exp(-q * HULL_T), abs=1e-8)


def test_gamma_and_vega_identical_for_call_and_put():
    call_g = greeks(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
    put_g = greeks(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put")
    assert call_g.gamma == pytest.approx(put_g.gamma)
    assert call_g.vega == pytest.approx(put_g.vega)


def test_greeks_at_t_zero_are_boundary_indicator():
    itm = greeks(105.0, 100.0, 0.0, 0.05, 0.2, "call")
    assert itm.delta == pytest.approx(1.0)
    assert itm.gamma == 0.0
    assert itm.vega == 0.0

    otm = greeks(95.0, 100.0, 0.0, 0.05, 0.2, "call")
    assert otm.delta == pytest.approx(0.0)


def test_delta_via_finite_difference():
    bump = 1e-4
    up = price(HULL_S + bump, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
    down = price(HULL_S - bump, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
    fd_delta = (up - down) / (2 * bump)
    g = greeks(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
    assert g.delta == pytest.approx(fd_delta, abs=1e-4)


def test_vega_via_finite_difference():
    bump = 1e-4
    up = price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA + bump, "call")
    down = price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA - bump, "call")
    fd_vega = (up - down) / (2 * bump)
    g = greeks(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
    assert g.vega == pytest.approx(fd_vega, abs=1e-3)
