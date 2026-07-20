"""Tests for the CRR binomial tree: convergence to Black-Scholes and the
American >= European invariant."""

import pytest

from vol_surface.pricing.binomial import price as binomial_price
from vol_surface.pricing.black_scholes import price as bs_price

S, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_converges_to_black_scholes_at_n200(option_type):
    bs = bs_price(S, K, T, R, SIGMA, option_type, Q)
    tree = binomial_price(S, K, T, R, SIGMA, option_type, Q, N=200, american=False)
    assert tree == pytest.approx(bs, rel=0.005)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_converges_tighter_at_n1000(option_type):
    bs = bs_price(S, K, T, R, SIGMA, option_type, Q)
    tree = binomial_price(S, K, T, R, SIGMA, option_type, Q, N=1000, american=False)
    assert tree == pytest.approx(bs, rel=0.001)


@pytest.mark.parametrize(
    "S_,K_,option_type",
    [
        (100.0, 100.0, "call"),
        (100.0, 100.0, "put"),
        (80.0, 100.0, "put"),
        (120.0, 100.0, "call"),
        (100.0, 120.0, "put"),
    ],
)
def test_american_at_least_european_for_all_cases(S_, K_, option_type):
    european = binomial_price(S_, K_, T, R, SIGMA, option_type, Q, N=300, american=False)
    american = binomial_price(S_, K_, T, R, SIGMA, option_type, Q, N=300, american=True)
    assert american >= european - 1e-9


def test_american_call_equals_european_call_with_no_dividends():
    # With q=0, early exercise of a call is never optimal, so American ==
    # European (a standard theoretical result, not just an inequality).
    european = binomial_price(S, K, T, R, SIGMA, "call", 0.0, N=300, american=False)
    american = binomial_price(S, K, T, R, SIGMA, "call", 0.0, N=300, american=True)
    assert american == pytest.approx(european, abs=1e-6)


def test_american_put_exceeds_european_put_when_deep_itm():
    # Deep ITM put with a meaningful rate has real early-exercise premium.
    european = binomial_price(80.0, 100.0, 1.0, 0.08, 0.2, "put", 0.0, N=300, american=False)
    american = binomial_price(80.0, 100.0, 1.0, 0.08, 0.2, "put", 0.0, N=300, american=True)
    assert american > european + 1e-3


def test_t_zero_returns_intrinsic():
    assert binomial_price(105.0, 100.0, 0.0, 0.05, 0.2, "call") == pytest.approx(5.0)
    assert binomial_price(95.0, 100.0, 0.0, 0.05, 0.2, "call") == pytest.approx(0.0)


def test_invalid_sigma_raises():
    with pytest.raises(ValueError):
        binomial_price(S, K, T, R, 0.0, "call")


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        binomial_price(S, K, T, R, SIGMA, "call", N=0)
