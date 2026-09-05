"""Tests for pricing/pde.py.

The solver is pinned from three directions, because on a local vol surface
there is nothing to compare it to directly. Against Black-Scholes and the
binomial tree it must reproduce known answers on a flat surface. Against
the Monte Carlo it must agree on a surface neither of them has a formula
for -- the check that actually exercises the local vol path, and the one
that would catch a coordinate error the flat-surface tests cannot see.
Against itself it must be internally consistent: two independent LCP
solvers, and a European solve that is the American solve with the
constraint switched off.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vol_surface.pricing.binomial import price as tree_price
from vol_surface.pricing.black_scholes import greeks as bs_greeks
from vol_surface.pricing.black_scholes import price as bs_price
from vol_surface.pricing.monte_carlo import price_european
from vol_surface.pricing.pde import (
    PREMIUM_COLUMNS,
    _cell_averaged_payoff,
    american_premium_chain,
    constant_vol,
    default_time_steps,
    early_exercise_premium,
    solve,
    surface_vol,
)
from vol_surface.surface.local_vol import LocalVolSurface

AS_OF = dt.datetime(2026, 1, 1)
S0, T, R, Q, SIGMA = 100.0, 1.0, 0.05, 0.0, 0.25
STRIKES = (80.0, 95.0, 100.0, 105.0, 120.0)

# A fine grid, used where the point is accuracy rather than speed.
FINE = {"n_space": 801, "n_time": 400}

K_GRID = np.linspace(-0.8, 0.8, 60)
T_GRID = np.array([0.05, 0.25, 0.5, 1.0])


def _skewed_surface(base=0.25, skew=-0.15, term=0.05) -> LocalVolSurface:
    """A downward-skewed local vol surface with a mild upward term structure.

    Not derived from a fit: the point is a surface with real structure in
    both axes that no method here has a closed form for, so agreement
    between the PDE and the Monte Carlo on it is agreement about the
    surface rather than about a shared formula.
    """
    grid = base + skew * K_GRID[None, :] + term * T_GRID[:, None]
    return LocalVolSurface(k=K_GRID, T=T_GRID, local_vol=grid, implied_vol=grid)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", STRIKES)
def test_european_matches_black_scholes(option_type, K):
    result = solve(S0, K, T, R, Q, option_type, SIGMA, **FINE)

    assert result.price == pytest.approx(bs_price(S0, K, T, R, SIGMA, option_type, Q), abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_grid_greeks_match_the_closed_form(option_type):
    result = solve(S0, 100.0, T, R, Q, option_type, SIGMA, **FINE)
    expected = bs_greeks(S0, 100.0, T, R, SIGMA, option_type, Q)

    assert result.delta == pytest.approx(expected.delta, abs=1e-4)
    assert result.gamma == pytest.approx(expected.gamma, abs=1e-5)
    # Theta is a one-sided difference across the last time step, so it
    # carries the step size as its error rather than the grid spacing.
    assert result.theta == pytest.approx(expected.theta, abs=1e-2)


def test_european_prices_satisfy_put_call_parity():
    K = 105.0
    call = solve(S0, K, T, R, Q, "call", SIGMA, **FINE).price
    put = solve(S0, K, T, R, Q, "put", SIGMA, **FINE).price

    assert call - put == pytest.approx(S0 * np.exp(-Q * T) - K * np.exp(-R * T), abs=1e-3)


def test_error_falls_as_the_grid_refines():
    exact = bs_price(S0, 100.0, T, R, SIGMA, "call", Q)
    errors = [
        abs(solve(S0, 100.0, T, R, Q, "call", SIGMA, n_space=n, n_time=n // 2).price - exact)
        for n in (101, 201, 401, 801)
    ]

    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < errors[0] / 10


@pytest.mark.parametrize("K", STRIKES)
def test_american_put_matches_the_binomial_tree(K):
    # An independent method for the same optimal stopping problem: backward
    # induction on a lattice against a free-boundary PDE solve.
    result = solve(S0, K, T, R, Q, "put", SIGMA, american=True, **FINE)
    expected = tree_price(S0, K, T, R, SIGMA, "put", Q, N=4000, american=True)

    assert result.price == pytest.approx(expected, abs=2e-3)


def test_american_call_matches_the_binomial_tree_under_dividends():
    q = 0.06  # a call is only ever exercised early against a dividend yield
    result = solve(S0, 100.0, T, R, q, "call", SIGMA, american=True, **FINE)
    expected = tree_price(S0, 100.0, T, R, SIGMA, "call", q, N=4000, american=True)

    assert result.price == pytest.approx(expected, abs=2e-3)


def test_american_call_has_no_early_exercise_premium_without_dividends():
    # The standard theoretical result, and a sharp test of the LCP solver:
    # the constraint must never bind, so the answer has to be the European
    # price to solver precision rather than merely close to it.
    premium, _, _ = early_exercise_premium(S0, 100.0, T, R, 0.0, "call", SIGMA, **FINE)

    assert premium == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("K", STRIKES)
def test_american_is_never_worth_less_than_european(K):
    premium, american, european = early_exercise_premium(S0, K, T, R, Q, "put", SIGMA, **FINE)

    assert premium >= 0
    assert american.price >= european.price


def test_american_put_premium_grows_with_the_rate():
    # Early exercise on a put buys the interest on the strike, so the
    # premium has to be increasing in r -- the economic content of the
    # free boundary, not just its existence.
    premiums = [early_exercise_premium(S0, 110.0, T, r, Q, "put", SIGMA, **FINE)[0] for r in (0.0, 0.02, 0.05, 0.10)]

    assert premiums == sorted(premiums)
    assert premiums[0] == pytest.approx(0.0, abs=1e-3)  # no rate, (almost) no reason to exercise


def test_the_two_lcp_solvers_agree():
    # Brennan-Schwartz is exact only because a vanilla's exercise region is
    # a single interval. PSOR assumes nothing of the kind, so agreement
    # between them is what licenses using the fast one by default.
    grid = {"n_space": 201, "n_time": 100}
    direct = solve(S0, 105.0, T, R, Q, "put", SIGMA, american=True, lcp="brennan-schwartz", **grid)
    iterative = solve(S0, 105.0, T, R, Q, "put", SIGMA, american=True, lcp="psor", **grid)

    assert direct.price == pytest.approx(iterative.price, abs=1e-8)
    np.testing.assert_allclose(direct.values, iterative.values, atol=1e-8)


def test_the_two_lcp_solvers_agree_on_a_call_under_dividends():
    # The call's exercise region sits at high spot, so Brennan-Schwartz
    # sweeps the other way; PSOR is direction-agnostic and pins that down.
    grid = {"n_space": 201, "n_time": 100}
    direct = solve(S0, 95.0, T, R, 0.08, "call", SIGMA, american=True, lcp="brennan-schwartz", **grid)
    iterative = solve(S0, 95.0, T, R, 0.08, "call", SIGMA, american=True, lcp="psor", **grid)

    assert direct.price == pytest.approx(iterative.price, abs=1e-8)


def test_rannacher_startup_keeps_gamma_non_negative():
    # Crank-Nicolson alone rings at the payoff kink: the price stays close
    # but gamma oscillates and goes negative, which is impossible for a
    # convex payoff. Two implicit steps damp the modes that cause it.
    grid = {"n_space": 401, "n_time": 20}
    undamped = solve(S0, 100.0, 0.25, R, Q, "call", 0.3, rannacher=0, **grid)
    damped = solve(S0, 100.0, 0.25, R, Q, "call", 0.3, rannacher=2, **grid)

    def min_gamma(result):
        return float(np.gradient(np.gradient(result.values, result.S), result.S).min())

    assert min_gamma(undamped) < -1e-3, "the fixture is supposed to ring without damping"
    # Not exactly zero: the residual is the differencing noise of reading
    # gamma off the grid, four orders of magnitude below the ringing.
    assert min_gamma(damped) >= -1e-6


def test_cell_averaged_payoff_matches_quadrature():
    # The terminal condition is an integral, not a sample; if the closed
    # form is wrong the whole solve starts from the wrong place.
    x = np.linspace(np.log(80), np.log(125), 21)
    h = x[1] - x[0]

    for option_type in ("call", "put"):
        averaged = _cell_averaged_payoff(x, 100.0, option_type)
        for node, expected in zip(x, averaged):
            fine = np.linspace(node - h / 2, node + h / 2, 20_001)
            payoff = np.maximum(np.exp(fine) - 100.0, 0.0) if option_type == "call" else np.maximum(100.0 - np.exp(fine), 0.0)
            assert expected == pytest.approx(np.trapezoid(payoff, fine) / h, rel=1e-6, abs=1e-9)


def test_price_does_not_depend_on_where_the_strike_falls_between_nodes():
    # What cell-averaging buys: strikes that land at different points
    # within a cell must not price differently for that reason alone.
    prices = [solve(S0, K, T, R, Q, "call", SIGMA, n_space=201, n_time=100).price for K in (100.0, 100.05, 100.1)]
    exact = [bs_price(S0, K, T, R, SIGMA, "call", Q) for K in (100.0, 100.05, 100.1)]

    np.testing.assert_allclose(np.diff(prices), np.diff(exact), atol=2e-4)


def test_exercise_boundary_is_below_the_strike_and_rises_toward_expiry():
    result = solve(S0, 100.0, T, R, Q, "put", SIGMA, american=True, **FINE)
    boundary = result.exercise_boundary

    assert result.tau[-1] == pytest.approx(T)
    finite = boundary[np.isfinite(boundary)]
    assert len(finite) > 0
    assert np.all(finite < 100.0)  # exercise early only below the strike
    # tau counts up to T as calendar time counts down, so the boundary
    # falls along the array and rises toward the strike as expiry nears.
    assert np.all(np.diff(finite) <= 1e-9)


def test_european_solve_reports_no_exercise_boundary():
    result = solve(S0, 100.0, T, R, Q, "put", SIGMA, american=False)

    assert result.exercise_boundary is None
    assert np.isnan(result.critical_spot_now)
    assert not result.early_exercise_is_optimal_now


def test_deep_in_the_money_american_put_is_exercised_immediately():
    result = solve(S0, 200.0, T, 0.08, Q, "put", SIGMA, american=True, **FINE)

    assert result.early_exercise_is_optimal_now
    assert result.price == pytest.approx(100.0, abs=1e-6)  # exactly intrinsic
    assert result.critical_spot_now > S0


def test_at_the_money_american_put_is_not_exercised_immediately():
    result = solve(S0, 100.0, T, R, Q, "put", SIGMA, american=True, **FINE)

    assert not result.early_exercise_is_optimal_now
    assert result.critical_spot_now < S0


@pytest.mark.parametrize("option_type,mny", [("call", 1.0), ("call", 1.05), ("put", 0.95)])
def test_pde_and_monte_carlo_agree_on_the_same_local_vol_surface(option_type, mny):
    # The cross-method check. Backward linear algebra against forward
    # random sampling, reading one surface through one sampler; the Monte
    # Carlo is given enough steps to push its O(dt) freezing bias below
    # its own standard error, which is the term that separates the two.
    surface = _skewed_surface()
    K = S0 * mny
    pde_price = solve(S0, K, 0.5, R, Q, option_type, surface_vol(surface), n_space=601, n_time=400).price
    mc = price_european(
        surface, S0, K, 0.5, r=R, q=Q, option_type=option_type, n_paths=100_000, n_steps=1000, seed=11
    )

    assert pde_price == pytest.approx(mc.price, abs=3 * mc.std_error)


def test_monte_carlo_bias_shrinks_toward_the_pde_answer():
    # The Monte Carlo freezes local vol over each step, an O(dt) bias its
    # own tests can only show is decreasing. The PDE supplies the value it
    # is decreasing *towards*, which turns a trend into a measurement.
    surface = _skewed_surface()
    reference = solve(S0, S0, 0.5, R, Q, "call", surface_vol(surface), n_space=601, n_time=800).price

    coarse = price_european(surface, S0, S0, 0.5, r=R, q=Q, n_paths=100_000, n_steps=32, seed=5)
    fine = price_european(surface, S0, S0, 0.5, r=R, q=Q, n_paths=100_000, n_steps=512, seed=5)

    assert abs(fine.price - reference) < abs(coarse.price - reference) / 3


def test_constant_vol_helper_ignores_moneyness_and_time():
    sigma = constant_vol(0.3)

    np.testing.assert_allclose(sigma(np.array([-1.0, 0.0, 2.0]), 0.5), 0.3)


def test_default_time_steps_is_daily_with_a_floor():
    assert default_time_steps(1.0) == 252
    assert default_time_steps(1.0 / 365.0) == 25


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"option_type": "straddle"}, "option_type must be one of"),
        ({"T": 0.0}, "T must be > 0"),
        ({"n_space": 3}, "n_space must be >= 5"),
        ({"lcp": "newton"}, "lcp must be one of"),
    ],
)
def test_rejects_bad_inputs(kwargs, message):
    call = {"S0": S0, "K": 100.0, "T": T, "r": R, "q": Q, "option_type": "put", "sigma": SIGMA, **kwargs}
    with pytest.raises(ValueError, match=message):
        solve(**call)


def _chain(expiry: dt.datetime, strikes=(90.0, 100.0, 110.0)) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "expiry": expiry,
                "strike": strike,
                "moneyness": strike / S0,
                "log_moneyness": float(np.log(strike / S0)),
                "iv": 0.25,
                "option_type": "put" if strike < S0 else "call",
            }
            for strike in strikes
        ]
    )


def test_american_premium_chain_reports_a_premium_on_puts_and_none_on_calls():
    surface = _skewed_surface()
    expiry = AS_OF + dt.timedelta(days=int(0.5 * 365))
    premium = american_premium_chain(surface, _chain(expiry), S0=S0, r=R, q=Q, as_of=AS_OF)

    assert list(premium.columns) == PREMIUM_COLUMNS
    assert not premium.empty

    puts = premium[premium["option_type"] == "put"]
    calls = premium[premium["option_type"] == "call"]
    assert (puts["premium"] > 0).all()
    np.testing.assert_allclose(calls["premium"], 0.0, atol=1e-10)  # q = 0
    assert (premium["american"] >= premium["european"] - 1e-10).all()


def test_american_premium_chain_skips_quotes_outside_the_surface_window():
    surface = _skewed_surface()
    inside = AS_OF + dt.timedelta(days=int(0.5 * 365))
    outside = AS_OF + dt.timedelta(days=int(5 * 365))  # past the surface's last row

    premium = american_premium_chain(
        surface, pd.concat([_chain(inside), _chain(outside)], ignore_index=True), S0=S0, r=R, q=Q, as_of=AS_OF
    )

    assert set(pd.to_datetime(premium["expiry"])) == {pd.Timestamp(inside)}
