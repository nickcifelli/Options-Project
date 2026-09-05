"""Tests for pricing/monte_carlo.py.

A flat local vol surface is the one case with a closed-form answer, so it
carries the convergence checks: simulating it must return the
Black-Scholes price. Everything else is pinned down by properties that
hold for any surface -- the discounted spot is a martingale, put-call
parity holds pathwise, the standard error falls as 1/sqrt(N), and the
discretization bias falls with the step count rather than the path count.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vol_surface.pricing.black_scholes import price as bs_price
from vol_surface.pricing.monte_carlo import (
    MCResult,
    REPRICE_COLUMNS,
    atm_implied_vol,
    default_steps,
    price_european,
    price_from_terminal,
    reprice_chain,
    simulate_terminal,
)
from vol_surface.surface.local_vol import LocalVolSurface

AS_OF = dt.datetime(2026, 1, 1)
S0, K, T, R, Q = 100.0, 105.0, 0.5, 0.03, 0.01
SIGMA = 0.22

K_GRID = np.linspace(-0.6, 0.6, 60)
T_GRID = np.array([0.05, 0.25, 0.5, 1.0])


def _flat(sigma: float = SIGMA) -> LocalVolSurface:
    grid = np.full((len(T_GRID), len(K_GRID)), sigma)
    return LocalVolSurface(k=K_GRID, T=T_GRID, local_vol=grid, implied_vol=grid)


def _skewed() -> LocalVolSurface:
    """Local vol falling in k (equity skew) and rising in T."""
    grid = 0.30 - 0.18 * K_GRID[None, :] + 0.04 * T_GRID[:, None]
    return LocalVolSurface(k=K_GRID, T=T_GRID, local_vol=grid, implied_vol=grid)


def test_flat_surface_converges_to_black_scholes():
    # Plain Monte Carlo, no variance reduction: the estimate has to land
    # within its own error bar of the analytic price.
    analytic = bs_price(S0, K, T, R, SIGMA, "call", Q)

    result = price_european(
        _flat(), S0, K, T, r=R, q=Q, n_paths=200_000, antithetic=False, control_variate=False, seed=7
    )

    assert abs(result.price - analytic) < 3 * result.std_error


def test_control_variate_is_exact_when_the_control_is_the_process():
    # On a flat surface the constant-vol control *is* the simulated
    # diffusion, so every path's correction cancels exactly and the
    # estimator collapses onto Black-Scholes with essentially no variance.
    # The sharpest available check that the control is wired up correctly.
    analytic = bs_price(S0, K, T, R, SIGMA, "call", Q)

    result = price_european(_flat(), S0, K, T, r=R, q=Q, n_paths=20_000, control_variate=True, seed=7)

    assert result.price == pytest.approx(analytic, abs=1e-5)
    assert result.std_error < 1e-4


def test_antithetic_sampling_reduces_the_standard_error():
    kwargs = dict(r=R, q=Q, n_paths=60_000, n_steps=50, control_variate=False, seed=7)

    plain = price_european(_skewed(), S0, K, T, antithetic=False, **kwargs)
    antithetic = price_european(_skewed(), S0, K, T, antithetic=True, **kwargs)

    assert antithetic.std_error < plain.std_error


def test_control_variate_reduces_the_standard_error():
    kwargs = dict(r=R, q=Q, n_paths=60_000, n_steps=50, antithetic=True, seed=7)

    plain = price_european(_skewed(), S0, K, T, control_variate=False, **kwargs)
    controlled = price_european(_skewed(), S0, K, T, control_variate=True, **kwargs)

    assert controlled.std_error < plain.std_error / 2


def test_standard_error_scales_as_one_over_sqrt_paths():
    kwargs = dict(r=R, q=Q, n_steps=50, antithetic=False, control_variate=False, seed=3)

    coarse = price_european(_flat(), S0, K, T, n_paths=50_000, **kwargs)
    fine = price_european(_flat(), S0, K, T, n_paths=200_000, **kwargs)

    # Four times the paths halves the error.
    assert coarse.std_error / fine.std_error == pytest.approx(2.0, abs=0.25)


def test_discounted_spot_is_a_martingale():
    # E[e^{-rT} S_T] == S0 e^{-qT} is the drift term's acceptance check: it
    # fails immediately if the -sigma**2/2 Ito correction is wrong.
    paths = simulate_terminal(_skewed(), S0, T, r=R, q=Q, n_paths=200_000, seed=11)

    discounted = np.exp(-R * T) * paths.spot
    std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))

    assert abs(discounted.mean() - S0 * np.exp(-Q * T)) < 3 * std_error


def test_put_call_parity_holds_across_shared_paths():
    paths = simulate_terminal(_skewed(), S0, T, r=R, q=Q, n_paths=100_000, seed=5)

    call = price_from_terminal(paths, S0, K, T, r=R, q=Q, option_type="call", control_variate=False)
    put = price_from_terminal(paths, S0, K, T, r=R, q=Q, option_type="put", control_variate=False)

    theoretical = S0 * np.exp(-Q * T) - K * np.exp(-R * T)
    assert call.price - put.price == pytest.approx(theoretical, abs=1e-2)


def test_discretization_bias_shrinks_with_more_steps():
    # Freezing local vol over each step biases the price by O(dt). That bias
    # is not Monte Carlo noise and adding paths will not remove it, so it is
    # measured against a finely-stepped reference at a fixed seed.
    kwargs = dict(r=R, q=Q, n_paths=50_000, antithetic=True, control_variate=True, seed=99)
    reference = price_european(_skewed(), S0, K, T, n_steps=500, **kwargs)

    coarse = price_european(_skewed(), S0, K, T, n_steps=5, **kwargs)
    finer = price_european(_skewed(), S0, K, T, n_steps=80, **kwargs)

    assert abs(coarse.price - reference.price) > abs(finer.price - reference.price)


def test_antithetic_paths_mirror_each_other():
    paths = simulate_terminal(_flat(), S0, T, r=R, q=Q, n_paths=1000, antithetic=True, seed=1)

    half = len(paths.brownian) // 2
    np.testing.assert_allclose(paths.brownian[:half], -paths.brownian[half:])


def test_seeding_makes_runs_reproducible():
    kwargs = dict(r=R, q=Q, n_paths=20_000, n_steps=25)

    first = price_european(_skewed(), S0, K, T, seed=42, **kwargs)
    second = price_european(_skewed(), S0, K, T, seed=42, **kwargs)
    different = price_european(_skewed(), S0, K, T, seed=43, **kwargs)

    assert first.price == second.price
    assert first.price != different.price


def test_simulate_terminal_rejects_expired_options():
    with pytest.raises(ValueError, match="T must be > 0"):
        simulate_terminal(_flat(), S0, 0.0)


def test_price_from_terminal_rejects_unknown_option_type():
    paths = simulate_terminal(_flat(), S0, T, n_paths=1000, n_steps=5, seed=1)

    with pytest.raises(ValueError, match="option_type must be one of"):
        price_from_terminal(paths, S0, K, T, option_type="straddle")


def test_confidence_interval_brackets_the_price():
    result = MCResult(price=5.0, std_error=0.1, n_paths=1000, n_steps=50)

    low, high = result.confidence_interval()

    assert low < result.price < high
    assert high - low == pytest.approx(2 * 1.96 * 0.1)


def test_default_steps_is_daily_with_a_floor():
    assert default_steps(1.0) == 252
    assert default_steps(1.0 / 365.0) == 20  # floor, not a two-step path


def test_atm_implied_vol_reads_the_surface_at_the_money():
    surface = _skewed()

    # k = 0 with the skew above: 0.30 + 0.04*T, independent of k.
    assert atm_implied_vol(surface, 0.25) == pytest.approx(0.30 + 0.04 * 0.25)


def _chain_surface(expiry: dt.datetime, strikes=(95.0, 100.0, 105.0)) -> pd.DataFrame:
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


def test_reprice_chain_returns_one_row_per_otm_quote():
    expiry = AS_OF + dt.timedelta(days=182)
    surface = _chain_surface(expiry)

    result = reprice_chain(_flat(), surface, S0, r=R, q=Q, as_of=AS_OF, n_paths=5_000, seed=1)

    assert len(result) == 3
    assert list(result.columns) == REPRICE_COLUMNS
    assert result["mc_iv"].notna().all()
    assert result["iv_std_error"].notna().all()


def test_reprice_chain_recovers_the_surfaces_own_vol():
    # A flat 0.22 local vol surface priced by simulation and inverted must
    # come back at 0.22 implied, whatever the quoted market_iv column says.
    expiry = AS_OF + dt.timedelta(days=182)

    result = reprice_chain(_flat(), _chain_surface(expiry), S0, r=R, q=Q, as_of=AS_OF, n_paths=20_000, seed=4)

    np.testing.assert_allclose(result["mc_iv"].to_numpy(dtype=float), SIGMA, atol=5e-3)


def test_reprice_chain_skips_expiries_outside_the_surface():
    # The surface spans T in [0.05, 1.0]; a two-year expiry has no fitted
    # local vol to reprice against and must be left out rather than clamped.
    far = AS_OF + dt.timedelta(days=730)

    result = reprice_chain(_flat(), _chain_surface(far), S0, r=R, q=Q, as_of=AS_OF, n_paths=2_000, seed=1)

    assert result.empty


def test_reprice_chain_skips_strikes_outside_the_moneyness_window():
    # The surface spans k in [-0.6, 0.6]; a strike at k ~ -1.6 would be
    # priced against the clamped edge of the grid rather than against local
    # vol, so it must be left out rather than scored as a repricing error.
    expiry = AS_OF + dt.timedelta(days=182)
    surface = _chain_surface(expiry, strikes=(20.0, 100.0, 105.0))

    result = reprice_chain(_flat(), surface, S0, r=R, q=Q, as_of=AS_OF, n_paths=5_000, seed=1)

    assert set(result["strike"]) == {100.0, 105.0}
