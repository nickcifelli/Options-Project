"""Tests for surface/svi.py using synthetic smiles generated from known SVI params."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vol_surface.surface.svi import MIN_POINTS, SVIParams, _balance_domain, fit_svi_slice, fit_svi_surface

AS_OF = dt.datetime(2026, 1, 1)
EXPIRY = dt.datetime(2026, 4, 1)
T = 90.0 / 365.0
TRUE_PARAMS = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.05, sigma=0.2)


def test_fit_svi_slice_recovers_known_curve():
    k = np.linspace(-0.4, 0.4, 15)
    iv = TRUE_PARAMS.implied_vol(k, T)

    result = fit_svi_slice(k, iv, T)

    assert result.ok
    fitted_iv = result.params.implied_vol(k, T)
    np.testing.assert_allclose(fitted_iv, iv, atol=1e-3)


def test_fit_svi_slice_fails_with_too_few_points():
    k = np.linspace(-0.1, 0.1, MIN_POINTS - 1)
    iv = TRUE_PARAMS.implied_vol(k, T)

    result = fit_svi_slice(k, iv, T)

    assert not result.ok
    assert "need >=" in result.reason


def test_svi_params_min_total_variance_matches_brute_force_minimum():
    k_grid = np.linspace(-5.0, 5.0, 200_000)
    brute_force_min = TRUE_PARAMS.total_variance(k_grid).min()
    assert TRUE_PARAMS.min_total_variance == pytest.approx(brute_force_min, abs=1e-6)


def test_fit_svi_surface_fits_one_slice_per_otm_expiry():
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
    spot = 100.0
    log_moneyness = np.log(strikes / spot)
    iv = TRUE_PARAMS.implied_vol(log_moneyness, T)

    rows = [
        {
            "expiry": EXPIRY,
            "strike": strike,
            "moneyness": strike / spot,
            "log_moneyness": k,
            "iv": sigma,
            "option_type": "put" if strike < spot else "call",
        }
        for strike, k, sigma in zip(strikes, log_moneyness, iv)
    ]
    surface = pd.DataFrame(rows)

    fits = fit_svi_surface(surface, as_of=AS_OF)

    assert set(fits) == {EXPIRY}
    assert fits[EXPIRY].ok
    np.testing.assert_allclose(fits[EXPIRY].params.implied_vol(log_moneyness, T), iv, atol=1e-3)


def test_fit_svi_surface_excludes_itm_legs():
    # Both call and put quoted at every strike; only the OTM leg of each
    # should reach the fit, so ITM-only garbage IVs must not affect it.
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
    spot = 100.0
    log_moneyness = np.log(strikes / spot)
    good_iv = TRUE_PARAMS.implied_vol(log_moneyness, T)

    rows = []
    for strike, k, otm_iv in zip(strikes, log_moneyness, good_iv):
        otm_type = "put" if strike < spot else "call"
        itm_type = "call" if otm_type == "put" else "put"
        rows.append(
            {"expiry": EXPIRY, "strike": strike, "moneyness": strike / spot, "log_moneyness": k, "iv": otm_iv, "option_type": otm_type}
        )
        rows.append(
            {"expiry": EXPIRY, "strike": strike, "moneyness": strike / spot, "log_moneyness": k, "iv": 5.0, "option_type": itm_type}
        )
    surface = pd.DataFrame(rows)

    fits = fit_svi_surface(surface, as_of=AS_OF)

    assert fits[EXPIRY].ok
    np.testing.assert_allclose(fits[EXPIRY].params.implied_vol(log_moneyness, T), good_iv, atol=1e-3)


def test_balance_domain_truncates_to_the_shorter_wing():
    k = np.array([-0.9, -0.5, -0.2, -0.05, 0.01, 0.03, 0.08])
    iv = np.full_like(k, 0.2)

    k_bal, iv_bal = _balance_domain(k, iv)

    np.testing.assert_array_equal(k_bal, np.array([-0.05, 0.01, 0.03, 0.08]))
    assert len(iv_bal) == len(k_bal)


def test_balance_domain_is_a_noop_when_already_symmetric():
    k = np.linspace(-0.3, 0.3, 7)
    iv = np.full_like(k, 0.2)

    k_bal, iv_bal = _balance_domain(k, iv)

    np.testing.assert_array_equal(k_bal, k)
    np.testing.assert_array_equal(iv_bal, iv)


def test_fit_svi_slice_does_not_flatten_the_short_wing_when_ladder_is_lopsided():
    # A SPY-shaped chain: a deep, densely quoted put wing (real crash-skew
    # richness) against a narrow, shallow call wing. An unweighted fit over
    # the *full* ladder lets the put wing's much larger variance range
    # dominate the objective and drags rho to its -1 boundary, which makes
    # the SVI curve degenerate into a flat line past the minimum -- exactly
    # the flattening this test guards against.
    put_k = np.linspace(-0.9, -0.01, 30)
    put_iv = np.linspace(0.65, 0.17, 30)
    call_k = np.linspace(0.01, 0.2, 12)
    call_iv = 0.13 + 0.6 * (call_k - 0.05) ** 2  # mild dip-then-rise smile

    k = np.concatenate([put_k, call_k])
    iv = np.concatenate([put_iv, call_iv])

    result = fit_svi_slice(k, iv, T)

    assert result.ok
    assert abs(result.params.rho) < 0.95, "rho pinned at its bound is the signature of a flattened wing"

    fitted_call_iv = result.params.implied_vol(call_k, T)
    assert np.ptp(fitted_call_iv) > 0.01, "fitted call wing should curve, not collapse to a flat line"
