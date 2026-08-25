"""Tests for surface/arbitrage.py.

The butterfly and calendar conditions are checked against curves built
from known SVI params -- both curves that satisfy them and curves
deliberately constructed to violate them, since a checker that never
fires is indistinguishable from one that always passes.
"""

import dataclasses
import datetime as dt

import numpy as np
import pytest

from vol_surface.surface.arbitrage import (
    butterfly_g,
    check_butterfly,
    check_calendar,
    risk_neutral_density,
    total_variance_derivatives,
)
from vol_surface.surface.svi import SVIFitResult, SVIParams

AS_OF = dt.datetime(2026, 1, 1)
T = 90.0 / 365.0

# A well-behaved equity-index-shaped smile: downward skew, no arbitrage.
CLEAN = SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.05, sigma=0.2)

# Same family, but with the curvature cranked far past what any real smile
# shows. Total variance stays positive everywhere -- `fit_svi_slice`'s
# constraint would accept this curve -- yet the implied density goes
# negative, which is exactly the gap `check_butterfly` exists to close.
ARBITRAGEABLE = SVIParams(a=0.01, b=1.2, rho=-0.95, m=0.0, sigma=0.02)


def _fits(params_by_expiry, k_range=(-0.4, 0.4)):
    return {expiry: SVIFitResult(params, k_range=k_range) for expiry, params in params_by_expiry.items()}


def _flat_term_structure(vol=0.2, days=(30, 60, 90)):
    """Constant-vol slices: w = vol**2 * T, strictly increasing in T."""
    return {AS_OF + dt.timedelta(days=d): SVIParams(a=vol**2 * d / 365.0, b=0.0, rho=0.0, m=0.0, sigma=0.1) for d in days}


def test_total_variance_derivatives_match_finite_differences():
    k = np.linspace(-0.6, 0.6, 25)
    _, dw, d2w = total_variance_derivatives(CLEAN, k)

    h = 1e-5
    np.testing.assert_allclose(dw, (CLEAN.total_variance(k + h) - CLEAN.total_variance(k - h)) / (2 * h), atol=1e-8)
    expected_d2w = (CLEAN.total_variance(k + h) - 2 * CLEAN.total_variance(k) + CLEAN.total_variance(k - h)) / h**2
    np.testing.assert_allclose(d2w, expected_d2w, atol=1e-5)


def test_total_variance_derivatives_agree_with_svi_params():
    k = np.linspace(-0.6, 0.6, 25)
    w, _, _ = total_variance_derivatives(CLEAN, k)
    np.testing.assert_allclose(w, CLEAN.total_variance(k))


def test_butterfly_g_is_the_dupire_denominator():
    # Gatheral's g(k) and the denominator of the Dupire local variance
    # (The Volatility Surface, eq. 1.10) are the same expression written
    # two ways. surface/local_vol.py divides by `butterfly_g` on the
    # strength of this identity, so it is asserted rather than assumed.
    k = np.linspace(-0.6, 0.6, 50)
    w, dw, d2w = total_variance_derivatives(CLEAN, k)

    dupire_denominator = 1 - (k / w) * dw + 0.25 * (-0.25 - 1 / w + k**2 / w**2) * dw**2 + 0.5 * d2w

    np.testing.assert_allclose(butterfly_g(CLEAN, k), dupire_denominator, atol=1e-14)


def test_risk_neutral_density_integrates_to_one():
    k = np.linspace(-12.0, 12.0, 400_001)
    assert np.trapezoid(risk_neutral_density(CLEAN, k), k) == pytest.approx(1.0, abs=1e-6)


def test_density_is_negative_exactly_where_g_is():
    k = np.linspace(-1.0, 1.0, 2001)
    np.testing.assert_array_equal(butterfly_g(ARBITRAGEABLE, k) < 0, risk_neutral_density(ARBITRAGEABLE, k) < 0)


def test_butterfly_condition_is_stronger_than_non_negative_variance():
    # The premise of this module: `fit_svi_slice` constrains total
    # variance to stay non-negative, and a curve can clear that bar while
    # still implying a negative density.
    assert ARBITRAGEABLE.min_total_variance > 0
    assert butterfly_g(ARBITRAGEABLE, np.linspace(-1.0, 1.0, 2001)).min() < 0


def test_butterfly_g_is_invariant_to_the_forward_shift():
    # Evaluating a spot-coordinate fit at `k + drift` must equal shifting
    # the curve's own `m` by the same drift -- the check that the forward
    # conversion moves the curve rather than distorting it.
    k = np.linspace(-0.5, 0.5, 50)
    drift = 0.04 * T

    shifted = dataclasses.replace(CLEAN, m=CLEAN.m - drift)

    np.testing.assert_allclose(butterfly_g(CLEAN, k, drift=drift), butterfly_g(shifted, k, drift=0.0), atol=1e-14)


def test_check_butterfly_passes_a_well_behaved_smile():
    result = check_butterfly(_fits({AS_OF + dt.timedelta(days=90): CLEAN}), as_of=AS_OF)

    assert len(result) == 1
    assert not result["flagged"].any()
    assert result["n_violations"].iloc[0] == 0
    assert result["min_g"].iloc[0] > 0


def test_check_butterfly_flags_a_negative_density_slice():
    result = check_butterfly(_fits({AS_OF + dt.timedelta(days=90): ARBITRAGEABLE}), as_of=AS_OF)

    assert result["flagged"].all()
    assert result["n_violations"].iloc[0] > 0
    assert result["min_g"].iloc[0] < 0
    assert abs(result["k_at_min_g"].iloc[0]) <= 0.4  # reported inside the fitted window


def test_check_butterfly_skips_unconverged_and_expired_slices():
    fits = {
        AS_OF + dt.timedelta(days=90): SVIFitResult(CLEAN, k_range=(-0.4, 0.4)),
        AS_OF + dt.timedelta(days=45): SVIFitResult(None, reason="did not converge"),
        AS_OF - dt.timedelta(days=10): SVIFitResult(CLEAN, k_range=(-0.4, 0.4)),
    }

    result = check_butterfly(fits, as_of=AS_OF)

    assert len(result) == 1


def test_check_calendar_passes_a_monotone_term_structure():
    result = check_calendar(_fits(_flat_term_structure()), as_of=AS_OF)

    assert len(result) == 2  # three expiries -> two adjacent pairs
    assert not result["flagged"].any()
    assert (result["max_variance_drop"] < 0).all()


def test_check_calendar_flags_a_total_variance_drop():
    # The 60-day slice carries more total variance than the 90-day one --
    # a calendar spread between them is free money, and must be flagged.
    slices = _flat_term_structure(days=(30, 60))
    slices[AS_OF + dt.timedelta(days=90)] = SVIParams(a=0.005, b=0.0, rho=0.0, m=0.0, sigma=0.1)

    result = check_calendar(_fits(slices), as_of=AS_OF)

    flagged = result[result["flagged"]]
    assert len(flagged) == 1
    assert flagged["far_expiry"].iloc[0] == AS_OF + dt.timedelta(days=90)
    assert flagged["max_variance_drop"].iloc[0] > 0


def test_check_calendar_reports_pairs_in_expiry_order():
    result = check_calendar(_fits(_flat_term_structure(days=(90, 30, 60))), as_of=AS_OF)

    assert list(result["near_expiry"]) == [AS_OF + dt.timedelta(days=30), AS_OF + dt.timedelta(days=60)]
    assert list(result["far_expiry"]) == [AS_OF + dt.timedelta(days=60), AS_OF + dt.timedelta(days=90)]


def test_check_calendar_skips_pairs_with_no_common_window():
    slices = _flat_term_structure(days=(30, 60))
    fits = {expiry: SVIFitResult(p, k_range=(-0.4, 0.4)) for expiry, p in slices.items()}
    fits[AS_OF + dt.timedelta(days=60)] = dataclasses.replace(fits[AS_OF + dt.timedelta(days=60)], k_range=(2.0, 3.0))

    assert check_calendar(fits, as_of=AS_OF).empty


def test_check_calendar_needs_at_least_two_slices():
    assert check_calendar(_fits({AS_OF + dt.timedelta(days=30): CLEAN}), as_of=AS_OF).empty
