"""Tests for surface/local_vol.py.

Dupire local vol has two properties that pin it down without needing a
reference implementation to compare against: it collapses to the constant
vol of a flat surface, and near the money it is roughly twice as skewed as
the implied vol it came from. Both are asserted here on synthetic surfaces
built from known SVI params.
"""

import dataclasses
import datetime as dt

import numpy as np
import pytest

from vol_surface.surface.local_vol import MIN_EXPIRY_GAP, build_local_vol_surface
from vol_surface.surface.svi import SVIFitResult, SVIParams

AS_OF = dt.datetime(2026, 1, 1)
DAYS = (30, 60, 90, 180, 365)


def _fits(params_by_expiry, k_range=(-0.4, 0.4)):
    return {expiry: SVIFitResult(params, k_range=k_range) for expiry, params in params_by_expiry.items()}


def _flat(vol=0.22, days=DAYS):
    """Constant vol, no smile: w = vol**2 * T with b = 0."""
    return {AS_OF + dt.timedelta(days=d): SVIParams(a=vol**2 * d / 365.0, b=0.0, rho=0.0, m=0.0, sigma=0.1) for d in days}


def _skewed(days=DAYS):
    """A downward-skewed surface whose total variance scales linearly in T."""
    return {
        AS_OF + dt.timedelta(days=d): SVIParams(a=0.04 * d / 365.0, b=0.10 * d / 365.0, rho=-0.7, m=0.0, sigma=0.15)
        for d in days
    }


def _atm_slope(k, values, width=0.05):
    near_money = np.abs(k) < width
    return float(np.polyfit(k[near_money], values[near_money], 1)[0])


def test_flat_surface_round_trips_to_its_constant_vol():
    # The one case with a known closed-form answer: no smile and no term
    # structure beyond w = sigma**2 * T means local vol == implied vol ==
    # the constant that generated the surface, everywhere.
    lv = build_local_vol_surface(_fits(_flat(vol=0.22)), r=0.0, q=0.0, as_of=AS_OF)

    assert lv.coverage == 1.0
    np.testing.assert_allclose(lv.local_vol, 0.22, atol=1e-10)
    np.testing.assert_allclose(lv.implied_vol, 0.22, atol=1e-10)


def test_local_vol_skew_is_about_twice_the_implied_skew_near_the_money():
    # Derman's rule of thumb, and the standard sanity check on any local
    # vol implementation: for a near-linear smile the local vol skew is
    # about double the implied vol skew around the forward.
    lv = build_local_vol_surface(_fits(_skewed()), r=0.0, q=0.0, as_of=AS_OF, n_k=401)

    front = int(np.argmin(lv.T))
    ratio = _atm_slope(lv.k, lv.local_vol[front]) / _atm_slope(lv.k, lv.implied_vol[front])

    assert ratio == pytest.approx(2.0, abs=0.25)


def test_derman_ratio_decays_with_maturity():
    # The 2x rule is a short-maturity approximation, so the ratio should
    # sit near 2 at the front and fall away with T rather than holding
    # flat -- a check that dw/dT is actually entering the result.
    lv = build_local_vol_surface(_fits(_skewed()), r=0.0, q=0.0, as_of=AS_OF, n_k=401)

    ratios = [_atm_slope(lv.k, lv.local_vol[i]) / _atm_slope(lv.k, lv.implied_vol[i]) for i in np.argsort(lv.T)]

    assert ratios[0] > ratios[-1]
    assert all(earlier >= later for earlier, later in zip(ratios, ratios[1:]))


def test_implied_vol_grid_matches_the_fitted_slices():
    fits = _fits(_skewed())
    lv = build_local_vol_surface(fits, r=0.0, q=0.0, as_of=AS_OF)

    for expiry, fit in fits.items():
        T = (expiry - AS_OF).days / 365.0
        row = int(np.argmin(np.abs(lv.T - T)))
        np.testing.assert_allclose(lv.implied_vol[row], fit.params.implied_vol(lv.k, T), atol=1e-12)


def test_grid_is_the_intersection_of_the_fitted_windows():
    fits = _fits(_flat())
    narrow = AS_OF + dt.timedelta(days=60)
    fits[narrow] = dataclasses.replace(fits[narrow], k_range=(-0.1, 0.2))

    lv = build_local_vol_surface(fits, r=0.0, q=0.0, as_of=AS_OF)

    assert lv.k.min() == pytest.approx(-0.1)
    assert lv.k.max() == pytest.approx(0.2)


def test_forward_shift_moves_the_grid_off_spot_moneyness():
    # With a non-zero drift the common window is measured against the
    # forward, so the grid shifts left relative to the spot-coordinate fits.
    fits = _fits(_flat())

    spot_grid = build_local_vol_surface(fits, r=0.0, q=0.0, as_of=AS_OF).k
    forward_grid = build_local_vol_surface(fits, r=0.05, q=0.0, as_of=AS_OF).k

    assert forward_grid.max() < spot_grid.max()


def test_calendar_arbitrage_is_masked_rather_than_square_rooted():
    # Total variance that falls with T makes dw/dT negative, so local
    # variance is negative and local vol does not exist. Those points must
    # come back as NaN, not as a warning-laden sqrt of a negative number.
    slices = _flat(days=(30, 60))
    slices[AS_OF + dt.timedelta(days=90)] = SVIParams(a=0.001, b=0.0, rho=0.0, m=0.0, sigma=0.1)

    lv = build_local_vol_surface(_fits(slices), r=0.0, q=0.0, as_of=AS_OF)

    assert lv.coverage < 1.0
    assert np.isnan(lv.local_vol).any()


def test_needs_at_least_two_slices_to_difference_in_time():
    with pytest.raises(ValueError, match="need >= 2 converged SVI slices"):
        build_local_vol_surface(_fits(_flat(days=(30,))), as_of=AS_OF)


def test_unconverged_slices_do_not_count_toward_the_minimum():
    fits = _fits(_flat(days=(30,)))
    fits[AS_OF + dt.timedelta(days=60)] = SVIFitResult(None, reason="did not converge")

    with pytest.raises(ValueError, match="need >= 2 converged SVI slices"):
        build_local_vol_surface(fits, as_of=AS_OF)


def test_raises_when_fitted_windows_do_not_overlap():
    fits = _fits(_flat(days=(30, 60)))
    disjoint = AS_OF + dt.timedelta(days=60)
    fits[disjoint] = dataclasses.replace(fits[disjoint], k_range=(2.0, 3.0))

    with pytest.raises(ValueError, match="no common log-moneyness window"):
        build_local_vol_surface(fits, as_of=AS_OF)


def test_surface_shapes_line_up():
    lv = build_local_vol_surface(_fits(_skewed()), r=0.0, q=0.0, as_of=AS_OF, n_k=64)

    assert lv.local_vol.shape == (len(DAYS), 64)
    assert lv.implied_vol.shape == lv.local_vol.shape
    assert lv.k.shape == (64,)
    assert np.all(np.diff(lv.T) > 0)


def test_slices_closer_than_the_gap_are_dropped_from_the_time_derivative():
    # A daily-expiry front (30, 31, 32) collapses to a single slice at the
    # default one-week gap, leaving the weekly ladder behind it intact.
    lv = build_local_vol_surface(_fits(_flat(days=(30, 31, 32, 60, 90))), r=0.0, q=0.0, as_of=AS_OF)

    np.testing.assert_allclose(lv.T * 365, [30, 60, 90], atol=1e-9)


def test_gap_thinning_can_be_disabled():
    lv = build_local_vol_surface(_fits(_flat(days=(30, 31, 32))), r=0.0, q=0.0, as_of=AS_OF, min_expiry_gap=0.0)

    assert len(lv.T) == 3


def test_gap_thinning_falls_back_rather_than_starving_the_difference():
    # Every expiry inside one week: thinning would leave a single slice and
    # no dw/dT at all, so the full ladder is kept and the caller is left to
    # judge the result from `coverage` rather than handed an exception.
    lv = build_local_vol_surface(_fits(_flat(days=(30, 31, 32))), r=0.0, q=0.0, as_of=AS_OF)

    assert len(lv.T) == 3
    assert MIN_EXPIRY_GAP == pytest.approx(7.0 / 365.0)


def test_front_slice_derivative_is_anchored_at_zero_total_variance():
    # w(k, 0) = 0 is known exactly, so the front slice's dw/dT must be a
    # difference against the origin rather than np.gradient's backward
    # extrapolation from the next two expiries. Total variance rising
    # steeply from 0 to the first expiry and then flattening is where the
    # two disagree: the unanchored rule would read the flat part.
    slices = {
        AS_OF + dt.timedelta(days=d): SVIParams(a=a, b=0.0, rho=0.0, m=0.0, sigma=0.1)
        for d, a in ((30, 0.02), (60, 0.021), (90, 0.022))
    }
    lv = build_local_vol_surface(_fits(slices), r=0.0, q=0.0, as_of=AS_OF)

    front_local_variance = lv.local_vol[0] ** 2
    anchored = 0.02 / (30 / 365)  # w0 / T0, the average over [0, T0]
    unanchored = (0.021 - 0.02) / (30 / 365)  # what the flat tail alone implies

    assert abs(front_local_variance[0] - anchored) < abs(front_local_variance[0] - unanchored)
