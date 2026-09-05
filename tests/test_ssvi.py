"""Tests for surface/ssvi.py.

Two things have to be true for the global fit to be worth its lost
flexibility, and they are tested separately. It must *be* SSVI -- the
reparameterization into raw SVI is exact, so everything downstream is
entitled to treat a slice of it as an ordinary SVI slice. And it must be
arbitrage-free *by construction*, which is checked by feeding it a term
structure that is calendar-arbitrageable on its face and asserting that
the unmodified `check_calendar` finds nothing in the result.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vol_surface.surface.arbitrage import CALENDAR_TOL, check_butterfly, check_calendar
from vol_surface.surface.local_vol import build_local_vol_surface
from vol_surface.surface.ssvi import (
    GAMMA_BOUNDS,
    MIN_SLICES,
    SSVIParams,
    fit_ssvi_surface,
)
from vol_surface.surface.svi import fit_svi_surface

AS_OF = dt.datetime(2026, 1, 1)
SPOT = 100.0
R, Q = 0.04, 0.0
DAYS = (30, 60, 90, 180, 365)
TRUTH = SSVIParams(rho=-0.45, eta=1.1, gamma=0.3)


def _surface(atm_vol_by_day: dict[int, float], params: SSVIParams = TRUTH, strikes=None) -> pd.DataFrame:
    """A synthetic chain generated from known SSVI params.

    `atm_vol_by_day` sets the at-the-money vol per expiry, which is the
    only per-expiry freedom SSVI has -- so a surface built this way is
    exactly representable, and anything the fit fails to recover is the
    optimizer's doing rather than the model's.
    """
    strikes = np.linspace(70, 130, 25) if strikes is None else np.asarray(strikes, dtype=float)
    rows = []
    for days, atm_vol in atm_vol_by_day.items():
        expiry = AS_OF + dt.timedelta(days=days)
        T = days / 365.0
        theta = atm_vol**2 * T
        spot_k = np.log(strikes / SPOT)
        # SSVI is a statement about forward log-moneyness, so the truth is
        # generated there; the surface itself carries spot log-moneyness,
        # which is the column `build_surface` produces.
        iv = np.sqrt(np.asarray(params.total_variance(spot_k - (R - Q) * T, theta), dtype=float) / T)
        for strike, log_moneyness, sigma in zip(strikes, spot_k, iv):
            rows.append(
                {
                    "expiry": expiry,
                    "strike": strike,
                    "moneyness": strike / SPOT,
                    "log_moneyness": log_moneyness,
                    "iv": sigma,
                    "option_type": "put" if strike < SPOT else "call",
                }
            )
    return pd.DataFrame(rows)


FLAT_TERM_STRUCTURE = {days: 0.20 for days in DAYS}

# Total variance that *falls* with maturity: w = 0.30**2*(30/365) = 0.0074
# at the front against 0.10**2*(365/365) = 0.0100 -- no, the front is
# cheaper here, so the arbitrage is engineered explicitly below.
INVERTED_TERM_STRUCTURE = {30: 0.60, 60: 0.40, 90: 0.30, 180: 0.20, 365: 0.13}


def test_slice_params_reproduce_ssvi_exactly():
    # The whole design rests on an SSVI slice *being* a raw SVI slice, so
    # this is held to machine precision rather than to a tolerance.
    k = np.linspace(-0.8, 0.8, 41)
    for theta in (0.005, 0.04, 0.25, 1.0):
        np.testing.assert_allclose(
            TRUTH.slice_params(theta).total_variance(k),
            TRUTH.total_variance(k, theta),
            rtol=0,
            atol=1e-14,
        )


def test_slice_params_keep_total_variance_positive():
    # SSVI's minimum total variance is theta*(1 - rho**2), which is what
    # the raw-SVI slice's own `min_total_variance` must reproduce.
    for theta in (0.01, 0.1, 0.5):
        assert TRUTH.slice_params(theta).min_total_variance == pytest.approx(theta * (1 - TRUTH.rho**2))


def test_atm_total_variance_is_theta():
    for theta in (0.01, 0.1, 0.5):
        assert TRUTH.total_variance(0.0, theta) == pytest.approx(theta)


def test_atm_skew_is_theta_rho_phi():
    # dw/dk at k=0 works out to theta*rho*phi(theta), which is the identity
    # `_initial_shape` inverts to read the power law off the data.
    theta, h = 0.04, 1e-6
    slope = (TRUTH.total_variance(h, theta) - TRUTH.total_variance(-h, theta)) / (2 * h)
    assert slope == pytest.approx(theta * TRUTH.rho * TRUTH.phi(theta), rel=1e-6)


def test_fit_recovers_known_ssvi_params():
    fit = fit_ssvi_surface(_surface(FLAT_TERM_STRUCTURE), r=R, q=Q, as_of=AS_OF)

    assert fit.ok, fit.reason
    assert fit.params.rho == pytest.approx(TRUTH.rho, abs=1e-3)
    assert fit.params.eta == pytest.approx(TRUTH.eta, abs=1e-3)
    assert fit.params.gamma == pytest.approx(TRUTH.gamma, abs=1e-3)
    assert fit.rmse_vol < 1e-4


def test_fit_recovers_the_atm_variance_term_structure():
    fit = fit_ssvi_surface(_surface(FLAT_TERM_STRUCTURE), r=R, q=Q, as_of=AS_OF)

    for days in DAYS:
        expiry = pd.Timestamp(AS_OF + dt.timedelta(days=days))
        assert fit.theta[expiry] == pytest.approx(0.20**2 * days / 365.0, rel=1e-3)


def test_fit_uses_three_shared_parameters_plus_one_per_expiry():
    fit = fit_ssvi_surface(_surface(FLAT_TERM_STRUCTURE), r=R, q=Q, as_of=AS_OF)

    assert fit.n_parameters == 3 + len(DAYS)
    assert fit.n_parameters < 5 * len(DAYS)  # the point of the exercise


def test_theta_is_non_decreasing_even_when_the_market_is_not():
    # An inverted term structure: front-month total variance sitting above
    # the back. Independent per-expiry fits reproduce it faithfully, which
    # is precisely the calendar arbitrage. SSVI cannot represent it at all.
    surface = _surface(INVERTED_TERM_STRUCTURE)
    fit = fit_ssvi_surface(surface, r=R, q=Q, as_of=AS_OF)

    assert fit.ok, fit.reason
    theta = [fit.theta[e] for e in sorted(fit.theta)]
    assert np.all(np.diff(theta) >= 0), theta


def test_the_same_calendar_checker_flags_the_independent_fit_and_clears_ssvi():
    # The comparison that justifies the module: same quotes, same checker,
    # different fit. Nothing about `check_calendar` is relaxed for SSVI.
    surface = _surface(INVERTED_TERM_STRUCTURE)

    independent = check_calendar(fit_svi_surface(surface, as_of=AS_OF), r=R, q=Q, as_of=AS_OF)
    coupled = check_calendar(fit_ssvi_surface(surface, r=R, q=Q, as_of=AS_OF).slices, r=R, q=Q, as_of=AS_OF)

    assert independent["flagged"].sum() > 0, "the fixture is supposed to be arbitrageable"
    assert coupled["flagged"].sum() == 0
    assert coupled["max_variance_drop"].max() <= CALENDAR_TOL


def test_fitted_surface_is_butterfly_clean():
    fit = fit_ssvi_surface(_surface(FLAT_TERM_STRUCTURE), r=R, q=Q, as_of=AS_OF)

    butterfly = check_butterfly(fit.slices, r=R, q=Q, as_of=AS_OF)
    assert butterfly["flagged"].sum() == 0
    assert butterfly["min_g"].min() > 0


def test_fit_respects_the_butterfly_constraints_it_was_given():
    fit = fit_ssvi_surface(_surface(FLAT_TERM_STRUCTURE), r=R, q=Q, as_of=AS_OF)

    for theta in fit.theta.values():
        assert min(fit.params.butterfly_margins(theta)) >= 0


def test_butterfly_quantities_are_non_decreasing_in_theta():
    # Why constraining at the largest fitted theta covers the whole
    # surface: both quantities in Theorem 4.2 rise with theta when
    # gamma <= 1/2, so their suprema sit at the longest expiry.
    theta = np.geomspace(1e-4, 1.0, 200)
    for gamma in (GAMMA_BOUNDS[0], 0.25, GAMMA_BOUNDS[1]):
        params = SSVIParams(rho=-0.5, eta=1.0, gamma=gamma)
        margins = np.array([params.butterfly_margins(t) for t in theta])
        assert np.all(np.diff(margins, axis=0) <= 1e-12), gamma  # margin falls => quantity rises


def test_calendar_margin_is_positive_for_every_admissible_parameter_pair():
    # The derivation in the module docstring, held to account numerically:
    # with the power law, Theorem 4.1's second condition is slack for every
    # (rho, gamma) the fit can reach, which is why monotone theta is the
    # whole calendar condition.
    for rho in np.linspace(-0.999, 0.999, 41):
        for gamma in np.linspace(*GAMMA_BOUNDS, 11):
            assert SSVIParams(rho=rho, eta=1.0, gamma=gamma).calendar_margin > 0


def test_k_window_union_widens_every_slice_to_the_widest_ladder():
    # A front expiry quoted over a narrow ladder against back months quoted
    # much deeper -- the SPY shape that caps the local vol window.
    narrow = np.linspace(95, 105, 9)
    wide = np.linspace(60, 140, 25)
    frames = [_surface({30: 0.20}, strikes=narrow)]
    frames += [_surface({days: 0.20}, strikes=wide) for days in (90, 180, 365)]
    surface = pd.concat(frames, ignore_index=True)

    per_slice = fit_ssvi_surface(surface, r=R, q=Q, as_of=AS_OF)
    union = fit_ssvi_surface(surface, r=R, q=Q, as_of=AS_OF, k_window="union")

    front = pd.Timestamp(AS_OF + dt.timedelta(days=30))
    assert per_slice.slices[front].k_range[0] > np.log(0.9)  # its own narrow ladder

    # The windows are balanced around the forward before the union is
    # taken, so the widest is the back months' shorter (call-side) wing,
    # not their deepest listed put.
    widest = max(np.ptp(fit.k_range) for fit in union.slices.values())
    assert all(np.ptp(fit.k_range) == pytest.approx(widest) for fit in union.slices.values())
    assert np.ptp(union.slices[front].k_range) > 4 * np.ptp(per_slice.slices[front].k_range)


def test_union_window_widens_the_local_vol_surface():
    narrow = np.linspace(95, 105, 9)
    wide = np.linspace(60, 140, 25)
    surface = pd.concat(
        [_surface({30: 0.20}, strikes=narrow)] + [_surface({days: 0.20}, strikes=wide) for days in (90, 180, 365)],
        ignore_index=True,
    )

    def width(k_window):
        fits = fit_ssvi_surface(surface, r=R, q=Q, as_of=AS_OF, k_window=k_window).slices
        surface_lv = build_local_vol_surface(fits, r=R, q=Q, as_of=AS_OF)
        return float(np.ptp(surface_lv.k))

    assert width("union") > 3 * width("slice")


def test_fit_rejects_an_unknown_k_window():
    with pytest.raises(ValueError, match="k_window must be one of"):
        fit_ssvi_surface(_surface(FLAT_TERM_STRUCTURE), r=R, q=Q, as_of=AS_OF, k_window="everything")


def test_fit_explains_itself_instead_of_raising_on_too_few_expiries():
    fit = fit_ssvi_surface(_surface({30: 0.20}), r=R, q=Q, as_of=AS_OF)

    assert not fit.ok
    assert f"need >= {MIN_SLICES}" in fit.reason
    assert fit.slices == {}


def test_fit_survives_noisy_quotes_and_stays_arbitrage_free():
    rng = np.random.default_rng(0)
    surface = _surface(FLAT_TERM_STRUCTURE)
    surface["iv"] += rng.normal(0, 0.01, len(surface))

    fit = fit_ssvi_surface(surface, r=R, q=Q, as_of=AS_OF)

    assert fit.ok, fit.reason
    assert check_calendar(fit.slices, r=R, q=Q, as_of=AS_OF)["flagged"].sum() == 0
    assert check_butterfly(fit.slices, r=R, q=Q, as_of=AS_OF)["flagged"].sum() == 0
