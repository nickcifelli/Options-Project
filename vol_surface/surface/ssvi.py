"""SSVI: one global surface fit that couples the expiry slices.

`surface/svi.py` fits each expiry independently. That is the standard way
to get a smooth smile, and it is also the reason `surface/arbitrage.py`
reports calendar violations: five free parameters per slice with nothing
tying them together in `T` means nothing stops a near slice's total
variance from crossing above a far one's. On a live 21-expiry SPY chain,
7 of 20 adjacent pairs crossed.

SSVI (Gatheral & Jacquier, *Arbitrage-free SVI volatility surfaces*, 2014)
removes the freedom that allows it. Total variance is written as a single
function of log-moneyness and the at-the-money total variance `theta_T`:

    w(k, theta) = (theta/2) * (1 + rho*phi(theta)*k
                               + sqrt((phi(theta)*k + rho)**2 + 1 - rho**2))

`rho` and the function `phi` are shared by *every* expiry; the only thing
an expiry owns is its `theta_T`. A 20-expiry chain goes from 100 free
parameters to 23 -- `rho`, the two parameters of `phi`, and one `theta`
per expiry -- and the term structure is now a property of the surface
rather than an accident of 20 separate optimizations.

**Every SSVI slice is a raw SVI slice.** Expanding the square root above
and matching coefficients against `a + b*(rho*(k-m) + sqrt((k-m)**2 +
sigma**2))` gives an exact reparameterization (`SSVIParams.slice_params`):

    a = theta*(1 - rho**2)/2    b = theta*phi/2
    m = -rho/phi                sigma = sqrt(1 - rho**2)/phi

so this module hands the rest of the project the same `SVIFitResult`
objects `fit_svi_surface` does. The arbitrage checks, the Dupire local vol
construction, the Monte Carlo repricing and the plots all run unchanged --
which is the point, because it means the calendar checker that flagged the
independent fit is the *same unmodified code* that certifies this one.

**Why the no-arbitrage conditions become constraints on three numbers.**
Gatheral and Jacquier state both conditions on the SSVI family directly.
With the power law `phi(theta) = eta * theta**-gamma`, `gamma` in
`(0, 1/2]`, they collapse to something an optimizer can carry:

*Calendar* (their Theorem 4.1) is `d(theta)/dt >= 0` together with

    0 <= d/dtheta (theta*phi(theta)) <= (1 + sqrt(1-rho**2))/rho**2 * phi(theta)

Under the power law `theta*phi = eta*theta**(1-gamma)`, so the middle term
is `(1-gamma)*phi(theta)`, and `phi > 0` divides out of the whole chain:
the condition is `0 <= 1 - gamma <= (1 + sqrt(1-rho**2))/rho**2`. The right
side is decreasing in `rho**2` with minimum 1 at `rho**2 = 1`, and
`1 - gamma < 1`, so it holds identically -- for every `theta`, not just the
fitted ones. **Calendar arbitrage-freedom is therefore exactly the
statement that `theta` is non-decreasing**, which `_theta_from_increments`
imposes by construction rather than by penalty. There is no residual
condition left to check, and none to trade off against fit quality.

*Butterfly* (their Theorem 4.2) is sufficient, not necessary:

    theta*phi(theta) * (1 + |rho|) < 4        theta*phi(theta)**2 * (1 + |rho|) <= 4

Under the power law these are `eta*theta**(1-gamma)` and
`eta**2*theta**(1-2*gamma)`, both non-decreasing in `theta` for
`gamma <= 1/2`, so their suprema over the fitted term structure sit at the
longest expiry and two scalar constraints cover the whole surface. Unlike
the calendar condition these are *not* automatic -- the power law breaks
them at large enough `theta` -- which is why they are imposed on the
optimizer instead of assumed, and why `check_butterfly` is still worth
running afterwards.

**What it costs.** Three shared parameters cannot follow a per-expiry
smile as closely as five free ones can. The fit is worse by construction,
and `SSVIFit.rmse_vol` reports how much worse in vol points so the trade
is measured rather than argued. The interesting question is not which fits
the quotes better -- the independent fit always will -- but which reprices
them better once the local vol surface is built on top, since the
independent fit pays for its closeness with slices that cross and a local
vol that fails to exist where they do.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, minimize

from vol_surface.surface.build import year_fraction
from vol_surface.surface.svi import SVIFitResult, SVIParams, _balance_domain, otm_quotes

# theta_T alone is one parameter per expiry, so a slice earns its place in
# the global fit with far fewer points than an independent 5-parameter fit
# would need. Two still pins a level and a slope through the shared shape.
MIN_SLICE_POINTS = 3
MIN_SLICES = 2  # a "surface" fit over one expiry is just a slice fit

# Gatheral-Jacquier Theorem 4.2 bounds `theta*phi*(1+|rho|)` by 4 strictly.
# Fitting to the boundary would leave the surface exactly marginal, so the
# constraint is imposed with a small margin -- and `check_butterfly` then
# verifies the density numerically rather than trusting the margin.
BUTTERFLY_BOUND = 4.0
BUTTERFLY_MARGIN = 1e-6

# gamma <= 1/2 is what makes both butterfly quantities non-decreasing in
# theta, so their suprema sit at the longest expiry (see module docstring).
GAMMA_BOUNDS = (0.01, 0.5)
ETA_BOUNDS = (1e-4, 50.0)
RHO_BOUND = 0.999

K_WINDOWS = ("slice", "union")


@dataclass(frozen=True)
class SSVIParams:
    """The three numbers shared by every expiry on the surface.

    `rho` is the skew and `phi(theta) = eta * theta**-gamma` the power law
    controlling how the smile flattens as total variance grows. An expiry
    contributes only its at-the-money total variance `theta`, which is why
    these three plus one number per expiry are the whole surface.
    """

    rho: float
    eta: float
    gamma: float

    def phi(self, theta: np.ndarray | float) -> np.ndarray | float:
        return self.eta * np.asarray(theta, dtype=float) ** -self.gamma

    def total_variance(self, k: np.ndarray | float, theta: float) -> np.ndarray | float:
        k = np.asarray(k, dtype=float)
        phi_k = self.phi(theta) * k
        return 0.5 * theta * (1 + self.rho * phi_k + np.sqrt((phi_k + self.rho) ** 2 + 1 - self.rho**2))

    def slice_params(self, theta: float) -> SVIParams:
        """The exact raw-SVI parameters of the slice at `theta`.

        SSVI is a subfamily of raw SVI, not an alternative to it -- the map
        is a coefficient match, not an approximation, and it is what lets
        every downstream module take an SSVI surface without knowing it did.
        `test_ssvi.py` pins the two forms together to machine precision.
        """
        phi = float(self.phi(theta))
        return SVIParams(
            a=0.5 * theta * (1 - self.rho**2),
            b=0.5 * theta * phi,
            rho=self.rho,
            m=-self.rho / phi,
            sigma=np.sqrt(1 - self.rho**2) / phi,
        )

    def butterfly_margins(self, theta: float) -> tuple[float, float]:
        """Slack in Gatheral-Jacquier Theorem 4.2's two conditions at `theta`.

        Both are `4 - (quantity)`, so non-negative means satisfied. Both
        quantities are non-decreasing in `theta` for `gamma <= 1/2`, so
        evaluating at the largest fitted `theta` bounds the whole surface.
        """
        skew = 1 + abs(self.rho)
        return (
            BUTTERFLY_BOUND - self.eta * theta ** (1 - self.gamma) * skew,
            BUTTERFLY_BOUND - self.eta**2 * theta ** (1 - 2 * self.gamma) * skew,
        )

    @property
    def calendar_margin(self) -> float:
        """Slack in the non-trivial half of Theorem 4.1's second condition.

        `(1 + sqrt(1-rho**2))/rho**2 - (1 - gamma)`, which the module
        docstring shows is positive for every admissible `(rho, gamma)`.
        Computed rather than asserted so the test suite can hold the
        derivation to account instead of restating it.
        """
        rho_sq = self.rho**2
        if rho_sq == 0:
            return np.inf
        return (1 + np.sqrt(1 - rho_sq)) / rho_sq - (1 - self.gamma)


@dataclass(frozen=True)
class SSVIFit:
    """A fitted SSVI surface, plus the per-expiry slices it projects to.

    `slices` is the drop-in for `fit_svi_surface`'s return value: one
    `SVIFitResult` per expiry, carrying the exact raw-SVI parameters of
    that slice. `theta` is the fitted at-the-money total variance term
    structure, non-decreasing by construction. `rmse_vol` is the residual
    in vol points across every quote the fit saw, directly comparable to
    the same figure for an independent per-expiry fit.
    """

    params: SSVIParams | None
    theta: dict[pd.Timestamp, float] = field(default_factory=dict)
    slices: dict[pd.Timestamp, SVIFitResult] = field(default_factory=dict)
    rmse_vol: float = float("nan")
    n_quotes: int = 0
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.params is not None

    @property
    def n_parameters(self) -> int:
        """3 shared + 1 per expiry, against 5 per expiry fit independently."""
        return 3 + len(self.theta)


def _theta_from_increments(increments: np.ndarray) -> np.ndarray:
    """Non-negative increments -> a non-decreasing `theta` term structure.

    Monotone `theta` *is* the calendar no-arbitrage condition for this
    family (module docstring), so expressing it as a cumulative sum of
    bounded-below variables hands it to the optimizer as a box constraint
    rather than a nonlinear one. There is no way for the optimizer to step
    to a calendar-arbitrageable surface, not merely a penalty discouraging
    it.
    """
    return np.cumsum(increments)


def _slice_data(
    surface: pd.DataFrame, as_of: dt.datetime, r: float, q: float
) -> list[tuple[pd.Timestamp, float, float, np.ndarray, np.ndarray]]:
    """Per-expiry `(expiry, T, drift, k_forward, iv)`, OTM only and balanced.

    **The log-moneyness here is forward, not spot.** `surface/build.py`
    works in `log(K/S)`; every no-arbitrage statement about SSVI is in
    `log(K/F)`, and the two differ by a drift `(r-q)T` that grows with
    maturity. Fitting a monotone `theta` in spot coordinates would make the
    slices monotone along a set of curves that slide sideways as `T` grows,
    which is not the calendar condition and does not imply it -- a
    synthetic inverted term structure fit that way still produced four
    flagged pairs, which is the test that put this conversion here.
    `fit_ssvi_surface` maps the fitted slices back to spot coordinates on
    the way out, so callers see the same convention every other fit uses.

    The other two conventions match `fit_svi_surface`: OTM quotes only, and
    a symmetric window so a deep put wing cannot out-vote a shallow call
    wing on point count alone (`svi._balance_domain`).
    """
    slices = []
    for expiry, group in otm_quotes(surface).groupby("expiry"):
        T = year_fraction(expiry, as_of)
        if T <= 0:
            continue
        drift = (r - q) * T
        k, iv = _balance_domain(group["log_moneyness"].to_numpy(float) - drift, group["iv"].to_numpy(float))
        if len(k) >= MIN_SLICE_POINTS:
            slices.append((expiry, T, drift, k, iv))
    return sorted(slices, key=lambda s: s[1])


def _to_spot_coordinates(params: SVIParams, drift: float) -> SVIParams:
    """Shift a forward-coordinate raw-SVI slice into spot log-moneyness.

    `w_spot(k) = w_fwd(k - drift)`, and raw SVI depends on `k` only through
    `k - m`, so the whole conversion is `m -> m + drift`. Everything
    downstream (`surface/arbitrage.py`, `surface/local_vol.py`, the plots)
    expects spot-coordinate params and converts back itself, so this keeps
    SSVI slices interchangeable with independently fit ones.
    """
    return SVIParams(a=params.a, b=params.b, rho=params.rho, m=params.m + drift, sigma=params.sigma)


def _initial_shape(theta: np.ndarray, slices: list, rho: float) -> tuple[float, float]:
    """Estimate `(eta, gamma)` from the observed at-the-money skew.

    The at-the-money slope of SSVI total variance is `dw/dk|_0 =
    theta*rho*phi(theta)`, so each slice's own near-the-money slope gives a
    point estimate `phi_i = slope_i / (rho * theta_i)`. Regressing
    `log(phi_i)` on `log(theta_i)` is then a straight line whose intercept
    and slope are `log(eta)` and `-gamma` -- the power law read directly off
    the data instead of started from a constant.
    """
    phis = []
    for (_, T, _, k, iv), theta_i in zip(slices, theta):
        near = np.abs(k) <= 0.1
        if near.sum() >= 2 and theta_i > 0:
            slope = np.polyfit(k[near], iv[near] ** 2 * T, 1)[0]
            phi = slope / (rho * theta_i)
            if phi > 0:
                phis.append((theta_i, phi))

    if len(phis) < 2:
        return 1.0, 0.35

    log_theta, log_phi = np.log(np.array(phis)).T
    if np.ptp(log_theta) < 1e-8:  # one distinct theta fixes no slope
        return 1.0, 0.35

    design = np.column_stack([log_theta, np.ones_like(log_theta)])
    slope, intercept = np.linalg.lstsq(design, log_phi, rcond=None)[0]
    gamma = float(np.clip(-slope, *GAMMA_BOUNDS))
    eta = float(np.clip(np.exp(intercept), *ETA_BOUNDS))
    return eta, gamma


def _initial_theta(slices: list) -> np.ndarray:
    """At-the-money total variance per slice, forced non-decreasing.

    `np.interp` at `k = 0` reads the market's own at-the-money vol where
    the ladder brackets it and clamps to the nearest quote where it does
    not. `maximum.accumulate` then lifts any dip into the monotone region
    the parameterization lives in, so the optimizer starts inside its own
    feasible set rather than being projected onto its boundary.
    """
    atm_var = [float(np.interp(0.0, k, iv)) ** 2 * T for _, T, _, k, iv in slices]
    return np.maximum.accumulate(np.maximum(atm_var, 1e-8))


def fit_ssvi_surface(
    surface: pd.DataFrame,
    r: float = 0.04,
    q: float = 0.0,
    as_of: dt.datetime | None = None,
    k_window: str = "slice",
) -> SSVIFit:
    """Fit one SSVI surface to every expiry at once.

    The objective is squared error in **implied vol**, not total variance.
    Total variance grows roughly linearly in `T`, so a global fit that
    minimized it would let a seven-month slice outweigh a one-week slice by
    the ratio of their maturities and quietly become a long-dated fit. Vol
    points put every expiry on one scale -- the same scale the Monte Carlo
    repricing check reports in.

    `r` and `q` are not decoration: they define the forward, and the whole
    arbitrage argument lives in forward log-moneyness. The fit runs there
    and the slices come back in spot coordinates (`_to_spot_coordinates`),
    which is why they have to match what the arbitrage checks and the local
    vol construction are later given.

    `k_window` decides the log-moneyness range each returned slice claims:

    * `"slice"` -- its own quoted (domain-balanced) ladder, matching
      `fit_svi_slice`. Conservative, and the honest default.
    * `"union"` -- the widest ladder quoted anywhere on the surface. This
      is only defensible *because* the fit is global: a slice's wings are
      set by `rho` and `phi`, which every expiry's quotes helped estimate,
      so a front-week slice evaluated at a strike only the back months list
      is being interpolated in the shared parameters rather than
      extrapolated from three of its own points. It widens the local vol
      window -- the intersection of the slice windows -- which is what
      caps the range the repricing check can speak to.

    Returns `SSVIFit(params=None, reason=...)` rather than raising when
    there is too little to fit or the optimizer fails, matching
    `fit_svi_slice`'s convention of explaining itself instead of returning
    a garbage surface.
    """
    if k_window not in K_WINDOWS:
        raise ValueError(f"k_window must be one of {K_WINDOWS}, got {k_window!r}")

    as_of = as_of or dt.datetime.now()
    slices = _slice_data(surface, as_of, r, q)
    if len(slices) < MIN_SLICES:
        return SSVIFit(None, reason=f"need >= {MIN_SLICES} expiries to fit a surface, got {len(slices)}")

    n = len(slices)
    k_all = np.concatenate([k for *_, k, _ in slices])
    iv_all = np.concatenate([iv for *_, iv in slices])
    T_all = np.concatenate([np.full(len(k), T) for _, T, _, k, _ in slices])
    index = np.concatenate([np.full(len(k), i) for i, (*_, k, _) in enumerate(slices)])

    def unpack(p: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        return p[0], p[1], p[2], _theta_from_increments(p[3:])

    def sum_sq_error(p: np.ndarray) -> float:
        rho, eta, gamma, theta = unpack(p)
        theta_pt = theta[index]
        phi_k = eta * theta_pt**-gamma * k_all
        w = 0.5 * theta_pt * (1 + rho * phi_k + np.sqrt((phi_k + rho) ** 2 + 1 - rho**2))
        return float(np.mean((np.sqrt(w / T_all) - iv_all) ** 2))

    def butterfly(p: np.ndarray) -> np.ndarray:
        """Theorem 4.2's two margins at the largest theta, which bounds all."""
        rho, eta, gamma, theta = unpack(p)
        return np.array(SSVIParams(rho, eta, gamma).butterfly_margins(float(theta[-1])))

    theta0 = _initial_theta(slices)
    increments0 = np.diff(theta0, prepend=0.0)
    # theta is a cumulative sum, so "non-decreasing" is a lower bound of 0
    # on every increment past the first -- a box the optimizer cannot leave.
    bounds = [(-RHO_BOUND, RHO_BOUND), ETA_BOUNDS, GAMMA_BOUNDS, (1e-8, None)] + [(0.0, None)] * (n - 1)

    best = None
    for rho0 in (-0.8, -0.6, -0.4, -0.2):
        eta0, gamma0 = _initial_shape(theta0, slices, rho0)
        result = minimize(
            sum_sq_error,
            np.concatenate([[rho0, eta0, gamma0], increments0]),
            method="SLSQP",
            bounds=bounds,
            constraints=[NonlinearConstraint(butterfly, BUTTERFLY_MARGIN, np.inf)],
            options={"ftol": 1e-12, "maxiter": 400},
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result

    if best is None:
        return SSVIFit(None, reason="SSVI fit did not converge from any starting point")

    rho, eta, gamma, theta = unpack(best.x)
    params = SSVIParams(rho=rho, eta=eta, gamma=gamma)

    windows = [(float(k.min()), float(k.max())) for *_, k, _ in slices]
    if k_window == "union":
        windows = [(min(lo for lo, _ in windows), max(hi for _, hi in windows))] * n

    # Back to spot log-moneyness, the convention every other fit returns.
    return SSVIFit(
        params=params,
        theta={expiry: float(t) for (expiry, *_), t in zip(slices, theta)},
        slices={
            expiry: SVIFitResult(
                _to_spot_coordinates(params.slice_params(float(t)), drift),
                k_range=(lo + drift, hi + drift),
            )
            for (expiry, _, drift, _, _), t, (lo, hi) in zip(slices, theta, windows)
        },
        rmse_vol=float(np.sqrt(best.fun)),
        n_quotes=len(k_all),
    )
