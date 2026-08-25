"""Static no-arbitrage checks on the fitted SVI surface.

Fitting a smooth curve through market IVs does not make the result a valid
surface. Two conditions have to hold, and goodness of fit implies neither:

* **Butterfly** (within one expiry slice): the implied risk-neutral density
  must be non-negative. Gatheral's `g(k) >= 0` is exactly that condition
  rewritten in terms of total variance -- a slice with `g(k) < 0` somewhere
  prices an infinitesimal butterfly spread negatively, i.e. pays you to
  hold a non-negative payoff.
* **Calendar** (across slices): total implied variance must be
  non-decreasing in `T` at fixed forward log-moneyness. A drop means a
  longer-dated option is cheaper than a shorter-dated one covering the
  same event, which a calendar spread monetizes directly.

`fit_svi_slice` already constrains each slice to non-negative *variance*,
which is strictly weaker than either condition here: a curve can be
positive everywhere and still imply a negative density.

**Coordinates.** Both conditions are stated in *forward* log-moneyness
`k = log(K/F)`, `F = S*e^((r-q)T)`, while `surface/build.py` works in spot
log-moneyness `log(K/S)`. The two differ by the drift `(r-q)T`, so every
function here takes a `drift` and evaluates the fitted curve at
`k + drift` while using the forward `k` in the explicit-`k` terms.
`drift=0` means the params are already in forward coordinates -- the
convenient case for synthetic tests, and the reason the argument exists
rather than being folded in silently.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from vol_surface.surface.build import year_fraction
from vol_surface.surface.svi import SVIFitResult, SVIParams

BUTTERFLY_COLUMNS = ["expiry", "T", "n_points", "n_violations", "min_g", "k_at_min_g", "flagged"]
CALENDAR_COLUMNS = [
    "near_expiry",
    "far_expiry",
    "n_points",
    "n_violations",
    "max_variance_drop",
    "k_at_max_drop",
    "flagged",
]

# Total variance across adjacent slices is compared by subtraction, so an
# exactly-flat region of a legitimately non-decreasing surface can register
# a drop of a few ulps. Only drops larger than this count as violations.
CALENDAR_TOL = 1e-12

# One converged expiry slice ready to evaluate in forward coordinates:
# its expiry, the fit, the year fraction to it, and the forward drift
# `(r-q)T` that maps forward log-moneyness back onto the fitted curve.
Slice = tuple[pd.Timestamp, SVIFitResult, float, float]


def total_variance_derivatives(
    params: SVIParams, k: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`w`, `dw/dk`, `d2w/dk2` for a raw-SVI slice, in closed form.

    Raw SVI is differentiable by hand, so the strike-axis derivatives that
    both the butterfly condition and the Dupire formula need carry no
    finite-difference error at all. With `x = k - m` and
    `D = sqrt(x**2 + sigma**2)`:

        w   = a + b*(rho*x + D)
        w'  = b*(rho + x/D)
        w'' = b*sigma**2 / D**3
    """
    k = np.asarray(k, dtype=float)
    x = k - params.m
    D = np.sqrt(x**2 + params.sigma**2)
    w = np.asarray(params.total_variance(k), dtype=float)
    return w, params.b * (params.rho + x / D), params.b * params.sigma**2 / D**3


def butterfly_g(params: SVIParams, k: np.ndarray | float, drift: float = 0.0) -> np.ndarray:
    """Gatheral's `g(k)`, whose sign is the butterfly-arbitrage condition.

        g(k) = (1 - k*w'/(2w))**2 - (w'**2/4)*(1/w + 1/4) + w''/2

    `g(k) >= 0` for all `k` iff the slice implies a non-negative density.
    This same expression is the denominator of the Dupire local variance
    (see `surface/local_vol.py`) -- not a coincidence but the same
    quantity, which is why a slice that fails here cannot produce a real
    local vol.
    """
    k = np.asarray(k, dtype=float)
    w, dw, d2w = total_variance_derivatives(params, k + drift)
    return (1 - k * dw / (2 * w)) ** 2 - (dw**2 / 4) * (1 / w + 0.25) + d2w / 2


def risk_neutral_density(params: SVIParams, k: np.ndarray | float, drift: float = 0.0) -> np.ndarray:
    """Risk-neutral density of `log(S_T/F)` implied by the fitted slice.

    `p(k) = g(k) * exp(-d_minus(k)**2 / 2) / sqrt(2*pi*w(k))`, the density
    obtained by differentiating the Black-Scholes call price twice in
    strike and substituting the smile. `g` enters as a plain factor, so
    "negative density" and "negative `g`" are the same statement -- this
    is the intuition `butterfly_g` compresses into a sign check.
    """
    k = np.asarray(k, dtype=float)
    w, _, _ = total_variance_derivatives(params, k + drift)
    d_minus = -k / np.sqrt(w) - np.sqrt(w) / 2
    return butterfly_g(params, k, drift) * np.exp(-(d_minus**2) / 2) / np.sqrt(2 * np.pi * w)


def forward_window(fit: SVIFitResult, drift: float) -> tuple[float, float]:
    """The fit's spot log-moneyness window expressed in forward coordinates."""
    lo, hi = fit.k_range
    return lo - drift, hi - drift


def converged_slices(
    fits: dict[pd.Timestamp, SVIFitResult], r: float, q: float, as_of: dt.datetime
) -> list[Slice]:
    """Converged fits as `(expiry, fit, T, drift)`, sorted by time to expiry.

    Slices at `T <= 0` are dropped: an expired slice has no forward to
    measure moneyness against.
    """
    slices = [
        (expiry, fit, T, (r - q) * T)
        for expiry, fit in fits.items()
        if fit.ok and (T := year_fraction(expiry, as_of)) > 0
    ]
    return sorted(slices, key=lambda s: s[2])


def check_butterfly(
    fits: dict[pd.Timestamp, SVIFitResult],
    r: float = 0.04,
    q: float = 0.0,
    as_of: dt.datetime | None = None,
    n_k: int = 400,
) -> pd.DataFrame:
    """Evaluate `g(k)` across each converged slice's own fitted window.

    One row per expiry. `min_g` is the worst value found and `k_at_min_g`
    where it sits, so a flagged slice can be inspected rather than just
    counted. Each slice is checked only over the log-moneyness range it
    was actually fit on -- SVI extrapolates past the observed ladder, and
    flagging arbitrage in a region no quote informed would be measuring
    the extrapolation, not the market.
    """
    as_of = as_of or dt.datetime.now()
    rows = []

    for expiry, fit, T, drift in converged_slices(fits, r, q, as_of):
        k_grid = np.linspace(*forward_window(fit, drift), n_k)
        g = butterfly_g(fit.params, k_grid, drift)
        i_min = int(np.argmin(g))
        rows.append(
            {
                "expiry": expiry,
                "T": T,
                "n_points": n_k,
                "n_violations": int((g < 0).sum()),
                "min_g": float(g[i_min]),
                "k_at_min_g": float(k_grid[i_min]),
                "flagged": bool((g < 0).any()),
            }
        )

    return pd.DataFrame(rows, columns=BUTTERFLY_COLUMNS)


def check_calendar(
    fits: dict[pd.Timestamp, SVIFitResult],
    r: float = 0.04,
    q: float = 0.0,
    as_of: dt.datetime | None = None,
    n_k: int = 400,
) -> pd.DataFrame:
    """Compare total variance between adjacent expiries at fixed forward `k`.

    One row per adjacent pair, checked over the overlap of the two slices'
    fitted windows. `max_variance_drop` is the largest `w_near - w_far`
    found; positive means the near slice carries *more* total variance
    than the far one, which is the arbitrage. Pairs whose fitted windows
    don't overlap are skipped rather than reported as clean -- there is no
    common `k` at which the comparison is even defined.
    """
    as_of = as_of or dt.datetime.now()
    slices = converged_slices(fits, r, q, as_of)
    rows = []

    for (near_expiry, near_fit, _, near_drift), (far_expiry, far_fit, _, far_drift) in zip(slices, slices[1:]):
        near_lo, near_hi = forward_window(near_fit, near_drift)
        far_lo, far_hi = forward_window(far_fit, far_drift)
        lo, hi = max(near_lo, far_lo), min(near_hi, far_hi)
        if lo >= hi:
            continue

        k_grid = np.linspace(lo, hi, n_k)
        drop = np.asarray(near_fit.params.total_variance(k_grid + near_drift), dtype=float) - np.asarray(
            far_fit.params.total_variance(k_grid + far_drift), dtype=float
        )
        i_max = int(np.argmax(drop))
        rows.append(
            {
                "near_expiry": near_expiry,
                "far_expiry": far_expiry,
                "n_points": n_k,
                "n_violations": int((drop > CALENDAR_TOL).sum()),
                "max_variance_drop": float(drop[i_max]),
                "k_at_max_drop": float(k_grid[i_max]),
                "flagged": bool((drop > CALENDAR_TOL).any()),
            }
        )

    return pd.DataFrame(rows, columns=CALENDAR_COLUMNS)
