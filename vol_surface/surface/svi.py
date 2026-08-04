"""SVI (stochastic volatility inspired) fit of the smile, per expiry.

Gatheral's raw parameterization models total implied variance as a function
of log-moneyness `k = log(K/S)`:

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)**2 + sigma**2))

where `w = sigma_BS**2 * T`. It's the standard industry way to turn a
handful of noisy per-strike market IVs into a smooth, parametric smile --
used here to fit each expiry slice of the surface built in `surface/build.py`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, minimize

from vol_surface.surface.build import year_fraction

MIN_POINTS = 6  # 5 free parameters; need at least one degree of freedom to fit


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters. `a`/`b` set the level and slope of total
    variance, `rho` its skew, `m` its horizontal shift, `sigma` the
    curvature at its minimum."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray | float) -> np.ndarray | float:
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def implied_vol(self, k: np.ndarray | float, T: float) -> np.ndarray | float:
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / T)

    @property
    def min_total_variance(self) -> float:
        """w(k) at its minimum (attained at k = m - rho*sigma/sqrt(1-rho**2)).

        Must be >= 0, or the fit implies a negative variance somewhere on
        the smile -- the constraint `fit_svi_slice` enforces.
        """
        return self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)


@dataclass(frozen=True)
class SVIFitResult:
    """params is None on failure; reason explains why rather than returning a garbage fit.

    `k_range` is the (possibly domain-balanced, see `_balance_domain`)
    log-moneyness window the fit was actually computed over, so plotting
    code can avoid drawing the curve past where it was fit.
    """

    params: SVIParams | None
    reason: str | None = None
    k_range: tuple[float, float] | None = None

    @property
    def ok(self) -> bool:
        return self.params is not None


def _initial_guess(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    m0 = k[np.argmin(w)]
    sigma0 = max(float(np.std(k)), 0.1)
    a0 = max(float(np.min(w)), 1e-6)
    b0 = (float(np.ptp(w)) or 1e-3) / (2 * (float(np.ptp(k)) or 1.0))
    rho0 = -0.5  # equity indices skew toward richer downside puts; a reasonable starting bias
    return np.array([a0, b0, rho0, m0, sigma0])


def _balance_domain(k: np.ndarray, iv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate to a symmetric log-moneyness window sized to the shorter wing.

    Listed strike ladders are rarely symmetric around spot (SPY, e.g.,
    lists much deeper puts than calls) -- an unweighted least-squares fit
    over the full ladder lets whichever wing has more points, or spans a
    wider variance range, dictate `rho` almost by itself, collapsing the
    other wing's curve to a nearly straight line. Matching both wings to
    the same domain keeps the fit answerable to both.
    """
    if k.min() >= 0 or k.max() <= 0:
        return k, iv
    k_max = min(-k.min(), k.max())
    mask = np.abs(k) <= k_max
    return k[mask], iv[mask]


def fit_svi_slice(log_moneyness: np.ndarray, iv: np.ndarray, T: float) -> SVIFitResult:
    """Fit SVI total variance to one expiry slice by least squares.

    Constrained to `min_total_variance >= 0` so the fit can't imply a
    negative variance between the observed strikes, not just at them.
    """
    k, iv = _balance_domain(np.asarray(log_moneyness, dtype=float), np.asarray(iv, dtype=float))
    w = iv**2 * T

    if len(k) < MIN_POINTS:
        return SVIFitResult(None, f"need >= {MIN_POINTS} points to fit SVI, got {len(k)}")

    def sum_sq_error(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        model_w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        return float(np.sum((model_w - w) ** 2))

    def min_total_variance(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        return a + b * sigma * np.sqrt(1 - rho**2)

    k_pad = float(np.ptp(k)) or 1.0
    bounds = [
        (1e-8, None),  # a >= 0
        (1e-8, None),  # b >= 0: total variance is non-decreasing away from its minimum
        (-0.999, 0.999),  # rho
        (k.min() - k_pad, k.max() + k_pad),  # m: kept near the observed strike range
        (1e-4, None),  # sigma > 0
    ]

    result = minimize(
        sum_sq_error,
        _initial_guess(k, w),
        method="SLSQP",
        bounds=bounds,
        constraints=[NonlinearConstraint(min_total_variance, 0.0, np.inf)],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not result.success:
        return SVIFitResult(None, f"SVI fit did not converge: {result.message}")

    return SVIFitResult(SVIParams(*result.x), k_range=(float(k.min()), float(k.max())))


def fit_svi_surface(surface: pd.DataFrame, as_of: dt.datetime | None = None) -> dict[pd.Timestamp, SVIFitResult]:
    """Fit one SVI slice per expiry in a surface built by `build_surface`.

    Only OTM quotes are used per strike (puts below spot, calls at/above
    it) -- the liquid, tightly-quoted side of the chain, and the usual
    convention for smile construction so a strike isn't double-counted
    from both legs.
    """
    as_of = as_of or dt.datetime.now()
    otm = surface[
        ((surface["option_type"] == "put") & (surface["moneyness"] < 1))
        | ((surface["option_type"] == "call") & (surface["moneyness"] >= 1))
    ]

    return {
        expiry: fit_svi_slice(group["log_moneyness"].to_numpy(), group["iv"].to_numpy(), year_fraction(expiry, as_of))
        for expiry, group in otm.groupby("expiry")
    }
