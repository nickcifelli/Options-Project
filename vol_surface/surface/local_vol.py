"""Dupire local volatility, derived from the fitted SVI surface.

Implied vol is an *average* of volatility along all paths to one strike and
expiry: it answers "what single constant vol reprices this contract?".
Local vol is the *instantaneous* vol the underlying must have at a given
spot and time to reproduce the entire observed surface. Dupire's result is
that the surface pins that function down uniquely -- so local vol is not a
new model fit to the data, it is the market's own surface read in a
different coordinate.

In terms of total implied variance `w(k, T)` with `k = log(K/F)` forward
log-moneyness (Gatheral, *The Volatility Surface*, eq. 1.10):

    sigma_LV**2(k, T) = (dw/dT) / g(k)

where `g` is exactly the butterfly function from `surface/arbitrage.py`.
Writing it that way makes the dependency between the two modules explicit
rather than incidental: the denominator of local variance *is* the
butterfly condition, so a slice with a negative density produces a
negative local variance at precisely the same strikes. Local vol is not
merely nicer on an arbitrage-free surface -- it fails to exist without one,
and this module masks those points rather than returning a complex number
under a square root.

**The strike axis is exact; the time axis is not.** Raw SVI is
differentiable in closed form, so `dw/dk` and `d2w/dk2` carry no
discretization error (see `total_variance_derivatives`). `dw/dT` cannot:
there are only as many `T` samples as fitted expiries, so it is a finite
difference across neighbouring slices. That is the accuracy floor of the
whole surface, and it is a property of the listed expiry ladder rather
than of the method.

That floor collapses when the ladder is dense. SPY lists *daily*
expiries at the front, and differencing total variance across a one-day
gap divides each slice's own fit residual by `1/365` -- amplifying fit
noise roughly 365-fold. Measured on a live 19-expiry SPY chain, every
row whose neighbours sat one day apart produced local vols spanning
0.007 to 0.905 with up to 54% of the row undefined, while every row with
a gap of five days or more stayed inside a 0.08-0.40 band with no gaps
at all. `min_expiry_gap` thins the slices used for `dw/dT` accordingly.

The thinning is deliberately *not* applied to the arbitrage checks in
`surface/arbitrage.py`: a total-variance drop between two consecutive
daily expiries is a real property of the fitted surface and worth
reporting. It is only the division by a tiny `dT` that is
ill-conditioned, so only the differencing is thinned.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vol_surface.surface.arbitrage import Slice, butterfly_g, converged_slices, forward_window
from vol_surface.surface.svi import SVIFitResult

MIN_SLICES = 2  # dw/dT needs at least two expiries to difference across

# Minimum spacing, in years, between slices used for the dw/dT difference.
# One week: enough to put the difference safely above per-slice fit noise,
# while still tracking the weekly cadence of the listed ladder.
MIN_EXPIRY_GAP = 7.0 / 365.0


@dataclass(frozen=True)
class LocalVolSurface:
    """Local vol on a regular `(T, k)` grid, `NaN` where it doesn't exist.

    `local_vol` and `implied_vol` are both `(len(T), len(k))`, sampled on
    the same grid so the two can be compared point for point. A `NaN` in
    `local_vol` marks a grid point where the surface implies no real
    instantaneous vol -- either a negative `dw/dT` (calendar arbitrage) or
    a non-positive `g` (butterfly arbitrage).
    """

    k: np.ndarray
    T: np.ndarray
    local_vol: np.ndarray
    implied_vol: np.ndarray

    @property
    def coverage(self) -> float:
        """Fraction of grid points where local vol exists. Below 1.0 means
        the fitted surface is not fully arbitrage-free."""
        return float(np.isfinite(self.local_vol).mean())


def fill_holes(local_vol: np.ndarray) -> np.ndarray:
    """Fill `NaN` gaps so a path can be stepped through them.

    Holes are where the fitted surface implies no real local vol, so any
    fill is an admission that the surface is incomplete rather than a
    recovery of missing information. Within a row the gaps are bridged by
    linear interpolation in `k`, where local vol is smooth; a row that is
    entirely undefined takes the nearest row that isn't. Callers should
    read `LocalVolSurface.coverage` alongside any result that needed this.
    """
    filled = np.array(local_vol, dtype=float, copy=True)
    columns = np.arange(filled.shape[1])

    for row in filled:
        valid = np.isfinite(row)
        if valid.any() and not valid.all():
            row[~valid] = np.interp(columns[~valid], columns[valid], row[valid])

    complete = np.array([np.isfinite(row).all() for row in filled])
    if not complete.any():
        raise ValueError("local vol surface is undefined everywhere; nothing to simulate")

    usable = np.flatnonzero(complete)
    for i in np.flatnonzero(~complete):
        filled[i] = filled[usable[np.argmin(np.abs(usable - i))]]
    return filled


def curve_at_time(filled: np.ndarray, T_grid: np.ndarray, t: float) -> np.ndarray:
    """Local vol as a function of `k` at time `t`, linear in `T` between rows.

    Clamped outside the grid: the surface's earliest expiry is the earliest
    information there is, and extrapolating local vol past the quoted
    ladder would be inventing term structure.
    """
    if t <= T_grid[0]:
        return filled[0]
    if t >= T_grid[-1]:
        return filled[-1]

    upper = int(np.searchsorted(T_grid, t))
    lower = upper - 1
    weight = (t - T_grid[lower]) / (T_grid[upper] - T_grid[lower])
    return (1 - weight) * filled[lower] + weight * filled[upper]


@dataclass(frozen=True)
class LocalVolSampler:
    """A `sigma(k, t)` view of a `LocalVolSurface`: holes filled, edges clamped.

    Both numerical engines that consume the surface -- the Monte Carlo in
    `pricing/monte_carlo.py` and the finite-difference solver in
    `pricing/pde.py` -- need the same thing from it: local vol at an
    arbitrary forward log-moneyness and time, not just at grid points.
    Sharing one sampler is what makes them cross-checkable: when the PDE
    and the Monte Carlo agree on a European price they have agreed on the
    surface too, rather than on two different readings of it.

    Holes are filled once, up front (`fill_holes`), because a path or a
    grid node can land in one and neither method can step through a `NaN`.
    That is an admission the surface is incomplete, not a repair of it --
    read `LocalVolSurface.coverage` alongside anything this produces.
    """

    k: np.ndarray
    T: np.ndarray
    values: np.ndarray

    @classmethod
    def from_surface(cls, surface: LocalVolSurface) -> LocalVolSampler:
        return cls(k=surface.k, T=surface.T, values=fill_holes(surface.local_vol))

    def __call__(self, k: np.ndarray | float, t: float) -> np.ndarray:
        """Local vol at forward log-moneyness `k` and time `t`.

        Linear in `T` between rows and linear in `k` within one, clamped
        outside the fitted window in both axes -- extrapolating local vol
        past the quoted ladder would be inventing term structure, and
        `reprice_chain` skips quotes that land there rather than scoring
        the clamp.
        """
        return np.interp(k, self.k, curve_at_time(self.values, self.T, t))


def _thin_by_expiry_gap(slices: list[Slice], min_gap: float) -> list[Slice]:
    """Greedily drop slices closer than `min_gap` to the last one kept.

    Falls back to the full list rather than raising if thinning would
    leave too few slices to difference: a chain quoted entirely inside one
    week (SPY's front-month dailies, say) still gets a surface, just a
    noisy one that `LocalVolSurface.coverage` will report honestly.
    """
    kept = slices[:1]
    for candidate in slices[1:]:
        if candidate[2] - kept[-1][2] >= min_gap:
            kept.append(candidate)
    return kept if len(kept) >= MIN_SLICES else slices


def build_local_vol_surface(
    fits: dict[pd.Timestamp, SVIFitResult],
    r: float = 0.04,
    q: float = 0.0,
    as_of: dt.datetime | None = None,
    n_k: int = 200,
    min_expiry_gap: float = MIN_EXPIRY_GAP,
) -> LocalVolSurface:
    """Apply Dupire's formula to the fitted SVI slices.

    The grid spans the log-moneyness window *common to every* converged
    slice, so `dw/dT` is always differenced between curves that were both
    fit at that `k`. Widening it past the intersection would difference a
    fitted curve against an extrapolated one and read the result as term
    structure.

    `min_expiry_gap` (in years) is the minimum spacing between the slices
    used for `dw/dT`; closer ones are dropped, since differencing across
    them measures fit noise rather than term structure. See the module
    docstring for the measurements behind the default.

    Raises `ValueError` if fewer than two slices converged, or if their
    fitted windows share no common `k`.
    """
    as_of = as_of or dt.datetime.now()
    slices = converged_slices(fits, r, q, as_of)
    if len(slices) < MIN_SLICES:
        raise ValueError(f"need >= {MIN_SLICES} converged SVI slices for dw/dT, got {len(slices)}")
    slices = _thin_by_expiry_gap(slices, min_expiry_gap)

    windows = [forward_window(fit, drift) for _, fit, _, drift in slices]
    lo, hi = max(w[0] for w in windows), min(w[1] for w in windows)
    if lo >= hi:
        raise ValueError("converged SVI slices share no common log-moneyness window")

    k_grid = np.linspace(lo, hi, n_k)
    T = np.array([T for _, _, T, _ in slices])
    w = np.vstack([np.asarray(fit.params.total_variance(k_grid + drift), dtype=float) for _, fit, _, drift in slices])
    g = np.vstack([butterfly_g(fit.params, k_grid, drift) for _, fit, _, drift in slices])

    # Central differences in the interior, correcting for the uneven expiry
    # spacing a listed ladder always has. The front slice is anchored at
    # T = 0, where total variance is known to be exactly zero, rather than
    # left to `np.gradient`'s edge rule -- which would extrapolate the front
    # derivative backwards from the *next two* expiries and discard a data
    # point already in hand. The Monte Carlo repricing check in
    # `pricing/monte_carlo.py` is what surfaced this: on a live SPY chain
    # the unanchored edge rule mispriced expiries inside 9 days by -1.50 vol
    # points, which anchoring cuts to -0.28 (and the 95th-percentile error
    # across all expiries from 3.07 to 2.61). It does cost a little at the
    # median, 0.50 to 0.56, because the front row also anchors the
    # interpolation out to the second expiry; the anchored derivative is
    # kept anyway, on the grounds that using a known boundary value beats
    # extrapolating past it.
    T_anchored = np.concatenate([[0.0], T])
    w_anchored = np.vstack([np.zeros_like(w[0]), w])
    dw_dT = np.gradient(w_anchored, T_anchored, axis=0)[1:]

    local_variance = np.divide(dw_dT, g, out=np.full_like(dw_dT, np.nan), where=g > 0)
    local_variance[local_variance < 0] = np.nan

    return LocalVolSurface(
        k=k_grid,
        T=T,
        local_vol=np.sqrt(local_variance),
        implied_vol=np.sqrt(w / T[:, None]),
    )
