"""Monte Carlo pricing under the fitted local vol surface.

This closes the loop the rest of the project opens. Market quotes become
implied vols (`pricing/implied_vol.py`), those become SVI slices
(`surface/svi.py`), and those become a Dupire local vol surface
(`surface/local_vol.py`). Dupire's construction says a diffusion carrying
that local vol reprices the vanillas it was built from -- simulating it and
recovering the original market prices is what turns that claim into a
measurement.

The check is not vacuous, because the surface it runs on is not perfect.
Each expiry is fit independently, so the slices can and do cross in `T`
(`surface/arbitrage.py` reports how often). Where they cross, the local
vol that repricing depends on doesn't exist. The repricing error is
therefore a direct, in-basis-points readout of what the slice-independent
fit costs, and it should be largest exactly where the calendar check fires.

**Scheme.** Log-Euler on `ln S`, which removes the drift discretization
entirely and keeps `S` positive by construction:

    ln S_{t+dt} = ln S_t + (r - q - sigma**2/2) dt + sigma sqrt(dt) Z

with `sigma = sigma_LV(k_t, t)` frozen over each step, and `k_t` the
*forward* log-moneyness of the path, `log(S_t/S_0) - (r-q)t`, matching the
coordinate `LocalVolSurface` is built in. Freezing sigma over the step
leaves an O(dt) discretization bias, separate from Monte Carlo noise and
not reduced by adding paths -- `n_steps` is what controls it.

**Variance reduction.** Two techniques, both switchable so their effect can
be measured rather than asserted:

* *Antithetic variates*: every path is simulated alongside its mirror
  `-Z`. The two are not independent, so the standard error is computed
  across antithetic *pairs* rather than across paths -- averaging the
  pair first is what keeps the error bar honest.
* *A control variate*: the same Brownian increments also drive a constant-vol
  GBM whose exact price is known from Black-Scholes, and the estimator
  corrects by the control's sampling error, `beta` fit by least squares.
  Because the control's mean is known analytically the estimator stays
  unbiased whatever `beta` is; `beta` only decides how much variance goes
  away. The reference vol is the surface's *at-the-money* implied vol for
  the expiry -- deliberately not the per-strike market vol, which would
  make the control almost exactly the answer and hollow out the repricing
  test it is meant to support.

Simulating the control costs nothing extra: log-Euler is exact for
constant-vol GBM, so accumulating the Brownian path `W_T` during the same
loop is enough to price it in closed form at the end.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vol_surface.pricing.black_scholes import greeks as bs_greeks
from vol_surface.pricing.black_scholes import price as bs_price
from vol_surface.pricing.implied_vol import implied_vol
from vol_surface.surface.build import year_fraction
from vol_surface.surface.local_vol import LocalVolSurface
from vol_surface.surface.svi import otm_quotes

_VALID_TYPES = ("call", "put")

REPRICE_COLUMNS = [
    "expiry",
    "strike",
    "option_type",
    "T",
    "market_iv",
    "mc_iv",
    "mc_price",
    "mc_std_error",
    "iv_std_error",
    "iv_error",
]

# Below this vega, a price standard error doesn't translate into a
# meaningful vol standard error -- the same near-zero-vega regime that
# makes near-expiry wings hard to invert in the first place.
MIN_VEGA = 1e-6

STEPS_PER_YEAR = 252  # daily stepping
MIN_STEPS = 20


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo price with the error bar that makes it meaningful.

    `std_error` is the standard error of the mean, computed across
    antithetic pairs when antithetic sampling is on. A price without it is
    not a result, which is why they travel together.
    """

    price: float
    std_error: float
    n_paths: int
    n_steps: int

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        return self.price - z * self.std_error, self.price + z * self.std_error


def default_steps(T: float) -> int:
    """Daily stepping, floored so very short expiries still get a path."""
    return max(MIN_STEPS, int(np.ceil(T * STEPS_PER_YEAR)))


def _fill_holes(local_vol: np.ndarray) -> np.ndarray:
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


def _curve_at_time(filled: np.ndarray, T_grid: np.ndarray, t: float) -> np.ndarray:
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
class TerminalPaths:
    """Terminal spots under the local vol diffusion, plus what the control needs.

    `brownian` is the accumulated `W_T` of the same increments that drove
    the paths, which prices the constant-vol control exactly without
    simulating it separately.
    """

    spot: np.ndarray
    brownian: np.ndarray
    sigma_ref: float
    n_steps: int

    @property
    def n_paths(self) -> int:
        return len(self.spot)


def atm_implied_vol(local_vol: LocalVolSurface, T: float) -> float:
    """The surface's at-the-money implied vol at `T`, used as the control's vol."""
    curve = _curve_at_time(local_vol.implied_vol, local_vol.T, T)
    return float(np.interp(0.0, local_vol.k, curve))


def simulate_terminal(
    local_vol: LocalVolSurface,
    S0: float,
    T: float,
    r: float = 0.04,
    q: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int | None = None,
    antithetic: bool = True,
    seed: int | None = None,
) -> TerminalPaths:
    """Step the local vol diffusion out to `T` and return the terminal spots.

    Paths are simulated once per expiry and reused across every strike at
    that expiry -- the terminal distribution doesn't depend on `K`, so
    re-simulating per strike would be pure waste and would also make
    strikes at one expiry needlessly independent of each other.

    With `antithetic=True` the path count is rounded up to an even number
    and the second half mirrors the first.
    """
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")

    n_steps = n_steps or default_steps(T)
    rng = np.random.default_rng(seed)

    n_sim = 2 * ((n_paths + 1) // 2) if antithetic else n_paths

    def draw() -> np.ndarray:
        """One step's normals, mirrored in the second half when antithetic."""
        if antithetic:
            z = rng.standard_normal(n_sim // 2)
            return np.concatenate([z, -z])
        return rng.standard_normal(n_sim)

    filled = _fill_holes(local_vol.local_vol)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    drift_rate = r - q

    log_spot = np.full(n_sim, np.log(S0))
    brownian = np.zeros_like(log_spot)

    for step in range(n_steps):
        t = step * dt
        # Forward log-moneyness of each path, the coordinate the surface
        # is indexed by; np.interp clamps at the edges of the fitted window.
        k = log_spot - np.log(S0) - drift_rate * t
        sigma = np.interp(k, local_vol.k, _curve_at_time(filled, local_vol.T, t))

        z = draw()
        log_spot += (drift_rate - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
        brownian += sqrt_dt * z

    return TerminalPaths(
        spot=np.exp(log_spot),
        brownian=brownian,
        sigma_ref=atm_implied_vol(local_vol, T),
        n_steps=n_steps,
    )


def _payoff(spot: np.ndarray, K: float, option_type: str) -> np.ndarray:
    return np.maximum(spot - K, 0.0) if option_type == "call" else np.maximum(K - spot, 0.0)


def _pairwise_mean_and_error(values: np.ndarray, antithetic: bool) -> tuple[float, float]:
    """Mean and standard error, averaging antithetic partners before measuring.

    Antithetic paths are negatively correlated by design, so treating them
    as independent samples would overstate the error -- and overstating it
    would hide the very variance reduction the technique exists for.
    """
    if antithetic:
        half = len(values) // 2
        values = 0.5 * (values[:half] + values[half:])

    n = len(values)
    std_error = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(values.mean()), std_error


def price_from_terminal(
    paths: TerminalPaths,
    S0: float,
    K: float,
    T: float,
    r: float = 0.04,
    q: float = 0.0,
    option_type: str = "call",
    antithetic: bool = True,
    control_variate: bool = True,
) -> MCResult:
    """Price one strike off an already-simulated set of terminal spots."""
    if option_type not in _VALID_TYPES:
        raise ValueError(f"option_type must be one of {_VALID_TYPES}, got {option_type!r}")

    discount = np.exp(-r * T)
    values = discount * _payoff(paths.spot, K, option_type)

    if control_variate and paths.sigma_ref > 0:
        # Log-Euler is exact for constant-vol GBM, so the control's terminal
        # spot follows in closed form from the Brownian path already
        # accumulated -- no second simulation.
        sigma = paths.sigma_ref
        control_spot = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * paths.brownian)
        control = discount * _payoff(control_spot, K, option_type)
        control_mean = bs_price(S0, K, T, r, sigma, option_type, q)

        variance = control.var()
        if variance > 0:
            beta = float(np.cov(values, control, ddof=1)[0, 1] / variance)
            values = values - beta * (control - control_mean)

    price, std_error = _pairwise_mean_and_error(values, antithetic)
    return MCResult(price=price, std_error=std_error, n_paths=paths.n_paths, n_steps=paths.n_steps)


def price_european(
    local_vol: LocalVolSurface,
    S0: float,
    K: float,
    T: float,
    r: float = 0.04,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int | None = None,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> MCResult:
    """Monte Carlo price of one European option under the local vol surface."""
    paths = simulate_terminal(
        local_vol, S0, T, r=r, q=q, n_paths=n_paths, n_steps=n_steps, antithetic=antithetic, seed=seed
    )
    return price_from_terminal(
        paths, S0, K, T, r=r, q=q, option_type=option_type, antithetic=antithetic, control_variate=control_variate
    )


def _price_error_in_vol(
    std_error: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    sigma: float | None,
) -> float | None:
    """Convert a price standard error into vol points via vega.

    `dSigma ~ dPrice / vega` -- the same local linearisation the IV solver
    inverts, evaluated at the price the simulation actually produced.
    """
    if sigma is None:
        return None
    vega = bs_greeks(S0, K, T, r, sigma, option_type, q).vega
    return std_error / vega if vega > MIN_VEGA else None


def reprice_chain(
    local_vol: LocalVolSurface,
    surface: pd.DataFrame,
    S0: float,
    r: float = 0.04,
    q: float = 0.0,
    as_of: dt.datetime | None = None,
    n_paths: int = 50_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> pd.DataFrame:
    """Reprice the surface's own quotes by simulation and compare in vol points.

    One simulation per expiry, shared across that expiry's strikes. Only
    OTM quotes are repriced, via the same `otm_quotes` selection the smile
    was fit under, so the check cannot silently score a different set of
    quotes than the surface was built from.

    Quotes outside the surface's own `(k, T)` window are skipped rather
    than clamped, in both axes. Simulation clamps local vol at the edge of
    the grid, so a strike beyond the window is priced against the last
    fitted value rather than against local vol, and scoring that as a
    repricing error would measure the clamp instead of the surface.
    Measured on a live SPY chain, the median error inside the window was
    0.50 vol points against 3.06 outside it, rising monotonically with
    distance past the edge -- 2.2 vol points at 0.05-0.10 in log-moneyness,
    11.0 at 0.20-0.50. Note this is a real coverage limit and not only a
    reporting one: the window is the *intersection* of every slice's fitted
    range, so a single narrow front-week ladder pulls it in for everyone.

    The comparison is reported in implied vol rather than price: a cent of
    error means something different on a 1-week wing than on a 5-month
    at-the-money, and inverting both sides puts them on one scale. The
    Monte Carlo standard error is carried across the same way, divided by
    vega, so an error bar in vol points sits next to every point and a gap
    can be told apart from simulation noise.
    """
    as_of = as_of or dt.datetime.now()
    otm = otm_quotes(surface)
    T_lo, T_hi = local_vol.T.min(), local_vol.T.max()
    k_lo, k_hi = local_vol.k.min(), local_vol.k.max()
    rows = []

    for i, (expiry, group) in enumerate(otm.groupby("expiry")):
        T = year_fraction(expiry, as_of)
        if not (T_lo <= T <= T_hi):
            continue

        # Forward log-moneyness, the coordinate the surface is indexed by.
        k = np.log(group["strike"] / S0) - (r - q) * T
        group = group[(k >= k_lo) & (k <= k_hi)]
        if group.empty:
            continue

        paths = simulate_terminal(
            local_vol,
            S0,
            T,
            r=r,
            q=q,
            n_paths=n_paths,
            antithetic=antithetic,
            seed=None if seed is None else seed + i,
        )

        for _, quote in group.iterrows():
            result = price_from_terminal(
                paths,
                S0,
                quote["strike"],
                T,
                r=r,
                q=q,
                option_type=quote["option_type"],
                antithetic=antithetic,
                control_variate=control_variate,
            )
            inverted = implied_vol(result.price, S0, quote["strike"], T, r, quote["option_type"], q)
            rows.append(
                {
                    "expiry": expiry,
                    "strike": quote["strike"],
                    "option_type": quote["option_type"],
                    "T": T,
                    "market_iv": quote["iv"],
                    "mc_iv": inverted.sigma,
                    "mc_price": result.price,
                    "mc_std_error": result.std_error,
                    "iv_std_error": _price_error_in_vol(
                        result.std_error, S0, quote["strike"], T, r, q, quote["option_type"], inverted.sigma
                    ),
                    "iv_error": None if inverted.sigma is None else inverted.sigma - quote["iv"],
                }
            )

    return pd.DataFrame(rows, columns=REPRICE_COLUMNS)
