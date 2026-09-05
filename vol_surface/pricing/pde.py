"""Finite-difference pricing under the local vol surface, European or American.

The README opens on why American exercise is hard: there is no closed form,
because the holder's right to exercise early makes pricing an optimal
stopping problem rather than an expectation. `pricing/binomial.py` solves
that for a *constant* vol. `pricing/monte_carlo.py` handles the fitted
local vol surface but only for European payoffs -- forward simulation has
no natural way to ask "would the holder have exercised here?", because the
continuation value at a path's current state is exactly what has not been
computed yet.

This module closes the gap: a backward PDE solve carries the whole
continuation surface at every time step, so the early-exercise test is a
pointwise comparison already in hand. Written on `x = log S`, the
Black-Scholes-Merton equation in time-to-expiry `tau = T - t` is

    dV/dtau = (1/2)*sigma(x, tau)**2 * d2V/dx2
              + (r - q - (1/2)*sigma(x, tau)**2) * dV/dx
              - r*V

with `sigma` read from the fitted surface at the node's own forward
log-moneyness. Log space is not cosmetic: it makes the diffusion
coefficient independent of the node (up to `sigma`), so a *uniform* grid is
already well-scaled, and it keeps `S > 0` by construction the same way
log-Euler does in the Monte Carlo.

**Why this is a real check and not a third opinion.** The Monte Carlo and
this solver read the surface through the same `LocalVolSampler`, and they
share nothing else: one integrates forward with random draws, the other
integrates backward with linear algebra. Agreement on a European price is
therefore a statement about the surface rather than about a common
implementation -- and the American price is the same solver with one extra
line, so the early-exercise premium it reports is a difference between two
numbers computed identically apart from the constraint that defines it.

**Discretization.** Crank-Nicolson (`theta = 1/2`, second order in both
axes) with two fully implicit steps at the start -- Rannacher startup.
Crank-Nicolson is only *A*-stable, not L-stable: it damps high-frequency
error slowly, and the kink in a vanilla payoff is exactly high-frequency
error. Left alone it produces the familiar sawtooth in gamma near the
strike, which then contaminates delta and the exercise boundary. Two
implicit steps annihilate those modes before the scheme goes second order,
which is why the startup is not optional here.

The terminal condition is *cell-averaged* rather than sampled: node `i`
gets the exact integral of the payoff over `[x_i - h/2, x_i + h/2]`, which
is available in closed form in log space. Sampling a kinked payoff makes
the answer depend on where the strike happens to fall between two nodes --
a source of error that looks like noise across a strike ladder and does
not shrink cleanly under refinement. Averaging removes that dependence.

The grid is centred so that `log S0` is exactly a node (`n_space` is forced
odd), so the price is read off, not interpolated.

**Early exercise.** Each step solves a linear complementarity problem
rather than a linear system:

    (M V - b) >= 0,   V >= payoff,   (M V - b)(V - payoff) = 0

Two solvers, which is the point of having two:

* `brennan-schwartz` (default) -- a single modified tridiagonal sweep, run
  in the direction that meets the exercise region last. It is *exact* for
  the vanilla American LCP, where the exercise region is a single interval
  (Jaillet, Lamberton & Lapeyre 1990), and costs the same as the European
  solve.
* `psor` -- projected successive over-relaxation, iterated to a tolerance.
  Slower and general, and it assumes nothing about the shape of the
  exercise region.

The test suite runs both on the same problem and requires them to agree,
which is what makes the fast one trustworthy: Brennan-Schwartz is exact
only under a hypothesis about the free boundary that a vanilla satisfies
and an arbitrary payoff need not.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

from vol_surface.pricing.black_scholes import greeks as bs_greeks
from vol_surface.surface.build import year_fraction
from vol_surface.surface.local_vol import LocalVolSampler, LocalVolSurface
from vol_surface.surface.svi import otm_quotes

_VALID_TYPES = ("call", "put")
LCP_SOLVERS = ("brennan-schwartz", "psor")

TIME_STEPS_PER_YEAR = 252  # daily stepping, matching the Monte Carlo
MIN_TIME_STEPS = 25
DEFAULT_SPACE_NODES = 401
RANNACHER_STEPS = 2

# Half-width of the log-spot grid in standard deviations. Six puts the
# Dirichlet boundaries far enough out that their exact asymptotic values
# are correct to well past machine noise at the money.
GRID_WIDTH_SDS = 6.0

PSOR_OMEGA = 1.2
PSOR_TOL = 1e-10
PSOR_MAX_ITER = 500

# A node counts as exercised when its value sits within this of intrinsic.
# The comparison is against a discounted price, so an absolute tolerance in
# currency units is the right scale.
EXERCISE_TOL = 1e-8

VolFunction = Callable[[np.ndarray, float], np.ndarray]

PREMIUM_COLUMNS = [
    "expiry",
    "strike",
    "moneyness",
    "option_type",
    "T",
    "market_iv",
    "european",
    "american",
    "premium",
    "premium_pct",
    "premium_vol_points",
    "critical_spot",
]

# A chain sweep is two solves per quote, so the per-solve grid is smaller
# than the default used for a single contract. The premium is a difference
# of two solves on an identical grid, where the leading discretization
# error cancels, so it tolerates a coarser grid than either price does.
CHAIN_SPACE_NODES = 201


@dataclass(frozen=True)
class PDEResult:
    """The `t = 0` solution across the whole grid, not just at spot.

    A backward solve produces the option value at every spot on the grid on
    its way to the one that was asked for, so the Greeks come out of the
    same array by differencing rather than from a second and third solve.
    That matters most for American options, where a bump-and-reprice delta
    costs as much as the price itself and inherits the free boundary's
    kink; here `delta` and `gamma` are read off the solution the free
    boundary was already computed against.

    `exercise_boundary` is the critical spot at each time step for an
    American solve, and `None` for a European one -- the free boundary the
    optimal stopping problem is really solving for.
    """

    price: float
    delta: float
    gamma: float
    theta: float
    intrinsic: float
    S: np.ndarray
    values: np.ndarray
    tau: np.ndarray
    exercise_boundary: np.ndarray | None
    n_space: int
    n_time: int

    @property
    def early_exercise_is_optimal_now(self) -> bool:
        """Whether the holder should exercise at today's spot, right now.

        True when the solve found the contract's value at spot pinned to
        its intrinsic value, which is what "the constraint is binding here"
        means. Out-of-the-money contracts have no intrinsic value to take,
        so this is False for every quote on the OTM side of a chain -- see
        `critical_spot_now` for the question that is interesting there.
        """
        return (
            self.exercise_boundary is not None
            and self.intrinsic > 0
            and self.price <= self.intrinsic + EXERCISE_TOL
        )

    @property
    def critical_spot_now(self) -> float:
        """Today's exercise boundary: the spot at which waiting stops paying.

        `NaN` for a European solve, and also when the boundary lies outside
        the solved grid -- reported rather than extrapolated, since a
        boundary the grid never reached is not a number this solve knows.
        """
        return np.nan if self.exercise_boundary is None else float(self.exercise_boundary[-1])


def constant_vol(sigma: float) -> VolFunction:
    """A `sigma(k, tau)` callable for a flat surface.

    The one case with a closed-form answer, so it is what the solver's
    convergence tests are written against.
    """

    def sigma_at(k: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(np.asarray(k, dtype=float), sigma)

    return sigma_at


def surface_vol(local_vol: LocalVolSurface) -> VolFunction:
    """A `sigma(k, t)` callable reading the fitted Dupire surface.

    Deliberately the same `LocalVolSampler` the Monte Carlo steps through,
    so the two methods cannot disagree because they read the surface
    differently.
    """
    return LocalVolSampler.from_surface(local_vol)


def _check_inputs(T: float, option_type: str, n_space: int, lcp: str) -> None:
    if option_type not in _VALID_TYPES:
        raise ValueError(f"option_type must be one of {_VALID_TYPES}, got {option_type!r}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if n_space < 5:
        raise ValueError(f"n_space must be >= 5, got {n_space}")
    if lcp not in LCP_SOLVERS:
        raise ValueError(f"lcp must be one of {LCP_SOLVERS}, got {lcp!r}")


def default_time_steps(T: float) -> int:
    """Daily stepping, floored so a one-week expiry still gets a real solve."""
    return max(MIN_TIME_STEPS, int(np.ceil(T * TIME_STEPS_PER_YEAR)))


def _log_spot_grid(S0: float, K: float, T: float, vol_ref: float, n_space: int, width: float) -> np.ndarray:
    """Uniform log-spot grid, centred so `log(S0)` is exactly the middle node.

    Centring on spot rather than on the strike means the price is read from
    a node instead of interpolated between two, which removes an error term
    that would otherwise vary from strike to strike across a ladder. The
    half-width is whichever is larger: `width` diffusive standard
    deviations, or enough to keep the strike well inside the grid -- a deep
    out-of-the-money strike must not sit on the boundary where the value is
    imposed rather than solved for.
    """
    sd = vol_ref * np.sqrt(T)
    half_width = max(width * sd, abs(np.log(K / S0)) + 2 * sd, 0.05)
    n_space = n_space if n_space % 2 else n_space + 1  # odd => log(S0) is a node
    return np.linspace(np.log(S0) - half_width, np.log(S0) + half_width, n_space)


def _cell_averaged_payoff(x: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """Exact average of the payoff over each grid cell, in log space.

    For a call the integrand is `max(e^u - K, 0)` over `[x-h/2, x+h/2]`,
    which integrates in closed form once the cell is clipped to the side of
    `log K` where the payoff is non-zero. Sampling the payoff instead would
    make every price depend on where `K` happened to land between two
    nodes; averaging makes the terminal condition a projection of the true
    payoff onto the grid, and the strike's position stops mattering.
    """
    h = x[1] - x[0]
    lo, hi, log_K = x - h / 2, x + h / 2, np.log(K)

    if option_type == "call":
        a = np.clip(lo, log_K, None)
        return np.where(hi > log_K, (np.exp(hi) - np.exp(a) - K * (hi - a)) / h, 0.0)

    b = np.clip(hi, None, log_K)
    return np.where(lo < log_K, (K * (b - lo) - (np.exp(b) - np.exp(lo))) / h, 0.0)


def _boundary_values(
    S_lo: float, S_hi: float, K: float, tau: float, r: float, q: float, option_type: str, american: bool
) -> tuple[float, float]:
    """Dirichlet values at the two ends, from the known deep-ITM/OTM limits.

    Far enough from the strike the option is either worthless or a forward:
    a deep in-the-money call is `S*e^(-q*tau) - K*e^(-r*tau)` and a deep
    out-of-the-money one is zero. For an American contract the boundary
    also cannot fall below intrinsic, since the holder could simply
    exercise -- taking the max is what lets the deep in-the-money put
    boundary switch to `K - S` once immediate exercise dominates waiting.
    """
    forward_lo, forward_hi = S_lo * np.exp(-q * tau), S_hi * np.exp(-q * tau)
    strike_pv = K * np.exp(-r * tau)

    if option_type == "call":
        low, high = 0.0, max(forward_hi - strike_pv, 0.0)
        if american:
            high = max(high, S_hi - K)
    else:
        low, high = max(strike_pv - forward_lo, 0.0), 0.0
        if american:
            low = max(low, K - S_lo)

    return low, high


def _brennan_schwartz(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray, obstacle: np.ndarray, exercise_low: bool) -> np.ndarray:
    """Exact tridiagonal LCP solve for a single-interval exercise region.

    The elimination runs from the end of the grid that is *always* in the
    continuation region, so by the time the substitution reaches the
    constrained end every value it depends on has already been fixed. That
    ordering is the whole trick: it turns the free boundary from something
    to be located into something the sweep runs into. For a put the
    exercise region sits at low spot (`exercise_low=True`) so the sweep
    ends there; for a call it is at high spot and the directions reverse.

    Exact only because a vanilla's exercise region is one interval --
    `psor` is the solver to use when that cannot be assumed, and the test
    suite holds the two against each other.
    """
    n = len(diag)
    d, b = diag.copy(), rhs.copy()
    v = np.empty(n)

    if exercise_low:
        for i in range(n - 2, -1, -1):  # eliminate the superdiagonal from the top end down
            factor = sup[i] / d[i + 1]
            d[i] -= factor * sub[i + 1]
            b[i] -= factor * b[i + 1]
        v[0] = max(b[0] / d[0], obstacle[0])
        for i in range(1, n):
            v[i] = max((b[i] - sub[i] * v[i - 1]) / d[i], obstacle[i])
    else:
        for i in range(1, n):  # eliminate the subdiagonal from the bottom end up
            factor = sub[i] / d[i - 1]
            d[i] -= factor * sup[i - 1]
            b[i] -= factor * b[i - 1]
        v[n - 1] = max(b[n - 1] / d[n - 1], obstacle[n - 1])
        for i in range(n - 2, -1, -1):
            v[i] = max((b[i] - sup[i] * v[i + 1]) / d[i], obstacle[i])

    return v


def _psor(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray, obstacle: np.ndarray, guess: np.ndarray) -> np.ndarray:
    """Projected SOR: Gauss-Seidel with over-relaxation, projected each sweep.

    Makes no assumption about the shape of the exercise region, which is
    what makes it the right reference for `_brennan_schwartz` -- and the
    wrong default, since it costs a few hundred sweeps where the direct
    solve costs one.
    """
    v = np.maximum(guess, obstacle)
    n = len(diag)

    for _ in range(PSOR_MAX_ITER):
        change = 0.0
        for i in range(n):
            lower = sub[i] * v[i - 1] if i > 0 else 0.0
            upper = sup[i] * v[i + 1] if i < n - 1 else 0.0
            candidate = max(v[i] + PSOR_OMEGA * (rhs[i] - lower - diag[i] * v[i] - upper) / diag[i], obstacle[i])
            change = max(change, abs(candidate - v[i]))
            v[i] = candidate
        if change < PSOR_TOL:
            break

    return v


def _exercise_boundary(values: np.ndarray, intrinsic: np.ndarray, S: np.ndarray, exercise_low: bool) -> float:
    """The critical spot separating exercise from continuation, or `NaN`.

    Read off as the last node on the exercise side whose value has been
    lifted to intrinsic. `NaN` means the constraint is not binding anywhere
    on the grid at this time step -- for a put that is the (correct)
    statement that immediate exercise is nowhere optimal yet, not a
    failure to find something.
    """
    exercised = (values <= intrinsic + EXERCISE_TOL) & (intrinsic > 0)
    if not exercised.any():
        return np.nan
    indices = np.flatnonzero(exercised)
    return float(S[indices[-1] if exercise_low else indices[0]])


def solve(
    S0: float,
    K: float,
    T: float,
    r: float = 0.04,
    q: float = 0.0,
    option_type: str = "put",
    sigma: VolFunction | float = 0.2,
    american: bool = False,
    n_space: int = DEFAULT_SPACE_NODES,
    n_time: int | None = None,
    lcp: str = "brennan-schwartz",
    rannacher: int = RANNACHER_STEPS,
    width: float = GRID_WIDTH_SDS,
) -> PDEResult:
    """Price one option by Crank-Nicolson on `log S`, with optional early exercise.

    `sigma` is either a constant or a callable `sigma(k, t)` taking forward
    log-moneyness and calendar time -- `surface_vol(local_vol_surface)`
    builds the latter. Vol is frozen at each step's *midpoint* in time,
    which keeps the scheme second order in `tau` rather than reducing it to
    first order by evaluating at an endpoint.

    `rannacher` is how many opening steps run fully implicit before
    Crank-Nicolson takes over; see the module docstring for why zero is a
    bad idea on a kinked payoff.
    """
    _check_inputs(T, option_type, n_space, lcp)

    sigma_at: VolFunction = constant_vol(float(sigma)) if np.isscalar(sigma) else sigma
    drift_rate = r - q
    vol_ref = float(np.mean(np.atleast_1d(sigma_at(np.zeros(1), T / 2))))
    if not np.isfinite(vol_ref) or vol_ref <= 0:
        raise ValueError(f"reference vol must be finite and positive, got {vol_ref}")

    x = _log_spot_grid(S0, K, T, vol_ref, n_space, width)
    S = np.exp(x)
    h = x[1] - x[0]
    n_time = n_time or default_time_steps(T)
    dtau = T / n_time

    values = _cell_averaged_payoff(x, K, option_type)
    intrinsic = np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)
    exercise_low = option_type == "put"

    interior = slice(1, -1)
    boundary = np.empty(n_time + 1)
    boundary[0] = np.nan  # at expiry every in-the-money node is "exercised"; not a boundary
    previous = values.copy()

    for step in range(n_time):
        tau_next = (step + 1) * dtau
        # Vol frozen at the step midpoint; `t` is calendar time, which the
        # surface is indexed by, so it counts down from T as tau counts up.
        k_mid = x - np.log(S0) - drift_rate * (T - tau_next + dtau / 2)
        sigma_step = np.asarray(sigma_at(k_mid, T - tau_next + dtau / 2), dtype=float)

        variance = sigma_step**2
        mu = drift_rate - 0.5 * variance
        a = 0.5 * variance / h**2 - mu / (2 * h)
        b = -variance / h**2 - r
        c = 0.5 * variance / h**2 + mu / (2 * h)

        weight = 1.0 if step < rannacher else 0.5
        low_new, high_new = _boundary_values(S[0], S[-1], K, tau_next, r, q, option_type, american)

        # Explicit half: (I + (1-weight)*dtau*L) applied to the old solution,
        # which already reaches the old boundary values through values[:-2]
        # and values[2:].
        rhs = values[interior] + (1 - weight) * dtau * (
            a[interior] * values[:-2] + b[interior] * values[interior] + c[interior] * values[2:]
        )
        # Implicit half's boundary contributions, at the new time level.
        rhs[0] += weight * dtau * a[1] * low_new
        rhs[-1] += weight * dtau * c[-2] * high_new

        sub = -weight * dtau * a[interior]
        diag = 1 - weight * dtau * b[interior]
        sup = -weight * dtau * c[interior]

        if american:
            obstacle = intrinsic[interior]
            if lcp == "brennan-schwartz":
                solution = _brennan_schwartz(sub, diag, sup, rhs, obstacle, exercise_low)
            else:
                solution = _psor(sub, diag, sup, rhs, obstacle, values[interior])
        else:
            banded = np.zeros((3, len(diag)))
            banded[0, 1:] = sup[:-1]
            banded[1] = diag
            banded[2, :-1] = sub[1:]
            solution = solve_banded((1, 1), banded, rhs)

        previous = values
        values = np.concatenate([[low_new], solution, [high_new]])
        if american:
            boundary[step + 1] = _exercise_boundary(values, intrinsic, S, exercise_low)

    centre = len(x) // 2
    dV_dx = (values[centre + 1] - values[centre - 1]) / (2 * h)
    d2V_dx2 = (values[centre + 1] - 2 * values[centre] + values[centre - 1]) / h**2

    return PDEResult(
        price=float(values[centre]),
        # x = log S, so d/dS = (1/S) d/dx and the second derivative picks up
        # the extra -dV/dx from the change of variables.
        delta=float(dV_dx / S0),
        gamma=float((d2V_dx2 - dV_dx) / S0**2),
        # tau counts down to expiry, so dV/dt = -dV/dtau.
        theta=float(-(values[centre] - previous[centre]) / dtau),
        intrinsic=float(intrinsic[centre]),
        S=S,
        values=values,
        tau=np.linspace(0.0, T, n_time + 1),
        exercise_boundary=boundary if american else None,
        n_space=len(x),
        n_time=n_time,
    )


def early_exercise_premium(
    S0: float,
    K: float,
    T: float,
    r: float = 0.04,
    q: float = 0.0,
    option_type: str = "put",
    sigma: VolFunction | float = 0.2,
    **kwargs,
) -> tuple[float, PDEResult, PDEResult]:
    """`(premium, american, european)` from two solves on an identical grid.

    Both solves share every discretization choice, so the difference is the
    early-exercise right and nothing else -- the leading discretization
    error is common to the two and cancels out of the subtraction, which
    matters because the premium is often smaller than either price's own
    error.
    """
    european = solve(S0, K, T, r, q, option_type, sigma, american=False, **kwargs)
    american = solve(S0, K, T, r, q, option_type, sigma, american=True, **kwargs)
    return american.price - european.price, american, european


def american_premium_chain(
    local_vol: LocalVolSurface,
    surface: pd.DataFrame,
    S0: float,
    r: float = 0.04,
    q: float = 0.0,
    as_of: dt.datetime | None = None,
    n_space: int = CHAIN_SPACE_NODES,
    lcp: str = "brennan-schwartz",
) -> pd.DataFrame:
    """What early exercise is worth, quote by quote, under the fitted surface.

    Every quote the surface covers is priced twice on an identical grid --
    once with the early-exercise constraint and once without -- and the
    difference is reported three ways: in currency, as a fraction of the
    European price, and in **vol points**, `premium / vega`. The last is the
    one that compares across the chain, and it is deliberately the same
    unit `reprice_chain` reports its errors in: an early-exercise premium
    smaller than the surface's own repricing error is not a number anyone
    should trade on, and putting both in vol points is what makes that
    judgement possible instead of rhetorical.

    Quotes outside the surface's own `(k, T)` window are skipped, exactly
    as in `reprice_chain` and for the same reason -- past the edge the
    solver reads a clamped local vol, so the premium would be a property of
    the clamp.

    `critical_spot` is where today's free boundary sits: the spot at which
    the holder of that contract should stop waiting and exercise. For an
    out-of-the-money quote it says how far the underlying has to move
    before the early-exercise right becomes live, which is the version of
    the question that has an answer there -- the contract is not exercisable
    now at any price. It is `NaN` when the boundary falls outside the
    solved grid rather than extrapolated to a number the solve never
    reached.

    With `q = 0` the American call premium is theoretically zero: there is
    never a reason to exercise a call early on a non-dividend-paying
    underlying. It is computed rather than assumed, so the zero column in
    the result is a measurement of the solver, not an assertion about it.
    """
    as_of = as_of or dt.datetime.now()
    sigma_at = surface_vol(local_vol)
    T_lo, T_hi = local_vol.T.min(), local_vol.T.max()
    k_lo, k_hi = local_vol.k.min(), local_vol.k.max()
    rows = []

    for expiry, group in otm_quotes(surface).groupby("expiry"):
        T = year_fraction(expiry, as_of)
        if not (T_lo <= T <= T_hi):
            continue

        k = np.log(group["strike"] / S0) - (r - q) * T
        group = group[(k >= k_lo) & (k <= k_hi)]

        for _, quote in group.iterrows():
            K, option_type = float(quote["strike"]), quote["option_type"]
            premium, american, european = early_exercise_premium(
                S0, K, T, r, q, option_type, sigma_at, n_space=n_space, lcp=lcp
            )
            vega = bs_greeks(S0, K, T, r, quote["iv"], option_type, q).vega
            rows.append(
                {
                    "expiry": expiry,
                    "strike": K,
                    "moneyness": K / S0,
                    "option_type": option_type,
                    "T": T,
                    "market_iv": quote["iv"],
                    "european": european.price,
                    "american": american.price,
                    "premium": premium,
                    "premium_pct": premium / european.price if european.price > 0 else np.nan,
                    "premium_vol_points": premium / vega if vega > 0 else np.nan,
                    "critical_spot": american.critical_spot_now,
                }
            )

    return pd.DataFrame(rows, columns=PREMIUM_COLUMNS)
