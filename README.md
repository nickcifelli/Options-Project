# Options Pricer & Implied Volatility Surface

[![tests](https://github.com/nickcifelli/Options-Project/actions/workflows/tests.yml/badge.svg)](https://github.com/nickcifelli/Options-Project/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

A small, self-contained Python project covering the core of derivatives
pricing: closed-form European pricing, American early-exercise pricing via
backward induction, implied vol extraction, an arbitrage-free volatility
surface fit to live SPY options, and the Dupire local vol surface that
follows from it -- validated by pricing the same quotes back two
independent ways.

This is a pricing/analysis tool, not a trading strategy. Nothing here claims
tradable edge.

![SPY implied vol smile, SSVI surface, Dupire local vol surface, Monte Carlo repricing, and early exercise premium](surface.png)

*Pictured: `--max-expiries 22 --fit ssvi --k-window union --reprice
--american` (see [Why the pictured run widens the
ladder](#why-the-pictured-run-widens-the-ladder) and [Widening the
window](#widening-the-window)).*

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run

```
.venv/bin/vol-surface --ticker SPY --out surface.png
```

or `python -m vol_surface.cli ...`. Runs `fetch -> build -> fit -> check ->
local vol -> reprice -> american -> plot`: pulls a live options chain,
solves implied vol per contract, fits the smile, checks put-call parity
plus the two static no-arbitrage conditions, derives the Dupire local vol
surface, and saves (or shows) the panels above.

Three flags do the interesting work:

- `--fit {svi,ssvi}` picks the smile fit. `svi` fits each expiry
  independently -- the closest fit to the quotes, and the one nothing
  prevents from crossing in `T`. `ssvi` fits **one surface to every expiry
  at once**, which gives up some closeness and rules calendar arbitrage out
  by construction instead of reporting it. See [Ruling calendar arbitrage
  out by construction](#ruling-calendar-arbitrage-out-by-construction).
- `--reprice` Monte Carlos the fitted local vol surface and checks it
  reprices the quotes it was built from.
- `--american` prices every quote again under early exercise, by
  Crank-Nicolson on the same surface, and reports what the right is worth.
  See [American exercise under local
  vol](#american-exercise-under-local-vol).

The last two are off by default because they are the stages that cost real
time (a few seconds and about half a minute respectively on the pictured
chain). See `--help` for the rest (rate, dividend yield, liquidity
thresholds, near-expiry cutoff, expiry-gap thinning, moneyness window, path
count, seed, cache control).

## Tests

```
.venv/bin/pytest
```

## Repo layout

```
vol_surface/
  pricing/
    black_scholes.py     # price + Greeks, closed form
    binomial.py           # CRR tree, European + American
    implied_vol.py         # brentq-based IV solver
    monte_carlo.py       # local vol MC + the repricing check
    pde.py                  # Crank-Nicolson on log S; American exercise under local vol
  data/
    chain.py               # fetch + clean live options chain via yfinance
  surface/
    build.py               # strike/expiry -> IV grid construction
    svi.py                   # SVI curve fit per expiry (Gatheral raw parameterization)
    ssvi.py                 # SSVI: one global surface fit, calendar-arbitrage-free by construction
    parity.py               # put-call parity check across the chain
    arbitrage.py            # butterfly + calendar no-arbitrage checks on the fit
    local_vol.py            # Dupire local vol, derived from the fitted slices
  viz/
    plots.py                # smile + fit, IV/local vol surfaces, repricing, early exercise
  cli.py                     # entry point: fetch -> ... -> american -> plot
tests/
data/                        # gitignored, cached raw chain snapshots
```

## Why American exercise is hard

The European Black-Scholes price is a closed form: given `(S, K, T, r, sigma)`
there's a formula. American exercise has no closed form because the holder
can exercise at any time before expiry, so pricing it means solving an
**optimal stopping problem**: at every point in time, compare the payoff from
exercising now against the expected discounted value of continuing to hold.

The binomial tree (`pricing/binomial.py`) solves this by backward induction.
At each node:

```
value = max(exercise_value, discounted_continuation_value)
```

That single `max()` is the entire reason American pricing is structurally
harder than European. There's no way to skip it and land on a formula --
the early-exercise boundary itself is part of what you're solving for.

That is also why the same problem comes back at the end of this README, in
a harder form: once the flat `sigma` is replaced by a fitted local vol
*surface*, the tree can no longer take the input and the Monte Carlo can no
longer answer the question. `pricing/pde.py` is what closes it.

## What's shown here

The pictured run against SPY (22 expiries, spanning ~3 days to ~5 months
out) pulled 5,340 quotes that passed the liquidity filter (non-zero
volume/OI, uncrossed bid-ask), solved implied vol for 4,211 of them, and
flagged 829 of 2,113 call/put pairs as deviating from put-call parity by
more than their combined bid-ask spread -- the rest of the drop from 5,340
to 4,211 is mostly contracts priced outside the no-arbitrage bounds implied
by the observed mid (stale quotes) plus the near-expiry exclusion described
below. Both fits converged on all 21 expiries that survived the near-expiry
cutoff.

**Skew shape.** SPY (like most equity index underlyings) shows a downward
skew: implied vol rises as strike falls below spot and is lowest near/just
above the money. This is the standard equity index pattern -- it reflects
persistent demand for downside protection (crash risk hedging) pricing OTM
puts richer than OTM calls, not a data artifact.

**Term structure.** Shorter-dated expiries trade at a visibly higher implied
vol than longer-dated ones, in both the smile panel and the 3D surface. The
*raw* smile panel is visibly noisier at the short end too (see the numerical
caveat below); the fitted 3D surface isn't, by construction.

**SVI fit.** `surface/svi.py` fits Gatheral's raw SVI parameterization to
each expiry's smile, subject to the constraint that keeps total variance
non-negative everywhere, not just at the observed strikes. One nuance worth
calling out: SPY's listed strikes go much deeper on the put side than the
call side, so an unweighted fit over the full ladder lets the put wing's far
larger variance range dominate the objective and drags `rho` to its -1
boundary -- degenerating the call wing into a flat line. `fit_svi_slice`
guards against this by truncating each slice to a symmetric log-moneyness
window sized to the shorter wing before fitting, so neither wing can
out-vote the other on point count or range alone. That guard turns out to
be load-bearing for the global fit too, and there it is [measured rather
than argued](#ruling-calendar-arbitrage-out-by-construction).

**3D surface.** The pictured surface (`plot_svi_surface_3d`) is built by
sampling the fitted curves, not by triangulating the raw quotes directly --
smooth, and free of the negative-variance arbitrage the fit already
excludes. `plot_surface_3d` (the raw triangulation) is still there as the
unsmoothed, purely diagnostic view; the CLI falls back to it automatically
if every fit in a run fails to converge.

**No-arbitrage checks.** A curve that fits the market points well can still
be an invalid surface, and `surface/arbitrage.py` checks the two conditions
that goodness of fit does not imply:

- **Butterfly** (within one expiry): the implied risk-neutral density must
  be non-negative. Gatheral's `g(k) >= 0` is that condition written in terms
  of total variance -- `g < 0` means the slice prices an infinitesimal
  butterfly negatively, i.e. pays you to hold a non-negative payoff. This is
  *strictly stronger* than the non-negative-variance constraint
  `fit_svi_slice` already enforces, and the test suite pins that down with a
  curve whose total variance stays positive everywhere while its density
  still goes negative.
- **Calendar** (across expiries): total implied variance must be
  non-decreasing in `T` at fixed forward log-moneyness. A drop means a
  longer-dated option is cheaper than a shorter-dated one covering the same
  events, which a calendar spread monetizes directly.

Both conditions are stated in *forward* log-moneyness `k = log(K/F)`, while
`surface/build.py` works in spot log-moneyness `log(K/S)`; the checks
convert by the drift `(r-q)T` rather than conflating the two. That
distinction is not pedantry -- it is exactly what the section below turned
out to depend on.

On the pictured chain, fitting each expiry independently leaves every slice
butterfly-clean (0 of 21) but **7 of 20 adjacent expiry pairs showing a
total-variance drop** -- worst, 2027-01-15 sitting above 2027-01-29 by
0.00098 at `k = +0.173`. Refitting the same quotes as one surface takes that
to **0 of 20**, and the rest of this section is about what that costs.

## Ruling calendar arbitrage out by construction

Fitting each expiry independently spends five free parameters per slice with
nothing tying them together in `T`. Nothing *prevents* the slices from
crossing, so on a real chain they do. Reporting the violations is the useful
half of the answer; the other half needs a different fit.

`surface/ssvi.py` is that fit. SSVI (Gatheral & Jacquier, *Arbitrage-free
SVI volatility surfaces*, 2014) writes total variance as a single function
of log-moneyness and the at-the-money total variance `theta_T`:

```
w(k, theta) = (theta/2) * (1 + rho*phi(theta)*k
                           + sqrt((phi(theta)*k + rho)**2 + 1 - rho**2))
```

`rho` and the function `phi` are shared by **every** expiry; the only thing
an expiry owns is its own `theta_T`. On the pictured chain that is 24 free
parameters across 21 expiries instead of 105, and the term structure becomes
a property of the surface rather than an accident of 21 separate
optimizations.

**Every SSVI slice is still a raw SVI slice.** Expanding the square root and
matching coefficients gives an exact reparameterization,

```
a = theta*(1 - rho**2)/2      b = theta*phi/2
m = -rho/phi                  sigma = sqrt(1 - rho**2)/phi
```

so `fit_ssvi_surface` hands the rest of the project the same
`SVIFitResult` objects the per-expiry fit does. The arbitrage checks, the
Dupire construction, the Monte Carlo and the plots all run unchanged --
which is the point, because it means **the calendar checker that flagged 7
of 20 pairs is the same unmodified code that now reports 0 of 20**. The test
suite holds the two parameterizations together to `1e-14`.

**The no-arbitrage conditions become constraints on three numbers.** With
the power law `phi(theta) = eta * theta**-gamma`, `gamma` in `(0, 1/2]`,
Gatheral and Jacquier's two theorems collapse into something an optimizer
can carry:

*Calendar* (their Theorem 4.1) is `d(theta)/dt >= 0` together with

```
0 <= d/dtheta (theta*phi(theta)) <= (1 + sqrt(1-rho**2))/rho**2 * phi(theta)
```

Under the power law `theta*phi = eta*theta**(1-gamma)`, so the middle term
is `(1-gamma)*phi(theta)` and `phi > 0` divides out of the whole chain: the
condition reduces to `1 - gamma <= (1 + sqrt(1-rho**2))/rho**2`. The right
side is decreasing in `rho**2` with minimum 1, and `1 - gamma < 1`, so it
holds identically -- for every `theta`, not just the fitted ones.
**Calendar-arbitrage freedom is therefore exactly the statement that `theta`
is non-decreasing**, which the fit imposes by parameterizing `theta` as a
cumulative sum of non-negative increments. It is a box constraint the
optimizer cannot step out of, not a penalty discouraging it. A test asserts
the inequality across the whole admissible `(rho, gamma)` rectangle rather
than restating the derivation.

*Butterfly* (their Theorem 4.2) is sufficient rather than necessary, and is
genuinely a constraint:

```
theta*phi(theta) * (1 + |rho|) < 4      theta*phi(theta)**2 * (1 + |rho|) <= 4
```

Under the power law both are non-decreasing in `theta` when `gamma <= 1/2`,
so their suprema over the fitted term structure sit at the longest expiry
and two scalars cover the whole surface. The power law breaks them at large
enough `theta`, which is why they are imposed on the optimizer instead of
assumed -- and why `check_butterfly` is still worth running afterwards. On
the pictured chain it comes back 0 of 21, with `min g = 0.270` against
0.129 for the independent fit.

**A coordinate bug the synthetic tests caught.** SSVI's guarantee is a
statement about *forward* log-moneyness, and the surface is built in spot.
Fitting a monotone `theta` in spot coordinates makes the slices monotone
along a set of curves that slide sideways as `T` grows, which is not the
calendar condition and does not imply it. A test that feeds the fit a
deliberately inverted term structure and asserts the unmodified checker
finds nothing came back with four flagged pairs, which is how the
conversion in `_slice_data` got there. The fit runs in forward coordinates
and maps the slices back to spot on the way out.

**What it costs.** Three shared parameters cannot follow a per-expiry smile
as closely as five free ones. Same chain, same quotes:

| | independent SVI | global SSVI |
| --- | --- | --- |
| free parameters | 105 | **24** |
| fit residual (RMSE, in-domain) | **0.63 vol pts** | 0.98 |
| butterfly violations | 0 / 21 | 0 / 21 |
| calendar violations | 7 / 20 | **0 / 20** |

The fit is worse by construction, and `SSVIFit.rmse_vol` reports how much
worse so the trade is measured rather than argued. Which is *better* is not
a question the fit residual answers, though -- that is what the repricing
check downstream is for, and the answer there runs the other way.

## Widening the window

The Dupire construction can only run where every slice was fit, so the local
vol surface spans the **intersection** of the slice windows. On the pictured
chain that intersection is 0.022 wide in log-moneyness for the independent
fit and 0.012 for SSVI -- about one percent either side of the forward.

Both edges are set by a single expiry. 2026-10-23 carries 26 liquid
out-of-the-money quotes, almost all of them within a percent of the
forward; after the wings are balanced its fitted window is
`[-0.0054, +0.0062]`, and that one thin ladder *is* the intersection --
both the lower and the upper edge, for all 21 expiries.

That is not only a reporting limit. The simulation clamps local vol at the
edge of the grid, so on a window 0.012 wide essentially every path leaves it
within one step and is stepped forward on a clamped value. The surface stops
being a local vol surface and becomes two constants. It shows up exactly
where you would expect: only 126 quotes fall inside the window at all, and
they reprice at 1.11 vol points.

`--k-window union` gives each SSVI slice the widest ladder quoted anywhere
on the surface. **This is defensible only because the fit is global**: a
slice's wings are set by `rho` and `phi`, which every expiry's quotes helped
estimate, so a front-week slice evaluated at a strike only the back months
list is being interpolated in the shared parameters rather than extrapolated
from its own three points. The same move on independently fitted slices
would be extrapolating a curve past the data that produced it.

It widens the local vol window from 0.012 to **0.874**, and the repricing
check goes from 126 quotes to 2,741. The cost is real and shows up as
distance past each slice's own fitted domain:

| distance outside the slice's own domain | quotes | fit error | repricing error |
| --- | --- | --- | --- |
| inside | 2,101 | **0.52 vol pts** | **0.63 vol pts** |
| 0.00 - 0.05 | 285 | 1.49 | 1.34 |
| 0.05 - 0.10 | 149 | 4.55 | 4.31 |
| 0.10 - 0.20 | 179 | 6.45 | 5.19 |
| 0.20 - 0.50 | 82 | 10.84 | 7.05 |

The two columns track each other closely, which is the useful part: the
repricing error out there is the *fit's* extrapolation error arriving
downstream, not a separate failure of the local vol construction.

On the 126 quotes all three configurations can speak to, the ordering is
unambiguous:

| | quotes repriced | median error | 95th pct |
| --- | --- | --- | --- |
| independent SVI | 244 | 1.20 vol pts | 2.37 |
| SSVI, slice window | 126 | 1.11 | 3.41 |
| SSVI, union window | 2,741 | **0.71** | **1.77** |

So the global fit gives up 0.35 vol points of RMSE against the quotes and
buys back 41% of the repricing error, on eleven times as many contracts as
the independent fit could speak to at all. That is the trade, and it only
goes that way because widening the window is what lets the diffusion see the
surface in the first place.

**The wing-balancing guard is what makes this work.** Truncating each slice
to a symmetric window before fitting was introduced for the per-expiry fit,
where one wing can out-vote the other within a single slice. It is not
obvious it should survive into a global fit, where `rho` is estimated from
every expiry at once -- so it was tested rather than assumed. Removing it:
`rho` drops from -0.473 to -0.599, the fit residual rises from 0.98 to 3.57
vol points, and the union window's repricing median goes from 0.80 to **2.24
vol points**. It stays.

## Local vol

`surface/local_vol.py` applies Dupire's formula to the fitted slices.
Implied vol is an *average* of volatility along all paths to one strike and
expiry; local vol is the *instantaneous* vol the underlying must have at a
given spot and time to reproduce the whole surface. In terms of total
variance,

```
sigma_LV**2(k, T) = (dw/dT) / g(k)
```

where `g` is exactly the butterfly function above -- not a coincidence but
the same expression, which the test suite asserts to machine precision
against an independent transcription of Gatheral eq. 1.10. That identity is
what ties the two modules together: the denominator of local variance *is*
the butterfly condition, so a slice implying a negative density produces a
negative local variance at precisely the same strikes. Local vol does not
merely look nicer on an arbitrage-free surface -- it fails to exist without
one, and those grid points are returned as `NaN` and rendered as holes
rather than square-rooted into a complex number.

Because raw SVI is differentiable in closed form, `dw/dk` and `d2w/dk2`
carry no discretization error at all. `dw/dT` is the exception, and it is
where the accuracy floor sits.

## Closing the loop: does the surface reprice its own quotes?

Dupire's construction says a diffusion carrying the fitted local vol
reprices the vanillas the surface was built from. `pricing/monte_carlo.py`
simulates that diffusion and checks it, which is what turns the claim into a
measurement -- and the check is not circular, because the surface it runs on
is not assumed to be perfect. Under the independent fit the slices cross in
`T`, and where they cross the local vol the repricing depends on doesn't
exist. Under SSVI they cannot cross, and what is left to measure is the
extrapolation the widened window buys.

**On the pictured run: median error 0.80 vol points across 2,741 quotes and
21 expiries** (0.63 inside each slice's own fitted domain, per the table
above).

**Scheme.** Log-Euler on `ln S`, which removes the drift discretization
entirely and keeps `S` positive by construction. Local vol is frozen over
each step and read at the path's *forward* log-moneyness, matching the
coordinate the surface is built in. Freezing leaves an `O(dt)` bias that is
separate from Monte Carlo noise and is not reduced by adding paths -- step
count is what controls it, and the test suite asserts it shrinks as steps
rise rather than assuming it. As of `pricing/pde.py` there is now a
reference value to measure it *against*; see below.

**Variance reduction**, both switchable so their effect is measured rather
than asserted:

- *Antithetic variates*: every path is simulated alongside its mirror. The
  two are not independent, so the standard error is computed across
  antithetic **pairs** rather than across paths -- averaging the pair first
  is what keeps the error bar honest rather than flattering.
- *A control variate*: the same Brownian increments also drive a
  constant-vol GBM whose exact price is known from Black-Scholes, with
  `beta` fit by least squares. Since the control's mean is known
  analytically the estimator stays unbiased whatever `beta` is; `beta` only
  decides how much variance goes away. Simulating it is free -- log-Euler is
  exact for constant-vol GBM, so accumulating the Brownian path during the
  same loop prices it in closed form at the end. On a flat surface, where
  the control *is* the simulated process, this collapses the standard error
  by five orders of magnitude (`4.0e-02` to `6.7e-07`) and returns the
  Black-Scholes price to seven decimal places.

The control's reference vol is deliberately the surface's **at-the-money**
implied vol for the expiry, not the per-strike market vol. The latter would
make the control almost exactly the answer and hollow out the repricing test
it exists to support.

**What the check caught.** Before any of this was tuned, expiries inside 9
days repriced 1.50 vol points low while everything past 15 days was near
perfect -- a bias too structured to be noise. The cause was the front slice's
`dw/dT`: `np.gradient`'s edge rule extrapolates the front derivative
backwards from the *next two* expiries, discarding something already known
exactly, that total variance is zero at `T = 0`. Anchoring the difference
there cut the front bias to 0.28 vol points and the 95th-percentile error
across all expiries from 3.07 to 2.60. It cost a little at the median,
because the front row also anchors the interpolation out to the second
expiry; the anchored derivative is kept anyway, on the grounds that using a
known boundary value beats extrapolating past it. Fixing what the validation
finds is the point of having built it -- but the fix was taken on that
reasoning, not by tuning the construction until the metric looked best.

## American exercise under local vol

The Monte Carlo prices European payoffs only, and not by choice. Forward
simulation has no natural way to ask "would the holder have exercised
here?", because the continuation value at a path's current state is exactly
what has not been computed yet. The binomial tree does answer that question,
but it takes a single `sigma` and cannot be handed a surface. So the two
stretch goals -- American Greeks, and American exercise under local vol --
were blocked by the same gap.

`pricing/pde.py` closes it. A backward PDE solve carries the entire
continuation surface at every time step, so the early-exercise test is a
pointwise comparison already in hand. On `x = log S`, in time-to-expiry
`tau = T - t`:

```
dV/dtau = (1/2)*sigma(x, tau)**2 * d2V/dx2
          + (r - q - (1/2)*sigma(x, tau)**2) * dV/dx - r*V
```

with `sigma` read from the fitted surface at each node's own forward
log-moneyness. Log space is not cosmetic: it makes the diffusion coefficient
independent of the node, so a uniform grid is already well scaled, and it
keeps `S > 0` the same way log-Euler does in the Monte Carlo.

**Discretization.** Crank-Nicolson, second order in both axes, with two
fully implicit steps at the start -- Rannacher startup. Crank-Nicolson is
A-stable but not L-stable: it damps high-frequency error slowly, and the
kink in a vanilla payoff *is* high-frequency error. Left alone it produces
the familiar sawtooth in gamma near the strike. That is not a cosmetic
complaint and the test suite states it as an invariant rather than an
impression: on a deliberately coarse time grid, plain Crank-Nicolson returns
a **negative gamma** for a convex payoff (-0.022), which is impossible; two
implicit steps take the minimum to -5e-8.

The terminal condition is *cell-averaged* rather than sampled -- node `i`
gets the exact integral of the payoff across its own cell, available in
closed form in log space. Sampling a kinked payoff makes the answer depend
on where the strike happens to fall between two nodes, which looks like
noise across a strike ladder and does not shrink cleanly under refinement.
The grid is centred so `log S0` is exactly a node, so the price is read off
rather than interpolated.

**Early exercise** turns each step into a linear complementarity problem.
Two solvers, which is the point of having two:

- `brennan-schwartz` (default) is a single modified tridiagonal sweep, run
  in the direction that meets the exercise region last. It is *exact* for
  the vanilla American LCP, where the exercise region is a single interval
  (Jaillet, Lamberton & Lapeyre, 1990), and costs the same as the European
  solve.
- `psor` is projected successive over-relaxation, iterated to a tolerance.
  Slower, and assumes nothing about the shape of the exercise region.

They agree to `1e-10` on the same problem, with the direct solve about 20x
faster. That agreement is what licenses the default: Brennan-Schwartz is
exact only under a hypothesis about the free boundary that a vanilla
satisfies and an arbitrary payoff need not.

**Greeks come out of the solve, not out of a second one.** The backward
sweep produces the option value at every spot on the grid on the way to the
one that was asked for, so delta and gamma are finite differences of an
array already in hand. That matters most for American contracts, where a
bump-and-reprice delta costs as much as the price and inherits the free
boundary's kink. Against Black-Scholes on a flat surface the grid delta and
gamma match to five decimal places.

**What it says about SPY.** On the pictured surface, 2,796 quotes priced
twice each:

| time to expiry | put quotes | median premium | as % of price |
| --- | --- | --- | --- |
| 0 - 0.05y | 499 | 0.012 vol pts | 0.46% |
| 0.05 - 0.1y | 472 | 0.037 | 0.92% |
| 0.1 - 0.2y | 287 | 0.057 | 0.97% |
| 0.2 - 0.5y | 569 | 0.090 | 1.39% |

The richest is the 770 put expiring 2027-01-29: $0.498, or 2.25% of its
European value, or **0.263 vol points**. And that is the number worth
stopping on. The surface's own repricing error is 0.63 vol points *inside*
its fitted domain. **On SPY at a 4% rate, the entire early-exercise premium
sits comfortably below the noise floor of the surface it is computed on.**
Reporting it in vol points is what makes that comparison possible instead of
rhetorical; reporting it in dollars would have made it look like a result.

The call side comes back at `2.8e-14`. With no dividend yield there is never
a reason to exercise a call early, and the solver is not told so -- the
column is a measurement of the LCP solver, which is why it is computed and
plotted rather than skipped.

**Where local vol actually changes the answer** is the free boundary, not
the premium. For that same 770 put, the critical spot today is **494.97**,
0.64x spot. Priced at a constant vol equal to the surface's at-the-money
implied vol for that expiry (0.136), it is **693.09**, 0.90x spot. The skew
means local vol rises steeply as spot falls, so the option the holder would
be giving up by exercising is worth far more than a flat-vol model thinks,
and the boundary drops a long way to compensate. The price moves too, 21.54
to 22.66. A flat-vol American model does not get the exercise decision
approximately right here; it gets it wrong by a third of spot.

## Two methods, one surface

The Monte Carlo and the PDE share exactly one thing: `LocalVolSampler`, the
single reading of the fitted surface both step through. Beyond that they
have nothing in common -- one integrates forward with random draws, the
other backward with linear algebra. So agreement on a European price is a
statement about the surface rather than about a shared implementation, and
it is the check that would catch a coordinate error the flat-surface tests
cannot see.

They agree. Across nine contracts on the pictured surface spanning
`T = 0.05` to `0.30` and 7% either side of the money, eight land within 1.4
Monte Carlo standard errors of the PDE.

The ninth is the interesting one, and it is the shortest-dated, furthest
out-of-the-money quote -- exactly where the Monte Carlo's `O(dt)` freezing
bias is largest relative to a small price. That bias was previously
something the test suite could only show was *decreasing*. The PDE supplies
the value it is decreasing towards:

| MC steps (ATM call, T=0.15) | price | bias vs. PDE |
| --- | --- | --- |
| 38 (daily) | 17.980 | +0.269 |
| 76 | 17.834 | +0.123 |
| 152 | 17.784 | +0.073 |
| 304 | 17.762 | +0.051 |
| 608 | 17.717 | +0.006 |

Halving `dt` roughly halves the bias, which is the `O(dt)` the module
docstring claims, and by 608 steps the two methods agree inside the Monte
Carlo's own standard error of 0.016. At the default daily stepping the
Monte Carlo is about 1.5% high on this contract -- worth knowing, and not
knowable before there was something to compare against.

The obvious next move was to fix the Monte Carlo scheme, and two candidates
were tried and dropped:

- Reading local vol at the step *midpoint* in time barely moves it: on the
  same contract the daily-stepping bias goes from +0.272 to +0.250. The bias
  is dominated by `sigma` depending on the path's own state, not on the
  clock, and moving the clock reading does not touch that.
- A naive predictor-corrector -- Euler-predict the step, re-read `sigma`
  there, average -- is much *worse*, and worse in a way that does not
  converge: the bias runs -2.06 at 38 steps and -2.72 at 608. Re-reading
  `sigma` at a predicted state correlates it with the very Brownian
  increment it then multiplies, so `E[sigma*Z]` is no longer zero and the
  scheme stops being a martingale. The project's own martingale acceptance
  check says so directly: `E[e^-rT S_T]` comes back 753.5 against a spot of
  770.2, where the plain scheme returns 770.3.

The bias is a property of freezing a state-dependent `sigma` over a step,
and step count is what controls it. Fixing it properly means a scheme that
carries `dsigma/dS` (Milstein and friends), which is a different piece of
work than a re-read.

## Why the pictured run widens the ladder

SPY lists *daily* expiries at the front, so the default `--max-expiries 6`
selects the first six listed expiries -- which is six consecutive *days*,
not the multi-month span the term structure lives on. The pictured run
therefore passes `--max-expiries 22` to reach out to ~5 months.

That density is also a numerical problem, not just a coverage one.
Differencing total variance across a one-day gap divides each slice's own
fit residual by `1/365`, amplifying it roughly 365-fold. Measured directly
on a 19-slice live chain: every row whose neighbours sat one day apart
produced local vols ranging from 0.007 to 0.905 with up to 54% of the row
undefined, while every row with a gap of five days or more stayed inside a
0.08-0.40 band with no gaps at all.

`build_local_vol_surface` therefore thins the slices used for `dw/dT` to a
minimum spacing (`--min-expiry-gap-days`, default 7). On the pictured run
that leaves 12 of 21 slices and takes local vol coverage to 100%. The
thinning is deliberately *not* applied to the arbitrage checks: a variance
drop between two consecutive daily expiries is a real property of the fitted
surface and worth reporting. Only the division by a tiny `dT` is
ill-conditioned, so only the differencing is thinned.

## A numerical caveat worth stating explicitly

Very short-dated, deep ITM/OTM contracts have near-zero vega (`dPrice/dSigma`
scales roughly with `sqrt(T)`), so a one-cent bid-ask difference can swing
the *inverted* implied vol by tens of vol points. This was confirmed on live
data: a 1-day SPY call with 56 contracts of volume, 35 open interest, and a
3% bid-ask spread -- a perfectly liquid quote -- still produced an outlier
IV relative to neighboring strikes purely because of low vega, not a
liquidity problem. `build_surface()` exposes a `min_T` parameter (CLI:
`--min-days-to-expiry`, default 2 days) to exclude the contracts where this
instability dominates. It's a real property of Black-Scholes inversion, not
something a liquidity filter can catch, and pushing the cutoff further out
would trade away real near-term term structure for cosmetic smoothness --
so some residual noise in the nearest expiry is left visible rather than
filtered away.

The same instability is why the smile panel's axes are held to the fitted
region. A live SPY chain lists puts out to `log(K/S) = -2`, where a
near-worthless quote inverts to an implied vol near 1.0; left unbounded
those points compress the actual smile into a sliver. They are still in the
data, and `check_butterfly` still declines to judge the curve out there for
the same reason.

## What this isn't

- No exotic options (barriers, Asians, lookbacks).
- No stochastic vol models (Heston, SABR) -- SVI and SSVI fit the smile
  shape, not its dynamics.
- No live trading, execution, or backtested P&L claims.
- No multi-asset portfolio risk.
- Dividend yield is a flat continuous `q`, not a real dividend schedule.
  This matters more now than it used to: the American call premium is zero
  *because* `q = 0`, and a real dividend schedule would put discrete
  early-exercise dates into the problem that a flat `q` cannot represent.
- Put-call parity checks are a data-quality sanity check on the chain, not a
  trading signal.
- Butterfly arbitrage is still *reported*, not enforced: the SSVI constraint
  is Gatheral-Jacquier's sufficient condition, so a surface satisfying it is
  butterfly-free, but the constraint is imposed at the fitted term structure
  rather than proven for all `theta`. The calendar condition *is* enforced,
  exactly and everywhere.
- The independent per-expiry fit (`--fit svi`, the default) makes no
  calendar guarantee at all. It is kept as the default because it is the
  closest fit to the quotes and because the comparison between the two is
  worth being able to run.
- Local vol is the Dupire surface implied by today's quotes, not a
  calibrated model with dynamics. It reprices vanillas by construction and
  says nothing about whether the smile *evolves* the way it assumes.
- The Monte Carlo prices European vanillas only. Early exercise is the
  PDE's job.
- The PDE prices one contract per solve. There is no dual/forward
  (Dupire-equation) formulation here that would price a whole strike ladder
  in one sweep, which is why `--american` costs about half a minute on a
  2,800-quote chain.

## Acceptance checks

- Binomial European price converges to the Black-Scholes analytic price as
  `N` grows (tested within 0.5% at `N=200`, 0.1% at `N=1000`).
- The IV solver round-trips: pricing a known `(S,K,T,r,sigma)` with
  Black-Scholes and recovering `sigma` matches to `1e-6`.
- American price >= European price holds as a direct, asserted invariant
  across ITM/OTM/ATM cases, on both the tree and the PDE (and American call
  == European call with no dividends, the standard theoretical result).
- The SVI fit round-trips: fitting a smile generated from known SVI
  parameters recovers the same curve to within `1e-3` in implied vol.
- An SSVI slice equals its raw-SVI reparameterization to `1e-14`, and
  fitting a surface generated from known `(rho, eta, gamma)` recovers all
  three to `1e-3`.
- Fed a deliberately inverted term structure, the *same unmodified*
  `check_calendar` flags the independent fit and clears the SSVI fit, whose
  fitted `theta` comes back non-decreasing.
- Gatheral-Jacquier's calendar condition is asserted slack across the whole
  admissible `(rho, gamma)` rectangle, and both butterfly quantities are
  asserted non-decreasing in `theta` -- the two facts that let the
  constraints be imposed on three numbers at one expiry.
- Gatheral's `g(k)` matches an independent transcription of the Dupire
  local-variance denominator (eq. 1.10) to `1e-14` -- the identity
  `surface/local_vol.py` relies on to divide by it.
- The risk-neutral density implied by a fitted slice integrates to `1.0`
  within `1e-6`, and goes negative on exactly the strikes where `g < 0`.
- Local vol round-trips: a flat surface (`w = sigma**2 * T`, no smile)
  returns local vol == implied vol == `sigma` to `1e-10`.
- Local vol reproduces Derman's rule of thumb -- near the money the local
  vol skew is ~2x the implied vol skew (asserted within 0.25 at the front
  expiry), and the ratio decays with maturity as the short-dated
  approximation predicts.
- Monte Carlo on a flat local vol surface lands within 3 standard errors of
  the Black-Scholes price, and with the control variate on -- where the
  control *is* the simulated process -- returns it to `1e-5` with a standard
  error under `1e-4`.
- The simulated discounted spot is a martingale, `E[e^-rT S_T] = S0 e^-qT`
  within 3 standard errors: the acceptance check on the Ito drift term.
- Put-call parity holds on Monte Carlo prices struck from shared paths, and
  on PDE prices to `1e-3`.
- The Monte Carlo standard error falls as `1/sqrt(N)` (4x the paths halves
  it, asserted within 0.25), and the discretization bias falls with step
  count -- separately, since only one of the two responds to adding paths --
  and towards the PDE's answer specifically.
- The PDE matches Black-Scholes to `1e-3` across five strikes and both
  option types, its grid delta and gamma match the closed forms to `1e-4`
  and `1e-5`, and its error falls by more than 10x under an 8x refinement.
- The American PDE matches a 4,000-step binomial tree to `2e-3` across five
  strikes, and for a call under a 6% dividend yield.
- The two LCP solvers agree to `1e-8` across the whole grid, on a put and on
  a dividend-paying call, whose exercise regions sit at opposite ends.
- Rannacher startup keeps gamma non-negative on a grid where plain
  Crank-Nicolson returns -0.022 for a convex payoff.
- The cell-averaged terminal condition matches numerical quadrature to
  `1e-6`, and prices are insensitive to where the strike falls between two
  nodes.
- The American put's early-exercise premium is increasing in `r`, and
  vanishes at `r = 0`.
- The free boundary stays below the strike and rises towards it as expiry
  approaches; a deep in-the-money American put is priced at exactly
  intrinsic.
- The PDE and the Monte Carlo agree within 3 standard errors on a skewed
  local vol surface neither of them has a closed form for.

## Stretch goals (not implemented)

- Real dividend yield data instead of a flat assumed `q`, and with it a
  discrete dividend schedule -- the thing that would make the American call
  side of the early-exercise panel non-trivial.
- A second underlying for cross-asset comparison.
- eSSVI proper: letting `rho` vary by expiry rather than being shared. It
  buys back some of the fit quality SSVI gives up, at the cost of
  no-arbitrage conditions that are pairwise between slices rather than
  three global inequalities.
- A forward (Dupire-equation) PDE, which would price a whole strike ladder
  per solve instead of one contract, and make `--american` roughly as cheap
  as `--reprice`.
- A Monte Carlo scheme whose bias is better than `O(dt)`. The PDE now
  measures the bias precisely, so the work is at least well posed; two
  cheap attempts are documented above as failures.
- American exercise on the *raw* per-expiry fit as well, so the
  early-exercise premium could be compared across the two surfaces the way
  the repricing error is.
