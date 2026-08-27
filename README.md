# Options Pricer & Implied Volatility Surface

[![tests](https://github.com/nickcifelli/Options-Project/actions/workflows/tests.yml/badge.svg)](https://github.com/nickcifelli/Options-Project/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

A small, self-contained Python project covering the core of derivatives
pricing: closed-form European pricing, American early-exercise pricing via
backward induction, implied vol extraction, and a real implied vol surface
built from live SPY options data.

This is a pricing/analysis tool, not a trading strategy. Nothing here claims
tradable edge.

![SPY implied vol smile, SVI surface, and Dupire local vol surface](surface.png)

*Pictured: `--max-expiries 22 --reprice` (see [Why the pictured run widens
the ladder](#why-the-pictured-run-widens-the-ladder)).*

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
local vol -> reprice -> plot`: pulls a live options chain, solves implied
vol per contract, fits an SVI curve to each expiry's smile, checks put-call
parity plus the two static no-arbitrage conditions, derives the Dupire local
vol surface, and saves (or shows) the panel above.

`--reprice` adds the Monte Carlo stage, which simulates the fitted local vol
surface and checks it reprices the quotes it was built from. It is off by
default because it is the only stage that costs real time. See `--help` for
the full set of flags (rate, dividend yield, liquidity thresholds,
near-expiry cutoff, expiry-gap thinning, path count, seed, cache control).

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
  data/
    chain.py               # fetch + clean live options chain via yfinance
  surface/
    build.py               # strike/expiry -> IV grid construction
    svi.py                   # SVI curve fit per expiry (Gatheral raw parameterization)
    parity.py               # put-call parity check across the chain
    arbitrage.py            # butterfly + calendar no-arbitrage checks on the fit
    local_vol.py            # Dupire local vol, derived from the SVI slices
  viz/
    plots.py                # smile + SVI fit, IV/local vol surfaces, repricing
  cli.py                     # entry point: fetch -> ... -> reprice -> plot
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

## What's shown here

The pictured run against SPY (22 expiries, spanning ~3 days to ~7 months
out) pulled 6,376 quotes that passed the liquidity filter (non-zero
volume/OI, uncrossed bid-ask), solved implied vol for 5,618 of them, and
flagged 604 of 2,363 call/put pairs as deviating from put-call parity by
more than their combined bid-ask spread -- the rest of the drop from 6,376
to 5,618 is mostly contracts priced outside the no-arbitrage bounds implied
by the observed mid (stale quotes) plus the near-expiry exclusion described
below. SVI converged on all 20 expiries that survived the near-expiry
cutoff.

**Skew shape.** SPY (like most equity index underlyings) shows a downward
skew: implied vol rises as strike falls below spot and is lowest near/just
above the money. This is the standard equity index pattern -- it reflects
persistent demand for downside protection (crash risk hedging) pricing OTM
puts richer than OTM calls, not a data artifact.

**Term structure.** Shorter-dated expiries trade at a visibly higher implied
vol than longer-dated ones, in both the smile panel and the 3D surface. The
*raw* smile panel is visibly noisier at the short end too (see the numerical
caveat below); the SVI-fit 3D surface isn't, by construction.

**SVI fit.** `surface/svi.py` fits Gatheral's raw SVI parameterization to
each expiry's smile (all 6 converge in the pictured run), subject to the
constraint that keeps total variance non-negative everywhere, not just at
the observed strikes. One nuance worth calling out: SPY's listed strikes go
much deeper on the put side than the call side, so an unweighted fit over
the full ladder lets the put wing's far larger variance range dominate the
objective and drags `rho` to its -1 boundary -- degenerating the call wing
into a flat line. `fit_svi_slice` guards against this by truncating each
slice to a symmetric log-moneyness window sized to the shorter wing before
fitting, so neither wing can out-vote the other on point count or range
alone; the fitted curve is only drawn over that window, and the raw market
points are still plotted across the full range in the smile panel.

**3D surface.** The pictured surface (`plot_svi_surface_3d`) is built by
sampling the fitted SVI curves, not by triangulating the raw quotes directly
-- smooth, and free of the negative-variance arbitrage the fit already
excludes. `plot_surface_3d` (the raw triangulation) is still there as the
unsmoothed, purely diagnostic view; the CLI falls back to it automatically
if every SVI fit in a run fails to converge.

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

Both are stated in *forward* log-moneyness `k = log(K/F)`, while the surface
is built in spot log-moneyness `log(K/S)`; the checks convert by the drift
`(r-q)T` rather than conflating the two.

On the pictured run every slice was butterfly-clean (0 of 20), but **8 of 19
adjacent expiry pairs showed a total-variance drop** -- worst, 2027-03-19
sitting above 2027-03-31 by 0.00165 at `k = +0.217`. That is the expected
result rather than a bug, and it is the honest limitation of what this
project does: fitting each expiry *independently* leaves nothing tying the
slices together in `T`, so nothing prevents them from crossing. Surface-level
parameterizations (eSSVI and friends) exist precisely to impose that
coupling. Reporting the violations is the useful half of the answer; fixing
them requires a different fit.

**Local vol.** `surface/local_vol.py` applies Dupire's formula to the fitted
slices. Implied vol is an *average* of volatility along all paths to one
strike and expiry; local vol is the *instantaneous* vol the underlying must
have at a given spot and time to reproduce the whole surface. In terms of
total variance,

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
is known to be imperfect. Each expiry is fit independently, so the slices
cross in `T`; where they cross, the local vol repricing depends on doesn't
exist. The repricing error is a direct readout, in vol points, of what the
slice-independent fit costs.

**On the pictured run: median error 0.54 vol points, 95th percentile 2.60,
across 1,523 quotes and 20 expiries.**

**Scheme.** Log-Euler on `ln S`, which removes the drift discretization
entirely and keeps `S` positive by construction. Local vol is frozen over
each step and read at the path's *forward* log-moneyness, matching the
coordinate the surface is built in. Freezing leaves an `O(dt)` bias that is
separate from Monte Carlo noise and is not reduced by adding paths -- step
count is what controls it, and the test suite asserts it shrinks as steps
rise rather than assuming it.

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

**Where the error actually is.** Quotes outside the surface's own `(k, T)`
window are skipped rather than clamped. Simulation clamps local vol at the
edge of the grid, so a strike past the window is priced against the last
fitted value rather than against local vol, and scoring that would measure
the clamp instead of the surface. The gap is not subtle -- median error by
distance past the edge, same run:

| distance outside window (log-moneyness) | quotes | median error |
| --- | --- | --- |
| inside | 1,508 | **0.50 vol pts** |
| 0.00 - 0.05 | 512 | 0.78 |
| 0.05 - 0.10 | 365 | 2.16 |
| 0.10 - 0.20 | 409 | 4.49 |
| 0.20 - 0.50 | 322 | 10.98 |
| > 0.50 | 35 | 18.60 |

This is a real coverage limit, not only a reporting one: the window is the
*intersection* of every slice's fitted range, so one narrow front-week
ladder pulls it in for everyone. Widening it is the most valuable thing left
undone here.

**What the check caught.** Before any of this was tuned, expiries inside 9
days repriced 1.50 vol points low while everything past 15 days was near
perfect -- a bias too structured to be noise. The cause was the front slice's
`dw/dT`: `np.gradient`'s edge rule extrapolates the front derivative
backwards from the *next two* expiries, discarding something already known
exactly, that total variance is zero at `T = 0`. Anchoring the difference
there cuts the front bias to 0.28 vol points and the 95th-percentile error
across all expiries from 3.07 to 2.60. It costs a little at the median (0.50
to 0.56), because the front row also anchors the interpolation out to the
second expiry; the anchored derivative is kept anyway, on the grounds that
using a known boundary value beats extrapolating past it. Fixing what the
validation finds is the point of having built it -- but the fix was taken on
that reasoning, not by tuning the construction until the metric looked best.

## Why the pictured run widens the ladder

SPY lists *daily* expiries at the front, so the default `--max-expiries 6`
selects the first six listed expiries -- which is six consecutive *days*,
not the multi-month span the term structure lives on. The pictured run
therefore passes `--max-expiries 22` to reach out to ~5 months.

That density is also a numerical problem, not just a coverage one.
Differencing total variance across a one-day gap divides each slice's own
fit residual by `1/365`, amplifying it roughly 365-fold. Measured directly
on the 19-slice live chain: every row whose neighbours sat one day apart
produced local vols ranging from 0.007 to 0.905 with up to 54% of the row
undefined, while every row with a gap of five days or more stayed inside a
0.08-0.40 band with no gaps at all.

`build_local_vol_surface` therefore thins the slices used for `dw/dT` to a
minimum spacing (`--min-expiry-gap-days`, default 7). On the pictured run
that leaves 14 of 20 slices and takes local vol coverage to 100%. The thinning is deliberately *not* applied to the arbitrage checks:
a variance drop between two consecutive daily expiries is a real property of
the fitted surface and worth reporting. Only the division by a tiny `dT` is
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

## What this isn't

- No exotic options (barriers, Asians, lookbacks).
- No stochastic vol models (Heston, SABR) -- SVI fits the smile shape, not its dynamics.
- No live trading, execution, or backtested P&L claims.
- No multi-asset portfolio risk.
- Dividend yield is a flat continuous `q`, not a real dividend schedule.
- Put-call parity checks are a data-quality sanity check on the chain, not a
  trading signal.
- The no-arbitrage conditions are *reported*, not *enforced*. Each expiry is
  fit independently, so nothing couples the slices in `T` and calendar
  violations are possible by construction -- see the note above on eSSVI.
- Local vol is the Dupire surface implied by today's quotes, not a
  calibrated model with dynamics. It reprices vanillas by construction and
  says nothing about whether the smile *evolves* the way it assumes.
- The Monte Carlo prices European vanillas only. No early exercise, no path
  dependence, no exotics -- the point of it here is validating the surface,
  not building a pricing library on top of it.

## Acceptance checks

- Binomial European price converges to the Black-Scholes analytic price as
  `N` grows (tested within 0.5% at `N=200`, 0.1% at `N=1000`).
- The IV solver round-trips: pricing a known `(S,K,T,r,sigma)` with
  Black-Scholes and recovering `sigma` matches to `1e-6`.
- American price >= European price holds as a direct, asserted invariant
  across ITM/OTM/ATM cases (and American call == European call with no
  dividends, the standard theoretical result).
- The SVI fit round-trips: fitting a smile generated from known SVI
  parameters recovers the same curve to within `1e-3` in implied vol.
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
- Put-call parity holds on Monte Carlo prices struck from shared paths.
- The Monte Carlo standard error falls as `1/sqrt(N)` (4x the paths halves
  it, asserted within 0.25), and the discretization bias falls with step
  count against a finely-stepped reference -- separately, since only one of
  the two responds to adding paths.

## Stretch goals (not implemented)

- Real dividend yield data instead of a flat assumed `q`.
- American option Greeks via finite-difference bump-and-reprice on the tree.
- A second underlying for cross-asset comparison.
- An eSSVI (or otherwise surface-level) fit that couples the expiry slices,
  so calendar arbitrage is ruled out by construction rather than reported
  after the fact.
- A wider local vol window. It is currently the intersection of every
  slice's fitted range, so the narrowest front-week ladder caps the
  moneyness the surface covers -- and with it the range the repricing check
  can speak to.
- American exercise under local vol, which the Monte Carlo can't price and
  the binomial tree can't take a surface for.
