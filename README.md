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

*Pictured: `--max-expiries 22` (see [Why the pictured run widens the
ladder](#why-the-pictured-run-widens-the-ladder)).*

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
local vol -> plot -> report`: pulls a live options chain, solves implied vol
per contract, fits an SVI curve to each expiry's smile, checks put-call
parity plus the two static no-arbitrage conditions, derives the Dupire local
vol surface, and saves (or shows) a smile + implied vol surface + local vol
surface plot. See `--help` for the full set of flags (rate, dividend yield,
liquidity thresholds, near-expiry cutoff, expiry-gap thinning, cache
control).

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
  data/
    chain.py               # fetch + clean live options chain via yfinance
  surface/
    build.py               # strike/expiry -> IV grid construction
    svi.py                   # SVI curve fit per expiry (Gatheral raw parameterization)
    parity.py               # put-call parity check across the chain
    arbitrage.py            # butterfly + calendar no-arbitrage checks on the fit
    local_vol.py            # Dupire local vol, derived from the SVI slices
  viz/
    plots.py                # smile + SVI fit, IV surfaces, local vol surface
  cli.py                     # entry point: fetch -> build -> fit -> check -> plot
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

The pictured run against SPY (22 expiries, spanning ~3 days to ~5 months
out) pulled 6,044 quotes that passed the liquidity filter (non-zero
volume/OI, uncrossed bid-ask), solved implied vol for 5,321 of them, and
flagged 475 of 2,204 call/put pairs as deviating from put-call parity by
more than their combined bid-ask spread -- the rest of the drop from 6,044
to 5,321 is mostly contracts priced outside the no-arbitrage bounds implied
by the observed mid (stale quotes) plus the near-expiry exclusion described
below. SVI converged on all 19 expiries that survived the near-expiry
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

On the pictured run every slice was butterfly-clean (0 of 19), but **8 of 18
adjacent expiry pairs showed a total-variance drop** -- worst, 2027-01-15
sitting above 2027-01-29 by 0.00089 at `k = +0.180`. That is the expected
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
that leaves 12 of 19 slices and takes local vol coverage from 93.9% to
100%. The thinning is deliberately *not* applied to the arbitrage checks:
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

## Stretch goals (not implemented)

- Real dividend yield data instead of a flat assumed `q`.
- American option Greeks via finite-difference bump-and-reprice on the tree.
- A second underlying for cross-asset comparison.
- An eSSVI (or otherwise surface-level) fit that couples the expiry slices,
  so calendar arbitrage is ruled out by construction rather than reported
  after the fact.
- Monte Carlo under the fitted local vol surface, to confirm it reprices the
  vanillas it was built from.
