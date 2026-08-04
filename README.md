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

![Implied vol smile and surface for SPY](surface.png)

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run

```
.venv/bin/vol-surface --ticker SPY --out surface.png
```

or `python -m vol_surface.cli ...`. Runs `fetch -> build -> fit -> plot ->
report`: pulls a live options chain, solves implied vol per contract, fits
an SVI curve to each expiry's smile, checks put-call parity as a
data-quality sanity check, and saves (or shows) a smile + 3D surface plot.
See `--help` for the full set of flags (rate, dividend yield, liquidity
thresholds, near-expiry cutoff, cache control).

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
  viz/
    plots.py                # smile + SVI fit, raw and SVI-sampled 3D surfaces
  cli.py                     # entry point: fetch -> build -> fit -> plot -> report
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

A live run against SPY (6 expiries, spanning ~1 to ~15 weeks out) pulled
2,061 quotes that passed the liquidity filter (non-zero volume/OI, uncrossed
bid-ask), solved implied vol for 2,023 of them, and flagged 219 of 801
call/put pairs as deviating from put-call parity by more than their combined
bid-ask spread -- the rest of the drop from 2,061 to 2,023 is mostly
contracts priced outside the no-arbitrage bounds implied by the observed mid
(stale quotes) plus the near-expiry exclusion described below.

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

## Stretch goals (not implemented)

- Real dividend yield data instead of a flat assumed `q`.
- American option Greeks via finite-difference bump-and-reprice on the tree.
- A second underlying for cross-asset comparison.
