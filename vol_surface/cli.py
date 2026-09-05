"""CLI entry point: fetch -> build -> fit -> check -> localvol -> reprice -> plot."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from vol_surface.data.chain import fetch_chain
from vol_surface.pricing.monte_carlo import reprice_chain
from vol_surface.surface.arbitrage import check_butterfly, check_calendar
from vol_surface.surface.build import build_surface
from vol_surface.surface.local_vol import build_local_vol_surface
from vol_surface.surface.parity import check_parity
from vol_surface.surface.ssvi import fit_ssvi_surface
from vol_surface.surface.svi import fit_svi_surface
from vol_surface.viz.plots import (
    plot_local_vol_surface_3d,
    plot_mc_reprice,
    plot_surface_3d,
    plot_svi_fit,
    plot_svi_surface_3d,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch an options chain, build an implied vol surface, and plot it.")
    parser.add_argument("--ticker", default="SPY", help="underlying ticker (default: SPY)")
    parser.add_argument("--max-expiries", type=int, default=6)
    parser.add_argument("--r", type=float, default=0.04, help="flat risk-free rate")
    parser.add_argument("--q", type=float, default=0.0, help="flat continuous dividend yield")
    parser.add_argument(
        "--fit",
        choices=("svi", "ssvi"),
        default="svi",
        help=(
            "smile fit. 'svi' fits each expiry independently: the closest fit to "
            "the quotes, and the one nothing prevents from crossing in T. 'ssvi' "
            "fits one surface to every expiry at once (Gatheral-Jacquier), which "
            "gives up some closeness and rules calendar arbitrage out by "
            "construction instead of reporting it. Default: svi."
        ),
    )
    parser.add_argument(
        "--k-window",
        choices=("slice", "union"),
        default="slice",
        help=(
            "log-moneyness range each SSVI slice claims (--fit ssvi only). "
            "'slice' is its own quoted ladder; 'union' is the widest ladder "
            "quoted anywhere on the surface, which is defensible only because "
            "the wings are fit globally -- it widens the local vol window, and "
            "with it the range the repricing check can speak to. Default: slice."
        ),
    )
    parser.add_argument("--min-volume", type=int, default=1)
    parser.add_argument("--min-open-interest", type=int, default=1)
    parser.add_argument(
        "--min-days-to-expiry",
        type=float,
        default=2.0,
        help=(
            "exclude contracts closer to expiry than this many days (default: 2). "
            "Near-zero-vega, near-expiry contracts turn tiny bid-ask noise into "
            "large IV swings under inversion -- a real numerical artifact, not a "
            "data-quality issue, but it makes the plotted smile spiky and misleading."
        ),
    )
    parser.add_argument(
        "--min-expiry-gap-days",
        type=float,
        default=7.0,
        help=(
            "minimum spacing between expiries used for the local vol dw/dT "
            "difference (default: 7). Underlyings with daily expiries put "
            "neighbouring slices one day apart, and differencing total "
            "variance across that gap amplifies each slice's fit residual by "
            "roughly 365x -- noise, not term structure."
        ),
    )
    parser.add_argument(
        "--reprice",
        action="store_true",
        help=(
            "Monte Carlo the fitted local vol surface and check it reprices the "
            "quotes it was built from. Off by default because it is the only "
            "stage that costs real time."
        ),
    )
    parser.add_argument(
        "--reprice-paths",
        type=int,
        default=50_000,
        help="paths per expiry for --reprice (default: 50000)",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed the Monte Carlo for reproducible runs")
    parser.add_argument("--no-cache", action="store_true", help="ignore the cached chain snapshot")
    parser.add_argument("--out", default=None, help="save the plot to this path instead of showing it interactively")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    print(f"Fetching options chain for {args.ticker}...")
    chain = fetch_chain(
        args.ticker,
        max_expiries=args.max_expiries,
        min_volume=args.min_volume,
        min_open_interest=args.min_open_interest,
        use_cache=not args.no_cache,
    )
    if chain.empty:
        print("No liquid quotes survived filtering; nothing to build.", file=sys.stderr)
        return 1
    print(f"  {len(chain)} liquid quotes across {chain['expiry'].nunique()} expiries")

    print("Building implied vol surface...")
    min_T = args.min_days_to_expiry / 365.0
    surface = build_surface(chain, r=args.r, q=args.q, min_T=min_T)
    print(f"  solved IV for {len(surface)}/{len(chain)} contracts")

    parity = check_parity(chain, r=args.r, q=args.q)
    n_flagged = int(parity["flagged"].sum()) if not parity.empty else 0
    print(f"  put-call parity: {n_flagged}/{len(parity)} pairs flagged beyond combined spread")

    if surface.empty:
        print("No IVs solved; skipping plot.", file=sys.stderr)
        return 1

    if args.fit == "ssvi":
        print("Fitting one SSVI surface across every expiry...")
        ssvi = fit_ssvi_surface(surface, r=args.r, q=args.q, k_window=args.k_window)
        if not ssvi.ok:
            print(f"  SSVI fit failed: {ssvi.reason}", file=sys.stderr)
            return 1
        svi_fits = ssvi.slices
        print(
            f"  rho={ssvi.params.rho:+.3f} eta={ssvi.params.eta:.3f} gamma={ssvi.params.gamma:.3f}"
            f" over {len(svi_fits)} expiries ({ssvi.n_parameters} parameters for {ssvi.n_quotes} quotes)"
        )
        print(f"  residual: {ssvi.rmse_vol * 100:.3f} vol points RMSE")
    else:
        print("Fitting SVI smile per expiry...")
        svi_fits = fit_svi_surface(surface)
    n_svi_ok = sum(fit.ok for fit in svi_fits.values())
    print(f"  {args.fit.upper()} fit converged for {n_svi_ok}/{len(svi_fits)} expiries")

    print("Checking static no-arbitrage conditions...")
    butterfly = check_butterfly(svi_fits, r=args.r, q=args.q)
    calendar = check_calendar(svi_fits, r=args.r, q=args.q)
    print(f"  butterfly: {int(butterfly['flagged'].sum())}/{len(butterfly)} slices imply a negative density somewhere")
    if butterfly["flagged"].any():
        worst = butterfly.loc[butterfly["min_g"].idxmin()]
        print(f"    worst: {pd.Timestamp(worst['expiry']).date()} g={worst['min_g']:.4f} at k={worst['k_at_min_g']:+.3f}")
    print(f"  calendar: {int(calendar['flagged'].sum())}/{len(calendar)} adjacent expiry pairs drop in total variance")
    if calendar["flagged"].any():
        worst = calendar.loc[calendar["max_variance_drop"].idxmax()]
        near, far = pd.Timestamp(worst["near_expiry"]).date(), pd.Timestamp(worst["far_expiry"]).date()
        print(f"    worst: {near} over {far} by {worst['max_variance_drop']:.5f} at k={worst['k_at_max_drop']:+.3f}")

    print("Deriving Dupire local vol surface...")
    try:
        local_vol = build_local_vol_surface(
            svi_fits, r=args.r, q=args.q, min_expiry_gap=args.min_expiry_gap_days / 365.0
        )
        print(f"  {len(local_vol.T)} slices after gap thinning; local vol defined at {local_vol.coverage:.1%} of grid points")
    except ValueError as exc:
        local_vol = None
        print(f"  skipped: {exc}", file=sys.stderr)

    reprice = None
    if args.reprice and local_vol is not None:
        print(f"Repricing the chain under the local vol surface ({args.reprice_paths:,} paths/expiry)...")
        reprice = reprice_chain(
            local_vol,
            surface,
            S0=float(chain["spot"].iloc[0]),
            r=args.r,
            q=args.q,
            n_paths=args.reprice_paths,
            seed=args.seed,
        )
        solved = reprice.dropna(subset=["mc_iv"])
        if solved.empty:
            print("  no quotes repriced; skipping the repricing panel", file=sys.stderr)
            reprice = None
        else:
            errors = solved["iv_error"].abs()
            print(
                f"  repriced {len(solved)} quotes across {solved['expiry'].nunique()} expiries "
                f"(in the surface's own (k, T) window; the rest would only measure the clamp)"
            )
            print(f"  |IV error|: median {errors.median() * 100:.2f} vol pts, 95th pct {errors.quantile(0.95) * 100:.2f}")
    elif args.reprice:
        print("  no local vol surface to reprice against; skipping", file=sys.stderr)

    n_panels = 2 + (local_vol is not None) + (reprice is not None)
    # Past three panels a single row is too wide to read, so the panels wrap.
    n_cols = n_panels if n_panels <= 3 else math.ceil(n_panels / 2)
    n_rows = math.ceil(n_panels / n_cols)
    fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows))

    def axes(index: int, projection: str | None = None):
        return fig.add_subplot(n_rows, n_cols, index, projection=projection)

    plot_svi_fit(surface, svi_fits, ax=axes(1))
    if n_svi_ok:
        plot_svi_surface_3d(svi_fits, ax=axes(2, "3d"))
    else:
        print("  no SVI fits converged; falling back to the raw triangulated surface", file=sys.stderr)
        plot_surface_3d(surface, ax=axes(2, "3d"))
    panel = 3
    if local_vol is not None:
        plot_local_vol_surface_3d(local_vol, ax=axes(panel, "3d"))
        panel += 1
    if reprice is not None:
        plot_mc_reprice(reprice, ax=axes(panel))
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
