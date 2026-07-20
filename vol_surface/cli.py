"""CLI entry point: fetch -> build -> plot -> report."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import matplotlib.pyplot as plt

from vol_surface.data.chain import fetch_chain
from vol_surface.surface.build import build_surface
from vol_surface.surface.parity import check_parity
from vol_surface.viz.plots import plot_smiles, plot_surface_3d


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch an options chain, build an implied vol surface, and plot it.")
    parser.add_argument("--ticker", default="SPY", help="underlying ticker (default: SPY)")
    parser.add_argument("--max-expiries", type=int, default=6)
    parser.add_argument("--r", type=float, default=0.04, help="flat risk-free rate")
    parser.add_argument("--q", type=float, default=0.0, help="flat continuous dividend yield")
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

    fig = plt.figure(figsize=(14, 6))
    plot_smiles(surface, ax=fig.add_subplot(1, 2, 1))
    plot_surface_3d(surface, ax=fig.add_subplot(1, 2, 2, projection="3d"))
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
