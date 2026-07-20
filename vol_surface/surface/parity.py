"""Put-call parity check across the chain.

This is strictly a data-quality / no-arbitrage-bound sanity check on the
quotes themselves -- not a trading signal. A flagged pair means the
observed call/put mid prices are inconsistent with parity by more than
the combined bid-ask spread, which usually points at a stale or crossed
quote rather than a real arbitrage.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from vol_surface.surface.build import year_fraction

PARITY_COLUMNS = [
    "expiry",
    "strike",
    "observed_diff",
    "theoretical_diff",
    "deviation",
    "combined_spread",
    "flagged",
]


def check_parity(
    chain: pd.DataFrame,
    r: float = 0.04,
    q: float = 0.0,
    as_of: dt.datetime | None = None,
) -> pd.DataFrame:
    """For each (strike, expiry) with both a call and put quote, check
    C - P == S*e^(-qT) - K*e^(-rT), flagging deviations beyond the
    combined bid-ask spread.
    """
    as_of = as_of or dt.datetime.now()

    calls = chain[chain["option_type"] == "call"]
    puts = chain[chain["option_type"] == "put"]
    merged = calls.merge(puts, on=["expiry", "strike"], suffixes=("_call", "_put"))

    if merged.empty:
        return pd.DataFrame(columns=PARITY_COLUMNS)

    T = merged["expiry"].apply(lambda e: year_fraction(e, as_of))
    live = T > 0
    merged, T = merged.loc[live].reset_index(drop=True), T.loc[live].reset_index(drop=True)

    S = merged["spot_call"]
    K = merged["strike"]

    theoretical_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
    observed_diff = merged["mid_call"] - merged["mid_put"]
    deviation = (observed_diff - theoretical_diff).abs()
    combined_spread = (merged["ask_call"] - merged["bid_call"]) + (merged["ask_put"] - merged["bid_put"])

    out = pd.DataFrame(
        {
            "expiry": merged["expiry"],
            "strike": K,
            "observed_diff": observed_diff,
            "theoretical_diff": theoretical_diff,
            "deviation": deviation,
            "combined_spread": combined_spread,
            "flagged": deviation > combined_spread,
        }
    )
    return out[PARITY_COLUMNS]
