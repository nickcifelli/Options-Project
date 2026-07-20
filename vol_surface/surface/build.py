"""Build a long-format implied vol surface from a cleaned options chain."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from vol_surface.pricing.implied_vol import implied_vol

SURFACE_COLUMNS = ["expiry", "strike", "moneyness", "log_moneyness", "iv", "option_type"]


def year_fraction(expiry, as_of: dt.datetime | None = None) -> float:
    """Act/365 year fraction from `as_of` (default: now) to `expiry`."""
    as_of = as_of or dt.datetime.now()
    expiry_ts = pd.Timestamp(expiry)
    delta_days = (expiry_ts - pd.Timestamp(as_of)).days
    return max(delta_days, 0) / 365.0


def build_surface(
    chain: pd.DataFrame,
    r: float = 0.04,
    q: float = 0.0,
    as_of: dt.datetime | None = None,
) -> pd.DataFrame:
    """Solve IV for every (strike, expiry, option_type) row with a valid mid price.

    Moneyness (K/S) and log-moneyness make strikes comparable across
    expiries, which raw strike values are not.
    """
    as_of = as_of or dt.datetime.now()
    rows = []

    for _, row in chain.iterrows():
        T = year_fraction(row["expiry"], as_of)
        if T <= 0:
            continue

        result = implied_vol(row["mid"], row["spot"], row["strike"], T, r, row["option_type"], q)
        if not result.ok:
            continue

        rows.append(
            {
                "expiry": row["expiry"],
                "strike": row["strike"],
                "moneyness": row["strike"] / row["spot"],
                "log_moneyness": float(np.log(row["strike"] / row["spot"])),
                "iv": result.sigma,
                "option_type": row["option_type"],
            }
        )

    return pd.DataFrame(rows, columns=SURFACE_COLUMNS)
