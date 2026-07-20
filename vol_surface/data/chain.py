"""Fetch and clean a live options chain via yfinance.

Caches raw pulls to `data/` (gitignored) so repeated dev runs during
surface-building don't re-hit the API every time.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data"


def _cache_path(ticker: str, max_expiries: int, cache_dir: Path) -> Path:
    today = dt.date.today().isoformat()
    return cache_dir / f"{ticker.upper()}_{today}_{max_expiries}exp_raw.csv"


def clean_chain(raw: pd.DataFrame, min_volume: int = 1, min_open_interest: int = 1) -> pd.DataFrame:
    """Compute mid price and drop zero-volume/zero-OI/stale quotes.

    These filters exist because a quote with no trading activity can carry
    a wide or crossed bid-ask that produces a garbage implied vol later.
    """
    df = raw.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2

    liquid = (
        (df["volume"].fillna(0) >= min_volume)
        & (df["openInterest"].fillna(0) >= min_open_interest)
        & (df["bid"] > 0)
        & (df["ask"] > 0)
        & (df["ask"] >= df["bid"])
    )
    return df.loc[liquid].reset_index(drop=True)


def fetch_chain(
    ticker: str = "SPY",
    max_expiries: int = 6,
    min_volume: int = 1,
    min_open_interest: int = 1,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch a cleaned, long-format options chain for `ticker`.

    Returns columns including expiry, option_type, strike, bid, ask, mid,
    volume, openInterest, and spot. Only the *raw* pull is cached (one
    snapshot per ticker/day/max_expiries) -- liquidity filtering is always
    applied fresh with the current min_volume/min_open_interest, so callers
    can re-filter a cached pull without a stale cache silently ignoring
    their thresholds.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(ticker, max_expiries, cache_dir)

    if use_cache and cache_file.exists():
        raw = pd.read_csv(cache_file, parse_dates=["expiry"])
    else:
        tk = yf.Ticker(ticker)
        expiries = tk.options[:max_expiries]
        if not expiries:
            raise ValueError(f"no listed expiries found for {ticker!r}")

        spot = tk.history(period="1d")["Close"].iloc[-1]

        frames = []
        for expiry in expiries:
            chain = tk.option_chain(expiry)
            for option_type, leg in (("call", chain.calls), ("put", chain.puts)):
                leg = leg.copy()
                leg["option_type"] = option_type
                leg["expiry"] = expiry
                frames.append(leg)

        raw = pd.concat(frames, ignore_index=True)
        raw["spot"] = spot
        raw["expiry"] = pd.to_datetime(raw["expiry"])
        raw.to_csv(cache_file, index=False)

    return clean_chain(raw, min_volume, min_open_interest)
