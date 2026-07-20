"""Implied volatility extraction by inverting Black-Scholes with Brent's method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from vol_surface.pricing.black_scholes import price as bs_price

_VALID_TYPES = ("call", "put")
_DEFAULT_BRACKET = (1e-6, 5.0)


@dataclass(frozen=True)
class IVResult:
    """sigma is None on failure; reason explains why rather than returning garbage."""

    sigma: float | None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.sigma is not None


def _no_arbitrage_bounds(S: float, K: float, T: float, r: float, q: float, option_type: str) -> tuple[float, float]:
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if option_type == "call":
        return max(S * disc_q - K * disc_r, 0.0), S * disc_q
    return max(K * disc_r - S * disc_q, 0.0), K * disc_r


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    bracket: tuple[float, float] = _DEFAULT_BRACKET,
) -> IVResult:
    """Invert Black-Scholes for sigma given an observed market price.

    Returns an IVResult with sigma=None and a human-readable reason when the
    price can't be explained by any vol in `bracket` -- e.g. it violates
    no-arbitrage bounds (stale/illiquid quote) or the root isn't bracketed.
    """
    if option_type not in _VALID_TYPES:
        raise ValueError(f"option_type must be one of {_VALID_TYPES}, got {option_type!r}")

    if T <= 0:
        return IVResult(None, "T must be > 0: expired option has no implied vol")

    if market_price <= 0:
        return IVResult(None, f"market_price must be positive, got {market_price}")

    lower, upper = _no_arbitrage_bounds(S, K, T, r, q, option_type)
    if not (lower <= market_price <= upper):
        return IVResult(
            None,
            f"price {market_price:.4f} outside no-arbitrage bounds "
            f"[{lower:.4f}, {upper:.4f}]; likely stale/illiquid quote",
        )

    low, high = bracket

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price

    f_low, f_high = objective(low), objective(high)
    if f_low * f_high > 0:
        return IVResult(
            None,
            f"root not bracketed in sigma in [{low}, {high}]; "
            "price may be at/near the intrinsic-value boundary",
        )

    try:
        sigma = brentq(objective, low, high, xtol=1e-10, rtol=1e-12)
    except (RuntimeError, ValueError) as exc:
        return IVResult(None, f"brentq failed to converge: {exc}")

    return IVResult(sigma, None)
