"""Closed-form Black-Scholes-Merton pricing and Greeks for European options."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_VALID_TYPES = ("call", "put")


def _check_inputs(T: float, sigma: float, option_type: str) -> None:
    if option_type not in _VALID_TYPES:
        raise ValueError(f"option_type must be one of {_VALID_TYPES}, got {option_type!r}")
    if T < 0:
        raise ValueError(f"T must be >= 0, got {T}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def _intrinsic(S: float, K: float, option_type: str) -> float:
    return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)


def _forward_intrinsic(S: float, K: float, T: float, r: float, q: float, option_type: str) -> float:
    """PV of intrinsic value at the forward price -- the sigma -> 0 limit of the BS price."""
    forward = S * np.exp((r - q) * T)
    payoff = _intrinsic(forward, K, option_type)
    return np.exp(-r * T) * payoff


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes-Merton price of a European option.

    Falls back to intrinsic value at T=0 and to the discounted forward
    intrinsic value at sigma=0, rather than dividing by zero.
    """
    _check_inputs(T, sigma, option_type)

    if T == 0:
        return _intrinsic(S, K, option_type)
    if sigma == 0:
        return _forward_intrinsic(S, K, T, r, q, option_type)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> Greeks:
    """Standard closed-form Greeks. vega and rho are per unit (not per 1%/1bp)."""
    _check_inputs(T, sigma, option_type)

    if T == 0 or sigma == 0:
        # At the boundary the payoff is piecewise-linear in S with a kink at K:
        # delta is the indicator of being in the money, all higher-order
        # Greeks vanish away from the kink itself.
        forward = S * np.exp((r - q) * T) if T > 0 else S
        in_the_money = forward > K if option_type == "call" else forward < K
        delta = float(in_the_money) * (1.0 if option_type == "call" else -1.0)
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * disc_q * pdf_d1 * sqrt_T

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_T)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_T)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
