"""Cox-Ross-Rubinstein binomial tree pricing for European and American options.

American exercise is solved by backward induction: at each node the holder
compares immediate exercise against the discounted continuation value and
takes the max. That per-node `max(exercise, continuation)` is an optimal
stopping problem -- it's what makes American pricing structurally harder
than European (which has a closed form) and is why a tree/lattice method
is used instead.
"""

from __future__ import annotations

import numpy as np

_VALID_TYPES = ("call", "put")


def _check_inputs(T: float, sigma: float, option_type: str, N: int) -> None:
    if option_type not in _VALID_TYPES:
        raise ValueError(f"option_type must be one of {_VALID_TYPES}, got {option_type!r}")
    if T < 0:
        raise ValueError(f"T must be >= 0, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")


def _intrinsic(S: np.ndarray, K: float, option_type: str) -> np.ndarray:
    return np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    N: int = 500,
    american: bool = False,
) -> float:
    """CRR binomial tree price.

    Backward induction is vectorized over node prices at each step (numpy
    array ops), not a per-node Python loop, so N in the 500-1000 range is
    fast enough for interactive use.
    """
    _check_inputs(T, sigma, option_type, N)

    if T == 0:
        return float(_intrinsic(np.array(S), K, option_type))

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    j = np.arange(N + 1)
    terminal_prices = S * u ** (N - j) * d**j
    values = _intrinsic(terminal_prices, K, option_type)

    for i in range(N - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            j = np.arange(i + 1)
            node_prices = S * u ** (i - j) * d**j
            values = np.maximum(values, _intrinsic(node_prices, K, option_type))

    return float(values[0])
