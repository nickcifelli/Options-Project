"""Tests for the options-chain cleaning logic (no network calls)."""

import pandas as pd

from vol_surface.data.chain import clean_chain


def _row(bid, ask, volume, open_interest):
    return {"bid": bid, "ask": ask, "volume": volume, "openInterest": open_interest}


def test_clean_chain_computes_mid_price():
    raw = pd.DataFrame([_row(1.0, 1.2, 10, 10)])
    cleaned = clean_chain(raw)
    assert cleaned.loc[0, "mid"] == 1.1


def test_clean_chain_drops_zero_volume():
    raw = pd.DataFrame([_row(1.0, 1.2, 0, 10)])
    assert clean_chain(raw).empty


def test_clean_chain_drops_zero_open_interest():
    raw = pd.DataFrame([_row(1.0, 1.2, 10, 0)])
    assert clean_chain(raw).empty


def test_clean_chain_drops_zero_bid():
    raw = pd.DataFrame([_row(0.0, 1.2, 10, 10)])
    assert clean_chain(raw).empty


def test_clean_chain_drops_crossed_quote():
    raw = pd.DataFrame([_row(1.5, 1.2, 10, 10)])
    assert clean_chain(raw).empty


def test_clean_chain_keeps_liquid_rows_and_filters_illiquid_ones():
    raw = pd.DataFrame(
        [
            _row(1.0, 1.2, 10, 10),  # liquid
            _row(1.0, 1.2, 0, 10),  # illiquid: no volume
            _row(2.0, 2.4, 5, 5),  # liquid
        ]
    )
    cleaned = clean_chain(raw)
    assert len(cleaned) == 2
    assert list(cleaned["mid"]) == [1.1, 2.2]


def test_clean_chain_respects_custom_liquidity_thresholds():
    raw = pd.DataFrame([_row(1.0, 1.2, 3, 3)])
    assert clean_chain(raw, min_volume=5, min_open_interest=1).empty
    assert not clean_chain(raw, min_volume=1, min_open_interest=1).empty
