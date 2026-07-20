"""End-to-end CLI test with fetch_chain faked out (no network calls)."""

import datetime as dt

import pandas as pd
import pytest

import vol_surface.cli as cli
from vol_surface.pricing.black_scholes import price as bs_price

R, Q = 0.04, 0.0


def _synthetic_chain() -> pd.DataFrame:
    as_of = dt.datetime.now()
    expiries = [as_of + dt.timedelta(days=30), as_of + dt.timedelta(days=90)]
    rows = []
    for expiry in expiries:
        T = (expiry - as_of).days / 365.0
        for strike in (90.0, 100.0, 110.0):
            for option_type in ("call", "put"):
                mid = bs_price(100.0, strike, T, R, 0.2, option_type, Q)
                rows.append(
                    {
                        "spot": 100.0,
                        "strike": strike,
                        "expiry": expiry,
                        "option_type": option_type,
                        "mid": mid,
                        "bid": mid - 0.01,
                        "ask": mid + 0.01,
                        "volume": 100,
                        "openInterest": 100,
                    }
                )
    return pd.DataFrame(rows)


def test_main_runs_fetch_build_plot_report(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "fetch_chain", lambda *a, **k: _synthetic_chain())

    out_path = tmp_path / "surface.png"
    exit_code = cli.main(["--ticker", "TEST", "--out", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_main_returns_nonzero_on_empty_chain(monkeypatch):
    monkeypatch.setattr(cli, "fetch_chain", lambda *a, **k: pd.DataFrame())
    exit_code = cli.main(["--ticker", "TEST"])
    assert exit_code == 1


def test_arg_parser_defaults():
    args = cli.build_arg_parser().parse_args([])
    assert args.ticker == "SPY"
    assert args.max_expiries == 6
    assert args.r == pytest.approx(0.04)
    assert args.q == pytest.approx(0.0)
