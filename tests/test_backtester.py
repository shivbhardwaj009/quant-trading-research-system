"""Unit tests for Phase 4: Backtesting Engine (backtester.py)."""

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    generate_trade_log,
    run_portfolio_backtest,
    run_single_backtest,
)


def test_no_look_ahead_bias_signal_shift():
    """Prove that signals generated at day t are strictly traded on day t+1.

    A signal generated on Day 0 must NOT earn Day 0 returns; it must only
    participate starting on Day 1.
    """
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 110.0, 120.0, 115.0], index=dates)

    # Signal = 1 generated on Day 0, then 0 on Day 2
    signals = pd.Series([1.0, 1.0, 0.0, 0.0], index=dates)

    res = run_single_backtest(
        prices=prices,
        signals=signals,
        starting_capital=100000.0,
        brokerage_pct=0.0,
        slippage_pct=0.0,
        stt_pct=0.0,
    )
    df = res.daily_results

    # Day 0: position must be 0 (no trade yet!), gross return must be 0
    assert df["position"].iloc[0] == 0.0
    assert df["strategy_gross_return"].iloc[0] == 0.0

    # Day 1: position becomes 1.0 (traded after Day 0 close), participates in Day 1 return: (110 - 100)/100 = 10%
    assert df["position"].iloc[1] == 1.0
    assert pytest.approx(df["strategy_gross_return"].iloc[1], rel=1e-4) == 0.10

    # Day 2: position is still 1.0 (signal became 0 on Day 2 close, so takes effect on Day 3)
    assert df["position"].iloc[2] == 1.0
    assert pytest.approx(df["strategy_gross_return"].iloc[2], rel=1e-4) == (120.0 - 110.0) / 110.0

    # Day 3: position becomes 0.0, does not participate in price drop from 120 to 115
    assert df["position"].iloc[3] == 0.0
    assert df["strategy_gross_return"].iloc[3] == 0.0


def test_transaction_costs_applied_on_position_changes():
    """Verify transaction costs are charged only on turnover days."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=dates)
    # Signal triggers entry on Day 0 (executed Day 1) and exit on Day 2 (executed Day 3)
    signals = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=dates)

    total_cost = 0.001  # 0.1% total cost per trade
    res = run_single_backtest(
        prices=prices,
        signals=signals,
        starting_capital=100000.0,
        brokerage_pct=total_cost,
        slippage_pct=0.0,
        stt_pct=0.0,
    )
    df = res.daily_results

    # Day 0: no position, no cost
    assert df["transaction_cost"].iloc[0] == 0.0

    # Day 1: position entered (0 -> 1), cost charged
    assert pytest.approx(df["transaction_cost"].iloc[1], rel=1e-4) == 0.001

    # Day 2: position held (1 -> 1), NO cost charged
    assert df["transaction_cost"].iloc[2] == 0.0

    # Day 3: position exited (1 -> 0), cost charged
    assert pytest.approx(df["transaction_cost"].iloc[3], rel=1e-4) == 0.001

    # Day 4: position flat (0 -> 0), NO cost charged
    assert df["transaction_cost"].iloc[4] == 0.0


def test_trade_log_generation():
    """Verify trade log tracks entries, exits, and PnL properly."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 110.0, 120.0, 130.0], index=dates)
    positions = pd.Series([0.0, 1.0, 1.0, 0.0], index=dates)

    log = generate_trade_log(prices=prices, positions=positions, starting_capital=100000.0)
    assert len(log) == 1
    trade = log.iloc[0]
    assert trade["entry_price"] == 110.0
    assert trade["exit_price"] == 130.0
    assert pytest.approx(trade["pnl_pct"], rel=1e-4) == (130.0 - 110.0) / 110.0


def test_portfolio_backtest_equal_weight():
    """Verify multi-stock portfolio backtest runs with equal weights."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices_a = pd.Series([100.0, 110.0, 120.0, 130.0], index=dates)
    prices_b = pd.Series([50.0, 55.0, 60.0, 65.0], index=dates)
    signals = pd.Series([1.0, 1.0, 1.0, 1.0], index=dates)

    res = run_portfolio_backtest(
        price_dict={"A": prices_a, "B": prices_b},
        signal_dict={"A": signals, "B": signals},
        starting_capital=100000.0,
        brokerage_pct=0.0,
        slippage_pct=0.0,
        stt_pct=0.0,
    )
    assert not res.daily_results.empty
    assert res.daily_results["equity_curve"].iloc[-1] > 100000.0
