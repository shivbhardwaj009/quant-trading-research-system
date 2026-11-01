"""Unit tests for Phase 5: Risk & Performance (risk.py) with hand-made examples."""

import numpy as np
import pandas as pd
import pytest

from src.risk import (
    calculate_all_metrics,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_drawdowns,
    calculate_historical_var,
    calculate_sharpe_ratio,
    calculate_trade_metrics,
)


def test_cagr_hand_made_example():
    """Hand-made test: Equity doubles in 252 trading days (1 year) -> CAGR = 100%."""
    # 252 days from 100 to 200
    equity = pd.Series(np.linspace(100.0, 200.0, 252))
    cagr = calculate_cagr(equity, trading_days=252)
    assert pytest.approx(cagr, rel=1e-3) == 1.0  # 100% CAGR

    # 504 days (2 years) from 100 to 121 -> CAGR = sqrt(1.21) - 1 = 10%
    equity_2y = pd.Series(np.linspace(100.0, 121.0, 504))
    cagr_2y = calculate_cagr(equity_2y, trading_days=252)
    assert pytest.approx(cagr_2y, rel=1e-3) == 0.10  # 10% CAGR


def test_drawdowns_hand_made_example():
    """Hand-made test: Peak at 120, drops to 90 -> Max Drawdown is exactly -25%."""
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    dd_curve, max_dd = calculate_drawdowns(equity)

    assert dd_curve.iloc[0] == 0.0
    assert dd_curve.iloc[1] == 0.0
    assert pytest.approx(dd_curve.iloc[2]) == -0.25  # (90 - 120) / 120 = -0.25
    assert pytest.approx(dd_curve.iloc[3]) == (110.0 - 120.0) / 120.0
    assert pytest.approx(max_dd) == -0.25


def test_calmar_ratio_hand_made_example():
    """Hand-made test: CAGR of 15% and Max Drawdown of 30% yields Calmar of 0.5."""
    calmar = calculate_calmar_ratio(cagr=0.15, max_drawdown=-0.30)
    assert pytest.approx(calmar, rel=1e-4) == 0.50


def test_var_95_hand_made_example():
    """Hand-made test: Exactly 100 sorted returns from -0.10 to +0.09.

    At 95% confidence level, 5th percentile is approximately 0.09.
    """
    returns = pd.Series(np.linspace(-0.10, 0.09, 100))
    var_95 = calculate_historical_var(returns, confidence_level=0.95)
    # The 5th percentile of 100 linearly spaced values around -0.10 to +0.09
    assert 0.08 <= var_95 <= 0.10


def test_trade_metrics_hand_made_example():
    """Hand-made test: 2 wins and 1 loss."""
    trade_df = pd.DataFrame(
        [
            {"trade_type": "LONG", "pnl_pct": 0.10, "pnl_amount": 1000.0},
            {"trade_type": "LONG", "pnl_pct": 0.20, "pnl_amount": 2000.0},
            {"trade_type": "LONG", "pnl_pct": -0.06, "pnl_amount": -600.0},
        ]
    )

    metrics = calculate_trade_metrics(trade_df)
    assert metrics["total_trades"] == 3
    assert pytest.approx(metrics["win_rate"]) == 2 / 3
    assert pytest.approx(metrics["avg_win"]) == 0.15  # mean(0.10, 0.20)
    assert pytest.approx(metrics["avg_loss"]) == -0.06
    assert pytest.approx(metrics["profit_factor"]) == 3000.0 / 600.0  # 5.0


def test_calculate_all_metrics_integration():
    """Verify calculate_all_metrics computes all expected keys without error."""
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    daily_returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
    equity_curve = 100000.0 * (1.0 + daily_returns).cumprod()

    res = calculate_all_metrics(daily_returns=daily_returns, equity_curve=equity_curve)

    assert "Total Return" in res
    assert "CAGR" in res
    assert "Annualized Volatility" in res
    assert "Sharpe Ratio" in res
    assert "Sortino Ratio" in res
    assert "Max Drawdown" in res
    assert "Max Daily Loss" in res
    assert "Calmar Ratio" in res
    assert "Historical VaR (95%)" in res
