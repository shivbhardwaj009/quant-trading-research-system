"""Unit tests for Phase 2: Data Processing (processing.py)."""

import numpy as np
import pandas as pd
import pytest

from src.processing import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_moving_average,
    calculate_rolling_volatility,
    compute_correlation_matrix,
    add_technical_indicators,
)


def test_simple_and_log_returns_example_check():
    """Verify returns check from spec: prices 100, 105, 110 yield 5% and 4.76%."""
    prices = pd.Series([100.0, 105.0, 110.0])

    simple_ret = calculate_simple_returns(prices)
    assert np.isnan(simple_ret.iloc[0])
    assert pytest.approx(simple_ret.iloc[1], rel=1e-3) == 0.05      # 5.00%
    assert pytest.approx(simple_ret.iloc[2], rel=1e-3) == 0.047619  # 4.76%

    log_ret = calculate_log_returns(prices)
    assert np.isnan(log_ret.iloc[0])
    assert pytest.approx(log_ret.iloc[1], rel=1e-3) == np.log(105.0 / 100.0)
    assert pytest.approx(log_ret.iloc[2], rel=1e-3) == np.log(110.0 / 105.0)


def test_moving_average():
    """Verify rolling simple moving average."""
    prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    ma3 = calculate_moving_average(prices, window=3)

    assert np.isnan(ma3.iloc[0])
    assert np.isnan(ma3.iloc[1])
    assert ma3.iloc[2] == 20.0
    assert ma3.iloc[3] == 30.0
    assert ma3.iloc[4] == 40.0

    with pytest.raises(ValueError):
        calculate_moving_average(prices, window=0)


def test_rolling_volatility():
    """Verify annualized rolling volatility calculation."""
    # Constant returns have zero standard deviation
    flat_returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
    vol = calculate_rolling_volatility(flat_returns, window=3, trading_days=252)

    assert pytest.approx(vol.iloc[2], abs=1e-6) == 0.0

    with pytest.raises(ValueError):
        calculate_rolling_volatility(flat_returns, window=1)


def test_compute_correlation_matrix():
    """Verify correlation matrix between two assets."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df_a = pd.DataFrame({"Adj Close": [100.0, 102.0, 104.0, 106.0, 108.0]}, index=dates)
    df_b = pd.DataFrame({"Adj Close": [50.0, 51.0, 52.0, 53.0, 54.0]}, index=dates)

    corr = compute_correlation_matrix({"A": df_a, "B": df_b})
    assert corr.shape == (2, 2)
    assert pytest.approx(corr.loc["A", "A"]) == 1.0
    assert pytest.approx(corr.loc["B", "B"]) == 1.0
    # Both series are increasing at constant rate, correlation should be 1.0
    assert pytest.approx(corr.loc["A", "B"]) == 1.0


def test_add_technical_indicators():
    """Verify enrichment of price dataframe with technical indicators."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    prices = np.linspace(100, 200, 60)
    df = pd.DataFrame({"Adj Close": prices}, index=dates)

    enriched = add_technical_indicators(df, fast_window=20, slow_window=50)

    assert "simple_return" in enriched.columns
    assert "log_return" in enriched.columns
    assert "MA_20" in enriched.columns
    assert "MA_50" in enriched.columns
    assert "rolling_vol_20d" in enriched.columns
    assert not np.isnan(enriched["MA_20"].iloc[19])
    assert not np.isnan(enriched["MA_50"].iloc[49])
