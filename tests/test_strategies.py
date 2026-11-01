"""Unit tests for Phase 3: Moving Average Crossover Strategy (strategies.py)."""

import numpy as np
import pandas as pd
import pytest

from src.strategies import (
    MovingAverageCrossoverStrategy,
    get_current_signal,
    moving_average_crossover,
)


def test_invalid_window_parameters():
    """Verify ValueError is raised if fast_window >= slow_window."""
    with pytest.raises(ValueError):
        MovingAverageCrossoverStrategy(fast_window=50, slow_window=20)

    with pytest.raises(ValueError):
        MovingAverageCrossoverStrategy(fast_window=20, slow_window=20)


def test_ma_crossover_long_only_signals():
    """Verify long-only signals produce 1 for Golden Cross and 0 for Death Cross."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Prices rising then falling
    prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 12.0, 10.0, 8.0, 6.0]
    df = pd.DataFrame({"Adj Close": prices}, index=dates)

    # Use small windows for testing: fast=2, slow=4
    strat = MovingAverageCrossoverStrategy(fast_window=2, slow_window=4, long_only=True)
    signals = strat.generate_signals(df)

    # During warmup (index 0, 1, 2), slow MA is NaN, so signal is 0
    assert signals.iloc[0] == 0.0
    assert signals.iloc[1] == 0.0
    assert signals.iloc[2] == 0.0

    # Index 3: fast MA of [12, 13] is 12.5; slow MA of [10, 11, 12, 13] is 11.5 -> fast > slow -> Long (1)
    assert signals.iloc[3] == 1.0

    # Later when prices drop severely, fast < slow -> Cash (0 in long-only)
    assert signals.iloc[-1] == 0.0


def test_ma_crossover_long_short_signals():
    """Verify long-short mode produces -1 when fast MA is below slow MA."""
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    prices = [20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0]
    df = pd.DataFrame({"Adj Close": prices}, index=dates)

    strat = MovingAverageCrossoverStrategy(fast_window=2, slow_window=3, long_only=False)
    signals = strat.generate_signals(df)

    # After warmup period, consistently downward trending price gives -1.0
    assert signals.iloc[-1] == -1.0


def test_get_current_signal():
    """Verify latest status interpretation."""
    assert get_current_signal(pd.Series([0.0, 1.0, 1.0])) == "BUY"
    assert get_current_signal(pd.Series([1.0, 1.0, 0.0])) == "SELL"
    assert get_current_signal(pd.Series([0.0, 0.0, -1.0])) == "SELL"
    assert get_current_signal(pd.Series([])) == "HOLD"
