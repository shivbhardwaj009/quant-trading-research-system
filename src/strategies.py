"""Strategy module for generating trading signals.

Defines a modular BaseStrategy interface and implementations including the
Moving Average Crossover Strategy (Golden Cross / Death Cross).
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_FAST_MA,
    DEFAULT_LONG_ONLY,
    DEFAULT_SLOW_MA,
)
from src.processing import calculate_moving_average


class BaseStrategy(ABC):
    """Abstract Base Class for trading strategies.

    All strategies must implement the generate_signals method accepting a
    price DataFrame and returning a Series of target positions / signals.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        price_col: str = "Adj Close",
    ) -> pd.Series:
        """Generate a series of target positions.

        Args:
            df: Historical price DataFrame.
            price_col: Column name containing prices.

        Returns:
            pd.Series containing position targets: 1 (long), 0 (cash), or -1 (short).
        """
        pass


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy.

    - 20-day MA crosses ABOVE 50-day MA -> Buy (signal = 1)
    - 20-day MA crosses BELOW 50-day MA -> Sell (signal = 0 in long-only mode, or -1 in long-short mode)
    - Between crossovers -> Hold (retains the active position)
    """

    def __init__(
        self,
        fast_window: int = DEFAULT_FAST_MA,
        slow_window: int = DEFAULT_SLOW_MA,
        long_only: bool = DEFAULT_LONG_ONLY,
        name: str = "MA_Crossover",
    ) -> None:
        super().__init__(name=name)
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be strictly less than slow_window ({slow_window})."
            )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.long_only = long_only

    def generate_signals(
        self,
        df: pd.DataFrame,
        price_col: str = "Adj Close",
    ) -> pd.Series:
        """Calculate MA crossover signals.

        Args:
            df: Historical price DataFrame with DatetimeIndex.
            price_col: Column containing price data.

        Returns:
            pd.Series indexed like df with values:
                1 (Long), 0 (Cash), or -1 (Short if long_only is False).
        """
        target_price = df[price_col] if price_col in df.columns else df["Close"]

        fast_ma = calculate_moving_average(target_price, window=self.fast_window)
        slow_ma = calculate_moving_average(target_price, window=self.slow_window)

        # Initialize signal series with 0.0 (Cash)
        signals = pd.Series(0.0, index=df.index, dtype=float)

        # Valid mask: where both MAs are computed (after warmup period)
        valid_mask = fast_ma.notna() & slow_ma.notna()

        # Generate signals based on relative positions
        # fast_ma > slow_ma -> Long (1.0)
        # fast_ma < slow_ma -> Cash (0.0) if long_only else Short (-1.0)
        if self.long_only:
            signals[valid_mask & (fast_ma > slow_ma)] = 1.0
            signals[valid_mask & (fast_ma < slow_ma)] = 0.0
        else:
            signals[valid_mask & (fast_ma > slow_ma)] = 1.0
            signals[valid_mask & (fast_ma < slow_ma)] = -1.0

        # Handle equal MAs by forward filling previous position
        signals[valid_mask & (fast_ma == slow_ma)] = np.nan
        signals = signals.ffill().fillna(0.0)

        signals.name = "signal"
        return signals


def moving_average_crossover(
    df: pd.DataFrame,
    fast_window: int = DEFAULT_FAST_MA,
    slow_window: int = DEFAULT_SLOW_MA,
    long_only: bool = DEFAULT_LONG_ONLY,
    price_col: str = "Adj Close",
) -> pd.Series:
    """Functional wrapper for MovingAverageCrossoverStrategy."""
    strat = MovingAverageCrossoverStrategy(
        fast_window=fast_window,
        slow_window=slow_window,
        long_only=long_only,
    )
    return strat.generate_signals(df, price_col=price_col)


def get_current_signal(signals: pd.Series) -> str:
    """Interpret the latest position and action for a signal Series.

    Returns:
        One of: 'BUY' (position is 1), 'SELL' (position is 0 or -1), or 'HOLD' (no signal yet).
    """
    if signals.empty or signals.dropna().empty:
        return "HOLD"

    last_signal = signals.dropna().iloc[-1]
    if last_signal > 0:
        return "BUY"
    elif last_signal < 0:
        return "SELL"
    else:
        # Check if previous was long (meaning we recently exited to cash)
        if len(signals) >= 2 and signals.dropna().iloc[-2] > 0:
            return "SELL"
        return "HOLD"
