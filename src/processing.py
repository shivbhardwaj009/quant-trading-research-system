"""Data processing module for calculating returns, moving averages, and volatility.

Provides vectorised calculations for simple and logarithmic returns, rolling moving
averages, annualized rolling volatility, and cross-asset correlation matrices.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_FAST_MA,
    DEFAULT_SLOW_MA,
    TRADING_DAYS_PER_YEAR,
)


def calculate_simple_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily simple returns: (P_t - P_{t-1}) / P_{t-1}.

    Args:
        prices: Series of asset prices.

    Returns:
        Series of simple percentage returns with NaN at index 0.
    """
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily logarithmic returns: ln(P_t / P_{t-1}).

    Args:
        prices: Series of asset prices.

    Returns:
        Series of log returns with NaN at index 0.
    """
    return np.log(prices / prices.shift(1))


def calculate_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Calculate rolling simple moving average over a specified window.

    Args:
        prices: Series of asset prices.
        window: Number of periods for rolling calculation.

    Returns:
        Series containing rolling average.

    Raises:
        ValueError: If window is less than 1.
    """
    if window < 1:
        raise ValueError("Window size must be at least 1.")
    return prices.rolling(window=window).mean()


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = DEFAULT_FAST_MA,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate annualized rolling standard deviation of daily returns.

    Args:
        returns: Series of daily simple or log returns.
        window: Rolling observation window in days (default: 20).
        trading_days: Annualization scaling factor (default: 252).

    Returns:
        Series of annualized rolling volatility.

    Raises:
        ValueError: If window is less than 2.
    """
    if window < 2:
        raise ValueError("Window size must be at least 2 for volatility calculation.")
    return returns.rolling(window=window).std() * np.sqrt(trading_days)


def compute_correlation_matrix(
    data_dict: Dict[str, Union[pd.DataFrame, pd.Series]],
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """Compute the correlation matrix of daily returns between multiple assets.

    Args:
        data_dict: Mapping of ticker symbol to DataFrame (or Series) of prices.
        price_col: Column name to use for prices if DataFrames are provided.

    Returns:
        Square DataFrame representing pairwise Pearson correlation coefficients.
    """
    price_series_dict = {}
    for ticker, data in data_dict.items():
        if isinstance(data, pd.DataFrame):
            if price_col in data.columns:
                price_series_dict[ticker] = data[price_col]
            elif "Close" in data.columns:
                price_series_dict[ticker] = data["Close"]
            else:
                continue
        elif isinstance(data, pd.Series):
            price_series_dict[ticker] = data

    if not price_series_dict:
        return pd.DataFrame()

    combined_prices = pd.DataFrame(price_series_dict).dropna(how="all")
    returns_df = combined_prices.pct_change().dropna(how="all")
    return returns_df.corr()


def add_technical_indicators(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    fast_window: int = DEFAULT_FAST_MA,
    slow_window: int = DEFAULT_SLOW_MA,
) -> pd.DataFrame:
    """Enrich price DataFrame with returns, MAs, and rolling volatility.

    Args:
        df: Input price DataFrame.
        price_col: Target price column (default 'Adj Close').
        fast_window: Fast MA window (default 20).
        slow_window: Slow MA window (default 50).

    Returns:
        DataFrame enriched with columns: 'simple_return', 'log_return',
        f'MA_{fast_window}', f'MA_{slow_window}', 'rolling_vol_20d'.
    """
    result = df.copy()
    target_price = result[price_col] if price_col in result.columns else result["Close"]

    result["simple_return"] = calculate_simple_returns(target_price)
    result["log_return"] = calculate_log_returns(target_price)
    result[f"MA_{fast_window}"] = calculate_moving_average(target_price, window=fast_window)
    result[f"MA_{slow_window}"] = calculate_moving_average(target_price, window=slow_window)
    result["rolling_vol_20d"] = calculate_rolling_volatility(
        result["simple_return"],
        window=fast_window,
        trading_days=TRADING_DAYS_PER_YEAR,
    )

    return result
