"""Data collection and caching module using Yahoo Finance.

Downloads daily historical prices for Indian stocks and indices, handles caching,
error recovery, and data cleaning (missing value forward-fill & cleanup).
"""

import os
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.config import (
    BENCHMARK_TICKER,
    CACHE_EXPIRY_HOURS,
    DATA_DIR,
    DEFAULT_PERIOD,
    DEFAULT_TICKERS,
)


def _sanitize_ticker_for_filename(ticker: str) -> str:
    """Sanitize ticker string into a valid filesystem filename.

    Example: '^NSEI' -> 'NSEI', 'TCS.NS' -> 'TCS_NS'.
    """
    cleaned = re.sub(r"[^\w\.-]", "_", ticker)
    return cleaned.strip("_")


def get_cache_filepath(ticker: str, cache_dir: Path = DATA_DIR) -> Path:
    """Return the absolute path to the cached CSV file for a given ticker."""
    filename = f"{_sanitize_ticker_for_filename(ticker)}.csv"
    return cache_dir / filename


def is_cache_fresh(filepath: Path, max_age_hours: int = CACHE_EXPIRY_HOURS) -> bool:
    """Check if the cached file exists and was modified within max_age_hours."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False
    mtime = datetime.datetime.fromtimestamp(filepath.stat().st_mtime)
    age = datetime.datetime.now() - mtime
    return age.total_seconds() < (max_age_hours * 3600)


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw price DataFrame.

    - Sort by date ascending.
    - Forward-fill small gaps (up to 5 consecutive trading days).
    - Drop remaining rows containing NaNs in critical price columns.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    # Normalize index to date without timezone
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Forward fill small gaps in pricing
    df = df.ffill(limit=5)

    # Drop any remaining rows with NaN in Close or Adj Close
    cols_to_check = [col for col in ["Adj Close", "Close"] if col in df.columns]
    if cols_to_check:
        df = df.dropna(subset=cols_to_check)

    return df


def print_data_summary(ticker: str, df: pd.DataFrame) -> None:
    """Print a short readable summary of the downloaded / loaded dataset."""
    if df.empty:
        print(f"[{ticker}] No data available.")
        return

    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    total_rows = len(df)
    missing_vals = int(df.isna().sum().sum())

    print(
        f"[{ticker}] Summary -> Date Range: {start_date} to {end_date} | "
        f"Rows: {total_rows} | Missing Values: {missing_vals}"
    )


def download_ticker_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    force_reload: bool = False,
    cache_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Download daily historical prices with yfinance and cache locally to CSV.

    Args:
        ticker: Ticker symbol (e.g., '^NSEI', 'TCS.NS', 'RELIANCE.NS').
        period: Historical period string (default: '10y').
        force_reload: If True, bypass cache and re-download.
        cache_dir: Directory where cached CSVs are saved.

    Returns:
        pd.DataFrame containing historical price and volume data.

    Raises:
        ValueError: If ticker is invalid or returns empty dataset.
        RuntimeError: If download fails and no cached copy is available.
    """
    cache_file = get_cache_filepath(ticker, cache_dir)

    # 1. Check fresh cache
    if not force_reload and is_cache_fresh(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                print(f"[{ticker}] Loaded from cache: {cache_file.name}")
                print_data_summary(ticker, df)
                return df
        except Exception as e:
            print(f"[{ticker}] Cache read failed ({e}). Re-downloading...")

    # 2. Download from yfinance
    print(f"[{ticker}] Downloading {period} data from Yahoo Finance...")
    try:
        raw_df = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        # Fall back to stale cache if available when offline
        if cache_file.exists():
            print(f"[{ticker}] Download failed ({exc}). Using offline cache.")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise RuntimeError(
            f"Failed to download data for '{ticker}' and no cache is available: {exc}"
        ) from exc

    # Handle multi-level columns in newer yfinance versions (Price, Ticker)
    if isinstance(raw_df.columns, pd.MultiIndex):
        # Extract level 0 ('Close', 'Adj Close', etc.)
        try:
            raw_df.columns = raw_df.columns.get_level_values(0)
        except Exception:
            pass

    if raw_df is None or raw_df.empty:
        # Check if stale cache exists
        if cache_file.exists():
            print(f"[{ticker}] Empty data received. Using stale cache.")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise ValueError(f"No price data found for ticker '{ticker}'. Please verify the symbol.")

    # 3. Ensure 'Adj Close' is present
    if "Adj Close" not in raw_df.columns and "Close" in raw_df.columns:
        raw_df["Adj Close"] = raw_df["Close"]

    # 4. Clean missing values and format index
    cleaned_df = clean_price_data(raw_df)

    if cleaned_df.empty:
        raise ValueError(f"Cleaned dataset for '{ticker}' contains no valid rows.")

    # 5. Save to CSV cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(cache_file)
    print(f"[{ticker}] Saved cleaned data to cache: {cache_file.name}")
    print_data_summary(ticker, cleaned_df)

    return cleaned_df


def load_multiple_tickers(
    tickers: Optional[List[str]] = None,
    period: str = DEFAULT_PERIOD,
    force_reload: bool = False,
    cache_dir: Path = DATA_DIR,
) -> Dict[str, pd.DataFrame]:
    """Download or load historical data for a list of tickers.

    Args:
        tickers: List of ticker symbols. Defaults to DEFAULT_TICKERS.
        period: Historical period string (e.g. '10y').
        force_reload: If True, re-downloads all tickers.
        cache_dir: Path to directory for caching CSV files.

    Returns:
        Dictionary mapping ticker symbol to price DataFrame.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    data_store: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = download_ticker_data(
                ticker=ticker,
                period=period,
                force_reload=force_reload,
                cache_dir=cache_dir,
            )
            data_store[ticker] = df
        except Exception as err:
            print(f"Error loading {ticker}: {err}")

    return data_store


if __name__ == "__main__":
    print("Testing data_loader.py independently...")
    data = load_multiple_tickers(tickers=DEFAULT_TICKERS, period="1y")
    print(f"Successfully loaded {len(data)} tickers: {list(data.keys())}")
