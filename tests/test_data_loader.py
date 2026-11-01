"""Unit tests for Phase 1: Data Collection & Caching (data_loader.py)."""

import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    _sanitize_ticker_for_filename,
    clean_price_data,
    get_cache_filepath,
    is_cache_fresh,
    download_ticker_data,
)


def test_sanitize_ticker_for_filename():
    """Verify ticker symbols are sanitized safely for Windows filesystems."""
    assert _sanitize_ticker_for_filename("^NSEI") == "NSEI"
    assert _sanitize_ticker_for_filename("TCS.NS") == "TCS.NS"
    assert _sanitize_ticker_for_filename("M&M.NS") == "M_M.NS"


def test_clean_price_data_forward_fill_and_dropna():
    """Verify data cleaning correctly forward-fills small gaps and drops leftover NaNs."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    raw_df = pd.DataFrame(
        {
            "Open": [np.nan, 100.0, 102.0, np.nan, 105.0, 106.0],
            "Adj Close": [np.nan, 100.0, 102.0, np.nan, 105.0, 106.0],
            "Close": [np.nan, 100.0, 102.0, np.nan, 105.0, 106.0],
            "Volume": [np.nan, 1000, 1200, np.nan, 1500, 1100],
        },
        index=dates,
    )

    cleaned = clean_price_data(raw_df)

    # 1st row has NaN at start that cannot be forward-filled, so it should be dropped
    assert len(cleaned) == 5
    # The 4th row (index 3) should have been forward-filled with 102.0
    assert cleaned.loc["2024-01-04", "Adj Close"] == 102.0
    # No NaNs in critical price columns
    assert cleaned["Adj Close"].isna().sum() == 0


def test_cache_freshness(tmp_path: Path):
    """Verify cache freshness calculation."""
    cache_file = tmp_path / "TEST.csv"

    # Non-existent file
    assert not is_cache_fresh(cache_file, max_age_hours=24)

    # Create empty file
    cache_file.touch()
    assert not is_cache_fresh(cache_file, max_age_hours=24)

    # Write data to file
    cache_file.write_text("Date,Adj Close\n2024-01-01,100\n")
    assert is_cache_fresh(cache_file, max_age_hours=24)


def test_cache_save_and_reload(tmp_path: Path, monkeypatch):
    """Verify download_ticker_data reuses cached CSV when fresh."""
    ticker = "TEST_STOCK"
    cache_file = tmp_path / f"{ticker}.csv"

    # Pre-populate cache
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    sample_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Adj Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Volume": [1000, 1000, 1000, 1000, 1000],
        },
        index=dates,
    )
    sample_df.to_csv(cache_file)

    # Call download_ticker_data pointing to tmp_path cache
    loaded_df = download_ticker_data(ticker=ticker, cache_dir=tmp_path)

    assert len(loaded_df) == 5
    assert "Adj Close" in loaded_df.columns
    assert loaded_df["Adj Close"].iloc[-1] == 104.0


def test_invalid_ticker_handling(tmp_path: Path):
    """Verify invalid ticker raises ValueError when no data is returned."""
    with pytest.raises(ValueError):
        download_ticker_data(ticker="INVALID_TICKER_XYZ_999", cache_dir=tmp_path)
