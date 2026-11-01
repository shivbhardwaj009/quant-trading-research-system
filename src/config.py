"""Configuration settings for the Quant Trading Research System.

Centralized settings for data downloading, backtesting, risk calculations,
and dashboard display.
"""

from pathlib import Path
from typing import List

# Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Cache Settings
CACHE_EXPIRY_HOURS: int = 24  # Max age of cached CSV file before refreshing

# Default Tickers (Indian Market & Benchmark)
DEFAULT_TICKERS: List[str] = [
    "^NSEI",         # Nifty 50 Benchmark
    "TCS.NS",        # Tata Consultancy Services
    "RELIANCE.NS",   # Reliance Industries
    "INFY.NS",       # Infosys
    "HDFCBANK.NS",   # HDFC Bank
]

BENCHMARK_TICKER: str = "^NSEI"
DEFAULT_PERIOD: str = "10y"

# Market & Trading Calendar
TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE: float = 0.065  # 6.5% annual risk-free rate for Indian G-Secs

# Strategy Parameters
DEFAULT_FAST_MA: int = 20
DEFAULT_SLOW_MA: int = 50
DEFAULT_LONG_ONLY: bool = True

# Capital & Transaction Costs
DEFAULT_STARTING_CAPITAL: float = 100000.0  # ₹1,00,000 INR
DEFAULT_BROKERAGE_PCT: float = 0.0003       # 0.03% brokerage per trade
DEFAULT_SLIPPAGE_PCT: float = 0.0005        # 0.05% estimated slippage
DEFAULT_STT_PCT: float = 0.001              # 0.1% Securities Transaction Tax (delivery)

# Risk Parameters
VAR_CONFIDENCE_LEVEL: float = 0.95          # 95% Historical Value at Risk
