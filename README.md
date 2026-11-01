# Quant Trading Research System

An institutional-grade Quantitative Trading Research System for the Indian stock market (NSE & Nifty 50). This platform downloads historical price data, generates technical and statistical signals, executes backtests with realistic transaction friction (brokerage, slippage, STT), evaluates comprehensive risk metrics, and visualizes strategy performance via an interactive Streamlit dashboard.

---

## Architecture & Project Structure

```text
quant_project/
  data/                 # Cached historical price CSV files
  src/
    config.py           # Centralized configuration (parameters, costs, rates)
    data_loader.py      # Historical data download with yfinance & cache engine
    processing.py       # Returns, moving averages, rolling volatility, correlation
    strategies.py       # Modular signal generation strategies (e.g. MA Crossover)
    backtester.py       # Vectorized & realistic backtest engine (no lookahead bias)
    risk.py             # Performance metrics (Sharpe, Sortino, Max Drawdown, VaR)
  dashboard/
    app.py              # Interactive Streamlit & Plotly analytics dashboard
  tests/                # Automated pytest suite for all phases
  requirements.txt      # Pinned dependency specifications
  README.md             # Project documentation and user guide
```

---

## Phase Implementation Status

- [x] **Phase 1: Data Collection & Caching** (`src/data_loader.py`)
  - Automated download of Indian equities (`TCS.NS`, `RELIANCE.NS`, `INFY.NS`, `HDFCBANK.NS`) and benchmark (`^NSEI`).
  - Intelligent local CSV caching with freshness check.
  - Forward-fill for small price gaps and error handling for missing/offline data.
  - Unit tests covering sanitization, caching, forward-fill, and error recovery.
- [ ] **Phase 2: Data Processing** (`src/processing.py`)
- [ ] **Phase 3: Strategy Engine** (`src/strategies.py`)
- [ ] **Phase 4: Realistic Backtester** (`src/backtester.py`)
- [ ] **Phase 5: Risk & Performance Metrics** (`src/risk.py`)
- [ ] **Phase 6: Interactive Dashboard** (`dashboard/app.py`)
- [ ] **Phase 7: Final Polish & Comprehensive Documentation**

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shivbhardwaj009/quant-trading-research-system.git
   cd "python trading platform"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Unit Tests:**
   ```bash
   python -m pytest tests/ -v
   ```
