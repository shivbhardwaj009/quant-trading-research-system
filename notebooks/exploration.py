"""Exploration script for interactive research and Jupyter-style analysis.

Demonstrates the 5 phases of the Quant Trading Research System step-by-step:
1. Data collection & caching
2. Feature engineering & technical indicators
3. Strategy signal generation
4. Backtesting with zero lookahead bias
5. Risk & performance evaluation
"""

from pathlib import Path
import sys

# Ensure root directory is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.backtester import run_portfolio_backtest, run_single_backtest
from src.config import BENCHMARK_TICKER, DEFAULT_TICKERS, RISK_FREE_RATE
from src.data_loader import download_ticker_data, load_multiple_tickers
from src.processing import add_technical_indicators, compute_correlation_matrix
from src.risk import build_comparison_table, calculate_all_metrics, calculate_drawdowns
from src.strategies import MovingAverageCrossoverStrategy, get_current_signal


def main() -> None:
    print("=" * 60)
    print("QUANT TRADING RESEARCH SYSTEM - INTERACTIVE EXPLORATION")
    print("=" * 60)

    # Step 1: Ingestion
    print("\n1. Loading Data for TCS.NS and Nifty 50 Benchmark...")
    tcs_df = download_ticker_data("TCS.NS", period="10y")
    nifty_df = download_ticker_data(BENCHMARK_TICKER, period="10y")

    # Step 2: Feature Engineering
    print("\n2. Enriching with 20-Day and 50-Day Moving Averages & Returns...")
    enriched_df = add_technical_indicators(tcs_df, fast_window=20, slow_window=50)
    print(enriched_df[["Adj Close", "simple_return", "MA_20", "MA_50", "rolling_vol_20d"]].tail())

    # Step 3: Strategy Signal Generation
    print("\n3. Generating Moving Average Crossover Signals...")
    strat = MovingAverageCrossoverStrategy(fast_window=20, slow_window=50, long_only=True)
    signals = strat.generate_signals(enriched_df)
    current_signal = get_current_signal(signals)
    print(f"Current Signal for TCS.NS: {current_signal}")

    # Step 4: Backtesting with Zero Look-Ahead Bias & Costs
    print("\n4. Running Vectorized Backtest (T+1 Shift with Transaction Friction)...")
    res = run_single_backtest(
        prices=enriched_df["Adj Close"],
        signals=signals,
        starting_capital=100000.0,
        benchmark_prices=nifty_df["Adj Close"],
        ticker="TCS.NS",
    )
    print(f"Ending Equity: INR {res.daily_results['equity_curve'].iloc[-1]:,.2f}")
    print(f"Total Completed Trades: {len(res.trade_log)}")

    # Step 5: Risk & Performance Evaluation
    print("\n5. Computing Institutional Risk Metrics...")
    strat_metrics = calculate_all_metrics(
        daily_returns=res.daily_results["strategy_net_return"],
        equity_curve=res.daily_results["equity_curve"],
        trade_log=res.trade_log,
        risk_free_rate=RISK_FREE_RATE,
    )
    bench_metrics = calculate_all_metrics(
        daily_returns=res.daily_results["benchmark_return"],
        equity_curve=res.daily_results["benchmark_equity"],
        trade_log=None,
        risk_free_rate=RISK_FREE_RATE,
    )

    metrics_table = build_comparison_table(strat_metrics, bench_metrics)
    print(metrics_table.to_string())

    # Step 6: Multi-Stock Correlation Analysis
    print("\n6. Cross-Asset Correlation Analysis...")
    all_data = load_multiple_tickers(DEFAULT_TICKERS, period="10y")
    corr = compute_correlation_matrix(all_data)
    print(corr.round(3).to_string())

    print("\nExploration completed successfully!")


if __name__ == "__main__":
    main()
