"""Quant Trading Research System - Command Line Interface (CLI) Entry Point.

Runs the complete end-to-end quantitative research pipeline:
Data Ingestion -> Processing -> Signal Generation -> Backtesting -> Risk Analytics.
"""

import argparse
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode symbols
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backtester import run_portfolio_backtest, run_single_backtest
from src.config import (
    BENCHMARK_TICKER,
    DEFAULT_BROKERAGE_PCT,
    DEFAULT_FAST_MA,
    DEFAULT_LONG_ONLY,
    DEFAULT_PERIOD,
    DEFAULT_SLIPPAGE_PCT,
    DEFAULT_SLOW_MA,
    DEFAULT_STARTING_CAPITAL,
    DEFAULT_STT_PCT,
    DEFAULT_TICKERS,
    RISK_FREE_RATE,
)
from src.data_loader import download_ticker_data, load_multiple_tickers
from src.processing import add_technical_indicators, compute_correlation_matrix
from src.risk import build_comparison_table, calculate_all_metrics
from src.strategies import MovingAverageCrossoverStrategy, get_current_signal


def run_pipeline(
    ticker: str = "TCS.NS",
    portfolio_mode: bool = False,
    period: str = DEFAULT_PERIOD,
    fast_ma: int = DEFAULT_FAST_MA,
    slow_ma: int = DEFAULT_SLOW_MA,
    long_only: bool = DEFAULT_LONG_ONLY,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
) -> None:
    """Execute complete quantitative research workflow."""
    print("=" * 70)
    print("       QUANT TRADING RESEARCH SYSTEM - EXECUTION PIPELINE")
    print("=" * 70)
    print(f"Target: {'Equal-Weight Portfolio' if portfolio_mode else ticker}")
    print(f"Period: {period} | Fast MA: {fast_ma} | Slow MA: {slow_ma} | Long Only: {long_only}")
    print(f"Capital: INR {starting_capital:,.2f} | Risk-Free Rate: {RISK_FREE_RATE * 100:.1f}%\n")

    # 1. Benchmark Data
    print(f"Ingesting Benchmark ({BENCHMARK_TICKER})...")
    benchmark_df = download_ticker_data(BENCHMARK_TICKER, period=period)
    benchmark_prices = benchmark_df["Adj Close"]

    # 2. Strategy Execution
    strat = MovingAverageCrossoverStrategy(
        fast_window=fast_ma,
        slow_window=slow_ma,
        long_only=long_only,
    )

    if not portfolio_mode:
        print(f"\nIngesting Asset Data ({ticker})...")
        df = download_ticker_data(ticker, period=period)
        enriched_df = add_technical_indicators(df, fast_window=fast_ma, slow_window=slow_ma)

        signals = strat.generate_signals(enriched_df, price_col="Adj Close")
        current_status = get_current_signal(signals)
        latest_price = enriched_df["Adj Close"].iloc[-1]
        print(f"\n>>> CURRENT SIGNAL FOR {ticker}: [{current_status}] (Latest Close: INR {latest_price:,.2f}) <<<\n")

        # Run Backtest
        print("Running Backtest Engine (Shifted T+1 Execution with Friction)...")
        result = run_single_backtest(
            prices=enriched_df["Adj Close"],
            signals=signals,
            starting_capital=starting_capital,
            brokerage_pct=DEFAULT_BROKERAGE_PCT,
            slippage_pct=DEFAULT_SLIPPAGE_PCT,
            stt_pct=DEFAULT_STT_PCT,
            benchmark_prices=benchmark_prices,
            ticker=ticker,
        )
    else:
        tickers = [t for t in DEFAULT_TICKERS if t != BENCHMARK_TICKER]
        print(f"\nIngesting Portfolio Equities: {tickers}...")
        data_dict = load_multiple_tickers(tickers=tickers, period=period)

        price_dict = {}
        signal_dict = {}
        for t, df_t in data_dict.items():
            price_dict[t] = df_t["Adj Close"]
            s = strat.generate_signals(df_t, price_col="Adj Close")
            signal_dict[t] = s
            curr = get_current_signal(s)
            print(f" - {t:14s}: Current Signal = [{curr:4s}] (Price: INR {df_t['Adj Close'].iloc[-1]:,.2f})")

        print("\nRunning Multi-Asset Equal-Weight Portfolio Backtest...")
        result = run_portfolio_backtest(
            price_dict=price_dict,
            signal_dict=signal_dict,
            starting_capital=starting_capital,
            brokerage_pct=DEFAULT_BROKERAGE_PCT,
            slippage_pct=DEFAULT_SLIPPAGE_PCT,
            stt_pct=DEFAULT_STT_PCT,
            benchmark_prices=benchmark_prices,
        )

    # 3. Risk & Performance Analysis
    strat_daily = result.daily_results["strategy_net_return"]
    strat_eq = result.daily_results["equity_curve"]
    bench_daily = result.daily_results["benchmark_return"]
    bench_eq = result.daily_results["benchmark_equity"]

    strat_metrics = calculate_all_metrics(
        daily_returns=strat_daily,
        equity_curve=strat_eq,
        trade_log=result.trade_log,
        risk_free_rate=RISK_FREE_RATE,
    )
    bench_metrics = calculate_all_metrics(
        daily_returns=bench_daily,
        equity_curve=bench_eq,
        trade_log=None,
        risk_free_rate=RISK_FREE_RATE,
    )

    comparison_table = build_comparison_table(strat_metrics, bench_metrics)

    print("\n" + "=" * 70)
    print("                PERFORMANCE & RISK METRICS REPORT")
    print("=" * 70)
    print(comparison_table.to_string())
    print("=" * 70)

    # 4. Recent Trade Log
    if not result.trade_log.empty:
        print("\nRECENT COMPLETED TRADES (Last 5):")
        print(result.trade_log.tail(5).to_string(index=False))
    else:
        print("\nNo trades were generated in the evaluation period.")

    print("\nTo view interactive charts, run: streamlit run dashboard/app.py\n")


def main() -> None:
    """Parse arguments and execute pipeline."""
    parser = argparse.ArgumentParser(description="Quant Trading Research System CLI")
    parser.add_argument("--ticker", type=str, default="TCS.NS", help="NSE ticker symbol (e.g. TCS.NS, RELIANCE.NS)")
    parser.add_argument("--portfolio", action="store_true", help="Run multi-stock equal-weight portfolio")
    parser.add_argument("--period", type=str, default="10y", help="Historical timeframe (e.g. 5y, 10y, max)")
    parser.add_argument("--fast-ma", type=int, default=DEFAULT_FAST_MA, help="Fast Moving Average Window")
    parser.add_argument("--slow-ma", type=int, default=DEFAULT_SLOW_MA, help="Slow Moving Average Window")
    parser.add_argument("--capital", type=float, default=DEFAULT_STARTING_CAPITAL, help="Starting capital in INR")
    parser.add_argument("--long-only", action="store_true", default=True, help="Enable Long-Only mode")

    args = parser.parse_args()
    run_pipeline(
        ticker=args.ticker,
        portfolio_mode=args.portfolio,
        period=args.period,
        fast_ma=args.fast_ma,
        slow_ma=args.slow_ma,
        long_only=args.long_only,
        starting_capital=args.capital,
    )


if __name__ == "__main__":
    main()
