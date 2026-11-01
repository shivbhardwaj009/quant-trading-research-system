"""Institutional-grade backtesting engine with realistic friction and zero look-ahead bias.

Supports single-stock and equal-weighted multi-stock portfolios.
Applies realistic Indian market transaction costs (brokerage, slippage, STT)
strictly on execution days.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_BROKERAGE_PCT,
    DEFAULT_SLIPPAGE_PCT,
    DEFAULT_STARTING_CAPITAL,
    DEFAULT_STT_PCT,
)
from src.processing import calculate_simple_returns


@dataclass
class BacktestResult:
    """Encapsulates backtest outputs."""

    daily_results: pd.DataFrame
    trade_log: pd.DataFrame
    starting_capital: float
    ticker: str


def generate_trade_log(
    prices: pd.Series,
    positions: pd.Series,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    ticker: str = "ASSET",
) -> pd.DataFrame:
    """Generate discrete trade logs from positions and prices.

    Tracks entry date, exit date, entry price, exit price, trade type,
    percentage return, and rupee profit/loss.
    """
    trades: List[Dict[str, Union[str, float, pd.Timestamp]]] = []
    in_trade = False
    entry_date: Optional[pd.Timestamp] = None
    entry_price = 0.0
    trade_type = "LONG"

    # Iterate through chronological positions
    for current_date, pos in positions.items():
        price = prices.loc[current_date]
        prev_pos = 0.0 if not in_trade else (1.0 if trade_type == "LONG" else -1.0)

        # Position opens or changes
        if not in_trade and pos != 0:
            in_trade = True
            entry_date = current_date
            entry_price = float(price)
            trade_type = "LONG" if pos > 0 else "SHORT"
        elif in_trade and pos != prev_pos:
            # Trade exit or reversal
            exit_date = current_date
            exit_price = float(price)
            if trade_type == "LONG":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            pnl_amount = starting_capital * pnl_pct

            trades.append(
                {
                    "ticker": ticker,
                    "trade_type": trade_type,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "pnl_pct": float(pnl_pct),
                    "pnl_amount": float(pnl_amount),
                }
            )

            # Check if immediately reversing
            if pos != 0:
                in_trade = True
                entry_date = current_date
                entry_price = float(price)
                trade_type = "LONG" if pos > 0 else "SHORT"
            else:
                in_trade = False

    # Close any currently open trade on final date
    if in_trade and entry_date is not None:
        last_date = positions.index[-1]
        last_price = float(prices.iloc[-1])
        if trade_type == "LONG":
            pnl_pct = (last_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - last_price) / entry_price
        trades.append(
            {
                "ticker": ticker,
                "trade_type": f"{trade_type} (OPEN)",
                "entry_date": entry_date,
                "exit_date": last_date,
                "entry_price": float(entry_price),
                "exit_price": float(last_price),
                "pnl_pct": float(pnl_pct),
                "pnl_amount": float(starting_capital * pnl_pct),
            }
        )

    if not trades:
        return pd.DataFrame(
            columns=[
                "ticker",
                "trade_type",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "pnl_pct",
                "pnl_amount",
            ]
        )

    return pd.DataFrame(trades)


def run_single_backtest(
    prices: pd.Series,
    signals: pd.Series,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    brokerage_pct: float = DEFAULT_BROKERAGE_PCT,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    stt_pct: float = DEFAULT_STT_PCT,
    benchmark_prices: Optional[pd.Series] = None,
    ticker: str = "STRATEGY",
) -> BacktestResult:
    """Run single-asset backtest without look-ahead bias.

    Rule: Signals generated on day t are executed on day t+1.
          Positions are lagged by 1 day: position = signal.shift(1).

    Args:
        prices: Asset closing prices Series.
        signals: Raw target signals Series (1, 0, -1).
        starting_capital: Initial portfolio value in INR.
        brokerage_pct: Brokerage fee per trade turnover.
        slippage_pct: Slippage cost per trade turnover.
        stt_pct: Securities Transaction Tax per trade turnover.
        benchmark_prices: Optional benchmark price Series (e.g. ^NSEI).
        ticker: Asset identifier.

    Returns:
        BacktestResult containing daily table and trade logs.
    """
    # Align prices and signals
    df = pd.DataFrame({"price": prices, "signal": signals}).dropna(subset=["price"])
    df["signal"] = df["signal"].fillna(0.0)

    # CRITICAL: Prevent Look-Ahead Bias by lagging signal by 1 day for actual position
    df["position"] = df["signal"].shift(1).fillna(0.0)

    # Asset return: (P_t / P_{t-1}) - 1
    df["asset_return"] = calculate_simple_returns(df["price"]).fillna(0.0)

    # Strategy gross return = position * asset_return
    df["strategy_gross_return"] = df["position"] * df["asset_return"]

    # Transaction costs apply only on turnover days (when position changes)
    total_cost_rate = brokerage_pct + slippage_pct + stt_pct
    df["position_change"] = df["position"].diff().abs().fillna(df["position"].abs())
    df["transaction_cost"] = df["position_change"] * total_cost_rate

    # Strategy net return = gross return - transaction costs
    df["strategy_net_return"] = df["strategy_gross_return"] - df["transaction_cost"]

    # Strategy Equity Curve
    df["equity_curve"] = starting_capital * (1.0 + df["strategy_net_return"]).cumprod()

    # Benchmark calculation
    if benchmark_prices is not None:
        bench = benchmark_prices.reindex(df.index).ffill()
        bench_ret = calculate_simple_returns(bench).fillna(0.0)
        df["benchmark_return"] = bench_ret
        df["benchmark_equity"] = starting_capital * (1.0 + bench_ret).cumprod()
    else:
        # Default benchmark: Buy & hold the asset itself
        df["benchmark_return"] = df["asset_return"]
        df["benchmark_equity"] = starting_capital * (1.0 + df["asset_return"]).cumprod()

    # Generate trade logs
    trade_log = generate_trade_log(
        prices=df["price"],
        positions=df["position"],
        starting_capital=starting_capital,
        ticker=ticker,
    )

    return BacktestResult(
        daily_results=df,
        trade_log=trade_log,
        starting_capital=starting_capital,
        ticker=ticker,
    )


def run_portfolio_backtest(
    price_dict: Dict[str, pd.Series],
    signal_dict: Dict[str, pd.Series],
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    brokerage_pct: float = DEFAULT_BROKERAGE_PCT,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    stt_pct: float = DEFAULT_STT_PCT,
    benchmark_prices: Optional[pd.Series] = None,
) -> BacktestResult:
    """Run equal-weighted multi-asset portfolio backtest.

    Distributes initial capital equally across all provided assets.

    Args:
        price_dict: Mapping of ticker to price Series.
        signal_dict: Mapping of ticker to signal Series.
        starting_capital: Total portfolio starting capital in INR.
        brokerage_pct: Brokerage rate.
        slippage_pct: Slippage rate.
        stt_pct: STT rate.
        benchmark_prices: Benchmark index prices (e.g. ^NSEI).

    Returns:
        BacktestResult representing combined portfolio performance.
    """
    tickers = [t for t in price_dict if t in signal_dict]
    if not tickers:
        raise ValueError("No common tickers found between price_dict and signal_dict.")

    num_assets = len(tickers)
    capital_per_asset = starting_capital / num_assets

    single_results: Dict[str, BacktestResult] = {}
    trade_logs: List[pd.DataFrame] = []

    for t in tickers:
        res = run_single_backtest(
            prices=price_dict[t],
            signals=signal_dict[t],
            starting_capital=capital_per_asset,
            brokerage_pct=brokerage_pct,
            slippage_pct=slippage_pct,
            stt_pct=stt_pct,
            ticker=t,
        )
        single_results[t] = res
        if not res.trade_log.empty:
            trade_logs.append(res.trade_log)

    combined_trade_log = (
        pd.concat(trade_logs, ignore_index=True)
        if trade_logs
        else pd.DataFrame()
    )

    # Combine daily returns with equal weights
    daily_returns_df = pd.DataFrame(
        {t: res.daily_results["strategy_net_return"] for t, res in single_results.items()}
    ).fillna(0.0)

    portfolio_net_return = daily_returns_df.mean(axis=1)

    combined_df = pd.DataFrame(index=daily_returns_df.index)
    combined_df["strategy_net_return"] = portfolio_net_return
    combined_df["equity_curve"] = starting_capital * (1.0 + portfolio_net_return).cumprod()

    # Benchmark calculation
    if benchmark_prices is not None:
        bench = benchmark_prices.reindex(combined_df.index).ffill()
        bench_ret = calculate_simple_returns(bench).fillna(0.0)
        combined_df["benchmark_return"] = bench_ret
        combined_df["benchmark_equity"] = starting_capital * (1.0 + bench_ret).cumprod()

    return BacktestResult(
        daily_results=combined_df,
        trade_log=combined_trade_log,
        starting_capital=starting_capital,
        ticker="PORTFOLIO_EQUAL_WEIGHT",
    )
