"""Risk & Performance module for quantitative evaluation.

Calculates institutional risk metrics: Total Return, CAGR, Annualized Volatility,
Sharpe Ratio, Sortino Ratio, Drawdown curve, Maximum Drawdown, Max Single-Day Loss,
Calmar Ratio, Trade stats (Win Rate, Avg Win/Loss), and 95% Historical Value at Risk.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def calculate_cagr(
    equity_curve: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).

    CAGR = (Ending Value / Beginning Value) ** (252 / N) - 1.

    Args:
        equity_curve: Portfolio value series over time.
        trading_days: Annualization trading days constant (default: 252).

    Returns:
        Annualized compound growth rate as float.
    """
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0])
    if total_return <= 0:
        return -1.0

    num_periods = len(equity_curve)
    years = num_periods / trading_days
    if years <= 0:
        return 0.0

    return float(total_return ** (1.0 / years) - 1.0)


def calculate_drawdowns(equity_curve: pd.Series) -> Tuple[pd.Series, float]:
    """Calculate the rolling drawdown series and Maximum Drawdown.

    Drawdown_t = (Equity_t - CumulativeMax_t) / CumulativeMax_t.

    Args:
        equity_curve: Portfolio equity series.

    Returns:
        Tuple of (drawdown_series, max_drawdown as float between -1.0 and 0.0).
    """
    if equity_curve.empty:
        return pd.Series(dtype=float), 0.0

    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = float(drawdown.min())
    return drawdown, max_dd


def calculate_sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized Sharpe Ratio.

    Sharpe = (Annualized Return - Risk Free Rate) / Annualized Volatility.
    """
    clean_rets = daily_returns.dropna()
    if clean_rets.empty or clean_rets.std() == 0:
        return 0.0

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0
    excess_returns = clean_rets - daily_rf

    std_dev = clean_rets.std()
    if std_dev == 0 or np.isnan(std_dev):
        return 0.0

    # Annualize excess mean return and volatility
    annual_excess_return = excess_returns.mean() * trading_days
    annual_volatility = std_dev * np.sqrt(trading_days)

    return float(annual_excess_return / annual_volatility)


def calculate_sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized Sortino Ratio focusing on downside risk."""
    clean_rets = daily_returns.dropna()
    if clean_rets.empty:
        return 0.0

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0
    excess_returns = clean_rets - daily_rf

    negative_excess = excess_returns[excess_returns < 0]
    if len(negative_excess) == 0:
        return float("inf") if excess_returns.mean() > 0 else 0.0

    downside_std = np.sqrt(np.mean(negative_excess**2)) * np.sqrt(trading_days)
    if downside_std == 0:
        return 0.0

    annual_excess_return = excess_returns.mean() * trading_days
    return float(annual_excess_return / downside_std)


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio: CAGR / abs(Max Drawdown)."""
    abs_dd = abs(max_drawdown)
    if abs_dd == 0:
        return 0.0
    return float(cagr / abs_dd)


def calculate_historical_var(
    daily_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Historical Value at Risk (VaR) at specified confidence level.

    Returns positive percentage representing maximum expected loss at given confidence.
    """
    clean_rets = daily_returns.dropna()
    if clean_rets.empty:
        return 0.0

    percentile = (1.0 - confidence_level) * 100.0
    var_cutoff = float(np.percentile(clean_rets, percentile))
    return float(-var_cutoff) if var_cutoff < 0 else 0.0


def calculate_trade_metrics(trade_log: Optional[pd.DataFrame]) -> Dict[str, Union[float, int]]:
    """Compute trade-level statistics from the trade log."""
    if trade_log is None or trade_log.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }

    # Exclude open trades for win rate calculation if needed, or include them
    trades = trade_log[~trade_log["trade_type"].str.contains("OPEN", na=False)]
    if trades.empty:
        trades = trade_log

    total_trades = len(trades)
    winning_trades = trades[trades["pnl_pct"] > 0]
    losing_trades = trades[trades["pnl_pct"] < 0]

    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    avg_win = float(winning_trades["pnl_pct"].mean()) if not winning_trades.empty else 0.0
    avg_loss = float(losing_trades["pnl_pct"].mean()) if not losing_trades.empty else 0.0

    gross_profit = winning_trades["pnl_amount"].sum() if not winning_trades.empty else 0.0
    gross_loss = abs(losing_trades["pnl_amount"].sum()) if not losing_trades.empty else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def calculate_all_metrics(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    trade_log: Optional[pd.DataFrame] = None,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, Union[float, int]]:
    """Calculate comprehensive suite of institutional risk and performance metrics."""
    clean_rets = daily_returns.dropna()
    start_equity = equity_curve.iloc[0] if not equity_curve.empty else 1.0
    end_equity = equity_curve.iloc[-1] if not equity_curve.empty else 1.0
    total_return = (end_equity - start_equity) / start_equity if start_equity > 0 else 0.0

    cagr = calculate_cagr(equity_curve, trading_days=trading_days)
    ann_vol = float(clean_rets.std() * np.sqrt(trading_days)) if not clean_rets.empty else 0.0
    sharpe = calculate_sharpe_ratio(clean_rets, risk_free_rate=risk_free_rate, trading_days=trading_days)
    sortino = calculate_sortino_ratio(clean_rets, risk_free_rate=risk_free_rate, trading_days=trading_days)
    _, max_dd = calculate_drawdowns(equity_curve)
    max_daily_loss = float(clean_rets.min()) if not clean_rets.empty else 0.0
    calmar = calculate_calmar_ratio(cagr, max_dd)
    var_95 = calculate_historical_var(clean_rets, confidence_level=0.95)
    trade_stats = calculate_trade_metrics(trade_log)

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Max Daily Loss": max_daily_loss,
        "Calmar Ratio": calmar,
        "Historical VaR (95%)": var_95,
        "Total Trades": trade_stats["total_trades"],
        "Win Rate": trade_stats["win_rate"],
        "Avg Win": trade_stats["avg_win"],
        "Avg Loss": trade_stats["avg_loss"],
        "Profit Factor": trade_stats["profit_factor"],
    }


def build_comparison_table(
    strategy_metrics: Dict[str, Union[float, int]],
    benchmark_metrics: Dict[str, Union[float, int]],
) -> pd.DataFrame:
    """Format strategy vs benchmark metrics side-by-side into a presentation DataFrame."""
    keys = list(strategy_metrics.keys())
    data = []

    percent_keys = {
        "Total Return",
        "CAGR",
        "Annualized Volatility",
        "Max Drawdown",
        "Max Daily Loss",
        "Historical VaR (95%)",
        "Win Rate",
        "Avg Win",
        "Avg Loss",
    }

    for k in keys:
        strat_v = strategy_metrics.get(k, 0.0)
        bench_v = benchmark_metrics.get(k, 0.0)

        if k in percent_keys:
            strat_fmt = f"{strat_v * 100:.2f}%"
            bench_fmt = f"{bench_v * 100:.2f}%" if k in benchmark_metrics else "N/A"
        elif isinstance(strat_v, int) or k == "Total Trades":
            strat_fmt = f"{int(strat_v)}"
            bench_fmt = f"{int(bench_v)}" if k in benchmark_metrics and bench_v != 0 else "N/A"
        else:
            strat_fmt = f"{strat_v:.2f}"
            bench_fmt = f"{bench_v:.2f}" if k in benchmark_metrics else "N/A"

        data.append({"Metric": k, "Strategy": strat_fmt, "Benchmark": bench_fmt})

    return pd.DataFrame(data).set_index("Metric")
