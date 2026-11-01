"""Quant Trading Research System - Streamlit Dashboard.

Interactive web application displaying quantitative trading signals, realistic backtests,
institutional risk analysis, trade logs, and cross-asset correlations with Plotly.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add workspace root to sys.path to ensure src imports smoothly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backtester import BacktestResult, run_portfolio_backtest, run_single_backtest
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
from src.data_loader import download_ticker_data
from src.processing import (
    add_technical_indicators,
    compute_correlation_matrix,
)
from src.risk import (
    build_comparison_table,
    calculate_all_metrics,
    calculate_drawdowns,
)
from src.strategies import (
    MovingAverageCrossoverStrategy,
    get_current_signal,
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Quant Trading Research System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Institutional Visual Polish
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .signal-badge-buy {
        display: inline-block;
        padding: 8px 24px;
        background-color: #00c853;
        color: white;
        font-weight: 800;
        font-size: 26px;
        border-radius: 6px;
        letter-spacing: 1.5px;
    }
    .signal-badge-sell {
        display: inline-block;
        padding: 8px 24px;
        background-color: #d50000;
        color: white;
        font-weight: 800;
        font-size: 26px;
        border-radius: 6px;
        letter-spacing: 1.5px;
    }
    .signal-badge-hold {
        display: inline-block;
        padding: 8px 24px;
        background-color: #ff9100;
        color: white;
        font-weight: 800;
        font-size: 26px;
        border-radius: 6px;
        letter-spacing: 1.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_cached_ticker(ticker: str, force_reload: bool = False) -> pd.DataFrame:
    """Cached wrapper around download_ticker_data."""
    return download_ticker_data(ticker=ticker, period=DEFAULT_PERIOD, force_reload=force_reload)


def main() -> None:
    """Streamlit Application Entry Point."""
    st.title("📈 Quant Trading Research System")
    st.caption("Institutional Indian Market Backtesting & Risk Analytics Platform")

    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.header("🕹️ Strategy & Portfolio Controls")

    # Mode: Single Asset vs Multi-Asset Portfolio
    mode = st.sidebar.radio("Analysis Mode", ["Single Asset Backtest", "Multi-Stock Portfolio"])

    # Ticker Selection
    if mode == "Single Asset Backtest":
        available_tickers = DEFAULT_TICKERS.copy()
        custom_input = st.sidebar.text_input("Add Custom NSE Symbol (e.g., SBIN.NS)", "")
        if custom_input.strip() and custom_input.strip() not in available_tickers:
            available_tickers.append(custom_input.strip())

        selected_ticker = st.sidebar.selectbox("Select Asset", available_tickers, index=1)
        selected_tickers = [selected_ticker]
    else:
        selected_tickers = st.sidebar.multiselect(
            "Select Portfolio Assets",
            options=DEFAULT_TICKERS,
            default=[t for t in DEFAULT_TICKERS if t != BENCHMARK_TICKER],
        )
        if not selected_tickers:
            st.warning("Please select at least one stock for portfolio backtesting.")
            return

    # Benchmark Selection
    benchmark_choice = st.sidebar.selectbox(
        "Benchmark",
        options=[BENCHMARK_TICKER, "None"],
        index=0,
    )

    # Strategy Parameters
    st.sidebar.subheader("⚙️ Moving Average Parameters")
    fast_ma = st.sidebar.slider("Fast MA Window (Days)", min_value=5, max_value=100, value=DEFAULT_FAST_MA)
    slow_ma = st.sidebar.slider("Slow MA Window (Days)", min_value=20, max_value=250, value=DEFAULT_SLOW_MA)

    if fast_ma >= slow_ma:
        st.sidebar.error("Fast MA window must be strictly less than Slow MA window.")
        return

    long_only = st.sidebar.checkbox("Long-Only Mode (Cash on Sell)", value=DEFAULT_LONG_ONLY)

    # Friction & Capital
    st.sidebar.subheader("💰 Capital & Realistic Costs")
    starting_capital = st.sidebar.number_input(
        "Starting Capital (₹)",
        min_value=1000.0,
        value=DEFAULT_STARTING_CAPITAL,
        step=10000.0,
    )
    brokerage_pct = st.sidebar.number_input(
        "Brokerage % (per trade)",
        min_value=0.0,
        max_value=5.0,
        value=DEFAULT_BROKERAGE_PCT * 100.0,
        step=0.01,
    ) / 100.0
    slippage_pct = st.sidebar.number_input(
        "Slippage %",
        min_value=0.0,
        max_value=5.0,
        value=DEFAULT_SLIPPAGE_PCT * 100.0,
        step=0.01,
    ) / 100.0
    stt_pct = st.sidebar.number_input(
        "STT / Other Charges %",
        min_value=0.0,
        max_value=5.0,
        value=DEFAULT_STT_PCT * 100.0,
        step=0.01,
    ) / 100.0

    force_reload = st.sidebar.button("🔄 Refresh Market Data")

    # ------------------ DATA LOADING ------------------
    data_dict: Dict[str, pd.DataFrame] = {}
    with st.spinner("Loading market data..."):
        for t in set(selected_tickers + ([benchmark_choice] if benchmark_choice != "None" else [])):
            try:
                df = fetch_cached_ticker(t, force_reload=force_reload)
                data_dict[t] = df
            except Exception as e:
                st.error(f"Error loading {t}: {e}")

    if not data_dict:
        st.error("Could not load market data. Please verify your internet connection or ticker symbols.")
        return

    benchmark_prices = (
        data_dict[benchmark_choice]["Adj Close"]
        if benchmark_choice in data_dict and benchmark_choice != "None"
        else None
    )

    # ------------------ DATE FILTERING ------------------
    # Common date range determination
    all_dates = pd.concat([df.index.to_series() for df in data_dict.values()])
    min_market_date = all_dates.min().date()
    max_market_date = all_dates.max().date()

    st.sidebar.subheader("📅 Date Range Filter")
    date_range = st.sidebar.date_input(
        "Select Backtest Period",
        value=[min_market_date, max_market_date],
        min_value=min_market_date,
        max_value=max_market_date,
    )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_d, end_d = pd.to_datetime(min_market_date), pd.to_datetime(max_market_date)

    # Filter data to date range
    filtered_data = {
        t: df.loc[(df.index >= start_d) & (df.index <= end_d)].copy()
        for t, df in data_dict.items()
    }
    if benchmark_prices is not None:
        benchmark_prices = benchmark_prices.loc[(benchmark_prices.index >= start_d) & (benchmark_prices.index <= end_d)]

    # ------------------ EXECUTE STRATEGY & BACKTEST ------------------
    strat = MovingAverageCrossoverStrategy(
        fast_window=fast_ma,
        slow_window=slow_ma,
        long_only=long_only,
    )

    if mode == "Single Asset Backtest":
        target_ticker = selected_tickers[0]
        df_target = filtered_data[target_ticker]

        if df_target.empty:
            st.warning("No data found in the selected date range.")
            return

        signals = strat.generate_signals(df_target, price_col="Adj Close")
        current_signal = get_current_signal(signals)
        latest_price = df_target["Adj Close"].iloc[-1]
        latest_date = df_target.index[-1].strftime("%Y-%m-%d")

        # Top Signal Status Banner
        badge_class = (
            "signal-badge-buy"
            if current_signal == "BUY"
            else ("signal-badge-sell" if current_signal == "SELL" else "signal-badge-hold")
        )
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; background: #1f2937; padding: 18px 24px; border-radius: 10px; margin-bottom: 25px;">
                <div>
                    <h3 style="margin: 0; color: #f3f4f6;">{target_ticker} Current Status</h3>
                    <p style="margin: 0; color: #9ca3af;">Latest Close: <b>₹{latest_price:,.2f}</b> | Date: {latest_date}</p>
                </div>
                <div>
                    <span class="{badge_class}">{current_signal}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Run Backtest
        backtest_result = run_single_backtest(
            prices=df_target["Adj Close"],
            signals=signals,
            starting_capital=starting_capital,
            brokerage_pct=brokerage_pct,
            slippage_pct=slippage_pct,
            stt_pct=stt_pct,
            benchmark_prices=benchmark_prices,
            ticker=target_ticker,
        )

    else:
        # Multi-Stock Portfolio Mode
        price_dict = {}
        signal_dict = {}
        current_signals_map = {}

        for t in selected_tickers:
            df_t = filtered_data[t]
            if not df_t.empty:
                price_dict[t] = df_t["Adj Close"]
                s = strat.generate_signals(df_t, price_col="Adj Close")
                signal_dict[t] = s
                current_signals_map[t] = get_current_signal(s)

        # Current Signal Grid for Portfolio
        st.subheader("Current Market Positions")
        cols = st.columns(len(current_signals_map))
        for idx, (t, sig) in enumerate(current_signals_map.items()):
            badge_class = (
                "signal-badge-buy"
                if sig == "BUY"
                else ("signal-badge-sell" if sig == "SELL" else "signal-badge-hold")
            )
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="background: #1f2937; padding: 12px; border-radius: 8px; text-align: center;">
                        <h4 style="margin: 0; color: #f3f4f6;">{t}</h4>
                        <div style="margin-top: 8px;"><span class="{badge_class}" style="font-size: 16px; padding: 4px 12px;">{sig}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        backtest_result = run_portfolio_backtest(
            price_dict=price_dict,
            signal_dict=signal_dict,
            starting_capital=starting_capital,
            brokerage_pct=brokerage_pct,
            slippage_pct=slippage_pct,
            stt_pct=stt_pct,
            benchmark_prices=benchmark_prices,
        )

    # ------------------ RISK & PERFORMANCE CALCULATION ------------------
    strat_daily_rets = backtest_result.daily_results["strategy_net_return"]
    strat_equity = backtest_result.daily_results["equity_curve"]

    bench_daily_rets = backtest_result.daily_results.get("benchmark_return", pd.Series(dtype=float))
    bench_equity = backtest_result.daily_results.get("benchmark_equity", pd.Series(dtype=float))

    strat_metrics = calculate_all_metrics(
        daily_returns=strat_daily_rets,
        equity_curve=strat_equity,
        trade_log=backtest_result.trade_log,
        risk_free_rate=RISK_FREE_RATE,
    )

    bench_metrics = (
        calculate_all_metrics(
            daily_returns=bench_daily_rets,
            equity_curve=bench_equity,
            trade_log=None,
            risk_free_rate=RISK_FREE_RATE,
        )
        if not bench_equity.empty
        else {}
    )

    # High-level Summary Metric KPIs
    kpi_cols = st.columns(4)
    kpi_cols[0].metric(
        "Total Return",
        f"{strat_metrics['Total Return'] * 100:.2f}%",
        f"{(strat_metrics['Total Return'] - bench_metrics.get('Total Return', 0.0)) * 100:+.2f}% vs Bench"
        if bench_metrics
        else None,
    )
    kpi_cols[1].metric(
        "CAGR",
        f"{strat_metrics['CAGR'] * 100:.2f}%",
        f"{(strat_metrics['CAGR'] - bench_metrics.get('CAGR', 0.0)) * 100:+.2f}%"
        if bench_metrics
        else None,
    )
    kpi_cols[2].metric("Sharpe Ratio", f"{strat_metrics['Sharpe Ratio']:.2f}")
    kpi_cols[3].metric("Max Drawdown", f"{strat_metrics['Max Drawdown'] * 100:.2f}%")

    # ------------------ MAIN TABS ------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Price & MA Signals",
            "📈 Equity Curve",
            "📉 Drawdown",
            "📋 Risk Metrics",
            "📜 Trade Log",
            "🔥 Correlation Heatmap",
        ]
    )

    # TAB 1: PRICE & MA SIGNALS
    with tab1:
        st.subheader("Price Action with Moving Average Crossovers")
        if mode == "Single Asset Backtest":
            target_df = filtered_data[selected_tickers[0]]
            enriched_df = add_technical_indicators(
                target_df, fast_window=fast_ma, slow_window=slow_ma
            )

            fig_price = go.Figure()
            # Price line
            fig_price.add_trace(
                go.Scatter(
                    x=enriched_df.index,
                    y=enriched_df["Adj Close"],
                    mode="lines",
                    name="Adj Close",
                    line=dict(color="#3b82f6", width=1.5),
                )
            )
            # Fast MA
            fig_price.add_trace(
                go.Scatter(
                    x=enriched_df.index,
                    y=enriched_df[f"MA_{fast_ma}"],
                    mode="lines",
                    name=f"{fast_ma}-Day MA",
                    line=dict(color="#10b981", width=1.5),
                )
            )
            # Slow MA
            fig_price.add_trace(
                go.Scatter(
                    x=enriched_df.index,
                    y=enriched_df[f"MA_{slow_ma}"],
                    mode="lines",
                    name=f"{slow_ma}-Day MA",
                    line=dict(color="#f59e0b", width=1.5),
                )
            )

            # Signal Markers from daily results
            res_df = backtest_result.daily_results
            buy_points = res_df[(res_df["signal"] == 1.0) & (res_df["signal"].shift(1) != 1.0)]
            sell_points = res_df[(res_df["signal"] <= 0.0) & (res_df["signal"].shift(1) > 0.0)]

            if not buy_points.empty:
                fig_price.add_trace(
                    go.Scatter(
                        x=buy_points.index,
                        y=buy_points["price"],
                        mode="markers",
                        name="Buy Signal",
                        marker=dict(symbol="triangle-up", size=11, color="#22c55e"),
                    )
                )

            if not sell_points.empty:
                fig_price.add_trace(
                    go.Scatter(
                        x=sell_points.index,
                        y=sell_points["price"],
                        mode="markers",
                        name="Sell Signal",
                        marker=dict(symbol="triangle-down", size=11, color="#ef4444"),
                    )
                )

            fig_price.update_layout(
                template="plotly_dark",
                height=520,
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("In Multi-Stock Portfolio mode, select individual stocks via 'Single Asset Backtest' to view granular crossover charts.")

    # TAB 2: EQUITY CURVE
    with tab2:
        st.subheader("Strategy Equity Growth vs Nifty 50 Benchmark")
        fig_eq = go.Figure()

        fig_eq.add_trace(
            go.Scatter(
                x=strat_equity.index,
                y=strat_equity,
                mode="lines",
                name="Strategy (Net of Costs)",
                line=dict(color="#10b981", width=2.2),
            )
        )

        if not bench_equity.empty:
            fig_eq.add_trace(
                go.Scatter(
                    x=bench_equity.index,
                    y=bench_equity,
                    mode="lines",
                    name=f"Benchmark ({benchmark_choice}) Buy & Hold",
                    line=dict(color="#94a3b8", width=1.5, dash="dot"),
                )
            )

        fig_eq.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # TAB 3: DRAWDOWN ANALYSIS
    with tab3:
        st.subheader("Underwater Drawdown Profile")
        strat_dd, _ = calculate_drawdowns(strat_equity)

        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=strat_dd.index,
                y=strat_dd * 100.0,
                fill="tozeroy",
                mode="lines",
                name="Strategy Drawdown",
                line=dict(color="#ef4444", width=1.5),
                fillcolor="rgba(239, 68, 68, 0.2)",
            )
        )

        if not bench_equity.empty:
            bench_dd, _ = calculate_drawdowns(bench_equity)
            fig_dd.add_trace(
                go.Scatter(
                    x=bench_dd.index,
                    y=bench_dd * 100.0,
                    mode="lines",
                    name="Benchmark Drawdown",
                    line=dict(color="#94a3b8", width=1.2, dash="dash"),
                )
            )

        fig_dd.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # TAB 4: RISK METRICS TABLE
    with tab4:
        st.subheader("Comprehensive Risk & Performance Table")
        comparison_df = build_comparison_table(strat_metrics, bench_metrics)
        st.dataframe(comparison_df, use_container_width=True)

    # TAB 5: TRADE LOG
    with tab5:
        st.subheader("Executed Trade Log")
        log_df = backtest_result.trade_log
        if log_df.empty:
            st.info("No trades were executed during this backtest window.")
        else:
            display_log = log_df.copy()
            if "entry_date" in display_log.columns:
                display_log["entry_date"] = pd.to_datetime(display_log["entry_date"]).dt.strftime("%Y-%m-%d")
            if "exit_date" in display_log.columns:
                display_log["exit_date"] = pd.to_datetime(display_log["exit_date"]).dt.strftime("%Y-%m-%d")

            display_log["pnl_pct"] = (display_log["pnl_pct"] * 100).map("{:+.2f}%".format)
            display_log["pnl_amount"] = display_log["pnl_amount"].map("₹{:,.2f}".format)

            st.dataframe(display_log, use_container_width=True)

            csv_data = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Trade Log as CSV",
                data=csv_data,
                file_name=f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    # TAB 6: CORRELATION HEATMAP
    with tab6:
        st.subheader("Cross-Asset Return Correlation Heatmap")
        corr_matrix = compute_correlation_matrix(filtered_data, price_col="Adj Close")
        if not corr_matrix.empty:
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
            )
            fig_corr.update_layout(template="plotly_dark", height=480)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Insufficient multi-stock data available to construct correlation matrix.")


if __name__ == "__main__":
    main()
