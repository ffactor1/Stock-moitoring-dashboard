# ---- Imports ----
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from utils import (
    fetch_data,
    plot_candlestick_vwap,
    plot_volume,
    plot_detector,
    plot_3mths_candlestick,
    compute_intraday_indicators,
    plot_price_with_mas,
    plot_rsi_macd
)

# ---- Page Config ----
st.set_page_config(page_title="Stock Details", layout="wide")

# ---- Auto-refresh (every 5 minutes) ----
st_autorefresh(interval=300 * 1000, key="refresh_detail")

# ---- Check if stocks are set in session ----
if "stocks" not in st.session_state or not st.session_state.stocks:
    st.warning("⚠️ No stocks selected. Please go back to Home and enter tickers.")

else:
    st.title("📑 Stock Details")

    # ---- Create one tab per stock ----
    tabs = st.tabs(st.session_state.stocks)

    for i, code in enumerate(st.session_state.stocks):
        with tabs[i]:
            st.subheader(f"📊 {code} — Detailed View")

            # ---- Fetch Data ----
            df, msg = fetch_data(code)
            st.info(msg)

            if not df.empty:
                indicators = compute_intraday_indicators(df)

                if not indicators.empty:
                    st.header("🧮 Intraday Technical Indicators")

                    latest = indicators.iloc[-1]
                    first_close = indicators["Close"].iloc[0]
                    if pd.notna(first_close) and first_close != 0:
                        pct_move = ((latest["Close"] / first_close) - 1) * 100
                    else:
                        pct_move = 0.0

                    col_price, col_rsi, col_macd = st.columns(3)
                    col_price.metric(
                        "Last Price (SGD)",
                        f"{latest['Close']:.3f}",
                        f"{pct_move:+.2f}%"
                    )
                    col_rsi.metric("RSI (14)", f"{latest['RSI_14']:.1f}")
                    col_macd.metric("MACD Hist", f"{latest['MACD_hist']:.3f}")

                    fig_ma = plot_price_with_mas(indicators, code)
                    if fig_ma:
                        st.pyplot(fig_ma)

                    fig_rsi_macd = plot_rsi_macd(indicators, code)
                    if fig_rsi_macd:
                        st.pyplot(fig_rsi_macd)

                    st.caption(
                        "Indicators are computed on 1-minute bars using standard periods: "
                        "SMA20, EMA20, EMA50, RSI14, MACD (12-26-9)."
                    )

                    st.dataframe(
                        indicators[
                            [
                                "Datetime",
                                "Close",
                                "SMA_20",
                                "EMA_20",
                                "EMA_50",
                                "RSI_14",
                                "MACD",
                                "MACD_signal",
                                "MACD_hist",
                            ]
                        ].tail(20)
                    )
                else:
                    st.info(
                        "No intraday indicator data is available for the current session yet. "
                        "The dashboard will populate once fresh 1-minute bars arrive."
                    )

                # ---- 3-Month Candlestick Overview ----
                fig_3m = plot_3mths_candlestick(df, code, months=3)
                if fig_3m:
                    st.header("📈 3-Month Candlestick Overview")
                    st.caption("Note: If fewer than 3 months of data are available, all data will be shown.")
                    st.pyplot(fig_3m)

                # ---- Today's Candlestick + VWAP ----
                st.header("📉 Today's Candlestick + VWAP")
                st.pyplot(plot_candlestick_vwap(df, code))

                # ---- Volume Chart ----
                st.header("📊 Today's Volume")
                st.pyplot(plot_volume(df, code))

                # ---- Detector Chart + Signals ----
                st.header("🔍 Today's Detector Signals")
                fig, signals = plot_detector(df, code)
                st.pyplot(fig)
                st.subheader("Latest Signals (Last 20 Rows)")
                st.dataframe(signals.tail(20))

            else:
                st.warning(f"⚠️ No data available for {code}")
