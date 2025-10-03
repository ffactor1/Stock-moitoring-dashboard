# ---- Imports ----
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from utils import (
    fetch_data,
    plot_candlestick_vwap,
    plot_volume,
    plot_detector,
    plot_3mths_candlestick
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
