# ---- Imports ----
import pytz
import streamlit as st
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

from utils import fetch_data, plot_intraday_line


# ---- Page Config ----
st.set_page_config(page_title="Stock Monitor", layout="wide")

# ---- Auto-refresh (every 5 minutes) ----
st_autorefresh(interval=300 * 1000, key="refresh")

# ---- Page Title ----
st.title("📈 Stock Monitoring Dashboard")

# ---- Input Box ----
codes = st.text_input(
    "Enter stock codes (comma separated)", 
    "S58.SI,S59.SI"
).upper().strip()

# ---- Session State Init ----
if "stocks" not in st.session_state:
    st.session_state.stocks = []

# ---- Button: Monitor ----
if st.button("Monitor"):
    st.session_state.stocks = [c.strip() for c in codes.split(",") if c.strip()]

# ---- Data Availability Check ----
df_check = False

# ---- Show Summaries for Each Stock ----
for code in st.session_state.stocks:
    df, msg = fetch_data(code)
    st.info(msg)

    if df.empty:
        st.error(f"⚠️ No data for {code}")
        continue

    # Filter today's data
    today_sgt = datetime.now(pytz.timezone("Asia/Singapore")).date()
    today_df = df[df["Datetime"].dt.date == today_sgt]

    if today_df.empty:
        st.warning(f"ℹ️ No data yet today for {code}")
        continue

    # ---- Compute Daily Stats ----
    last_close = float(today_df["Close"].iloc[-1])
    today_open = float(today_df["Open"].iloc[0])
    pct_change = ((last_close - today_open) / today_open) * 100

    st.subheader(f"{code}: {last_close:.3f} SGD  ({pct_change:+.2f}%)")
    st.write(
        f"**Today so far** — "
        f"Open: {today_open:.3f}, "
        f"High: {today_df['High'].max():.3f}, "
        f"Low: {today_df['Low'].min():.3f}"
    )

    # ---- Plot Intraday Line ----
    fig = plot_intraday_line(df, code)  # df is full dataset; function filters to today
    if fig:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(fig, use_container_width=True)
        with col2:
            st.write("")  # reserved for future use
        df_check = True
    else:
        st.warning(f"ℹ️ No intraday data yet for {code}")

# ---- Footer Message ----
if df_check:
    st.title("ℹ️ Please visit the **Stock Details Page** for more information.")
else:
    st.title("ℹ️ No data available yet. Please check back later.")
