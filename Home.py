import streamlit as st
from streamlit_autorefresh import st_autorefresh
from utils import fetch_data
from datetime import datetime, timedelta
import pytz                   
from utils import fetch_data, plot_intraday_line 

df_check = '0'

st.set_page_config(page_title="Stock Monitor", layout="wide")

# 🔹 Auto-refresh every 300s
st_autorefresh(interval=300*1000, key="refresh")

st.title("📈 Stock Monitoring Dashboard")

# 🔹 Input box
codes = st.text_input("Enter stock codes (comma separated)", "S58.SI,S59.SI").upper().strip()

if "stocks" not in st.session_state:
    st.session_state.stocks = []

if st.button("Monitor"):
    st.session_state.stocks = [c.strip() for c in codes.split(",") if c.strip()]

# 🔹 Show summaries
for code in st.session_state.stocks:
    df, msg = fetch_data(code)
    st.info(msg)

    if df.empty:
        df_check = '0'
        st.error(f"No data for {code}")
    else:
        today_sgt = datetime.now(pytz.timezone("Asia/Singapore")).date()
        today_df = df[df["Datetime"].dt.date == today_sgt]

        if not today_df.empty:
            last_close = float(today_df["Close"].iloc[-1])
            today_open = float(today_df["Open"].iloc[0])
            pct_change = ((last_close - today_open) / today_open) * 100

            st.subheader(f"{code}: {last_close:.3f} SGD  ({pct_change:+.2f}%)")
            st.write(f"**Today so far** — Open: {today_open:.3f}, "
                     f"High: {today_df['High'].max():.3f}, "
                     f"Low: {today_df['Low'].min():.3f}")

            # ✅ Display candlestick + VWAP for today's data
            fig = plot_intraday_line(df, code)  # df is the full dataset; the function filters to today
            if fig:
                col1, col2 = st.columns([2,1])
                with col1:
                    st.pyplot(fig, use_container_width=True)
                with col2:
                    st.write("")
                df_check = '1'
            else:
                st.warning(f"No data yet today for {code}")


        else:
            st.warning(f"No data yet today for {code}")

if df_check == '1':
    st.title("ℹ️ Please visit the **Stock Details Page** for more information.")
else:
    st.title("ℹ️ No data available yet. Please check back later.")

