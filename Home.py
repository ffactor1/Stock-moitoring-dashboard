import streamlit as st
from streamlit_autorefresh import st_autorefresh
from utils import fetch_data

st.set_page_config(page_title="Stock Monitor", layout="wide")

# Auto-refresh every 60s
st_autorefresh(interval=60*1000, key="refresh")

st.title("📈 Stock Monitoring Dashboard")

# Input
codes = st.text_input("Enter stock codes (comma separated)", "S58.SI,SATS.SI")

if "stocks" not in st.session_state:
    st.session_state.stocks = []

if st.button("Monitor"):
    st.session_state.stocks = [c.strip() for c in codes.split(",")]

# Display summary
for code in st.session_state.stocks:
    df = fetch_data(code)
    if df.empty:
        st.error(f"No data for {code}")
    else:
        last_close = df["Close"].iloc[-1]
        st.subheader(f"{code}: {last_close:.3f} SGD")
        st.markdown(f"[➡️ View details for {code}](/Stock_Detail?symbol={code})")
