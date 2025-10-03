import streamlit as st
from utils import fetch_data, plot_candlestick_vwap, plot_volume, plot_detector

st.set_page_config(page_title="Stock Detail", layout="wide")

query_params = st.query_params
symbol = query_params.get("symbol", "")

if symbol:
    st.title(f"📊 {symbol} Details")

    df = fetch_data(symbol)
    if df.empty:
        st.error("No data found")
    else:
        st.subheader("📉 Candlestick + VWAP")
        st.pyplot(plot_candlestick_vwap(df, symbol))

        st.subheader("📊 Volume")
        st.pyplot(plot_volume(df, symbol))

        st.subheader("🚨 Detector (Breakout / Blowoff)")
        fig, signals = plot_detector(df, symbol)
        st.pyplot(fig)

        st.write("### Signals (Latest Day)")
        st.dataframe(signals.tail(20))
else:
    st.warning("No stock selected. Go back to Home.")
