import streamlit as st
from streamlit_autorefresh import st_autorefresh
from utils import fetch_data, plot_candlestick_vwap, plot_volume, plot_detector, plot_3mths_candlestick

st.set_page_config(page_title="Stock Details", layout="wide")

# Auto-refresh every 5 minutes
st_autorefresh(interval=300*1000, key="refresh_detail")

# If no stocks were set in session, show warning
if "stocks" not in st.session_state or not st.session_state.stocks:
    st.warning("⚠️ No stocks selected. Please go back to Home and enter tickers.")
else:
    st.title("📑 Stock Details")

    # Create one tab per stock
    tabs = st.tabs(st.session_state.stocks)

    for i, code in enumerate(st.session_state.stocks):
        with tabs[i]:
            st.subheader(f"📊 {code} — Detailed View")
            df, msg = fetch_data(code)
            st.info(msg)

            if not df.empty:
                fig_3m = plot_3mths_candlestick(df, code, months=3)
                if fig_3m:
                    st.title("3-Month Candlestick Overview")
                    st.subheader("📈 3-Month Overview (Daily Candlestick)")
                    st.write("Note: If there are less than 3 months of data, the graph will show all available data.")
                    st.pyplot(fig_3m)

                # Full candlestick + VWAP
                st.title("Today's Candlestick + VWAP")
                st.pyplot(plot_candlestick_vwap(df, code))

                # Volume chart
                st.pyplot(plot_volume(df, code))

                # Detector chart + signals
                st.title("Totday's Detector Signals")
                fig, signals = plot_detector(df, code)
                st.pyplot(fig)
                st.write("### Signals (Latest Day)")
                st.dataframe(signals.tail(20))
            else:
                st.warning(f"No data for {code}")
