📈 Stock Monitoring Dashboard

A real-time stock monitoring dashboard built with Streamlit and Yahoo Finance (yfinance).
This app allows you to:

✅ Monitor multiple tickers at once
✅ Automatically refresh data every 5 minutes
✅ Store and update data locally (/stock_data folder)
✅ View intraday price + volume charts (per minute)
✅ Analyse intraday technical indicators (SMA, EMA, RSI, MACD)
✅ Access detailed tabs with:

Daily candlestick + VWAP

Volume trends

Breakout / blow-off detector

Signal table
✅ Long-term 3-month line chart overview with wider support/resistance levels

🚀 Features

Multi-stock input → Enter multiple tickers (e.g. S58.SI, SATS.SI) to track them in one dashboard

Auto-refresh → Data refreshes every 5 minutes (Yahoo Finance is delayed ~10 mins)

Local storage → New data appends to CSV files in stock_data/, preventing duplicate fetches

Home Page →

Clean intraday line chart (price + volume) for today

Key stats: Open, High, Low, Last Close, % change

Details Page (Tabs) →

Daily candlestick + VWAP

Intraday technical indicator dashboard (metrics, MA overlay, RSI & MACD panels)

Intraday volume bars

Detector chart with signals

3-month line chart with extended support/resistance bands

🛠️ Tech Stack

Python

Streamlit (dashboard)

yfinance (data fetching)

matplotlib (charting)

pandas (data handling)
