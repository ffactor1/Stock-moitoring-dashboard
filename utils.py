import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz

# ================= Data Fetch =================
def fetch_data(symbol, period="5d", interval="1m"):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert("Asia/Singapore")
    return df.dropna()

# ================= VWAP + Candlestick =================
def plot_candlestick_vwap(df, symbol):
    df = df.copy()
    df["date_sgt"] = df["Datetime"].dt.date
    today_sgt = datetime.now(pytz.timezone("Asia/Singapore")).date()
    last_date = df["date_sgt"].max() if today_sgt not in df["date_sgt"].unique() else today_sgt
    day = df[df["date_sgt"] == last_date].copy()

    # VWAP
    day["tp"] = (day["High"] + day["Low"] + day["Close"]) / 3
    day["pv"] = day["tp"] * day["Volume"]
    day["VWAP"] = day["pv"].cumsum() / day["Volume"].replace(0, np.nan).cumsum()

    last_close = float(day["Close"].iloc[-1])
    R1, R2 = last_close * 1.06, last_close * 1.12
    S1, S2 = last_close * 0.90, last_close * 0.85
    vwap_last = float(day["VWAP"].iloc[-1])

    # Candlestick
    day["dt_sgt_naive"] = day["Datetime"].dt.tz_convert("Asia/Singapore").dt.tz_localize(None)
    day["mdates"] = mdates.date2num(day["dt_sgt_naive"])
    bar_w = 60.0 / (24*60*60) * 0.9

    fig, ax = plt.subplots(figsize=(13,6))
    ax.set_title(f"{symbol} — {last_date} (1m Candlesticks)")

    # Wicks
    for _, r in day.iterrows():
        ax.vlines(r["mdates"], r["Low"], r["High"], linewidth=1)

    # Bodies
    for _, r in day.iterrows():
        lower, height = min(r["Open"], r["Close"]), abs(r["Close"]-r["Open"])
        rect = plt.Rectangle((r["mdates"]-bar_w/2, lower),
                             bar_w, height if height != 0 else 1e-10,
                             fill=(r["Close"] < r["Open"]), linewidth=1)
        ax.add_patch(rect)

    # VWAP + Levels
    ax.plot(day["mdates"], day["VWAP"], label="VWAP", linewidth=1.5)
    for y, lbl in [(S2,"S2"), (S1,"S1"), (vwap_last,"VWAP"), (R1,"R1"), (R2,"R2")]:
        ax.axhline(y, linestyle="--", linewidth=1)
        ax.text(day["mdates"].iloc[0], y, f" {lbl} {y:.3f}", va="bottom", fontsize=9)

    ax.legend()
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.set_xlabel("Time (SGT)"); ax.set_ylabel("Price (SGD)")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig

# ================= Volume =================
def plot_volume(df, symbol):
    df = df.copy()
    df["dt_sgt_naive"] = df["Datetime"].dt.tz_convert("Asia/Singapore").dt.tz_localize(None)
    bar_w = 60.0/(24*60*60)*0.9

    fig, ax = plt.subplots(figsize=(13,3.5))
    ax.set_title(f"{symbol} — Volume (1m)")
    ax.bar(df["dt_sgt_naive"], df["Volume"], width=bar_w)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time (SGT)"); ax.set_ylabel("Volume")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig

# ================= Breakout/Blowoff Detector =================
def plot_detector(df, symbol):
    df = df.copy().sort_values("Datetime").reset_index(drop=True)
    df["date_sgt"] = df["Datetime"].dt.date

    # VWAP
    df["tp"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["pv"] = df["tp"] * df["Volume"]
    df["cum_pv"]  = df.groupby("date_sgt")["pv"].cumsum()
    df["cum_vol"] = df.groupby("date_sgt")["Volume"].cumsum().replace(0, np.nan)
    df["VWAP"]    = df["cum_pv"] / df["cum_vol"]

    # Volume spike baseline
    VOL_WINDOW, VOL_MULT = 30, 2.0
    df["vol_base"]  = df.groupby("date_sgt")["Volume"].transform(lambda x: x.rolling(VOL_WINDOW, min_periods=1).median())
    df["vol_spike"] = df["Volume"] >= (VOL_MULT * df["vol_base"])

    # Breakout
    HH_LOOKBACK = 10
    df["prior_hh"] = df.groupby("date_sgt")["High"].shift(1).rolling(HH_LOOKBACK, min_periods=1).max()
    df["above_vwap"] = df["Close"] >= df["VWAP"]
    df["breakout"]   = df["vol_spike"] & df["above_vwap"] & (df["Close"] > df["prior_hh"])

    # Blowoff
    WICK_RATIO, REVERSAL_BARS = 0.6, 3
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    upper_wick = df["High"] - df[["Open","Close"]].max(axis=1)
    df["long_upper_wick"] = (upper_wick / rng) >= WICK_RATIO
    wick_blowoff = df["vol_spike"] & (~df["above_vwap"]) & df["long_upper_wick"]

    reversal_blowoff = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if df["breakout"].iat[i]:
            j_end = min(i+REVERSAL_BARS, len(df)-1)
            mask = slice(i+1, j_end+1)
            reversal_blowoff.iloc[mask] |= (~df["above_vwap"].iloc[mask]) & df["vol_spike"].iloc[mask]

    df["blowoff"] = wick_blowoff | reversal_blowoff

    # Last day
    last_date = df["date_sgt"].max()
    day = df[df["date_sgt"] == last_date].copy()
    day["dt_sgt_naive"] = day["Datetime"].dt.tz_convert("Asia/Singapore").dt.tz_localize(None)
    day["mdates"] = mdates.date2num(day["dt_sgt_naive"])
    bar_w = 60.0/(24*60*60)*0.9

    # Plot
    fig, ax = plt.subplots(figsize=(13,6))
    ax.set_title(f"{symbol} — {last_date} (Detector View)")

    for _, r in day.iterrows():
        ax.vlines(r["mdates"], r["Low"], r["High"], linewidth=1)
        lower, height = min(r["Open"], r["Close"]), abs(r["Close"]-r["Open"])
        ax.add_patch(plt.Rectangle((r["mdates"]-bar_w/2, lower),
                                   bar_w, height if height != 0 else 1e-10,
                                   fill=(r["Close"] < r["Open"]), linewidth=1))

    ax.plot(day["mdates"], day["VWAP"], linewidth=1.5, label="VWAP")

    bo = day[day["breakout"]]
    ax.plot(bo["mdates"], bo["Close"], marker="^", linestyle="None", markersize=8, label="Breakout")
    bf = day[day["blowoff"]]
    ax.plot(bf["mdates"], bf["Close"], marker="v", linestyle="None", markersize=8, label="Blow-off")

    ax.legend()
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.set_xlabel("Time (SGT)"); ax.set_ylabel("Price (SGD)")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig, df[df["date_sgt"] == last_date][["Datetime","Open","High","Low","Close","Volume","VWAP","breakout","blowoff"]]
