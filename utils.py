# ---- Imports ----
import os
import pytz
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ---- Data Directory & Timezone ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "stock_data")
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure folder exists

TARGET_TZ = pytz.timezone("Asia/Singapore")


# ---- Helpers ----
def _force_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLCV columns are numeric if present."""
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns and isinstance(df[col], (pd.Series, list, np.ndarray)):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _ensure_datetime_tz(df: pd.DataFrame, column: str = "Datetime", tz=TARGET_TZ) -> pd.DataFrame:
    """Coerce a datetime column to timezone-aware values in the configured timezone."""
    if column not in df.columns:
        return df

    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
    elif not pd.api.types.is_datetime64tz_dtype(df[column]):
        df[column] = df[column].dt.tz_localize("UTC")

    df[column] = df[column].dt.tz_convert(tz)
    df = df.dropna(subset=[column]).sort_values(column)
    return df.reset_index(drop=True)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace({0: np.nan})
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")


# ---- Fetch Data ----
def fetch_data(symbol: str, period="5d", interval="1m"):
    """
    Fetch stock data for a given symbol.
    - Saves CSV in stock_data/<SYMBOL>.csv
    - If CSV exists, append only new rows
    - Returns (DataFrame, message)
    """
    file_path = os.path.join(DATA_DIR, f"{symbol.replace('.', '_')}.csv")

    # If CSV exists, load and update
    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path, parse_dates=["Datetime"])
        df_old = _ensure_datetime_tz(_force_numeric(df_old))

        if df_old.empty:
            last_dt = None
        else:
            last_dt = df_old["Datetime"].max()

        df_new = pd.DataFrame()
        if last_dt is not None and pd.notna(last_dt):
            # Fetch new data after last_dt (pad by one bar to avoid gaps)
            start_dt = last_dt.astimezone(pytz.UTC) - timedelta(minutes=1)
            df_new = yf.download(
                symbol,
                start=start_dt.replace(tzinfo=None),
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
        else:
            df_new = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

        if not df_new.empty:
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = [c[0] for c in df_new.columns]
            df_new = df_new.reset_index()
            df_new = _force_numeric(df_new)
            df_new = _ensure_datetime_tz(df_new)

            before = len(df_old)
            combined = pd.concat([df_old, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

            # ✅ Ensure datetime consistency
            combined = _ensure_datetime_tz(combined)

            combined.to_csv(file_path, index=False)

            added = len(combined) - before
            msg = f"✅ Updated {symbol} with {added} new rows. Total rows: {len(combined)}"
            return combined, msg

        else:
            msg = f"ℹ️ No new data for {symbol}. Using {len(df_old)} existing rows."
            return df_old, msg

    # If no CSV exists, fetch fresh data
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame(), f"⚠️ No data returned for {symbol}"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df = _force_numeric(df)

    # ✅ Ensure datetime consistency
    df = _ensure_datetime_tz(df)

    df.to_csv(file_path, index=False)
    msg = f"📂 Created {symbol} dataset with {len(df)} rows."
    return df, msg


# ---- Plot: Intraday Line + Volume ----
def plot_intraday_line(df, symbol, tz_name: str = TARGET_TZ.zone, figsize=(8, 3)):
    """Today's Close vs Time (line) with Volume as secondary bar axis."""
    if df is None or df.empty:
        return None

    # Ensure tz-aware datetime
    tz = pytz.timezone(tz_name)
    df = _ensure_datetime_tz(df.copy(), tz=tz)

    # Filter to today only
    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    today_df = df[(df["Datetime"] >= start) & (df["Datetime"] < end)].copy()
    if today_df.empty:
        return None

    today_df["t_naive"] = today_df["Datetime"].dt.tz_localize(None)

    # Plot
    fig, ax1 = plt.subplots(figsize=figsize)

    # Line for Close
    ax1.plot(today_df["t_naive"], today_df["Close"], color="blue", linewidth=1.6, label="Close")
    ax1.set_ylabel("Price (SGD)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Bar for Volume (secondary axis)
    ax2 = ax1.twinx()
    ax2.bar(today_df["t_naive"], today_df["Volume"], width=0.0005, color="red", alpha=0.4, label="Volume")
    ax2.set_ylabel("Volume", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # X-axis formatting
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    ax1.set_title(f"{symbol} — Today’s Price & Volume")
    ax1.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


# ---- Technical Indicators ----
def compute_intraday_indicators(df: pd.DataFrame, tz_name: str = TARGET_TZ.zone) -> pd.DataFrame:
    """Return today's intraday data enriched with common technical indicators."""
    if df is None or df.empty:
        return pd.DataFrame()

    tz = pytz.timezone(tz_name)
    data = _ensure_datetime_tz(df.copy(), tz=tz)
    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    today_df = data[(data["Datetime"] >= start) & (data["Datetime"] < end)].copy()
    if today_df.empty:
        return pd.DataFrame()

    today_df = today_df.sort_values("Datetime").reset_index(drop=True)

    # Moving averages
    today_df["SMA_20"] = today_df["Close"].rolling(window=20, min_periods=1).mean()
    today_df["EMA_20"] = today_df["Close"].ewm(span=20, adjust=False, min_periods=1).mean()
    today_df["EMA_50"] = today_df["Close"].ewm(span=50, adjust=False, min_periods=1).mean()

    # RSI and MACD
    today_df["RSI_14"] = _compute_rsi(today_df["Close"], period=14)

    ema_fast = today_df["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
    ema_slow = today_df["Close"].ewm(span=26, adjust=False, min_periods=1).mean()
    today_df["MACD"] = ema_fast - ema_slow
    today_df["MACD_signal"] = today_df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()
    today_df["MACD_hist"] = today_df["MACD"] - today_df["MACD_signal"]

    return today_df


def plot_price_with_mas(df: pd.DataFrame, symbol: str, figsize=(10, 4)):
    """Plot close price with selected moving averages."""
    if df is None or df.empty:
        return None

    data = df.copy()
    data["t_naive"] = data["Datetime"].dt.tz_localize(None)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data["t_naive"], data["Close"], label="Close", color="tab:blue", linewidth=1.4)
    ax.plot(data["t_naive"], data["SMA_20"], label="SMA 20", color="tab:orange", linewidth=1.2)
    ax.plot(data["t_naive"], data["EMA_20"], label="EMA 20", color="tab:green", linewidth=1.2, linestyle="--")
    ax.plot(data["t_naive"], data["EMA_50"], label="EMA 50", color="tab:red", linewidth=1.2, linestyle=":")

    ax.set_title(f"{symbol} — Intraday Price with Moving Averages")
    ax.set_ylabel("Price (SGD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_rsi_macd(df: pd.DataFrame, symbol: str, figsize=(10, 5)):
    """Plot RSI and MACD panels for intraday data."""
    if df is None or df.empty:
        return None

    data = df.copy()
    data["t_naive"] = data["Datetime"].dt.tz_localize(None)

    fig, (ax_rsi, ax_macd) = plt.subplots(2, 1, sharex=True, figsize=figsize)

    ax_rsi.plot(data["t_naive"], data["RSI_14"], color="tab:purple", linewidth=1.2)
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.9)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.9)
    ax_rsi.set_ylabel("RSI (14)")
    ax_rsi.set_title(f"{symbol} — RSI & MACD")
    ax_rsi.grid(True, linestyle="--", alpha=0.4)

    ax_macd.plot(data["t_naive"], data["MACD"], label="MACD", color="tab:blue", linewidth=1.2)
    ax_macd.plot(data["t_naive"], data["MACD_signal"], label="Signal", color="tab:orange", linewidth=1.0)
    ax_macd.bar(data["t_naive"], data["MACD_hist"], width=0.0006, color="tab:gray", alpha=0.5, label="Histogram")
    ax_macd.axhline(0, color="black", linewidth=0.8)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend()
    ax_macd.grid(True, linestyle="--", alpha=0.4)

    ax_macd.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ---- Plot: Candlestick + VWAP ----
def plot_candlestick_vwap(df, symbol):
    """Plot candlestick chart for today with VWAP and support/resistance levels."""
    df = _ensure_datetime_tz(df.copy())
    df["date_sgt"] = df["Datetime"].dt.date
    today_sgt = datetime.now(TARGET_TZ).date()
    last_date = df["date_sgt"].max() if today_sgt not in df["date_sgt"].unique() else today_sgt
    day = df[df["date_sgt"] == last_date].copy()

    # VWAP
    day["tp"] = (day["High"] + day["Low"] + day["Close"]) / 3
    day["pv"] = day["tp"] * day["Volume"]
    day["VWAP"] = day["pv"].cumsum() / day["Volume"].replace(0, np.nan).cumsum()

    last_close = float(day["Close"].iloc[-1])
    R1, R2 = last_close * 1.01, last_close * 1.02
    S1, S2 = last_close * 0.99, last_close * 0.98
    vwap_last = float(day["VWAP"].iloc[-1])

    # Prepare candlesticks
    day["dt_sgt_naive"] = day["Datetime"].dt.tz_convert(TARGET_TZ).dt.tz_localize(None)
    day["mdates"] = mdates.date2num(day["dt_sgt_naive"])
    bar_w = 60.0 / (24 * 60 * 60) * 0.9

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_title(f"{symbol} — {last_date} (1m Candlesticks)")

    for _, r in day.iterrows():
        ax.vlines(r["mdates"], r["Low"], r["High"], linewidth=1)
        lower, height = min(r["Open"], r["Close"]), abs(r["Close"] - r["Open"])
        ax.add_patch(plt.Rectangle((r["mdates"] - bar_w / 2, lower),
                                   bar_w, height if height != 0 else 1e-10,
                                   fill=(r["Close"] < r["Open"]), linewidth=1))

    ax.plot(day["mdates"], day["VWAP"], label="VWAP", linewidth=1.5)

    # Add S/R levels
    for y, lbl in [(S2, "S2"), (S1, "S1"), (vwap_last, "VWAP"), (R1, "R1"), (R2, "R2")]:
        ax.axhline(y, linestyle="--", linewidth=1)
        ax.text(day["mdates"].iloc[0], y, f" {lbl} {y:.3f}", va="bottom", fontsize=9)

    ax.legend()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.set_xlabel("Time (SGT)")
    ax.set_ylabel("Price (SGD)")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig


# ---- Plot: Volume ----
def plot_volume(df, symbol):
    """Plot volume chart (1m bars)."""
    df = _ensure_datetime_tz(df.copy())
    df["dt_sgt_naive"] = df["Datetime"].dt.tz_convert(TARGET_TZ).dt.tz_localize(None)
    bar_w = 60.0 / (24 * 60 * 60) * 0.9

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_title(f"{symbol} — Volume (1m)")
    ax.bar(df["dt_sgt_naive"], df["Volume"], width=bar_w)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time (SGT)")
    ax.set_ylabel("Volume")
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig


# ---- Plot: Detector ----
def plot_detector(df, symbol):
    """Plot candlestick + VWAP + detector signals (breakout/blowoff)."""
    df = _ensure_datetime_tz(df.copy()).sort_values("Datetime").reset_index(drop=True)
    df["date_sgt"] = df["Datetime"].dt.date

    # VWAP
    df["tp"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["pv"] = df["tp"] * df["Volume"]
    df["cum_pv"] = df.groupby("date_sgt")["pv"].cumsum()
    df["cum_vol"] = df.groupby("date_sgt")["Volume"].cumsum().replace(0, np.nan)
    df["VWAP"] = df["cum_pv"] / df["cum_vol"]

    # Volume spike baseline
    VOL_WINDOW, VOL_MULT = 30, 2.0
    df["vol_base"] = df.groupby("date_sgt")["Volume"].transform(lambda x: x.rolling(VOL_WINDOW, min_periods=1).median())
    df["vol_spike"] = df["Volume"] >= (VOL_MULT * df["vol_base"])

    # Breakout detection
    HH_LOOKBACK = 10
    df["prior_hh"] = df.groupby("date_sgt")["High"].shift(1).rolling(HH_LOOKBACK, min_periods=1).max()
    df["above_vwap"] = df["Close"] >= df["VWAP"]
    df["breakout"] = df["vol_spike"] & df["above_vwap"] & (df["Close"] > df["prior_hh"])

    # Blowoff detection
    WICK_RATIO, REVERSAL_BARS = 0.6, 3
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["long_upper_wick"] = (upper_wick / rng) >= WICK_RATIO
    wick_blowoff = df["vol_spike"] & (~df["above_vwap"]) & df["long_upper_wick"]

    reversal_blowoff = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if df["breakout"].iat[i]:
            j_end = min(i + REVERSAL_BARS, len(df) - 1)
            mask = slice(i + 1, j_end + 1)
            reversal_blowoff.iloc[mask] |= (~df["above_vwap"].iloc[mask]) & df["vol_spike"].iloc[mask]

    df["blowoff"] = wick_blowoff | reversal_blowoff

    # Last day subset
    last_date = df["date_sgt"].max()
    day = df[df["date_sgt"] == last_date].copy()
    day["dt_sgt_naive"] = day["Datetime"].dt.tz_convert(TARGET_TZ).dt.tz_localize(None)
    day["mdates"] = mdates.date2num(day["dt_sgt_naive"])
    bar_w = 60.0 / (24 * 60 * 60) * 0.9

    # Plot
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_title(f"{symbol} — {last_date} (Detector View)")

    for _, r in day.iterrows():
        ax.vlines(r["mdates"], r["Low"], r["High"], linewidth=1)
        lower, height = min(r["Open"], r["Close"]), abs(r["Close"] - r["Open"])
        ax.add_patch(plt.Rectangle((r["mdates"] - bar_w / 2, lower),
                                   bar_w, height if height != 0 else 1e-10,
                                   fill=(r["Close"] < r["Open"]), linewidth=1))

    ax.plot(day["mdates"], day["VWAP"], linewidth=1.5, label="VWAP")

    # Mark signals
    bo = day[day["breakout"]]
    ax.plot(bo["mdates"], bo["Close"], marker="^", linestyle="None", markersize=8, label="Breakout")
    bf = day[day["blowoff"]]
    ax.plot(bf["mdates"], bf["Close"], marker="v", linestyle="None", markersize=8, label="Blow-off")

    ax.legend()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.set_xlabel("Time (SGT)")
    ax.set_ylabel("Price (SGD)")
    ax.grid(True, linestyle="--", alpha=0.6)

    return fig, df[df["date_sgt"] == last_date][["Datetime", "Open", "High", "Low", "Close", "Volume", "VWAP", "breakout", "blowoff"]]


# ---- Plot: 3-Month Candlestick ----
def plot_3mths_candlestick(df, symbol, months=3):
    """Plot Close prices for the last N months with support/resistance levels."""
    df = _ensure_datetime_tz(df.copy())

    # Filter to last N months
    cutoff = datetime.now(TARGET_TZ) - pd.DateOffset(months=months)
    df = df[df["Datetime"] >= cutoff]
    if df.empty:
        return None

    # Last close
    last_close = float(df["Close"].iloc[-1])

    # Wider ranges (±10% and ±20%)
    R1, R2 = last_close * 1.10, last_close * 1.20
    S1, S2 = last_close * 0.90, last_close * 0.80

    # Prepare datetime
    df["dt_naive"] = df["Datetime"].dt.tz_localize(None)

    # Plot
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["dt_naive"], df["Close"], label="Close Price", color="blue")

    # Add support/resistance lines
    for y, lbl in [(S2, "S2"), (S1, "S1"), (last_close, "Close"), (R1, "R1"), (R2, "R2")]:
        ax.axhline(y, linestyle="--", linewidth=1, color="green" if "R" in lbl else "red")
        ax.text(df["dt_naive"].iloc[0], y, f" {lbl} {y:.3f}", va="bottom", fontsize=9)

    # X-axis formatting: weekly ticks, rotated
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_title(f"{symbol} — Last {months} Months (Line Chart)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (SGD)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    return fig
