import warnings
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import forecasting as fc

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stock Forecaster", page_icon="📈", layout="wide")

HORIZON = 30  # business days ahead


# ───────────────────────── data ─────────────────────────
@st.cache_data(show_spinner="Fetching prices…")
def load_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end + timedelta(days=1),
        auto_adjust=False, progress=False,  # keeps both Close and Adj Close
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(subset=["Close"])


def load_csv(file) -> pd.DataFrame:
    file.seek(0)
    df = pd.read_csv(file)
    rename = {"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
              "adj close": "Adj Close", "adj_close": "Adj Close", "volume": "Volume"}
    df = df.rename(columns={c: rename.get(c.strip().lower(), c) for c in df.columns})
    if "Date" not in df:
        df = df.rename(columns={df.columns[0]: "Date"})
    if "Close" not in df:
        df = df.rename(columns={df.columns[1]: "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if "Adj Close" not in df:
        df["Adj Close"] = df["Close"]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna(subset=["Close"])


# ───────────────────────── cached model wrappers ─────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(file_id: str, _file):
    _file.seek(0)
    return joblib.load(_file)


def fid(f):
    return getattr(f, "file_id", f"{f.name}-{f.size}")


# ───────────────────────── sidebar (inputs) ─────────────────────────
st.title("📈 Stock Price Forecaster")
st.caption(f"Pick a ticker + history window, upload your trained models, get the next {HORIZON} business days.")

with st.sidebar:
    st.header("⚙️ Inputs")
    ticker = st.text_input("Ticker", "AAPL").upper().strip()
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", date.today() - timedelta(days=3 * 365))
    end = c2.date_input("End", date.today())
    csv_file = st.file_uploader("…or upload price CSV (Date, Open, High, Low, Close, Adj Close, Volume)", type="csv")

    st.header("🧠 Upload your models")
    ts_file = st.file_uploader("Time-series model (.pkl / .joblib)", type=["pkl", "joblib"], key="ts")
    ml_file = st.file_uploader("ML model (.pkl / .joblib)", type=["pkl", "joblib"], key="ml")

# ───────────────────────── load data ─────────────────────────
if not csv_file and start >= end:
    st.error("Start date must be before end date.")
    st.stop()

df = load_csv(csv_file) if csv_file else load_prices(ticker, start, end)
if csv_file and len(df):
    st.caption(f"Using uploaded CSV: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()} (sidebar dates ignored).")
if len(df) < 60:
    st.error(f"Not enough price data: found {len(df)} rows (need ~{60}+). "
             + ("Check that your CSV has a Date column and a numeric Close column." if csv_file
                else "Widen the date range or check the ticker."))
    st.stop()

prices = df["Close"]
y = prices.to_numpy(dtype=float)
last = float(y[-1])
f_idx = fc.future_index(prices.index[-1], HORIZON)

# ───────────────────────── load + configure uploaded models ─────────────────────────
ts_model = ml_model = None
ts_target, ts_resync, ml_normalized, ml_is_ohlcv = "levels", 0, True, False

with st.sidebar:
    try:
        if ts_file:
            ts_model = load_model_cached(fid(ts_file), ts_file)
            st.subheader("Time-series model settings")
            guess = fc.guess_ts_target(ts_model, last)
            labels = {"levels": "Price levels", "diff": "Daily price change ($, differenced)", "pct": "Daily % return"}
            ts_target = st.radio(
                "Your model forecasts…", list(labels), format_func=labels.get, index=list(labels).index(guess),
                help="If you trained on prices.diff(), pick 'Daily price change'. Output is converted back to prices.",
            )
            if ts_target == "diff":
                ts_resync = st.slider(
                    "Sync model with my data (last N days)", 0, 250, 60,
                    help="Feeds recent daily changes into the model state (no refit) so it starts from "
                         "your selected date range instead of where it was trained. 0 = off.",
                )
        if ml_file:
            ml_model = load_model_cached(fid(ml_file), ml_file)
            st.subheader("ML model settings")
            ml_is_ohlcv = fc.is_ohlcv_model(ml_model)
            if ml_is_ohlcv:
                st.caption("Detected OHLCV features: " + ", ".join(ml_model.feature_names_in_))
            else:
                ml_normalized = st.radio(
                    "Model input", ["Last N closes ÷ last close", "Raw last N closes"],
                    help="Must match how you trained it.",
                ).startswith("Last")
    except Exception as e:
        st.error(f"Couldn't load an uploaded model: {e}")
    st.warning("Only upload models you trust — pickle files can run code on load.", icon="⚠️")

# ───────────────────────── run models ─────────────────────────
available = []
if ts_model is not None:
    available.append("Uploaded time-series model")
if ml_model is not None:
    available.append("Uploaded ML model")
if available:
    chosen = st.multiselect("Models to show", available, default=available)
else:
    chosen = []
    st.info("👈 Upload a time-series model and/or an ML model (.pkl / .joblib) in the sidebar to see forecasts.")

forecasts: dict[str, np.ndarray] = {}
bands: dict[str, tuple] = {}
notes: list[str] = []

for name in chosen:
    try:
        if name == "Uploaded time-series model":
            m, lo, hi, note = fc.ts_forecast(ts_model, HORIZON, ts_target, last, y, ts_resync)
            forecasts[name] = m
            if lo is not None:
                bands[name] = (lo, hi)
            if note:
                notes.append(note)
        elif name == "Uploaded ML model":
            if ml_is_ohlcv:
                forecasts[name] = fc.ohlcv_forecast(ml_model, df, HORIZON)
                rng = fc.tree_value_range(ml_model)
                if rng and not (rng[0] * 0.9 <= last <= rng[1] * 1.1):
                    st.warning(
                        f"Latest close ({last:,.2f}) is outside the price range this model has seen "
                        f"({rng[0]:,.0f}–{rng[1]:,.0f}). Tree models can't extrapolate, so treat its output with care."
                    )
                notes.append("Uploaded ML model uses OHLCV inputs, which are unknown for future days: each step assumes "
                             "Open/High/Low/Adj Close = previous predicted price and Volume = 20-day average.")
            else:
                lags = int(getattr(ml_model, "n_features_in_", 10))
                forecasts[name] = fc.lag_forecast(ml_model, y, lags, HORIZON, ml_normalized)
    except Exception as e:
        st.warning(f"**{name}** failed: {e}")

# ───────────────────────── UI ─────────────────────────
tab_fc, tab_hist, tab_data = st.tabs(["🔮 Forecast", "📊 Historical trends", "🗂️ Data"])

with tab_fc:
    cols = st.columns(max(len(forecasts), 1) + 1)
    cols[0].metric("Last close", f"{last:,.2f}", prices.index[-1].strftime("%d %b %Y"))
    for col, (n, v) in zip(cols[1:], forecasts.items()):
        col.metric(f"{n} · day {HORIZON}", f"{v[-1]:,.2f}", f"{(v[-1] / last - 1) * 100:+.2f}%")

    lookback = st.slider("History shown (days)", 30, len(prices), min(250, len(prices)))
    fig = go.Figure()
    h = prices.iloc[-lookback:]
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="Historical", line=dict(color="#4c78a8")))
    for n, v in forecasts.items():
        if n in bands:
            lo, hi = bands[n]
            fig.add_trace(go.Scatter(
                x=list(f_idx) + list(f_idx[::-1]), y=list(hi) + list(lo[::-1]),
                fill="toself", opacity=0.15, line=dict(width=0), name=f"{n} 95% CI", hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(x=f_idx, y=v, name=n, mode="lines", line=dict(dash="dash")))
    fig.add_vrect(x0=str(f_idx[0].date()), x1=str(f_idx[-1].date()), fillcolor="grey", opacity=0.07, line_width=0)
    fig.update_layout(height=520, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
                      title=f"{ticker} — {HORIZON}-day forecast", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    for n in notes:
        st.info(n)

    if forecasts:
        out = pd.DataFrame(forecasts, index=f_idx).round(2)
        out.index.name = "Date"
        st.dataframe(out, width="stretch")
        st.download_button("⬇️ Download forecast CSV", out.to_csv().encode(), f"{ticker}_forecast.csv", "text/csv")

with tab_hist:
    ma = pd.DataFrame({"Close": prices, "MA20": prices.rolling(20).mean(), "MA50": prices.rolling(50).mean()})
    st.plotly_chart(go.Figure([go.Scatter(x=ma.index, y=ma[c], name=c) for c in ma]).update_layout(
        height=400, title="Price + moving averages", margin=dict(l=0, r=0, t=40, b=0)), width="stretch")
    ret = prices.pct_change().dropna()
    a, b = st.columns(2)
    a.plotly_chart(go.Figure(go.Histogram(x=ret * 100, nbinsx=60)).update_layout(
        height=320, title="Daily returns (%)", margin=dict(l=0, r=0, t=40, b=0)), width="stretch")
    vol = ret.rolling(21).std() * np.sqrt(252) * 100
    b.plotly_chart(go.Figure(go.Scatter(x=vol.index, y=vol)).update_layout(
        height=320, title="21-day rolling volatility (annualised %)", margin=dict(l=0, r=0, t=40, b=0)),
        width="stretch")

with tab_data:
    st.dataframe(df.sort_index(ascending=False), width="stretch")

st.caption("Educational demo — not financial advice. Forecasts of stock prices are highly uncertain.")