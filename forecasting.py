"""Pure forecasting helpers (no Streamlit) so they can be tested on their own."""
import numpy as np
import pandas as pd

OHLCV = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


def future_index(last: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)


# ───────────── uploaded ML model (lag features) ─────────────
def lag_forecast(model, history: np.ndarray, n_lags: int, steps: int, normalized: bool):
    """Recursive forecast for models that take the last N closes as input."""
    buf = list(history[-n_lags:])
    for _ in range(steps):
        w = np.array(buf[-n_lags:], dtype=float)
        if normalized:
            base = w[-1]
            pred = float(np.ravel(model.predict((w / base).reshape(1, -1)))[0]) * base
        else:
            pred = float(np.ravel(model.predict(w.reshape(1, -1)))[0])
        buf.append(pred)
    return np.array(buf[n_lags:])


# ───────────── uploaded ML model (OHLCV features) ─────────────
def is_ohlcv_model(model) -> bool:
    names = getattr(model, "feature_names_in_", None)
    return names is not None and set(names) <= OHLCV


def ohlcv_forecast(model, df: pd.DataFrame, steps: int, vol_window: int = 20):
    """Recursive forecast for a model that maps same-day O/H/L/AdjClose/Volume -> price.

    Future OHLV isn't known, so each step assumes Open=High=Low=Close=Adj Close =
    previous step's predicted price, and Volume = recent average volume.
    """
    names = list(model.feature_names_in_)
    if "Volume" in names and "Volume" not in df:
        raise ValueError("This model needs a Volume column in the data.")
    prev = float(df["Adj Close"].iloc[-1] if "Adj Close" in df else df["Close"].iloc[-1])
    vol = float(df["Volume"].tail(vol_window).mean()) if "Volume" in df else 0.0
    out = []
    for _ in range(steps):
        row = {c: prev for c in names}
        if "Volume" in row:
            row["Volume"] = vol
        prev = float(np.ravel(model.predict(pd.DataFrame([row], columns=names)))[0])
        out.append(prev)
    return np.array(out)


def tree_value_range(model):
    """(min, max) of values a forest/tree ensemble can output, or None."""
    try:
        v = np.concatenate([e.tree_.value.ravel() for e in model.estimators_])
        return float(v.min()), float(v.max())
    except Exception:
        return None


# ───────────── uploaded time-series model ─────────────
def _raw_ts_forecast(model, steps: int):
    if hasattr(model, "make_future_dataframe"):  # Prophet
        fut = model.make_future_dataframe(steps, freq="B")
        out = model.predict(fut).tail(steps)
        return out["yhat"].to_numpy(), out["yhat_lower"].to_numpy(), out["yhat_upper"].to_numpy()
    if hasattr(model, "get_forecast"):  # statsmodels ARIMA / SARIMAX / ETS
        fc = model.get_forecast(steps)
        mean = np.asarray(fc.predicted_mean)
        try:
            ci = np.asarray(fc.conf_int())
            return mean, ci[:, 0], ci[:, 1]
        except Exception:
            return mean, None, None
    if hasattr(model, "predict"):  # pmdarima
        try:
            mean, ci = model.predict(n_periods=steps, return_conf_int=True)
            return np.asarray(mean), ci[:, 0], ci[:, 1]
        except TypeError:
            pass
    if hasattr(model, "forecast"):
        return np.asarray(model.forecast(steps)), None, None
    raise ValueError("Unsupported time-series model type.")


def guess_ts_target(model, price_level: float) -> str:
    """'levels' if the model was trained on prices, else 'diff' (changes)."""
    try:
        e = np.asarray(model.model.endog, dtype=float).ravel()
    except Exception:
        return "levels"
    return "levels" if np.nanmedian(np.abs(e)) > 0.3 * price_level else "diff"


def ts_forecast(model, steps: int, target: str, last_price: float,
                recent_close: np.ndarray | None = None, resync: int = 0):
    """Forecast prices from an uploaded time-series model.

    target: 'levels' (model outputs prices), 'diff' (daily $ change), 'pct' (daily % return).
    resync: for 'diff' models, feed the last N daily changes of the chosen data into the
            model state (no refit) before forecasting. Returns (mean, lo, hi, note).
    """
    note = ""
    m = model
    if target == "diff" and resync and recent_close is not None and hasattr(m, "append"):
        try:
            m = m.append(np.diff(recent_close)[-resync:], refit=False)
            note = f"Model state updated with the last {min(resync, len(recent_close) - 1)} daily changes (no refit)."
        except Exception as e:
            note = f"Couldn't update model state ({type(e).__name__}); using it as saved."

    mean, lo, hi = _raw_ts_forecast(m, steps)
    if target == "levels":
        return mean, lo, hi, note
    if target == "diff":
        path = last_price + np.cumsum(mean)
        if lo is None:
            return path, None, None, note
        se = (hi - lo) / (2 * 1.96)
        cum = 1.96 * np.sqrt(np.cumsum(se ** 2))  # approx.: treats step errors as independent
        return path, path - cum, path + cum, note
    if target == "pct":
        return last_price * np.cumprod(1 + mean / 100.0), None, None, note
    raise ValueError(f"Unknown target: {target}")