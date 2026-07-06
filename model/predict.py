"""
model/predict.py
Loads trained models and generates predictions + future forecasts.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model as keras_load
from data.fetch_data import fetch_stock
from model.indicators import add_indicators, get_feature_columns

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
SEQ_LEN    = 60


def _load_artifacts(ticker: str):
    """Load all saved artifacts for a ticker."""
    paths = {
        "lstm":      os.path.join(MODELS_DIR, f"{ticker}_lstm.keras"),
        "rf":        os.path.join(MODELS_DIR, f"{ticker}_rf.pkl"),
        "gb":        os.path.join(MODELS_DIR, f"{ticker}_gb.pkl"),
        "scaler_x":  os.path.join(MODELS_DIR, f"{ticker}_scaler_x.pkl"),
        "scaler_y":  os.path.join(MODELS_DIR, f"{ticker}_scaler_y.pkl"),
        "meta":      os.path.join(MODELS_DIR, f"{ticker}_meta.json"),
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files for {ticker}: {missing}. "
            f"Please run: python model/train.py --tickers {ticker}"
        )

    lstm     = keras_load(paths["lstm"])
    rf       = joblib.load(paths["rf"])
    gb       = joblib.load(paths["gb"])
    scaler_x = joblib.load(paths["scaler_x"])
    scaler_y = joblib.load(paths["scaler_y"])
    with open(paths["meta"]) as f:
        meta = json.load(f)

    return lstm, rf, gb, scaler_x, scaler_y, meta


def get_latest_sequence(ticker: str, scaler_x):
    """Fetch the most recent data and return the last SEQ_LEN rows scaled."""
    df_raw = fetch_stock(ticker)
    df = add_indicators(df_raw)
    feat_cols = get_feature_columns()
    data = df[feat_cols].values
    data_scaled = scaler_x.transform(data)
    seq = data_scaled[-SEQ_LEN:]      # shape (SEQ_LEN, n_features)
    return seq, df


def predict_next_day(ticker: str) -> dict:
    """
    Predict the next trading day's closing price for a given ticker.
    Returns a dict with predictions from each model + ensemble.
    """
    lstm, rf, gb, scaler_x, scaler_y, meta = _load_artifacts(ticker)
    seq, df = get_latest_sequence(ticker, scaler_x)

    # LSTM
    x_lstm = seq.reshape(1, SEQ_LEN, -1)
    p_lstm = scaler_y.inverse_transform(lstm.predict(x_lstm)).ravel()[0]

    # RF + GB (use last row of sequence = most recent features)
    x_flat = seq[-1].reshape(1, -1)
    p_rf   = scaler_y.inverse_transform(rf.predict(x_flat).reshape(-1,1)).ravel()[0]
    p_gb   = scaler_y.inverse_transform(gb.predict(x_flat).reshape(-1,1)).ravel()[0]

    p_ensemble = 0.5 * p_lstm + 0.3 * p_rf + 0.2 * p_gb

    last_close = float(df["Close"].iloc[-1])
    last_date  = df.index[-1]

    # Next business day
    next_date = last_date + timedelta(days=1)
    while next_date.weekday() >= 5:   # skip weekends
        next_date += timedelta(days=1)

    result = {
        "ticker":      ticker,
        "last_date":   str(last_date.date()),
        "last_close":  round(last_close, 2),
        "next_date":   str(next_date.date()),
        "predictions": {
            "LSTM":            round(float(p_lstm), 2),
            "RandomForest":    round(float(p_rf), 2),
            "GradientBoosting":round(float(p_gb), 2),
            "Ensemble":        round(float(p_ensemble), 2),
        },
        "change": {
            "LSTM":            round((p_lstm - last_close) / last_close * 100, 3),
            "RandomForest":    round((p_rf   - last_close) / last_close * 100, 3),
            "GradientBoosting":round((p_gb   - last_close) / last_close * 100, 3),
            "Ensemble":        round((p_ensemble - last_close) / last_close * 100, 3),
        },
        "model_metrics": meta.get("metrics", {}),
        "trained_at":    meta.get("trained_at", ""),
    }
    return result


def predict_historical(ticker: str, days: int = 90) -> dict:
    """
    Run the ensemble model over the last N days of historical data
    so we can plot predicted vs actual.
    """
    lstm, rf, gb, scaler_x, scaler_y, meta = _load_artifacts(ticker)
    df_raw = fetch_stock(ticker)
    df     = add_indicators(df_raw)

    feat_cols = get_feature_columns()
    data       = df[feat_cols].values
    data_scaled = scaler_x.transform(data)
    closes     = df["Close"].values

    preds_lstm, preds_rf, preds_gb, actuals, dates = [], [], [], [], []

    start_idx = max(SEQ_LEN, len(data_scaled) - days - SEQ_LEN)
    for i in range(start_idx, len(data_scaled)):
        seq = data_scaled[i - SEQ_LEN: i]
        x_lstm = seq.reshape(1, SEQ_LEN, -1)
        x_flat = seq[-1].reshape(1, -1)

        p_lstm = scaler_y.inverse_transform(lstm.predict(x_lstm, verbose=0)).ravel()[0]
        p_rf   = scaler_y.inverse_transform(rf.predict(x_flat).reshape(-1,1)).ravel()[0]
        p_gb   = scaler_y.inverse_transform(gb.predict(x_flat).reshape(-1,1)).ravel()[0]

        preds_lstm.append(float(p_lstm))
        preds_rf.append(float(p_rf))
        preds_gb.append(float(p_gb))
        actuals.append(float(closes[i]))
        dates.append(str(df.index[i].date()))

    ensemble_preds = [0.5*l + 0.3*r + 0.2*g
                      for l, r, g in zip(preds_lstm, preds_rf, preds_gb)]

    return {
        "ticker": ticker,
        "dates":  dates,
        "actual": actuals,
        "predictions": {
            "LSTM":         preds_lstm,
            "RandomForest": preds_rf,
            "GradientBoosting": preds_gb,
            "Ensemble":     ensemble_preds,
        },
    }


def get_ohlcv_history(ticker: str, days: int = 365) -> dict:
    """Return OHLCV + indicator data for charting."""
    df_raw = fetch_stock(ticker)
    df     = add_indicators(df_raw)
    df     = df.tail(days)

    return {
        "ticker": ticker,
        "dates":  [str(d.date()) for d in df.index],
        "open":   df["Open"].round(2).tolist(),
        "high":   df["High"].round(2).tolist(),
        "low":    df["Low"].round(2).tolist(),
        "close":  df["Close"].round(2).tolist(),
        "volume": df["Volume"].tolist(),
        "sma20":  df["SMA_20"].round(2).tolist(),
        "sma50":  df["SMA_50"].round(2).tolist(),
        "ema12":  df["EMA_12"].round(2).tolist(),
        "rsi":    df["RSI"].round(2).tolist(),
        "macd":   df["MACD"].round(4).tolist(),
        "macd_signal": df["MACD_Signal"].round(4).tolist(),
        "bb_upper": df["BB_Upper"].round(2).tolist(),
        "bb_lower": df["BB_Lower"].round(2).tolist(),
    }
