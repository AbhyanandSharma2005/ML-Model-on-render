"""
model/indicators.py
Calculates a rich set of technical indicators used as ML features.
"""
import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends ~25 technical indicators to an OHLCV DataFrame.
    Requires columns: Open, High, Low, Close, Volume.
    """
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Trend Indicators ─────────────────────────────────────────────────────
    df["SMA_10"]  = close.rolling(10).mean()
    df["SMA_20"]  = close.rolling(20).mean()
    df["SMA_50"]  = close.rolling(50).mean()
    df["EMA_12"]  = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = close.ewm(span=26, adjust=False).mean()
    df["EMA_50"]  = close.ewm(span=50, adjust=False).mean()

    # ── MACD ─────────────────────────────────────────────────────────────────
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── RSI ──────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    df["RSI"] = 100 - 100 / (1 + rs)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (bb_mid + 1e-10)
    df["BB_Pct"]   = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-10)

    # ── ATR (Average True Range) ──────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # ── Volume Indicators ────────────────────────────────────────────────────
    df["Volume_SMA"] = vol.rolling(20).mean()
    df["Volume_Ratio"] = vol / (df["Volume_SMA"] + 1e-10)
    df["OBV"] = (np.sign(close.diff()) * vol).cumsum()

    # ── Price Return Features ─────────────────────────────────────────────────
    df["Return_1d"]  = close.pct_change(1)
    df["Return_5d"]  = close.pct_change(5)
    df["Return_20d"] = close.pct_change(20)
    df["Log_Return"] = np.log(close / close.shift(1))

    # ── High-Low Range ────────────────────────────────────────────────────────
    df["HL_Ratio"] = (high - low) / (close + 1e-10)

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["Momentum_10"] = close - close.shift(10)
    df["ROC_10"]      = close.pct_change(10) * 100   # Rate of Change

    # ── Stochastic Oscillator ─────────────────────────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    df.dropna(inplace=True)
    return df


def get_feature_columns() -> list:
    """Return the list of feature column names used in training."""
    return [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_10", "SMA_20", "SMA_50",
        "EMA_12", "EMA_26", "EMA_50",
        "MACD", "MACD_Signal", "MACD_Hist",
        "RSI",
        "BB_Upper", "BB_Lower", "BB_Width", "BB_Pct",
        "ATR",
        "Volume_SMA", "Volume_Ratio", "OBV",
        "Return_1d", "Return_5d", "Return_20d", "Log_Return",
        "HL_Ratio",
        "Momentum_10", "ROC_10",
        "Stoch_K", "Stoch_D",
    ]
