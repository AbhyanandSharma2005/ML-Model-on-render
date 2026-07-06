"""
model/train.py
Trains an LSTM neural network and a Random Forest model for stock price prediction.
Saves trained models + scalers to saved_models/.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from data.fetch_data import fetch_stock, save_stock
from model.indicators import add_indicators, get_feature_columns

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
SEQ_LEN      = 60       # look-back window for LSTM
FUTURE_DAYS  = 1        # predict 1 day ahead
TEST_SPLIT   = 0.15     # 15 % test set
VAL_SPLIT    = 0.10     # 10 % validation set
EPOCHS       = 100
BATCH_SIZE   = 32
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(ticker: str):
    """
    Fetch raw data, add indicators, scale, and build sequences.
    Returns train/val/test splits + scaler.
    """
    print(f"\n{'='*60}")
    print(f"  Preparing data for {ticker}")
    print(f"{'='*60}")

    # 1. Fetch + add indicators
    df_raw = fetch_stock(ticker)
    save_stock(ticker, df_raw)
    df = add_indicators(df_raw)

    feat_cols = get_feature_columns()
    target_col = "Close"

    data = df[feat_cols].values
    targets = df[target_col].values

    # 2. Scale features
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    data_scaled    = scaler_x.fit_transform(data)
    targets_scaled = scaler_y.fit_transform(targets.reshape(-1, 1)).ravel()

    # 3. Build sequences
    X, y = [], []
    for i in range(SEQ_LEN, len(data_scaled) - FUTURE_DAYS + 1):
        X.append(data_scaled[i - SEQ_LEN: i])
        y.append(targets_scaled[i + FUTURE_DAYS - 1])
    X, y = np.array(X), np.array(y)

    # 4. Split
    n = len(X)
    n_test = int(n * TEST_SPLIT)
    n_val  = int(n * VAL_SPLIT)
    n_train = n - n_test - n_val

    X_train = X[:n_train];        y_train = y[:n_train]
    X_val   = X[n_train: n_train+n_val]; y_val = y[n_train: n_train+n_val]
    X_test  = X[n_train+n_val:];  y_test  = y[n_train+n_val:]

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Also prepare flat features for RF/GB (use last time-step only)
    X_flat_train = X_train[:, -1, :]
    X_flat_val   = X_val[:, -1, :]
    X_flat_test  = X_test[:, -1, :]

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_flat_train, X_flat_val, X_flat_test,
            scaler_x, scaler_y, df)


# ─────────────────────────────────────────────────────────────────────────────
# Model Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm(input_shape):
    """Build a stacked LSTM model with dropout and batch normalization."""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_ticker(ticker: str) -> dict:
    """Full training pipeline for one ticker. Returns evaluation metrics."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     X_flat_train, X_flat_val, X_flat_test,
     scaler_x, scaler_y, df) = prepare_data(ticker)

    metrics = {}

    # ── 1. LSTM ───────────────────────────────────────────────────────────────
    print("\n[LSTM] Building and training …")
    lstm_model = build_lstm((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]

    history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate LSTM
    y_pred_lstm_scaled = lstm_model.predict(X_test).ravel()
    y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1,1)).ravel()
    y_true      = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()

    lstm_mae  = mean_absolute_error(y_true, y_pred_lstm)
    lstm_rmse = np.sqrt(mean_squared_error(y_true, y_pred_lstm))
    lstm_r2   = r2_score(y_true, y_pred_lstm)
    metrics["LSTM"] = {"MAE": round(lstm_mae,4), "RMSE": round(lstm_rmse,4), "R2": round(lstm_r2,4)}
    print(f"[LSTM] MAE={lstm_mae:.4f}  RMSE={lstm_rmse:.4f}  R²={lstm_r2:.4f}")

    # Save LSTM
    lstm_path = os.path.join(MODELS_DIR, f"{ticker}_lstm.keras")
    lstm_model.save(lstm_path)
    print(f"[LSTM] Saved -> {lstm_path}")

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    print("\n[RF] Training Random Forest …")
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=10,
        min_samples_leaf=5, random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_flat_train, y_train)

    y_pred_rf_scaled = rf_model.predict(X_flat_test)
    y_pred_rf = scaler_y.inverse_transform(y_pred_rf_scaled.reshape(-1,1)).ravel()

    rf_mae  = mean_absolute_error(y_true, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_true, y_pred_rf))
    rf_r2   = r2_score(y_true, y_pred_rf)
    metrics["RandomForest"] = {"MAE": round(rf_mae,4), "RMSE": round(rf_rmse,4), "R2": round(rf_r2,4)}
    print(f"[RF]   MAE={rf_mae:.4f}  RMSE={rf_rmse:.4f}  R²={rf_r2:.4f}")

    rf_path = os.path.join(MODELS_DIR, f"{ticker}_rf.pkl")
    joblib.dump(rf_model, rf_path)
    print(f"[RF]   Saved -> {rf_path}")

    # ── 3. Gradient Boosting ──────────────────────────────────────────────────
    print("\n[GB] Training Gradient Boosting …")
    gb_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    )
    gb_model.fit(X_flat_train, y_train)

    y_pred_gb_scaled = gb_model.predict(X_flat_test)
    y_pred_gb = scaler_y.inverse_transform(y_pred_gb_scaled.reshape(-1,1)).ravel()

    gb_mae  = mean_absolute_error(y_true, y_pred_gb)
    gb_rmse = np.sqrt(mean_squared_error(y_true, y_pred_gb))
    gb_r2   = r2_score(y_true, y_pred_gb)
    metrics["GradientBoosting"] = {"MAE": round(gb_mae,4), "RMSE": round(gb_rmse,4), "R2": round(gb_r2,4)}
    print(f"[GB]   MAE={gb_mae:.4f}  RMSE={gb_rmse:.4f}  R²={gb_r2:.4f}")

    gb_path = os.path.join(MODELS_DIR, f"{ticker}_gb.pkl")
    joblib.dump(gb_model, gb_path)

    # ── 4. Ensemble Predictions ───────────────────────────────────────────────
    y_pred_ensemble = (0.5 * y_pred_lstm + 0.3 * y_pred_rf + 0.2 * y_pred_gb)
    ens_mae  = mean_absolute_error(y_true, y_pred_ensemble)
    ens_rmse = np.sqrt(mean_squared_error(y_true, y_pred_ensemble))
    ens_r2   = r2_score(y_true, y_pred_ensemble)
    metrics["Ensemble"] = {"MAE": round(ens_mae,4), "RMSE": round(ens_rmse,4), "R2": round(ens_r2,4)}
    print(f"\n[ENS]  MAE={ens_mae:.4f}  RMSE={ens_rmse:.4f}  R²={ens_r2:.4f}")

    # ── 5. Save scalers + metadata ────────────────────────────────────────────
    joblib.dump(scaler_x, os.path.join(MODELS_DIR, f"{ticker}_scaler_x.pkl"))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, f"{ticker}_scaler_y.pkl"))

    meta = {
        "ticker":      ticker,
        "trained_at":  datetime.utcnow().isoformat(),
        "seq_len":     SEQ_LEN,
        "features":    get_feature_columns(),
        "train_rows":  int(len(X_train)),
        "test_rows":   int(len(X_test)),
        "metrics":     metrics,
        "last_close":  float(df["Close"].iloc[-1]),
        "last_date":   str(df.index[-1].date()),
    }
    meta_path = os.path.join(MODELS_DIR, f"{ticker}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[OK] {ticker} training complete. Metadata -> {meta_path}")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ML stock prediction models")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "TSLA", "MSFT"],
                        help="List of tickers to train on")
    args = parser.parse_args()

    all_metrics = {}
    for ticker in args.tickers:
        try:
            m = train_ticker(ticker)
            all_metrics[ticker] = m["metrics"]
        except Exception as exc:
            print(f"\n❌ Failed to train {ticker}: {exc}")

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for t, m in all_metrics.items():
        print(f"\n{t}:")
        for model, scores in m.items():
            print(f"  {model:<20} MAE={scores['MAE']:.4f}  R²={scores['R2']:.4f}")
