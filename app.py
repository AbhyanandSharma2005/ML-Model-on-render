"""
app.py — Flask REST API for ML Stock Price Prediction
"""
import os
import json
import logging
from functools import lru_cache

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

AVAILABLE_TICKERS = {
    "AAPL":  "Apple Inc.",
    "TSLA":  "Tesla Inc.",
    "GOOGL": "Alphabet Inc.",
    "MSFT":  "Microsoft Corp.",
    "AMZN":  "Amazon.com Inc.",
    "NVDA":  "NVIDIA Corp.",
    "META":  "Meta Platforms",
    "NFLX":  "Netflix Inc.",
}


def _model_exists(ticker: str) -> bool:
    required = [
        f"{ticker}_lstm.keras",
        f"{ticker}_rf.pkl",
        f"{ticker}_scaler_x.pkl",
        f"{ticker}_meta.json",
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/api/tickers", methods=["GET"])
def get_tickers():
    """Return available tickers and their training status."""
    result = {}
    for ticker, name in AVAILABLE_TICKERS.items():
        trained = _model_exists(ticker)
        meta = {}
        if trained:
            meta_path = os.path.join(MODELS_DIR, f"{ticker}_meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
        result[ticker] = {
            "name":       name,
            "trained":    trained,
            "last_close": meta.get("last_close"),
            "last_date":  meta.get("last_date"),
            "metrics":    meta.get("metrics", {}),
        }
    return jsonify(result)


@app.route("/api/predict/<ticker>", methods=["GET"])
def predict(ticker: str):
    """Return next-day price prediction for a ticker."""
    ticker = ticker.upper()
    if ticker not in AVAILABLE_TICKERS:
        return jsonify({"error": f"Ticker {ticker} not supported."}), 400
    if not _model_exists(ticker):
        return jsonify({
            "error": f"Model for {ticker} not trained yet.",
            "hint":  f"Run: python model/train.py --tickers {ticker}"
        }), 404

    try:
        from model.predict import predict_next_day
        result = predict_next_day(ticker)
        return jsonify(result)
    except Exception as exc:
        logger.exception(f"Prediction failed for {ticker}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/history/<ticker>", methods=["GET"])
def history(ticker: str):
    """Return OHLCV + indicator data for charting."""
    ticker = ticker.upper()
    days   = int(request.args.get("days", 365))

    if ticker not in AVAILABLE_TICKERS:
        return jsonify({"error": f"Ticker {ticker} not supported."}), 400

    try:
        from model.predict import get_ohlcv_history
        result = get_ohlcv_history(ticker, days=days)
        return jsonify(result)
    except Exception as exc:
        logger.exception(f"History fetch failed for {ticker}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/backtest/<ticker>", methods=["GET"])
def backtest(ticker: str):
    """Return historical predicted vs actual for model evaluation."""
    ticker = ticker.upper()
    days   = int(request.args.get("days", 90))

    if ticker not in AVAILABLE_TICKERS:
        return jsonify({"error": f"Ticker {ticker} not supported."}), 400
    if not _model_exists(ticker):
        return jsonify({"error": f"Model for {ticker} not trained yet."}), 404

    try:
        from model.predict import predict_historical
        result = predict_historical(ticker, days=days)
        return jsonify(result)
    except Exception as exc:
        logger.exception(f"Backtest failed for {ticker}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/train/<ticker>", methods=["POST"])
def train(ticker: str):
    """Trigger model training for a ticker (async-ish via background thread)."""
    ticker = ticker.upper()
    if ticker not in AVAILABLE_TICKERS:
        return jsonify({"error": f"Ticker {ticker} not supported."}), 400

    import threading
    def _train():
        try:
            from model.train import train_ticker
            train_ticker(ticker)
            logger.info(f"Training complete for {ticker}")
        except Exception as exc:
            logger.error(f"Training failed for {ticker}: {exc}")

    t = threading.Thread(target=_train, daemon=True)
    t.start()
    return jsonify({"status": "training_started", "ticker": ticker,
                    "message": f"Training {ticker} in background. Refresh in ~5 min."})


# ── Health Check ──────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_dir": MODELS_DIR})


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
