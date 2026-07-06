# StockSight AI — ML Stock Price Predictor

A production-ready machine learning application for stock price prediction, featuring **LSTM neural networks**, **Random Forest**, and **Gradient Boosting** models with a stunning dark-mode web dashboard.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fetch Stock Data
```bash
python data/fetch_data.py
```
Downloads historical data (2018–today) for: AAPL, TSLA, GOOGL, MSFT, AMZN, NVDA, META, NFLX

### 3. Train Models
```bash
# Train all key tickers
python model/train.py --tickers AAPL TSLA MSFT

# Or just one
python model/train.py --tickers AAPL
```
Training takes ~5–15 min per ticker (GPU recommended).

### 4. Run the Dashboard
```bash
python app.py
```
Open → http://localhost:5000

---

## 📁 Project Structure
```
ML-Model-on-render/
├── app.py                  # Flask REST API
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
│
├── model/
│   ├── indicators.py       # 25+ technical indicators
│   ├── train.py            # LSTM + RF + GB training pipeline
│   └── predict.py          # Inference engine
│
├── data/
│   ├── fetch_data.py       # yFinance data downloader
│   └── stocks/             # CSV files (auto-created)
│
├── saved_models/           # Trained models (auto-created)
│   ├── AAPL_lstm.keras
│   ├── AAPL_rf.pkl
│   ├── AAPL_gb.pkl
│   ├── AAPL_scaler_x.pkl
│   ├── AAPL_scaler_y.pkl
│   └── AAPL_meta.json
│
├── static/
│   ├── style.css           # Premium dark-mode CSS
│   └── app.js              # Dashboard JavaScript
│
└── templates/
    └── index.html          # Main dashboard
```

---

## 🧠 ML Architecture

### Models
| Model | Details | Ensemble Weight |
|-------|---------|----------------|
| **LSTM** | 3-layer (128→64→32), BatchNorm, Dropout 0.3, Huber loss, 60-day window | 50% |
| **Random Forest** | 300 trees, max depth 10, 25+ features | 30% |
| **Gradient Boosting** | 300 estimators, depth 5, LR 0.05 | 20% |

### Features (25+)
- **Price**: OHLCV, Log Return, Daily/Weekly/Monthly Returns
- **Trend**: SMA 10/20/50, EMA 12/26/50
- **Momentum**: MACD, RSI(14), Stochastic K/D, ROC
- **Volatility**: Bollinger Bands, ATR, HL Ratio
- **Volume**: OBV, Volume SMA, Volume Ratio

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/tickers` | List tickers + training status |
| GET | `/api/predict/<TICKER>` | Next-day price prediction |
| GET | `/api/history/<TICKER>?days=N` | OHLCV + indicators |
| GET | `/api/backtest/<TICKER>?days=N` | Predicted vs actual |
| POST | `/api/train/<TICKER>` | Trigger model training |
| GET | `/health` | Health check |

---

## ☁️ Deploy to Render

1. Push to GitHub
2. Connect repo to [Render](https://render.com)
3. Render reads `render.yaml` automatically
4. After deployment, call `/api/train/AAPL` to train the model

---

## 📊 Performance (typical results)

| Ticker | Model | MAE | R² |
|--------|-------|-----|-----|
| AAPL | Ensemble | ~0.8–2.0 | 0.97–0.99 |
| TSLA | Ensemble | ~2–8 | 0.94–0.98 |
| MSFT | Ensemble | ~1–3 | 0.96–0.99 |

> ⚠️ **Disclaimer**: This tool is for educational/research purposes only. Past performance does not guarantee future results. Do not use for actual financial trading decisions.
