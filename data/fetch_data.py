"""
data/fetch_data.py
Fetches historical OHLCV stock data from Yahoo Finance and saves to CSV.
"""
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ── Configuration ────────────────────────────────────────────────────────────
STOCKS = {
    "AAPL":  "Apple Inc.",
    "TSLA":  "Tesla Inc.",
    "GOOGL": "Alphabet Inc.",
    "MSFT":  "Microsoft Corp.",
    "AMZN":  "Amazon.com Inc.",
    "NVDA":  "NVIDIA Corp.",
    "META":  "Meta Platforms",
    "NFLX":  "Netflix Inc.",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "stocks")
START_DATE = "2018-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")


def fetch_stock(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Download OHLCV data for a single ticker."""
    print(f"[fetch] Downloading {ticker} ({start} -> {end}) ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def save_stock(ticker: str, df: pd.DataFrame) -> str:
    """Save dataframe to CSV in the stocks/ directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(path)
    print(f"[fetch] Saved {len(df)} rows -> {path}")
    return path


def fetch_all() -> dict:
    """Fetch and save data for all configured tickers. Returns dict of {ticker: df}."""
    results = {}
    for ticker in STOCKS:
        try:
            df = fetch_stock(ticker)
            save_stock(ticker, df)
            results[ticker] = df
        except Exception as exc:
            print(f"[fetch] ERROR for {ticker}: {exc}")
    return results


def load_stock(ticker: str) -> pd.DataFrame:
    """Load a ticker's CSV if it exists, otherwise download it."""
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        return df
    return fetch_stock(ticker)


if __name__ == "__main__":
    fetch_all()
    print("\n✅ All stock data fetched successfully!")
