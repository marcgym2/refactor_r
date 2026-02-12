"""
Step 02 — Download & Clean Ticker Data.

Downloads OHLCV data via yfinance, records metadata, and cleans NAs
using noisy interpolation.
"""

import os
import time
import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from config import DATA_DIR, TRAIN_START_DATE
from step_02a_data_cleaning_helpers import noisy_interpolation


def run() -> None:
    """Download stock data, clean NAs, and persist to disk."""

    # --- Load ticker metadata ---
    meta_path = os.path.join(DATA_DIR, "tickers_metadata.parquet")
    stock_names = pd.read_parquet(meta_path)

    tickers = stock_names["Symbol"].tolist()
    from_date = str(TRAIN_START_DATE)

    # --- Download ---
    stocks: dict[str, pd.DataFrame] = {}
    min_dates, max_dates, activities, missings = [], [], [], []

    for i, ticker in enumerate(tickers):
        pct = round((i + 1) / len(tickers), 3)
        print(f"[Step 02] Downloading {ticker}  ({pct})")
        time.sleep(0.5)
        try:
            df = yf.download(ticker, start=from_date, progress=False, auto_adjust=False)
            if df.empty:
                raise ValueError("Empty dataframe")
            # Flatten multi-level column index if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df = df.rename(columns={"Date": "index", "Adj Close": "Adjusted"})
            cols = ["index", "Open", "High", "Low", "Close", "Volume", "Adjusted"]
            df = df[cols]
            df["index"] = pd.to_datetime(df["index"]).dt.date

            min_dates.append(df["index"].min())
            max_dates.append(df["index"].max())
            tail = df.tail(min(101, len(df)))
            activities.append(float((tail["Volume"] * tail["Close"]).mean()))
            stocks[ticker] = df
            missings.append(int(df.isna().any(axis=1).sum()))
        except Exception as exc:
            print(f"  ⚠ Failed to download {ticker}: {exc}")
            min_dates.append(None)
            max_dates.append(None)
            activities.append(None)
            missings.append(0)

    stock_names["MinDate"] = min_dates
    stock_names["MaxDate"] = max_dates
    stock_names["Activity"] = activities
    stock_names["missings"] = missings

    downloaded = set(stocks.keys())
    missing_tickers = [t for t in tickers if t not in downloaded]
    if missing_tickers:
        print(f"  ⚠ {len(missing_tickers)} stock(s) missing: {missing_tickers}")

    # --- Clean NAs via noisy interpolation ---
    stocks_clean: dict[str, pd.DataFrame] = {}
    for ticker, df in stocks.items():
        na_count = df.drop(columns=["index"]).isna().any(axis=1).sum()
        if na_count > 0:
            print(f"[Step 02] Cleaning {ticker} — {na_count} rows with NAs")
            numeric_cols = [c for c in df.columns if c != "index"]
            cleaned = df.copy()
            for col in numeric_cols:
                cleaned[col] = noisy_interpolation(cleaned[col])
            stocks_clean[ticker] = cleaned
        else:
            stocks_clean[ticker] = df.copy()

    # --- Save ---
    data_path = os.path.join(DATA_DIR, "tickers_data_cleaned.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(stocks_clean, f)
    print(f"[Step 02] Saved cleaned data → {data_path}  ({len(stocks_clean)} tickers)")

    meta_out = os.path.join(DATA_DIR, "tickers_metadata.parquet")
    stock_names.to_parquet(meta_out, index=False)
    print(f"[Step 02] Updated metadata → {meta_out}")


if __name__ == "__main__":
    run()
