"""
Market data sync and cleaning.

Synchronizes OHLCV data into a local SQLite cache, only backfilling
recent missing days for cached tickers, then exports the cleaned
in-memory representation used by the rest of the pipeline.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from .config import DATA_DIR, TRAIN_START_DATE
from .market_data import load_ticker_history, sync_ticker_history


def noisy_interpolation(series: pd.Series) -> pd.Series:
    """Fill NA gaps by interpolation plus small in-series noise."""
    x = series.values.astype(float).copy()
    na_mask = np.isnan(x)
    if not na_mask.any():
        return series.copy()

    intervals: list[list[int]] = []
    current: list[int] = []
    for i, is_na in enumerate(na_mask):
        if is_na:
            current.append(i)
        elif current:
            intervals.append(current)
            current = []
    if current:
        intervals.append(current)

    x_omitted = x[~na_mask]
    noise = np.zeros_like(x)
    for interval in intervals:
        n = len(interval)
        if n <= len(x_omitted):
            start_idx = np.random.randint(0, len(x_omitted) - n + 1)
            x_sub = x_omitted[start_idx : start_idx + n].copy()
            x_sub -= x_sub[0]
            ramp = np.arange(n) / max(n - 1, 1) * x_sub[-1] if n > 1 else np.zeros(n)
            x_sub -= ramp
        else:
            x_sub = np.zeros(n)
        noise[interval] = x_sub

    interpolated = pd.Series(x).interpolate(method="linear").values
    result = interpolated + noise
    return pd.Series(result, index=series.index, name=series.name)


def run() -> None:
    """Incrementally sync stock data, clean NAs, and persist to disk."""

    # --- Load ticker metadata ---
    meta_path = os.path.join(DATA_DIR, "tickers_metadata.parquet")
    stock_names = pd.read_parquet(meta_path)

    tickers = stock_names["Symbol"].tolist()

    # --- Sync from remote into local SQLite cache ---
    stocks: dict[str, pd.DataFrame] = {}
    min_dates, max_dates, activities, missings = [], [], [], []

    for i, ticker in enumerate(tickers):
        pct = round((i + 1) / len(tickers), 3)
        try:
            sync_info = sync_ticker_history(ticker, TRAIN_START_DATE)
            print(
                f"[Step 02] Syncing {ticker} ({pct}) "
                f"[{sync_info['mode']}, from {sync_info['fetch_start']}, rows {sync_info['rows_written']}]"
            )
            df = load_ticker_history(ticker)
            if df.empty:
                raise ValueError("No cached market data after sync")

            min_dates.append(df["index"].min())
            max_dates.append(df["index"].max())
            tail = df.tail(min(101, len(df)))
            activities.append(float((tail["Volume"] * tail["Close"]).mean()))
            stocks[ticker] = df
            missings.append(int(df.isna().any(axis=1).sum()))
        except Exception as exc:
            print(f"  ⚠ Failed to sync {ticker}: {exc}")
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
