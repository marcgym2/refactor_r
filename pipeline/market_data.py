"""
SQLite-backed market data cache for incremental yfinance syncs.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .config import MARKET_DATA_DB


PRICE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    adjusted REAL,
    PRIMARY KEY (ticker, trade_date)
)
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(MARKET_DATA_DB), exist_ok=True)
    conn = sqlite3.connect(MARKET_DATA_DB)
    conn.execute(PRICE_TABLE_DDL)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices (ticker, trade_date)"
    )
    return conn


def _normalize_downloaded_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["trade_date", "open", "high", "low", "close", "volume", "adjusted"]
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    normalized = (
        df.reset_index()
        .rename(
            columns={
                "Date": "trade_date",
                "Adj Close": "adjusted",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )[["trade_date", "open", "high", "low", "close", "volume", "adjusted"]]
        .copy()
    )
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"]).dt.date.astype(str)
    return normalized


def latest_trade_date(ticker: str) -> date | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT MAX(trade_date) FROM prices WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    return date.fromisoformat(row[0])


def sync_ticker_history(ticker: str, start_date: date, lookback_days: int = 7) -> dict[str, object]:
    last_cached = latest_trade_date(ticker)
    if last_cached is None:
        fetch_start = start_date
        mode = "full"
    else:
        fetch_start = max(start_date, last_cached - timedelta(days=lookback_days))
        mode = "incremental"

    fetch_end = date.today() + timedelta(days=1)
    downloaded = yf.download(
        ticker,
        start=str(fetch_start),
        end=str(fetch_end),
        progress=False,
        auto_adjust=False,
    )
    normalized = _normalize_downloaded_frame(downloaded)
    if normalized.empty:
        return {
            "ticker": ticker,
            "mode": mode,
            "fetch_start": fetch_start,
            "rows_written": 0,
            "last_cached": last_cached,
        }

    rows = [
        (
            ticker,
            rec.trade_date,
            rec.open,
            rec.high,
            rec.low,
            rec.close,
            rec.volume,
            rec.adjusted,
        )
        for rec in normalized.itertuples(index=False)
    ]
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices (
                ticker, trade_date, open, high, low, close, volume, adjusted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return {
        "ticker": ticker,
        "mode": mode,
        "fetch_start": fetch_start,
        "rows_written": len(rows),
        "last_cached": latest_trade_date(ticker),
    }


def load_ticker_history(ticker: str) -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                trade_date AS "index",
                open AS "Open",
                high AS "High",
                low AS "Low",
                close AS "Close",
                volume AS "Volume",
                adjusted AS "Adjusted"
            FROM prices
            WHERE ticker = ?
            ORDER BY trade_date
            """,
            conn,
            params=(ticker,),
        )
    if df.empty:
        return df
    df["index"] = pd.to_datetime(df["index"]).dt.date
    return df
