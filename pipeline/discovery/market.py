"""Free market-data enrichment for discovery candidates."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf


SNAPSHOT_COLUMNS = [
    "symbol",
    "price",
    "price_change_today",
    "relative_volume",
    "dollar_volume",
    "market_cap",
    "volume",
    "trade_date",
]


def empty_market_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SNAPSHOT_COLUMNS)


def _snapshot_from_history(
    symbol: str,
    history: pd.DataFrame,
    as_of_date: date,
    volume_baseline_days: int,
) -> dict[str, Any] | None:
    if history.empty:
        return None

    frame = history.copy().reset_index()
    if "Date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "Date"})

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted",
        "Volume": "volume",
        "Date": "trade_date",
    }
    frame = frame.rename(columns=rename_map)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame = frame.loc[frame["trade_date"] <= as_of_date].dropna(subset=["close"]).copy()
    if len(frame) < 2:
        return None

    latest = frame.iloc[-1]
    previous = frame.iloc[-2]
    baseline = frame.iloc[:-1].tail(volume_baseline_days)
    avg_volume = float(baseline["volume"].mean()) if not baseline.empty else 0.0
    latest_volume = float(latest["volume"] or 0.0)
    latest_close = float(latest["close"] or 0.0)
    previous_close = float(previous["close"] or 0.0)

    return {
        "symbol": symbol,
        "price": latest_close,
        "price_change_today": ((latest_close / previous_close) - 1.0) if previous_close else None,
        "relative_volume": (latest_volume / avg_volume) if avg_volume else None,
        "dollar_volume": latest_close * latest_volume,
        "market_cap": None,
        "volume": latest_volume,
        "trade_date": latest["trade_date"].isoformat(),
    }


def _download_histories(symbols: list[str], start_date: date, end_date: date) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}

    downloaded = yf.download(
        tickers=" ".join(symbols),
        start=str(start_date),
        end=str(end_date),
        progress=False,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )
    if downloaded.empty:
        return {}

    if isinstance(downloaded.columns, pd.MultiIndex):
        return {
            symbol: downloaded[symbol].dropna(how="all")
            for symbol in symbols
            if symbol in downloaded.columns.get_level_values(0)
        }

    return {symbols[0]: downloaded.dropna(how="all")}


def _fetch_market_caps(symbols: list[str], limit: int) -> dict[str, float | None]:
    caps: dict[str, float | None] = {}
    for symbol in symbols[:limit]:
        try:
            fast_info = yf.Ticker(symbol).fast_info
            caps[symbol] = fast_info.get("marketCap")
        except Exception:  # pragma: no cover - network failures vary
            caps[symbol] = None
    return caps


def fetch_market_snapshot(
    symbols: list[str],
    run_date: date,
    config: dict,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Fetch free price/liquidity metadata for the current candidate set."""

    if not symbols:
        return empty_market_frame(), [{"source": "market", "status": "no_symbols"}]

    if config.get("mock_mode"):
        frame = pd.read_csv(config["mock"]["market_path"])
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date.astype(str)
        diagnostics = [{"source": "market", "status": "ok", "mode": "mock", "rows": str(len(frame))}]
        return frame[SNAPSHOT_COLUMNS], diagnostics

    market_config = config["market"]
    start_date = run_date - timedelta(days=int(market_config["lookback_days"]))
    end_date = run_date + timedelta(days=1)

    diagnostics: list[dict[str, str]] = []
    try:
        histories = _download_histories(symbols=symbols, start_date=start_date, end_date=end_date)
    except Exception as exc:  # pragma: no cover - yfinance internals vary
        return empty_market_frame(), [{"source": "market", "status": "error", "detail": str(exc)}]

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        snapshot = _snapshot_from_history(
            symbol=symbol,
            history=histories.get(symbol, pd.DataFrame()),
            as_of_date=run_date,
            volume_baseline_days=int(market_config["volume_baseline_days"]),
        )
        if snapshot is None:
            diagnostics.append({"source": "market", "status": "missing", "symbol": symbol})
            continue
        rows.append(snapshot)
        diagnostics.append({"source": "market", "status": "ok", "symbol": symbol})

    frame = pd.DataFrame(rows) if rows else empty_market_frame()

    if market_config.get("include_market_cap", True) and not frame.empty:
        caps = _fetch_market_caps(
            symbols=frame["symbol"].tolist(),
            limit=int(market_config["market_cap_lookup_limit"]),
        )
        frame["market_cap"] = frame["symbol"].map(caps)

    if frame.empty:
        return frame, diagnostics

    return frame[SNAPSHOT_COLUMNS], diagnostics
