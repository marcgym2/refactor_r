"""Stocktwits-based attention adapter."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..http import fetch_json
from ..normalize import normalize_author, normalize_symbol
from . import empty_mentions_frame


def _record(
    *,
    run_date: date,
    symbol: str,
    author: str | None,
    mention_time: str | None,
    text_id: str | None,
    body_excerpt: str | None,
    watchlist_count: float | None,
    signal_strength: float = 1.0,
    engagement: float | None = None,
) -> dict[str, Any]:
    return {
        "run_date": run_date.isoformat(),
        "source": "stocktwits",
        "symbol": symbol,
        "author": normalize_author(author),
        "community": "stocktwits",
        "mention_time": mention_time,
        "text_id": text_id,
        "content_type": "message",
        "signal_strength": signal_strength,
        "engagement": engagement,
        "watchlist_count": watchlist_count,
        "body_excerpt": (body_excerpt or "")[:240],
    }


def _load_mock_payload(mock_path: str) -> dict[str, Any]:
    return json.loads(Path(mock_path).read_text())


def _parse_mock_payload(run_date: date, payload: dict[str, Any], config: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    invalid_tokens = set(config["normalization"]["invalid_tokens"])
    max_len = int(config["normalization"]["bare_symbol_max_len"])
    min_len = int(config["normalization"]["bare_symbol_min_len"])

    for item in payload.get("trending_symbols", []):
        symbol = normalize_symbol(
            item.get("symbol"),
            invalid_tokens=invalid_tokens,
            max_len=max_len,
            min_len=min_len,
            allow_single_char=True,
        )
        if not symbol:
            continue
        watchlist_count = item.get("watchlist_count")
        for message in payload.get("streams", {}).get(symbol, []):
            rows.append(
                _record(
                    run_date=run_date,
                    symbol=symbol,
                    author=message.get("author"),
                    mention_time=message.get("created_at"),
                    text_id=str(message.get("id")),
                    body_excerpt=message.get("body"),
                    watchlist_count=watchlist_count,
                )
            )

    if not rows:
        return empty_mentions_frame()

    return pd.DataFrame(rows)


def collect(run_date: date, config: dict) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Collect symbol-level message activity from Stocktwits."""

    source_config = config["sources"]["stocktwits"]
    if not source_config.get("enabled", True):
        return empty_mentions_frame(), [{"source": "stocktwits", "status": "disabled"}]

    if config.get("mock_mode"):
        mentions = _parse_mock_payload(
            run_date=run_date,
            payload=_load_mock_payload(config["mock"]["stocktwits_path"]),
            config=config,
        )
        return mentions, [
            {
                "source": "stocktwits",
                "status": "ok",
                "mode": "mock",
                "rows": str(len(mentions)),
            }
        ]

    http_config = config["http"]
    diagnostics: list[dict[str, str]] = []
    invalid_tokens = set(config["normalization"]["invalid_tokens"])
    max_len = int(config["normalization"]["bare_symbol_max_len"])
    min_len = int(config["normalization"]["bare_symbol_min_len"])

    trending_payload = fetch_json(
        "https://api.stocktwits.com/api/2/trending/symbols.json",
        timeout_seconds=http_config["timeout_seconds"],
        user_agent=http_config["user_agent"],
    )
    excluded_exchanges = set(source_config.get("excluded_exchanges", []))

    candidate_symbols: list[tuple[str, float | None]] = []
    for item in trending_payload.get("symbols", [])[: int(source_config["trending_limit"])]:
        if item.get("exchange") in excluded_exchanges:
            continue
        symbol = normalize_symbol(
            item.get("symbol"),
            invalid_tokens=invalid_tokens,
            max_len=max_len,
            min_len=min_len,
            allow_single_char=True,
        )
        if symbol:
            candidate_symbols.append((symbol, item.get("watchlist_count")))

    rows: list[dict[str, Any]] = []
    for symbol, fallback_watchlist in candidate_symbols:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json?limit={int(source_config['stream_limit'])}"
        try:
            stream_payload = fetch_json(
                url,
                timeout_seconds=http_config["timeout_seconds"],
                user_agent=http_config["user_agent"],
            )
            symbol_info = stream_payload.get("symbol", {})
            watchlist_count = symbol_info.get("watchlist_count", fallback_watchlist)
            for message in stream_payload.get("messages", []):
                rows.append(
                    _record(
                        run_date=run_date,
                        symbol=symbol,
                        author=(message.get("user") or {}).get("username"),
                        mention_time=message.get("created_at"),
                        text_id=str(message.get("id")),
                        body_excerpt=message.get("body"),
                        watchlist_count=watchlist_count,
                    )
                )
            diagnostics.append(
                {
                    "source": "stocktwits",
                    "status": "ok",
                    "symbol": symbol,
                    "rows": str(len(stream_payload.get("messages", []))),
                }
            )
        except Exception as exc:  # pragma: no cover - network failures vary
            diagnostics.append(
                {
                    "source": "stocktwits",
                    "status": "error",
                    "symbol": symbol,
                    "detail": str(exc),
                }
            )

    if not rows:
        return empty_mentions_frame(), diagnostics

    return pd.DataFrame(rows), diagnostics
