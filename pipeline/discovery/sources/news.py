"""Optional lightweight news-count adapter using Google News RSS."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import pandas as pd

from ..http import fetch_text
from ..normalize import normalize_author
from . import empty_mentions_frame


def _record(
    *,
    run_date: date,
    symbol: str,
    publisher: str | None,
    mention_time: str | None,
    text_id: str | None,
    body_excerpt: str,
) -> dict[str, Any]:
    return {
        "run_date": run_date.isoformat(),
        "source": "google_news",
        "symbol": symbol,
        "author": normalize_author(publisher),
        "community": normalize_author(publisher),
        "mention_time": mention_time,
        "text_id": text_id,
        "content_type": "news",
        "signal_strength": 1.0,
        "engagement": None,
        "watchlist_count": None,
        "body_excerpt": body_excerpt[:240],
    }


def _load_mock_payload(mock_path: str) -> dict[str, Any]:
    return json.loads(Path(mock_path).read_text())


def _parse_xml(run_date: date, symbol: str, payload: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(payload)
    rows: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        guid = item.findtext("guid") or title
        pub_date = item.findtext("pubDate")
        source = item.findtext("source")
        rows.append(
            _record(
                run_date=run_date,
                symbol=symbol,
                publisher=source,
                mention_time=pub_date,
                text_id=guid,
                body_excerpt=title,
            )
        )
    return rows


def collect(
    run_date: date,
    config: dict,
    seed_symbols: list[str],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Collect optional per-symbol news counts for a small candidate set."""

    source_config = config["sources"]["news"]
    if not source_config.get("enabled", False):
        return empty_mentions_frame(), [{"source": "google_news", "status": "disabled"}]

    if not seed_symbols:
        return empty_mentions_frame(), [{"source": "google_news", "status": "no_seed_symbols"}]

    if config.get("mock_mode"):
        payload = _load_mock_payload(config["mock"]["news_path"])
        rows: list[dict[str, Any]] = []
        for symbol in seed_symbols[: int(source_config["max_symbols"])]:
            for item in payload.get(symbol, []):
                rows.append(
                    _record(
                        run_date=run_date,
                        symbol=symbol,
                        publisher=item.get("publisher"),
                        mention_time=item.get("published_at"),
                        text_id=item.get("id"),
                        body_excerpt=item.get("title", ""),
                    )
                )
        mentions = pd.DataFrame(rows) if rows else empty_mentions_frame()
        return mentions, [
            {
                "source": "google_news",
                "status": "ok",
                "mode": "mock",
                "rows": str(len(mentions)),
            }
        ]

    diagnostics: list[dict[str, str]] = []
    http_config = config["http"]
    rows: list[dict[str, Any]] = []
    lookback_days = int(source_config.get("lookback_days", 2))

    for symbol in seed_symbols[: int(source_config["max_symbols"])]:
        query = quote_plus(f"{symbol} stock when:{lookback_days}d")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        try:
            payload = fetch_text(
                url,
                timeout_seconds=http_config["timeout_seconds"],
                user_agent=http_config["user_agent"],
            )
            symbol_rows = _parse_xml(run_date=run_date, symbol=symbol, payload=payload)
            rows.extend(symbol_rows)
            diagnostics.append(
                {
                    "source": "google_news",
                    "status": "ok",
                    "symbol": symbol,
                    "rows": str(len(symbol_rows)),
                }
            )
        except Exception as exc:  # pragma: no cover - network failures vary
            diagnostics.append(
                {
                    "source": "google_news",
                    "status": "error",
                    "symbol": symbol,
                    "detail": str(exc),
                }
            )

    if not rows:
        return empty_mentions_frame(), diagnostics
    return pd.DataFrame(rows), diagnostics
