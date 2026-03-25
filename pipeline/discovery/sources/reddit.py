"""Reddit finance-community attention adapter."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from math import log1p
from pathlib import Path
from typing import Any

import pandas as pd

from ..http import fetch_json
from ..normalize import extract_symbols, normalize_author
from . import empty_mentions_frame


def _utc_iso(created_utc: float | int | None) -> str | None:
    if created_utc is None:
        return None
    return datetime.fromtimestamp(float(created_utc), tz=timezone.utc).isoformat()


def _record(
    *,
    run_date: date,
    symbol: str,
    subreddit: str,
    author: str | None,
    mention_time: str | None,
    text_id: str | None,
    content_type: str,
    engagement: float,
    body_excerpt: str,
) -> dict[str, Any]:
    return {
        "run_date": run_date.isoformat(),
        "source": "reddit",
        "symbol": symbol,
        "author": normalize_author(author),
        "community": subreddit,
        "mention_time": mention_time,
        "text_id": text_id,
        "content_type": content_type,
        "signal_strength": 1.0,
        "engagement": engagement,
        "watchlist_count": None,
        "body_excerpt": body_excerpt[:240],
    }


def _load_mock_payload(mock_path: str) -> dict[str, Any]:
    return json.loads(Path(mock_path).read_text())


def _parse_children(
    *,
    run_date: date,
    subreddit: str,
    rows: list[dict[str, Any]],
    children: list[dict[str, Any]],
    normalization_config: dict,
    content_type: str,
) -> None:
    for child in children:
        data = child.get("data", {})
        text_parts = [
            data.get("title") or "",
            data.get("selftext") or "",
            data.get("body") or "",
        ]
        text = " ".join(part for part in text_parts if part)
        symbols = extract_symbols(text, normalization_config=normalization_config)
        if not symbols:
            continue

        engagement = 1.0 + log1p(max(float(data.get("score", 0) or 0), 0.0))
        mention_time = _utc_iso(data.get("created_utc"))
        text_id = data.get("name") or data.get("id")
        for symbol in symbols:
            rows.append(
                _record(
                    run_date=run_date,
                    symbol=symbol,
                    subreddit=subreddit,
                    author=data.get("author"),
                    mention_time=mention_time,
                    text_id=str(text_id),
                    content_type=content_type,
                    engagement=engagement,
                    body_excerpt=text,
                )
            )


def _parse_mock_payload(run_date: date, payload: dict[str, Any], config: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    normalization_config = config["normalization"]

    for subreddit, subreddit_payload in payload.get("subreddits", {}).items():
        _parse_children(
            run_date=run_date,
            subreddit=subreddit,
            rows=rows,
            children=subreddit_payload.get("posts", []),
            normalization_config=normalization_config,
            content_type="post",
        )
        _parse_children(
            run_date=run_date,
            subreddit=subreddit,
            rows=rows,
            children=subreddit_payload.get("comments", []),
            normalization_config=normalization_config,
            content_type="comment",
        )

    if not rows:
        return empty_mentions_frame()
    return pd.DataFrame(rows)


def collect(run_date: date, config: dict) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Collect mentions from public Reddit JSON endpoints."""

    source_config = config["sources"]["reddit"]
    if not source_config.get("enabled", True):
        return empty_mentions_frame(), [{"source": "reddit", "status": "disabled"}]

    if config.get("mock_mode"):
        mentions = _parse_mock_payload(
            run_date=run_date,
            payload=_load_mock_payload(config["mock"]["reddit_path"]),
            config=config,
        )
        return mentions, [
            {
                "source": "reddit",
                "status": "ok",
                "mode": "mock",
                "rows": str(len(mentions)),
            }
        ]

    http_config = config["http"]
    normalization_config = config["normalization"]
    diagnostics: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for subreddit in source_config.get("subreddits", []):
        for listing in source_config.get("listings", []):
            url = (
                f"https://www.reddit.com/r/{subreddit}/{listing}.json"
                f"?limit={int(source_config['listing_limit'])}&raw_json=1"
            )
            try:
                payload = fetch_json(
                    url,
                    timeout_seconds=http_config["timeout_seconds"],
                    user_agent=http_config["user_agent"],
                )
                children = payload.get("data", {}).get("children", [])
                _parse_children(
                    run_date=run_date,
                    subreddit=subreddit,
                    rows=rows,
                    children=children,
                    normalization_config=normalization_config,
                    content_type="post",
                )
                diagnostics.append(
                    {
                        "source": "reddit",
                        "status": "ok",
                        "subreddit": subreddit,
                        "listing": listing,
                        "rows": str(len(children)),
                    }
                )
            except Exception as exc:  # pragma: no cover - network failures vary
                diagnostics.append(
                    {
                        "source": "reddit",
                        "status": "error",
                        "subreddit": subreddit,
                        "listing": listing,
                        "detail": str(exc),
                    }
                )

        if source_config.get("include_comments", True):
            comment_url = (
                f"https://www.reddit.com/r/{subreddit}/comments.json"
                f"?limit={int(source_config['comment_limit'])}&raw_json=1"
            )
            try:
                payload = fetch_json(
                    comment_url,
                    timeout_seconds=http_config["timeout_seconds"],
                    user_agent=http_config["user_agent"],
                )
                children = payload.get("data", {}).get("children", [])
                _parse_children(
                    run_date=run_date,
                    subreddit=subreddit,
                    rows=rows,
                    children=children,
                    normalization_config=normalization_config,
                    content_type="comment",
                )
                diagnostics.append(
                    {
                        "source": "reddit",
                        "status": "ok",
                        "subreddit": subreddit,
                        "listing": "comments",
                        "rows": str(len(children)),
                    }
                )
            except Exception as exc:  # pragma: no cover - network failures vary
                diagnostics.append(
                    {
                        "source": "reddit",
                        "status": "error",
                        "subreddit": subreddit,
                        "listing": "comments",
                        "detail": str(exc),
                    }
                )

    if not rows:
        return empty_mentions_frame(), diagnostics

    mentions = pd.DataFrame(rows)
    return mentions, diagnostics
