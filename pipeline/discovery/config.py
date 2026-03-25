"""Configuration loader for the discovery pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import tomllib


DEFAULT_CONFIG: dict[str, Any] = {
    "timezone": "America/New_York",
    "http": {
        "timeout_seconds": 20,
        "user_agent": "refactor-r-discovery-mvp/0.1",
    },
    "paths": {
        "data_dir": "data/discovery",
        "output_dir": "forecasts/discovery",
        "raw_mentions_dir": "data/discovery/raw_mentions",
        "history_path": "data/discovery/history/daily_attention_history.parquet",
        "normalized_daily_path": "data/discovery/history/daily_attention_latest.parquet",
    },
    "sources": {
        "stocktwits": {
            "enabled": True,
            "trending_limit": 30,
            "stream_limit": 30,
            "excluded_exchanges": ["CRYPTO", "FOREX"],
        },
        "reddit": {
            "enabled": True,
            "subreddits": [
                "wallstreetbets",
                "stocks",
                "investing",
                "pennystocks",
                "options",
            ],
            "listings": ["new", "hot"],
            "listing_limit": 75,
            "include_comments": True,
            "comment_limit": 75,
        },
        "news": {
            "enabled": False,
            "max_symbols": 25,
            "lookback_days": 2,
        },
    },
    "normalization": {
        "allow_bare_symbols": True,
        "bare_symbol_min_len": 2,
        "bare_symbol_max_len": 5,
        "invalid_tokens": [
            "AI",
            "ALL",
            "CEO",
            "CFO",
            "DD",
            "ETF",
            "EPS",
            "FDA",
            "FOMO",
            "GDP",
            "HODL",
            "IMO",
            "IRA",
            "IV",
            "IPO",
            "IRS",
            "LOL",
            "MOON",
            "NYSE",
            "OTC",
            "PDT",
            "SEC",
            "TLDR",
            "USA",
            "USD",
            "YOLO",
        ],
    },
    "market": {
        "lookback_days": 45,
        "volume_baseline_days": 20,
        "include_market_cap": True,
        "market_cap_lookup_limit": 50,
    },
    "thresholds": {
        "min_mentions": 1,
        "apply_price_filter": False,
        "min_price": 0.0,
        "apply_dollar_volume_filter": False,
        "min_dollar_volume": 0.0,
    },
    "ranking": {
        "top_k": 25,
        "weights": {
            "mention_count_today": 0.10,
            "mention_count_vs_5d_baseline": 0.18,
            "mention_count_vs_20d_baseline": 0.18,
            "mention_zscore": 0.18,
            "mention_acceleration_day_over_day": 0.08,
            "unique_authors": 0.08,
            "source_breadth": 0.05,
            "subreddit_breadth": 0.04,
            "optional_news_count_delta": 0.03,
            "price_change_today": 0.10,
            "relative_volume": 0.10,
            "dollar_volume": 0.06,
        },
    },
    "mock": {
        "history_path": "data/discovery/mock/history.csv",
        "stocktwits_path": "data/discovery/mock/stocktwits.json",
        "reddit_path": "data/discovery/mock/reddit.json",
        "news_path": "data/discovery/mock/news.json",
        "market_path": "data/discovery/mock/market_snapshot.csv",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(config)

    for section in ("paths", "mock"):
        for key, value in resolved.get(section, {}).items():
            if isinstance(value, str):
                resolved[section][key] = str(Path(value).expanduser().resolve())

    return resolved


def load_config(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load TOML config and merge it with defaults."""

    config = deepcopy(DEFAULT_CONFIG)
    if config_path is not None:
        with open(config_path, "rb") as handle:
            loaded = tomllib.load(handle)
        config = _deep_merge(config, loaded)

    if overrides:
        config = _deep_merge(config, overrides)

    return _resolve_paths(config)
