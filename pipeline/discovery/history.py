"""History persistence and daily aggregation helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


HISTORY_COLUMNS = [
    "date",
    "symbol",
    "mention_count_today",
    "unique_authors",
    "source_breadth",
    "subreddit_breadth",
    "community_breadth",
    "news_count_today",
    "stocktwits_mentions",
    "reddit_mentions",
    "avg_signal_strength",
    "avg_engagement",
    "max_watchlist_count",
    "source_list",
    "community_list",
]


def ensure_storage(config: dict) -> None:
    paths = config["paths"]
    Path(paths["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["raw_mentions_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["history_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["normalized_daily_path"]).parent.mkdir(parents=True, exist_ok=True)


def raw_mentions_path(run_date: date, config: dict) -> str:
    return str(Path(config["paths"]["raw_mentions_dir"]) / f"mentions_{run_date.isoformat()}.parquet")


def full_candidates_csv_path(run_date: date, config: dict) -> str:
    return str(Path(config["paths"]["output_dir"]) / f"candidates_{run_date.isoformat()}.csv")


def full_candidates_parquet_path(run_date: date, config: dict) -> str:
    return str(Path(config["paths"]["output_dir"]) / f"candidates_{run_date.isoformat()}.parquet")


def top_candidates_path(run_date: date, config: dict) -> str:
    return str(Path(config["paths"]["output_dir"]) / f"top_candidates_{run_date.isoformat()}.csv")


def diagnostics_path(run_date: date, config: dict) -> str:
    return str(Path(config["paths"]["output_dir"]) / f"diagnostics_{run_date.isoformat()}.json")


def persist_raw_mentions(raw_mentions: pd.DataFrame, run_date: date, config: dict) -> str:
    path = raw_mentions_path(run_date, config)
    raw_mentions.to_parquet(path, index=False)
    return path


def _event_key(row: pd.Series) -> str:
    text_id = str(row.get("text_id") or "").strip()
    if text_id:
        return f"{row['source']}|{text_id}|{row['symbol']}"

    mention_time = str(row.get("mention_time") or "")
    return f"{row['source']}|{row['symbol']}|{row['author']}|{mention_time}"


def _sorted_join(values: pd.Series) -> str:
    unique = sorted({value for value in values.dropna().astype(str) if value})
    return ",".join(unique)


def aggregate_daily_mentions(raw_mentions: pd.DataFrame, run_date: date) -> pd.DataFrame:
    """Aggregate raw normalized mention records into one row per symbol."""

    if raw_mentions.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    mentions = raw_mentions.copy()
    mentions["event_key"] = mentions.apply(_event_key, axis=1)
    mentions = mentions.drop_duplicates("event_key").reset_index(drop=True)

    grouped = mentions.groupby("symbol", observed=True)
    daily = grouped.agg(
        mention_count_today=("event_key", "count"),
        unique_authors=("author", "nunique"),
        source_breadth=("source", "nunique"),
        community_breadth=("community", "nunique"),
        news_count_today=("source", lambda s: int((s == "google_news").sum())),
        stocktwits_mentions=("source", lambda s: int((s == "stocktwits").sum())),
        reddit_mentions=("source", lambda s: int((s == "reddit").sum())),
        avg_signal_strength=("signal_strength", "mean"),
        avg_engagement=("engagement", "mean"),
        max_watchlist_count=("watchlist_count", "max"),
        source_list=("source", _sorted_join),
        community_list=("community", _sorted_join),
    ).reset_index()

    reddit_breadth = (
        mentions.loc[mentions["source"] == "reddit"]
        .groupby("symbol", observed=True)["community"]
        .nunique()
        .rename("subreddit_breadth")
        .reset_index()
    )
    daily = daily.merge(reddit_breadth, on="symbol", how="left")
    daily["subreddit_breadth"] = daily["subreddit_breadth"].fillna(0).astype(int)
    daily["date"] = run_date.isoformat()
    daily = daily[HISTORY_COLUMNS]
    daily = daily.sort_values(["mention_count_today", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return daily


def load_history(config: dict) -> pd.DataFrame:
    history_path = Path(config["paths"]["history_path"])
    if history_path.exists():
        history = pd.read_parquet(history_path)
    elif config.get("mock_mode") and Path(config["mock"]["history_path"]).exists():
        history = pd.read_csv(config["mock"]["history_path"])
    else:
        history = pd.DataFrame(columns=HISTORY_COLUMNS)

    if history.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    history = history.copy()
    history["date"] = pd.to_datetime(history["date"]).dt.date.astype(str)
    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = 0
    return history[HISTORY_COLUMNS]


def upsert_history(history: pd.DataFrame, daily_mentions: pd.DataFrame, config: dict) -> pd.DataFrame:
    if history.empty:
        updated = daily_mentions.copy()
    else:
        current_date = daily_mentions["date"].iloc[0]
        updated = history.loc[history["date"] != current_date].copy()
        updated = pd.concat([updated, daily_mentions], ignore_index=True)

    updated = updated.sort_values(["symbol", "date"]).reset_index(drop=True)
    updated.to_parquet(config["paths"]["history_path"], index=False)
    return updated
