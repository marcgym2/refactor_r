"""End-to-end runner for the discovery pipeline."""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .history import (
    aggregate_daily_mentions,
    diagnostics_path,
    ensure_storage,
    full_candidates_csv_path,
    full_candidates_parquet_path,
    load_history,
    persist_raw_mentions,
    top_candidates_path,
    upsert_history,
)
from .market import fetch_market_snapshot
from .scoring import score_candidates
from .sources.news import collect as collect_news
from .sources.reddit import collect as collect_reddit
from .sources.stocktwits import collect as collect_stocktwits


def _coerce_run_date(run_date: date | str | None) -> date:
    if isinstance(run_date, date):
        return run_date
    if isinstance(run_date, str):
        return date.fromisoformat(run_date)
    return datetime.now().date()


def _concat_mentions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    if len(non_empty) == 1:
        return non_empty[0].reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for frame in non_empty:
        records.extend(frame.to_dict("records"))
    return pd.DataFrame.from_records(records)


def _write_outputs(
    *,
    run_date: date,
    scored: pd.DataFrame,
    diagnostics: dict[str, Any],
    config: dict,
) -> dict[str, str]:
    full_csv = full_candidates_csv_path(run_date, config)
    full_parquet = full_candidates_parquet_path(run_date, config)
    top_csv = top_candidates_path(run_date, config)
    diagnostics_json = diagnostics_path(run_date, config)

    scored.to_csv(full_csv, index=False)
    scored.to_parquet(full_parquet, index=False)

    top_columns = [
        "date",
        "rank",
        "symbol",
        "attention_score",
        "mention_count_today",
        "mention_count_vs_5d_baseline",
        "mention_count_vs_20d_baseline",
        "mention_zscore",
        "mention_acceleration_day_over_day",
        "unique_authors",
        "source_breadth",
        "subreddit_breadth",
        "news_count_today",
        "price_change_today",
        "relative_volume",
        "dollar_volume",
        "market_cap",
        "why_ranked_high",
    ]
    top_frame = scored[top_columns].head(int(config["ranking"]["top_k"])).copy()
    top_frame.to_csv(top_csv, index=False)

    Path(diagnostics_json).write_text(json.dumps(diagnostics, indent=2))
    return {
        "full_csv_path": full_csv,
        "full_parquet_path": full_parquet,
        "top_candidates_path": top_csv,
        "diagnostics_path": diagnostics_json,
    }


def run(
    *,
    run_date: date | str | None = None,
    config_path: str | None = "config/discovery.toml",
    mock_mode: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Run the end-of-day discovery pipeline and persist ranked candidates."""

    run_dt = _coerce_run_date(run_date)
    merged_overrides = {"mock_mode": mock_mode}
    if overrides:
        merged_overrides.update(overrides)

    config = load_config(config_path=config_path, overrides=merged_overrides)
    ensure_storage(config)

    diagnostics: dict[str, Any] = {
        "run_date": run_dt.isoformat(),
        "mock_mode": bool(config.get("mock_mode")),
        "source_status": [],
        "warnings": [],
    }

    source_frames: list[pd.DataFrame] = []
    for collector in (collect_stocktwits, collect_reddit):
        try:
            frame, source_diagnostics = collector(run_dt, config)
            source_frames.append(frame)
            diagnostics["source_status"].extend(source_diagnostics)
        except Exception as exc:  # pragma: no cover - top-level resilience
            diagnostics["source_status"].append(
                {"source": collector.__module__.split(".")[-1], "status": "error", "detail": str(exc)}
            )

    provisional_mentions = _concat_mentions(source_frames)
    if provisional_mentions.empty:
        raise RuntimeError("No mentions were collected from enabled sources.")

    provisional_daily = aggregate_daily_mentions(provisional_mentions, run_dt)
    seed_symbols = provisional_daily["symbol"].tolist()

    try:
        news_mentions, news_diagnostics = collect_news(run_dt, config, seed_symbols=seed_symbols)
        diagnostics["source_status"].extend(news_diagnostics)
        if not news_mentions.empty:
            source_frames.append(news_mentions)
    except Exception as exc:  # pragma: no cover - top-level resilience
        diagnostics["source_status"].append({"source": "google_news", "status": "error", "detail": str(exc)})

    raw_mentions = _concat_mentions(source_frames)
    if raw_mentions.empty:
        raise RuntimeError("No mention records remained after source collection.")

    raw_mentions_path_value = persist_raw_mentions(raw_mentions, run_dt, config)
    daily_mentions = aggregate_daily_mentions(raw_mentions, run_dt)
    daily_mentions.to_parquet(config["paths"]["normalized_daily_path"], index=False)

    history = load_history(config)
    updated_history = upsert_history(history, daily_mentions, config)
    market_snapshot, market_diagnostics = fetch_market_snapshot(
        symbols=daily_mentions["symbol"].tolist(),
        run_date=run_dt,
        config=config,
    )
    diagnostics["source_status"].extend(market_diagnostics)

    scored = score_candidates(
        run_date=run_dt,
        updated_history=updated_history,
        market_snapshot=market_snapshot,
        config=config,
    )
    if scored.empty:
        raise RuntimeError("Scoring produced no candidates after optional filters.")

    diagnostics["raw_mentions_path"] = raw_mentions_path_value
    diagnostics["history_path"] = config["paths"]["history_path"]
    diagnostics["candidate_count"] = int(len(scored))
    diagnostics["top_symbols"] = scored[["symbol", "attention_score", "why_ranked_high"]].head(10).to_dict("records")

    return _write_outputs(run_date=run_dt, scored=scored, diagnostics=diagnostics, config=config)
