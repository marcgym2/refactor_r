"""Feature generation and ranking for discovery candidates."""

from __future__ import annotations

from datetime import date
from math import log1p

import numpy as np
import pandas as pd


def _rolling_prior_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def _rolling_prior_std(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=2).std()


def _rolling_prior_count(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).count()


def _safe_percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(0.0, index=series.index, dtype=float)
    if numeric.dropna().nunique() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return numeric.rank(pct=True, method="average", na_option="bottom").fillna(0.0)


def _describe_feature(feature: str, row: pd.Series) -> str:
    if feature == "mention_count_vs_5d_baseline":
        return f"mentions {row[feature]:.1f}x vs 5d"
    if feature == "mention_count_vs_20d_baseline":
        return f"mentions {row[feature]:.1f}x vs 20d"
    if feature == "mention_zscore":
        return f"mention z-score {row[feature]:.1f}"
    if feature == "mention_acceleration_day_over_day":
        return f"mentions +{row[feature]:.0f} vs prior day"
    if feature == "relative_volume":
        return f"relative volume {row[feature]:.1f}x"
    if feature == "price_change_today":
        return f"price change {row[feature] * 100:.1f}%"
    if feature == "source_breadth":
        return f"source breadth {row[feature]:.0f}"
    if feature == "subreddit_breadth":
        return f"subreddit breadth {row[feature]:.0f}"
    if feature == "unique_authors":
        return f"{row[feature]:.0f} unique authors"
    if feature == "dollar_volume":
        return f"dollar volume ${row[feature]:,.0f}"
    if feature == "mention_count_today":
        return f"{row[feature]:.0f} mentions today"
    if feature == "optional_news_count_delta":
        return f"news delta {row[feature]:.0f}"
    return feature


def _apply_optional_filters(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    thresholds = config["thresholds"]
    filtered = frame.loc[frame["mention_count_today"] >= int(thresholds["min_mentions"])].copy()

    if thresholds.get("apply_price_filter", False):
        filtered = filtered.loc[
            filtered["price"].isna() | (filtered["price"] >= float(thresholds["min_price"]))
        ].copy()

    if thresholds.get("apply_dollar_volume_filter", False):
        filtered = filtered.loc[
            filtered["dollar_volume"].isna()
            | (filtered["dollar_volume"] >= float(thresholds["min_dollar_volume"]))
        ].copy()

    return filtered


def score_candidates(
    *,
    run_date: date,
    updated_history: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Compute abnormal-attention features and final candidate ranking."""

    if updated_history.empty:
        return pd.DataFrame()

    panel = updated_history.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    grouped_mentions = panel.groupby("symbol", observed=True)["mention_count_today"]
    grouped_news = panel.groupby("symbol", observed=True)["news_count_today"]

    panel["mention_count_5d_baseline"] = grouped_mentions.transform(lambda s: _rolling_prior_mean(s, 5))
    panel["mention_count_20d_baseline"] = grouped_mentions.transform(lambda s: _rolling_prior_mean(s, 20))
    panel["mention_count_20d_std"] = grouped_mentions.transform(lambda s: _rolling_prior_std(s, 20))
    panel["previous_day_mentions"] = grouped_mentions.transform(lambda s: s.shift(1))
    panel["history_days_available"] = grouped_mentions.transform(lambda s: _rolling_prior_count(s, 20))
    panel["news_count_5d_baseline"] = grouped_news.transform(lambda s: _rolling_prior_mean(s, 5))

    current = panel.loc[panel["date"].dt.date == run_date].copy()
    if current.empty:
        return pd.DataFrame()

    current["mention_count_vs_5d_baseline"] = (
        (current["mention_count_today"] + 1.0) / (current["mention_count_5d_baseline"].fillna(0.0) + 1.0)
    )
    current["mention_count_vs_20d_baseline"] = (
        (current["mention_count_today"] + 1.0) / (current["mention_count_20d_baseline"].fillna(0.0) + 1.0)
    )

    zscore = (
        (current["mention_count_today"] - current["mention_count_20d_baseline"].fillna(0.0))
        / current["mention_count_20d_std"].replace(0, np.nan)
    )
    poisson_fallback = (
        (current["mention_count_today"] - current["mention_count_20d_baseline"].fillna(0.0))
        / np.sqrt(current["mention_count_20d_baseline"].fillna(0.0) + 1.0)
    )
    current["mention_zscore"] = zscore.fillna(poisson_fallback).fillna(np.log1p(current["mention_count_today"]))
    current["mention_acceleration_day_over_day"] = (
        current["mention_count_today"] - current["previous_day_mentions"].fillna(0.0)
    )
    current["optional_news_count_delta"] = (
        current["news_count_today"] - current["news_count_5d_baseline"].fillna(0.0)
    )

    if market_snapshot.empty:
        current["price"] = np.nan
        current["price_change_today"] = np.nan
        current["relative_volume"] = np.nan
        current["dollar_volume"] = np.nan
        current["market_cap"] = np.nan
        current["volume"] = np.nan
        current["trade_date"] = pd.NaT
    else:
        current = current.merge(market_snapshot, on="symbol", how="left")

    current = _apply_optional_filters(current, config=config)
    if current.empty:
        return pd.DataFrame()

    weights = config["ranking"]["weights"]
    transform_map = {
        "mention_count_today": lambda s: s.map(lambda value: log1p(max(float(value), 0.0))),
        "unique_authors": lambda s: s.map(lambda value: log1p(max(float(value), 0.0))),
        "dollar_volume": lambda s: s.map(lambda value: log1p(max(float(value), 0.0)) if pd.notna(value) else np.nan),
    }

    contribution_columns: list[str] = []
    for feature, weight in weights.items():
        if feature not in current.columns:
            current[feature] = 0.0

        values = pd.to_numeric(current[feature], errors="coerce")
        if feature in transform_map:
            values = transform_map[feature](values)
        rank_column = f"{feature}_rank"
        contribution_column = f"{feature}_contribution"
        current[rank_column] = _safe_percentile_rank(values)
        current[contribution_column] = current[rank_column] * float(weight)
        contribution_columns.append(contribution_column)

    current["attention_score"] = current[contribution_columns].sum(axis=1)
    current = current.sort_values(
        ["attention_score", "mention_count_today", "unique_authors", "symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    current["rank"] = np.arange(1, len(current) + 1)

    top_reason_columns = [f"{feature}_contribution" for feature in weights]
    reasons: list[str] = []
    for _, row in current.iterrows():
        ordered = sorted(
            (
                (column.replace("_contribution", ""), float(row[column]))
                for column in top_reason_columns
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        fragments = [_describe_feature(feature, row) for feature, value in ordered[:3] if value > 0]
        reasons.append("; ".join(fragments))

    current["why_ranked_high"] = reasons
    current["date"] = current["date"].dt.date.astype(str)
    return current
