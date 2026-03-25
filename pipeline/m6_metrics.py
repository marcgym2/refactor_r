"""
Utilities for M6-style Ranked Probability Score (RPS) and Information Ratio (IR).
"""

from __future__ import annotations

from datetime import date
import math

import numpy as np
import pandas as pd


RANK_COLUMNS = [f"Rank{i}" for i in range(1, 6)]
TARGET_RANK_COLUMNS = [f"TargetRank{i}" for i in range(1, 6)]


def _bucket_bounds(total_assets: int, bucket_count: int = 5) -> list[tuple[int, int]]:
    positions = np.arange(1, total_assets + 1)
    buckets = np.array_split(positions, bucket_count)
    bounds: list[tuple[int, int]] = []
    for bucket in buckets:
        if len(bucket) == 0:
            bounds.append((1, 0))
        else:
            bounds.append((int(bucket[0]), int(bucket[-1])))
    return bounds


def compute_tie_aware_rank_probabilities(returns: pd.Series) -> pd.DataFrame:
    """
    Map realized returns to M6 rank probabilities.

    Tied assets share the occupied rank positions and therefore split
    probability mass across any quintile buckets their tie range overlaps.
    """

    valid = returns.dropna().astype(float)
    if valid.empty:
        return pd.DataFrame(columns=RANK_COLUMNS, index=returns.index)

    ranking = valid.rename("Return").reset_index()
    asset_col = ranking.columns[0]
    ranking["Position"] = ranking["Return"].rank(method="min", ascending=True).astype(int)

    bounds = _bucket_bounds(len(ranking), bucket_count=len(RANK_COLUMNS))
    probs_by_position: dict[int, np.ndarray] = {}
    for position, count in ranking["Position"].value_counts().sort_index().items():
        start = int(position)
        end = start + int(count) - 1
        probs = np.zeros(len(RANK_COLUMNS), dtype=float)
        for i, (bucket_start, bucket_end) in enumerate(bounds):
            overlap = max(0, min(end, bucket_end) - max(start, bucket_start) + 1)
            probs[i] = overlap / float(count)
        probs_by_position[start] = probs

    for i, col in enumerate(RANK_COLUMNS):
        ranking[col] = ranking["Position"].map(lambda pos, idx=i: probs_by_position[int(pos)][idx])

    ranked = ranking.set_index(asset_col)[RANK_COLUMNS]
    ranked.index = ranked.index.astype(str)
    return ranked.reindex(valid.index.astype(str))


def build_group_target_frame(
    frame: pd.DataFrame,
    *,
    group_col: str = "Interval",
    id_col: str = "Ticker",
    return_col: str = "Return",
) -> pd.DataFrame:
    """Build mergeable fractional M6 target probabilities for each interval group."""

    pieces: list[pd.DataFrame] = []
    grouped = frame[[group_col, id_col, return_col]].dropna(subset=[return_col]).groupby(
        group_col, sort=False, observed=False
    )
    for group_value, group in grouped:
        probs = compute_tie_aware_rank_probabilities(
            group.set_index(id_col)[return_col].rename(return_col)
        )
        if probs.empty:
            continue
        probs = probs.rename(columns=dict(zip(RANK_COLUMNS, TARGET_RANK_COLUMNS))).reset_index()
        probs = probs.rename(columns={"index": id_col})
        probs.insert(0, group_col, group_value)
        pieces.append(probs)

    if not pieces:
        return pd.DataFrame(columns=[group_col, id_col] + TARGET_RANK_COLUMNS)
    return pd.concat(pieces, ignore_index=True)


def _normalize_submission(submission: pd.DataFrame) -> pd.DataFrame:
    required = {"ID", *RANK_COLUMNS, "Decision"}
    missing = sorted(required - set(submission.columns))
    if missing:
        raise ValueError(f"Submission missing required columns: {missing}")

    normalized = submission.copy()
    normalized["ID"] = normalized["ID"].astype(str)
    for column in RANK_COLUMNS + ["Decision"]:
        normalized[column] = normalized[column].astype(float)
    return normalized


def build_hist_data_from_stocks(
    *,
    stocks: dict[str, pd.DataFrame],
    asset_ids: list[str],
    end_date: date,
    price_col: str = "Adjusted",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    end_ts = pd.Timestamp(end_date)
    for asset_id in asset_ids:
        stock = stocks.get(asset_id)
        if stock is None or stock.empty:
            continue
        frame = stock.copy()
        frame["index"] = pd.to_datetime(frame["index"])
        frame = frame.loc[frame["index"] <= end_ts, ["index", price_col]].dropna()
        if frame.empty:
            continue
        frame = frame.rename(columns={"index": "date", price_col: "price"})
        frame["symbol"] = asset_id
        rows.append(frame[["date", "symbol", "price"]])
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "price"])
    return pd.concat(rows, ignore_index=True)


def _build_price_panel(
    hist_data: pd.DataFrame,
    *,
    asset_ids: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if hist_data.empty:
        return pd.DataFrame(columns=asset_ids)

    hist = hist_data.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist["symbol"] = hist["symbol"].astype(str)
    hist["price"] = hist["price"].astype(float)
    hist = hist.loc[(hist["symbol"].isin(asset_ids)) & (hist["date"] <= pd.Timestamp(end_date))]
    if hist.empty:
        return pd.DataFrame(columns=asset_ids)

    panel = hist.pivot_table(index="date", columns="symbol", values="price", aggfunc="last").sort_index()
    panel = panel.reindex(columns=asset_ids).ffill()
    panel = panel.loc[panel.index >= pd.Timestamp(start_date)]
    panel = panel.dropna(how="all")
    if panel.empty:
        return panel

    panel = panel.ffill()
    if panel[asset_ids].isna().any().any():
        missing = sorted(panel.columns[panel.isna().any()].tolist())
        raise ValueError(f"Unable to construct complete price panel for assets: {missing}")
    return panel


def compute_m6_rps(
    *,
    hist_data: pd.DataFrame,
    submission: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> dict[str, object]:
    submission_norm = _normalize_submission(submission)
    asset_ids = submission_norm["ID"].tolist()
    panel = _build_price_panel(hist_data, asset_ids=asset_ids, start_date=start_date, end_date=end_date)
    if len(panel) < 2:
        return {"RPS": math.nan, "details": pd.DataFrame(columns=["ID", "RPS"])}

    realized_returns = (panel.iloc[-1] - panel.iloc[0]) / panel.iloc[0]
    target_probs = compute_tie_aware_rank_probabilities(realized_returns.rename("Return"))
    submission_probs = submission_norm.set_index("ID")[RANK_COLUMNS].reindex(asset_ids)

    target_cdf = target_probs[RANK_COLUMNS].cumsum(axis=1)
    forecast_cdf = submission_probs[RANK_COLUMNS].cumsum(axis=1)
    per_asset_rps = ((target_cdf - forecast_cdf) ** 2).mean(axis=1)

    details = submission_norm[["ID"]].copy()
    details["RPS"] = details["ID"].map(per_asset_rps.to_dict())
    return {"RPS": float(per_asset_rps.mean()), "details": details}


def compute_m6_ir(
    *,
    hist_data: pd.DataFrame,
    submission: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> dict[str, object]:
    submission_norm = _normalize_submission(submission)
    asset_ids = submission_norm["ID"].tolist()
    panel = _build_price_panel(hist_data, asset_ids=asset_ids, start_date=start_date, end_date=end_date)
    if len(panel) < 3:
        return {"IR": math.nan, "details": []}

    weights = submission_norm.set_index("ID")["Decision"].reindex(asset_ids)
    weighted_returns = panel.pct_change().iloc[1:].mul(weights, axis=1).sum(axis=1)
    log_returns = np.log1p(weighted_returns.astype(float))
    if len(log_returns) < 2:
        return {"IR": math.nan, "details": log_returns.tolist()}

    std = float(log_returns.std(ddof=1))
    ir = math.nan if not np.isfinite(std) or std == 0.0 else float(log_returns.sum() / std)
    return {"IR": ir, "details": log_returns.tolist()}


def evaluate_submission_from_stocks(
    *,
    stocks: dict[str, pd.DataFrame],
    submission: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> dict[str, object]:
    submission_norm = _normalize_submission(submission)
    hist_data = build_hist_data_from_stocks(
        stocks=stocks,
        asset_ids=submission_norm["ID"].tolist(),
        end_date=end_date,
    )
    rps = compute_m6_rps(
        hist_data=hist_data,
        submission=submission_norm,
        start_date=start_date,
        end_date=end_date,
    )
    ir = compute_m6_ir(
        hist_data=hist_data,
        submission=submission_norm,
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "period_start": str(start_date),
        "period_end": str(end_date),
        "asset_count": int(len(submission_norm)),
        "RPS": rps["RPS"],
        "IR": ir["IR"],
        "rps_by_id": dict(zip(rps["details"]["ID"], rps["details"]["RPS"])),
        "daily_log_returns": ir["details"],
    }
