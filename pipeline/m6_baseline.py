"""
Shared M6 baseline portfolio selection helpers.
"""

from __future__ import annotations

import pandas as pd


RANK_COLUMNS = [f"Rank{i}" for i in range(1, 6)]

# Active strategy parameters.
LONG_SCORE = "spread"
SHORT_SCORE = "Rank1"
LONG_SELECTION_COUNT = 1
SHORT_SELECTION_COUNT = 0
TARGET_GROSS_EXPOSURE = 1.0
LONG_GROSS_SHARE = 1.0
MIN_GROSS_EXPOSURE = 0.25


def _compute_score(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "spread":
        return frame["Rank5"] - frame["Rank1"]
    if metric == "expected_rank":
        return sum(frame[f"Rank{i}"] * i for i in range(1, 6))
    if metric == "tail5":
        return frame["Rank5"] + 0.5 * frame["Rank4"]
    if metric == "tail1":
        return frame["Rank1"] + 0.5 * frame["Rank2"]
    if metric in frame.columns:
        return frame[metric]
    raise ValueError(f"Unknown score metric: {metric}")


def _select_unique_ids(
    frame: pd.DataFrame,
    *,
    score: pd.Series,
    count: int,
    exclude_ids: set[str] | None = None,
    secondary_score: pd.Series | None = None,
) -> list[str]:
    if count <= 0:
        return []

    excluded = exclude_ids or set()
    ranked = frame.loc[~frame["ID"].isin(excluded), ["ID"]].copy()
    ranked["PrimaryScore"] = score.loc[ranked.index].astype(float).values
    sort_columns = ["PrimaryScore"]
    ascending = [False]

    if secondary_score is not None:
        ranked["SecondaryScore"] = secondary_score.loc[ranked.index].astype(float).values
        sort_columns.append("SecondaryScore")
        ascending.append(True)

    ranked = ranked.sort_values(
        by=sort_columns + ["ID"],
        ascending=ascending + [True],
        kind="mergesort",
    )
    return ranked["ID"].astype(str).head(count).tolist()


def apply_m6_baseline_portfolio(forecast: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"ID", *RANK_COLUMNS}
    missing = sorted(required - set(forecast.columns))
    if missing:
        raise ValueError(f"Forecast missing required columns: {missing}")

    baseline = forecast.copy()
    baseline["ID"] = baseline["ID"].astype(str)
    for column in RANK_COLUMNS:
        baseline[column] = baseline[column].astype(float)

    long_score = _compute_score(baseline, LONG_SCORE)
    short_score = _compute_score(baseline, SHORT_SCORE)
    long_ids = _select_unique_ids(
        baseline,
        score=long_score,
        count=LONG_SELECTION_COUNT,
        secondary_score=short_score,
    )
    short_ids = _select_unique_ids(
        baseline,
        score=short_score,
        count=SHORT_SELECTION_COUNT,
        exclude_ids=set(long_ids),
        secondary_score=long_score,
    )

    baseline["Decision"] = 0.0
    long_gross = TARGET_GROSS_EXPOSURE if SHORT_SELECTION_COUNT == 0 else TARGET_GROSS_EXPOSURE * LONG_GROSS_SHARE
    short_gross = 0.0 if SHORT_SELECTION_COUNT == 0 else TARGET_GROSS_EXPOSURE - long_gross
    if long_ids:
        baseline.loc[baseline["ID"].isin(long_ids), "Decision"] = long_gross / float(len(long_ids))
    if short_ids and short_gross > 0.0:
        baseline.loc[baseline["ID"].isin(short_ids), "Decision"] = -short_gross / float(len(short_ids))

    baseline["Position"] = "Flat"
    baseline.loc[baseline["Decision"] > 0, "Position"] = "Long"
    baseline.loc[baseline["Decision"] < 0, "Position"] = "Short"
    baseline["Invest"] = baseline["Decision"] != 0.0

    gross_exposure = float(baseline["Decision"].abs().sum())
    summary = {
        "long_score": LONG_SCORE,
        "short_score": SHORT_SCORE,
        "long_ids": long_ids,
        "short_ids": short_ids,
        "gross_exposure": gross_exposure,
        "net_exposure": float(baseline["Decision"].sum()),
        "meets_minimum_exposure": gross_exposure + 1e-12 >= MIN_GROSS_EXPOSURE,
    }
    return baseline, summary


def build_m6_baseline_submission(
    frame: pd.DataFrame,
    *,
    id_col: str = "ID",
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {id_col, *RANK_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction frame missing required columns: {missing}")

    grouped = (
        frame[[id_col, *RANK_COLUMNS]]
        .copy()
        .rename(columns={id_col: "ID"})
        .groupby("ID", as_index=False)[RANK_COLUMNS]
        .mean()
    )
    submission, summary = apply_m6_baseline_portfolio(grouped)
    return submission[["ID", *RANK_COLUMNS, "Decision", "Position", "Invest"]], summary
