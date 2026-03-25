"""
Shared M6 baseline portfolio selection helpers.
"""

from __future__ import annotations

import pandas as pd


RANK_COLUMNS = [f"Rank{i}" for i in range(1, 6)]
LONG_SELECTION_COUNT = 2
SHORT_SELECTION_COUNT = 2
LEG_WEIGHT = 0.0625
MIN_GROSS_EXPOSURE = 0.25


def _select_unique_ids(
    frame: pd.DataFrame,
    *,
    score_col: str,
    count: int,
    exclude_ids: set[str] | None = None,
    secondary_col: str | None = None,
) -> list[str]:
    excluded = exclude_ids or set()
    columns = ["ID", score_col]
    ascending = [False]

    if secondary_col is not None:
        columns.append(secondary_col)
        ascending.append(True)

    ranked = frame.loc[~frame["ID"].isin(excluded), columns].copy()
    ranked = ranked.sort_values(
        by=columns[1:] + ["ID"],
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

    long_ids = _select_unique_ids(
        baseline,
        score_col="Rank5",
        count=LONG_SELECTION_COUNT,
        secondary_col="Rank1",
    )
    short_ids = _select_unique_ids(
        baseline,
        score_col="Rank1",
        count=SHORT_SELECTION_COUNT,
        exclude_ids=set(long_ids),
        secondary_col="Rank5",
    )

    baseline["Decision"] = 0.0
    baseline.loc[baseline["ID"].isin(long_ids), "Decision"] = LEG_WEIGHT
    baseline.loc[baseline["ID"].isin(short_ids), "Decision"] = -LEG_WEIGHT

    baseline["Position"] = "Flat"
    baseline.loc[baseline["Decision"] > 0, "Position"] = "Long"
    baseline.loc[baseline["Decision"] < 0, "Position"] = "Short"
    baseline["Invest"] = baseline["Decision"] != 0.0

    gross_exposure = float(baseline["Decision"].abs().sum())
    summary = {
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
