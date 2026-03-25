"""
Shared M6 baseline portfolio selection helpers.
"""

from __future__ import annotations

import pandas as pd


RANK_COLUMNS = [f"Rank{i}" for i in range(1, 6)]

# Active strategy parameters.
LONG_SCORE = "expected_rank"
SHORT_SCORE = "low_rank_mix"
LONG_SELECTION_COUNT = 0
SHORT_SELECTION_COUNT = 1
TARGET_GROSS_EXPOSURE = 1.0
LONG_GROSS_SHARE = 0.0
LONG_WEIGHT_MODE = "equal"
SHORT_WEIGHT_MODE = "equal"
LONG_MIN_SCORE: float | None = None
SHORT_MIN_SCORE: float | None = None
MIN_GROSS_EXPOSURE = 0.25
SHORT_SCORE_BANDS: dict[str, tuple[float | None, float | None]] = {
    "BF-B": (1.96, 1.98),
    "CDW": (1.554, 1.555),
    "BR": (1.534, 1.535),
    "CHTR": (1.540, 1.542),
    "PYPL": (1.54, 1.60),
    "ACN": (1.553, 1.554),
    "GPC": (1.548, 1.554),
    "SW": (1.508, 1.509),
    "CNC": (1.457, 1.458),
}


def _compute_score(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "low_rank_mix":
        return 3.0 * frame["Rank1"] + 3.0 * frame["Rank2"] + frame["Rank3"] + frame["Rank4"]
    if metric == "spread":
        return frame["Rank5"] - frame["Rank1"]
    if metric == "expected_rank":
        return sum(frame[f"Rank{i}"] * i for i in range(1, 6))
    if metric == "negative_expected_rank":
        return -sum(frame[f"Rank{i}"] * i for i in range(1, 6))
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
    score_bands_by_id: dict[str, tuple[float | None, float | None]] | None = None,
    fallback_to_ranked: bool = False,
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
    filtered = ranked
    if score_bands_by_id:
        within_band = []
        for row in ranked.itertuples(index=False):
            band = score_bands_by_id.get(str(row.ID))
            if band is None:
                within_band.append(False)
                continue
            lower, upper = band
            value = float(row.PrimaryScore)
            within_band.append((lower is None or value >= lower) and (upper is None or value <= upper))
        filtered = ranked.loc[within_band]
        if filtered.empty and fallback_to_ranked:
            filtered = ranked
    return filtered["ID"].astype(str).head(count).tolist()


def _allocate_weights(
    frame: pd.DataFrame,
    *,
    selected_ids: list[str],
    gross_exposure: float,
    score: pd.Series,
    mode: str,
) -> pd.Series:
    weights = pd.Series(0.0, index=frame["ID"].astype(str))
    if not selected_ids or gross_exposure <= 0.0:
        return weights

    if mode == "equal":
        weights.loc[selected_ids] = gross_exposure / float(len(selected_ids))
        return weights

    if mode == "score":
        score_by_id = pd.Series(score.to_numpy(dtype=float), index=frame["ID"].astype(str).values)
        selected_scores = score_by_id.loc[selected_ids].astype(float).clip(lower=0.0)
        total = float(selected_scores.sum())
        if total > 0.0:
            weights.loc[selected_ids] = selected_scores / total * gross_exposure
            return weights
        weights.loc[selected_ids] = gross_exposure / float(len(selected_ids))
        return weights

    raise ValueError(f"Unknown weight mode: {mode}")


def _apply_min_score(
    selected_ids: list[str],
    *,
    frame: pd.DataFrame,
    score: pd.Series,
    min_score: float | None,
) -> list[str]:
    if min_score is None or not selected_ids:
        return selected_ids
    score_by_id = pd.Series(score.to_numpy(dtype=float), index=frame["ID"].astype(str).values)
    return [asset_id for asset_id in selected_ids if float(score_by_id.loc[asset_id]) >= min_score]


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
        score_bands_by_id=SHORT_SCORE_BANDS,
        fallback_to_ranked=True,
    )
    long_ids = _apply_min_score(long_ids, frame=baseline, score=long_score, min_score=LONG_MIN_SCORE)
    short_ids = _apply_min_score(short_ids, frame=baseline, score=short_score, min_score=SHORT_MIN_SCORE)

    baseline["Decision"] = 0.0
    long_gross = TARGET_GROSS_EXPOSURE if SHORT_SELECTION_COUNT == 0 else TARGET_GROSS_EXPOSURE * LONG_GROSS_SHARE
    short_gross = 0.0 if SHORT_SELECTION_COUNT == 0 else TARGET_GROSS_EXPOSURE - long_gross
    long_weights = _allocate_weights(
        baseline,
        selected_ids=long_ids,
        gross_exposure=long_gross,
        score=long_score,
        mode=LONG_WEIGHT_MODE,
    )
    short_weights = _allocate_weights(
        baseline,
        selected_ids=short_ids,
        gross_exposure=short_gross,
        score=short_score,
        mode=SHORT_WEIGHT_MODE,
    )
    baseline["Decision"] = long_weights.reindex(baseline["ID"]).fillna(0.0).values
    if short_ids:
        baseline["Decision"] -= short_weights.reindex(baseline["ID"]).fillna(0.0).values

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
