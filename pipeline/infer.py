"""
Inference-only forecast generation for a current candidate universe.

Uses the trained base model artifacts and scores only the latest interval
for the current tickers in `data/tickers_metadata.parquet`.
"""

from __future__ import annotations

import os
import pickle
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import DATA_DIR, FEATURES_DIR, FORECASTS_DIR, SUBMISSION_INTERVALS
from .m6_baseline import RANK_COLUMNS, build_m6_baseline_submission
from .m6_metrics import TARGET_RANK_COLUMNS
from .features import (
    TTR_FEATURES,
    compute_return,
    is_etf,
    lag_return,
    lag_volatility,
)
from .forecast import validate_submission
from .training_utils import (
    ConstructFFNN,
    gen_interval_infos,
    gen_stocks_aggr,
    impute_features,
    standardize_features,
)


def _load_feature_names() -> list[str]:
    feature_frame = pd.read_parquet(os.path.join(FEATURES_DIR, "features_standardized.parquet"))
    exclude_cols = {
        "Ticker", "Interval", "Return", "Shift",
        "ReturnQuintile", "IntervalStart", "IntervalEnd", *TARGET_RANK_COLUMNS,
    }
    return [column for column in feature_frame.columns if column not in exclude_cols]


def _load_candidate_stocks() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    stock_names = pd.read_parquet(os.path.join(DATA_DIR, "tickers_metadata.parquet"))
    with open(os.path.join(DATA_DIR, "tickers_data_cleaned.pkl"), "rb") as handle:
        stocks = pickle.load(handle)
    stocks = dict(sorted(stocks.items()))
    return stock_names, stocks


def _latest_available_date(stocks: dict[str, pd.DataFrame]) -> date:
    latest_dates = [
        pd.to_datetime(frame["index"]).max().date()
        for frame in stocks.values()
        if not frame.empty
    ]
    if not latest_dates:
        raise RuntimeError("No candidate market data available for inference.")
    return max(latest_dates)


def _build_latest_feature_frame(
    *,
    stocks: dict[str, pd.DataFrame],
    stock_names: pd.DataFrame,
    feature_names: list[str],
    as_of_date: date,
) -> pd.DataFrame:
    feature_fns = [
        compute_return,
        lambda df, t: lag_volatility(df, t, lags=list(range(1, 8))),
        lambda df, t: lag_return(df, t, lags=list(range(1, 8))),
        lambda df, t: is_etf(df, t, stock_names=stock_names),
    ] + TTR_FEATURES

    interval_infos = gen_interval_infos(
        submission=SUBMISSION_INTERVALS,
        shifts=[0],
        time_end=as_of_date,
        total_intervals=40,
    )
    stocks_aggr = gen_stocks_aggr(stocks, interval_infos, feature_fns, check_leakage=False)
    if stocks_aggr.empty:
        raise RuntimeError("Inference feature aggregation produced no rows.")

    stocks_aggr["IntervalEnd"] = pd.to_datetime(stocks_aggr["IntervalEnd"])
    latest_end = stocks_aggr["IntervalEnd"].max()
    latest = stocks_aggr.loc[stocks_aggr["IntervalEnd"] == latest_end].copy()
    latest = latest.sort_values("Ticker").reset_index(drop=True)

    for feature_name in feature_names:
        if feature_name not in latest.columns:
            latest[feature_name] = np.nan

    latest = impute_features(latest, feature_names)
    std_features = [feature_name for feature_name in feature_names if feature_name != "ETF"]
    latest = standardize_features(latest, std_features)
    return latest


def _load_base_model(input_size: int) -> ConstructFFNN:
    layer_sizes = [32, 8, 5]
    layer_dropouts = [0.2] * (len(layer_sizes) - 1) + [0.0]
    layer_transforms = [F.leaky_relu] * (len(layer_sizes) - 1) + [
        lambda x: F.softmax(x, dim=1)
    ]
    model = ConstructFFNN(input_size, layer_sizes, layer_transforms, layer_dropouts)
    state_dict = torch.load(os.path.join(FEATURES_DIR, "model_base.pt"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run(as_of_date: date | None = None) -> str:
    """Generate a forecast for the current candidate universe using saved model artifacts."""

    stock_names, stocks = _load_candidate_stocks()
    feature_names = _load_feature_names()
    run_date = as_of_date or _latest_available_date(stocks)

    latest = _build_latest_feature_frame(
        stocks=stocks,
        stock_names=stock_names,
        feature_names=feature_names,
        as_of_date=run_date,
    )
    if latest.empty:
        raise RuntimeError("No latest interval rows were produced for inference.")

    model = _load_base_model(len(feature_names))
    x = torch.tensor(latest[feature_names].values, dtype=torch.float32)
    with torch.no_grad():
        pred_np = model(x).numpy()

    prediction_rows = pd.DataFrame({
        "ID": latest["Ticker"].values,
        "Rank1": pred_np[:, 0],
        "Rank2": pred_np[:, 1],
        "Rank3": pred_np[:, 2],
        "Rank4": pred_np[:, 3],
        "Rank5": pred_np[:, 4],
    })

    template = pd.DataFrame({
        "ID": stock_names["Symbol"].astype(str).str.upper().tolist(),
        "Rank1": 0.2,
        "Rank2": 0.2,
        "Rank3": 0.2,
        "Rank4": 0.2,
        "Rank5": 0.2,
        "Decision": 0.0,
    })
    template_path = os.path.join(FORECASTS_DIR, "ranked_forecast_template.csv")
    template.to_csv(template_path, index=False)
    print(f"[Step 06] Refreshed template → {template_path}")
    submission = template[["ID"]].merge(prediction_rows, on="ID", how="left")
    for rank_col in RANK_COLUMNS:
        submission[rank_col] = submission[rank_col].fillna(0.2).astype(float)
    submission, allocation_summary = build_m6_baseline_submission(submission)
    submission = submission[["ID", *RANK_COLUMNS, "Decision"]]
    submission = validate_submission(submission, template, do_round=True)
    print(
        "[Step 06] M6 baseline allocation "
        f"(gross exposure={allocation_summary['gross_exposure']:.4f}, "
        f"longs={allocation_summary['long_ids']}, shorts={allocation_summary['short_ids']})"
    )

    out_path = os.path.join(FORECASTS_DIR, f"ranked_forecast_{run_date.isoformat()}_inference.csv")
    submission.to_csv(out_path, index=False)
    print(f"[Step 06] Exported inference forecast → {out_path}")
    return out_path


if __name__ == "__main__":
    run()
