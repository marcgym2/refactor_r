"""
Forecast export and validation.

Loads meta-model quantile predictions, selects the validation split,
builds a submission CSV, validates it, and exports.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from .config import DATA_DIR, FEATURES_DIR, FORECASTS_DIR
from .m6_baseline import RANK_COLUMNS, build_m6_baseline_submission
from .m6_metrics import evaluate_submission_from_stocks


def _build_template(stock_names: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ID": stock_names["Symbol"],
        "Rank1": 0.2,
        "Rank2": 0.2,
        "Rank3": 0.2,
        "Rank4": 0.2,
        "Rank5": 0.2,
        "Decision": 0.0,
    })


def round_preserve_sum(x: np.ndarray, digits: int = 0) -> np.ndarray:
    """Round each element while preserving the total sum."""
    factor = 10 ** digits
    scaled = x * factor
    floored = np.floor(scaled)
    remainders = scaled - floored
    deficit = int(round(scaled.sum())) - int(floored.sum())
    indices = np.argsort(-remainders)[:deficit]
    floored[indices] += 1
    return floored / factor


def validate_submission(
    submission: pd.DataFrame,
    template: pd.DataFrame,
    do_round: bool = False,
) -> pd.DataFrame:
    """Validate a ranked-forecast submission against a template."""
    rank_cols = RANK_COLUMNS

    if do_round:
        orig = submission.copy()
        for i in range(len(submission)):
            row = submission.loc[i, rank_cols].values.astype(float)
            row = row / row.sum()
            submission.loc[i, rank_cols] = round_preserve_sum(row, digits=5)
        submission["Decision"] = round_preserve_sum(
            submission["Decision"].values.astype(float), digits=5
        )
        max_diff = (
            orig[rank_cols + ["Decision"]].values
            - submission[rank_cols + ["Decision"]].values
        )
        print(f"  Max rounding diff: {np.abs(max_diff).max():.2e}")

    assert list(template["ID"]) == list(submission["ID"]), "ID ordering mismatch"
    assert list(template.columns) == list(submission.columns), "Column mismatch"
    row_sums = submission[rank_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8), "Rank probs don't sum to 1"
    assert (submission[rank_cols] >= 0).all().all(), "Negative probabilities"
    assert (submission[rank_cols] <= 1).all().all(), "Probabilities > 1"
    assert submission["Decision"].abs().sum() > 0, "Decision sum is zero"
    assert submission["Decision"].abs().sum() <= 1, "Decision sum exceeds 1"
    return submission


def run() -> str:
    # --- Load metadata ---
    stock_names = pd.read_parquet(os.path.join(DATA_DIR, "tickers_metadata.parquet"))

    # --- Build template from the active metadata ---
    template_path = os.path.join(FORECASTS_DIR, "ranked_forecast_template.csv")
    template = _build_template(stock_names)
    template.to_csv(template_path, index=False)
    print(f"[Step 06] Refreshed template → {template_path}")

    # --- Load quantile predictions ---
    with open(os.path.join(FEATURES_DIR, "forecast_ranks_all.pkl"), "rb") as f:
        predictions = pickle.load(f)

    qp = predictions["meta"]
    val_data = qp[qp["Split"] == "Validation"].copy()

    period = f"{val_data['IntervalStart'].min()} - {val_data['IntervalEnd'].max()}"

    # Average rank predictions per ticker (across intervals and shifts)
    rank_cols = RANK_COLUMNS
    ticker_avg = val_data.groupby("Ticker")[rank_cols].mean().reset_index()
    ticker_avg = ticker_avg.rename(columns={"Ticker": "ID"})

    # Align with template — use uniform 0.2 for any missing tickers.
    submission = template[["ID"]].merge(ticker_avg, on="ID", how="left")
    for col in rank_cols:
        submission[col] = submission[col].fillna(0.2).astype(float)
    submission, allocation_summary = build_m6_baseline_submission(submission)
    submission = submission[["ID", *rank_cols, "Decision"]]

    submission = validate_submission(submission, template, do_round=True)
    print(
        "[Step 06] M6 baseline allocation "
        f"(gross exposure={allocation_summary['gross_exposure']:.4f}, "
        f"longs={allocation_summary['long_ids']}, shorts={allocation_summary['short_ids']})"
    )

    out_path = os.path.join(FORECASTS_DIR, f"ranked_forecast_{period}.csv")
    submission.to_csv(out_path, index=False)
    print(f"[Step 06] Exported forecast → {out_path}")

    data_path = os.path.join(DATA_DIR, "tickers_data_cleaned.pkl")
    if os.path.exists(data_path):
        try:
            with open(data_path, "rb") as handle:
                stocks = pickle.load(handle)
            metrics = evaluate_submission_from_stocks(
                stocks=stocks,
                submission=submission,
                start_date=pd.to_datetime(val_data["IntervalStart"].min()).date(),
                end_date=pd.to_datetime(val_data["IntervalEnd"].max()).date(),
            )
            metrics_path = out_path.replace(".csv", "_m6_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            print(f"[Step 06] Saved M6 metrics → {metrics_path}")
            print(f"[Step 06] M6 RPS: {metrics['RPS']:.5f}  IR: {metrics['IR']:.5f}")
        except Exception as exc:
            print(f"[Step 06] ⚠ Failed to compute M6 metrics: {exc}")
    return out_path


if __name__ == "__main__":
    run()
