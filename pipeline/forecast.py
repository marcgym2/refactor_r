"""
Forecast export and validation.

Loads meta-model quantile predictions, selects the validation split,
builds a submission CSV, validates it, and exports.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from .config import DATA_DIR, FEATURES_DIR, FORECASTS_DIR


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
    rank_cols = ["Rank1", "Rank2", "Rank3", "Rank4", "Rank5"]

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


def run() -> None:
    # --- Load metadata ---
    stock_names = pd.read_parquet(os.path.join(DATA_DIR, "tickers_metadata.parquet"))

    # --- Build template if it doesn't exist ---
    template_path = os.path.join(FORECASTS_DIR, "ranked_forecast_template.csv")
    if not os.path.exists(template_path):
        template = pd.DataFrame({
            "ID": stock_names["Symbol"],
            "Rank1": 0.2, "Rank2": 0.2, "Rank3": 0.2, "Rank4": 0.2, "Rank5": 0.2,
            "Decision": 0.01,
        })
        template.to_csv(template_path, index=False)
        print(f"[Step 06] Created template → {template_path}")
    template = pd.read_csv(template_path)

    # --- Load quantile predictions ---
    with open(os.path.join(FEATURES_DIR, "forecast_ranks_all.pkl"), "rb") as f:
        predictions = pickle.load(f)

    qp = predictions["meta"]
    val_data = qp[qp["Split"] == "Validation"].copy()

    period = f"{val_data['IntervalStart'].min()} - {val_data['IntervalEnd'].max()}"

    # Average rank predictions per ticker (across intervals and shifts)
    rank_cols = ["Rank1", "Rank2", "Rank3", "Rank4", "Rank5"]
    ticker_avg = val_data.groupby("Ticker")[rank_cols].mean().reset_index()
    ticker_avg = ticker_avg.rename(columns={"Ticker": "ID"})

    # Align with template — use uniform 0.2 for any missing tickers
    submission = template[["ID"]].merge(ticker_avg, on="ID", how="left")
    for col in rank_cols:
        submission[col] = submission[col].fillna(0.2).astype(float)
    submission["Decision"] = 0.01 * 0.25

    submission = validate_submission(submission, template, do_round=True)

    out_path = os.path.join(FORECASTS_DIR, f"ranked_forecast_{period}.csv")
    submission.to_csv(out_path, index=False)
    print(f"[Step 06] Exported forecast → {out_path}")


if __name__ == "__main__":
    run()
