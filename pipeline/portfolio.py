"""
Step 09 — Generate Portfolio Allocation.

Loads the ranked forecast, applies the shared M6 baseline portfolio rule,
merges sector + volatility metadata, and writes a 12-month monthly backtest.
"""

from __future__ import annotations

import json
import math
import os
import pickle

import numpy as np
import pandas as pd

from .config import DATA_DIR, FEATURES_DIR, FORECASTS_DIR
from .m6_baseline import MIN_GROSS_EXPOSURE, RANK_COLUMNS, build_m6_baseline_submission
from .m6_metrics import evaluate_submission_from_stocks


def _resolve_latest_forecast_path() -> str:
    forecast_files = [
        os.path.join(FORECASTS_DIR, filename)
        for filename in os.listdir(FORECASTS_DIR)
        if filename.startswith("ranked_forecast_")
        and filename.endswith(".csv")
        and filename != "ranked_forecast_template.csv"
    ]
    if not forecast_files:
        raise FileNotFoundError("No ranked_forecast_*.csv found. Run Step 06 first.")
    return max(forecast_files, key=os.path.getmtime)


def _sort_portfolio_rows(forecast: pd.DataFrame) -> pd.DataFrame:
    sort_columns: list[str] = []
    ascending: list[bool] = []

    if "Invest" in forecast.columns:
        sort_columns.append("Invest")
        ascending.append(False)
    if "Decision" in forecast.columns:
        sort_columns.append("Decision")
        ascending.append(False)
    if "Rank5" in forecast.columns:
        sort_columns.append("Rank5")
        ascending.append(False)
    if "Rank1" in forecast.columns:
        sort_columns.append("Rank1")
        ascending.append(False)

    if not sort_columns:
        return forecast.reset_index(drop=True)
    return forecast.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _compute_period_return(daily_log_returns: list[float]) -> float | None:
    if not daily_log_returns:
        return None
    values = np.asarray(daily_log_returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(np.expm1(values.sum()))


def _run_monthly_backtest(stocks: dict[str, pd.DataFrame]) -> tuple[str, str] | None:
    predictions_path = os.path.join(FEATURES_DIR, "forecast_ranks_all.pkl")
    if not os.path.exists(predictions_path):
        print("[Step 09] No forecast history found for 12M monthly backtest.")
        return None

    with open(predictions_path, "rb") as handle:
        predictions = pickle.load(handle)

    history = predictions.get("meta")
    if not isinstance(history, pd.DataFrame) or history.empty:
        print("[Step 09] No meta prediction history available for monthly backtest.")
        return None

    required = {"Ticker", "IntervalStart", "IntervalEnd", *RANK_COLUMNS}
    missing = sorted(required - set(history.columns))
    if missing:
        print(f"[Step 09] Skipping monthly backtest; prediction history missing columns: {missing}")
        return None

    backtest_frame = history.copy()
    if "Split" in backtest_frame.columns and (backtest_frame["Split"] == "Validation").any():
        backtest_frame = backtest_frame.loc[backtest_frame["Split"] == "Validation"].copy()

    backtest_frame["IntervalStart"] = pd.to_datetime(backtest_frame["IntervalStart"])
    backtest_frame["IntervalEnd"] = pd.to_datetime(backtest_frame["IntervalEnd"])

    intervals = (
        backtest_frame[["IntervalStart", "IntervalEnd"]]
        .drop_duplicates()
        .sort_values(["IntervalEnd", "IntervalStart"])
        .tail(12)
    )
    if intervals.empty:
        print("[Step 09] No intervals available for monthly backtest.")
        return None

    periods: list[dict[str, object]] = []
    for interval in intervals.itertuples(index=False):
        interval_mask = (
            (backtest_frame["IntervalStart"] == interval.IntervalStart)
            & (backtest_frame["IntervalEnd"] == interval.IntervalEnd)
        )
        submission, allocation_summary = build_m6_baseline_submission(
            backtest_frame.loc[interval_mask],
            id_col="Ticker",
        )
        submission = submission[["ID", *RANK_COLUMNS, "Decision"]]
        metrics = evaluate_submission_from_stocks(
            stocks=stocks,
            submission=submission,
            start_date=interval.IntervalStart.date(),
            end_date=interval.IntervalEnd.date(),
        )

        periods.append(
            {
                "PeriodStart": str(interval.IntervalStart.date()),
                "PeriodEnd": str(interval.IntervalEnd.date()),
                "Longs": ",".join(allocation_summary["long_ids"]),
                "Shorts": ",".join(allocation_summary["short_ids"]),
                "GrossExposure": allocation_summary["gross_exposure"],
                "NetExposure": allocation_summary["net_exposure"],
                "MeetsMinExposure": bool(allocation_summary["meets_minimum_exposure"]),
                "PeriodReturn": _compute_period_return(metrics["daily_log_returns"]),
                "RPS": _safe_float(metrics["RPS"]),
                "IR": _safe_float(metrics["IR"]),
            }
        )

    backtest_df = pd.DataFrame(periods)
    backtest_csv_path = os.path.join(FORECASTS_DIR, "portfolio_m6_backtest_12m.csv")
    backtest_df.to_csv(backtest_csv_path, index=False)

    valid_returns = backtest_df["PeriodReturn"].dropna().astype(float)
    valid_rps = backtest_df["RPS"].dropna().astype(float)
    valid_ir = backtest_df["IR"].dropna().astype(float)
    cumulative_return = (
        float(np.prod(1.0 + valid_returns.values) - 1.0)
        if len(valid_returns) > 0
        else None
    )
    summary = {
        "intervals_evaluated": int(len(backtest_df)),
        "lookback_months_requested": 12,
        "minimum_gross_exposure": MIN_GROSS_EXPOSURE,
        "average_period_return": float(valid_returns.mean()) if len(valid_returns) > 0 else None,
        "cumulative_return": cumulative_return,
        "positive_periods": int((valid_returns > 0).sum()) if len(valid_returns) > 0 else 0,
        "average_rps": float(valid_rps.mean()) if len(valid_rps) > 0 else None,
        "average_ir": float(valid_ir.mean()) if len(valid_ir) > 0 else None,
        "periods": periods,
    }
    backtest_json_path = os.path.join(FORECASTS_DIR, "portfolio_m6_backtest_12m_summary.json")
    with open(backtest_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[Step 09] Saved 12M monthly backtest → {backtest_csv_path}")
    print(f"[Step 09] Saved 12M monthly backtest summary → {backtest_json_path}")
    return backtest_csv_path, backtest_json_path


def run(*, forecast_path: str | None = None) -> str:
    # --- Find latest forecast CSV ---
    if forecast_path is None:
        forecast_path = _resolve_latest_forecast_path()
    print(f"[Step 09] Loading forecast: {forecast_path}")
    forecast = pd.read_csv(forecast_path)
    meta = pd.read_parquet(os.path.join(DATA_DIR, "tickers_metadata.parquet"))

    # Drop legacy or utility columns that are no longer part of the baseline output.
    for col in ["Decision", "Weight", "ExpectedRank", "ZScore", "BenchmarkID", "Position", "Invest"]:
        if col in forecast.columns:
            forecast.drop(columns=[col], inplace=True)

    forecast, allocation_summary = build_m6_baseline_submission(forecast)
    print(
        "[Step 09] M6 baseline allocation "
        f"(gross exposure={allocation_summary['gross_exposure']:.4f}, "
        f"longs={allocation_summary['long_ids']}, shorts={allocation_summary['short_ids']})"
    )
    if not allocation_summary["meets_minimum_exposure"]:
        print("[Step 09] Warning: portfolio did not reach the 25% minimum gross exposure.")

    # --- Merge sector info ---
    forecast = forecast.merge(
        meta[["Symbol", "Sector"]].rename(columns={"Symbol": "ID"}),
        on="ID", how="left",
    )

    # --- Merge volatility info ---
    data_path = os.path.join(DATA_DIR, "tickers_data_cleaned.pkl")
    stocks: dict[str, pd.DataFrame] | None = None
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            stocks = pickle.load(f)
        vol = {
            t: np.std(np.diff(np.log(df["Adjusted"].dropna().values)))
            for t, df in stocks.items()
        }
        vol_df = pd.DataFrame({"ID": list(vol.keys()), "Volatility": list(vol.values())})
        forecast = forecast.merge(vol_df, on="ID", how="left")

    forecast = _sort_portfolio_rows(forecast)

    # --- Save ---
    out_path = os.path.join(FORECASTS_DIR, "portfolio_expected_rank_full.csv")
    forecast.to_csv(out_path, index=False)
    print(f"[Step 09] Saved enriched portfolio → {out_path}")

    # --- Display ---
    display_cols = ["ID", "Position", "Decision", "Invest", "Sector", "Volatility"] + RANK_COLUMNS
    display_cols = [c for c in display_cols if c in forecast.columns]
    print(forecast[display_cols].to_string(index=False))

    if stocks is not None:
        _run_monthly_backtest(stocks)
    return out_path


if __name__ == "__main__":
    run()
