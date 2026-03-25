"""
Step 09 — Generate Portfolio Allocation.

Loads the ranked forecast, computes expected rank, Z-scores vs SPY,
flags investment candidates, and merges sector + volatility metadata.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from .config import DATA_DIR, FORECASTS_DIR


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


def _resolve_benchmark_id(forecast: pd.DataFrame, meta: pd.DataFrame) -> str | None:
    if "Benchmark" in meta.columns:
        benchmark_symbols = (
            meta.loc[meta["Benchmark"].fillna(False), "Symbol"].astype(str).tolist()
        )
        for symbol in benchmark_symbols:
            if symbol in set(forecast["ID"].astype(str)):
                return symbol

    for symbol in ["SPY", "IVV"]:
        if symbol in set(forecast["ID"].astype(str)):
            return symbol
    return None


def _sort_portfolio_rows(forecast: pd.DataFrame) -> pd.DataFrame:
    sort_columns: list[str] = []
    ascending: list[bool] = []

    if "Invest" in forecast.columns:
        sort_columns.append("Invest")
        ascending.append(False)
    if "Rank5" in forecast.columns:
        sort_columns.append("Rank5")
        ascending.append(False)

    if not sort_columns:
        return forecast.reset_index(drop=True)
    return forecast.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def run(*, forecast_path: str | None = None) -> str:
    # --- Find latest forecast CSV ---
    if forecast_path is None:
        forecast_path = _resolve_latest_forecast_path()
    print(f"[Step 09] Loading forecast: {forecast_path}")
    forecast = pd.read_csv(forecast_path)
    meta = pd.read_parquet(os.path.join(DATA_DIR, "tickers_metadata.parquet"))

    # Drop utility columns
    for col in ["Decision", "Weight"]:
        if col in forecast.columns:
            forecast.drop(columns=[col], inplace=True)

    # --- Expected rank ---
    rank_cols = ["Rank1", "Rank2", "Rank3", "Rank4", "Rank5"]
    forecast["ExpectedRank"] = sum(
        forecast[f"Rank{i}"] * i for i in range(1, 6)
    )

    # --- Z-score vs active benchmark ---
    benchmark_id = _resolve_benchmark_id(forecast, meta)
    benchmark_label = benchmark_id or "median"
    if benchmark_id is None:
        print("  ⚠ No benchmark asset found in forecast — using median expected rank.")
        benchmark_rank = float(forecast["ExpectedRank"].median())
    else:
        benchmark_rank = float(
            forecast.loc[forecast["ID"] == benchmark_id, "ExpectedRank"].iloc[0]
        )
        print(f"[Step 09] Using benchmark: {benchmark_id}")

    sigma = forecast["ExpectedRank"].std()
    if not np.isfinite(sigma) or sigma == 0:
        forecast["ZScore"] = 0.0
    else:
        forecast["ZScore"] = (forecast["ExpectedRank"] - benchmark_rank) / sigma

    # --- Flag tickers above threshold ---
    z_threshold = 1.0
    cutoff = benchmark_rank + z_threshold * sigma if np.isfinite(sigma) else benchmark_rank
    forecast["Invest"] = forecast["ExpectedRank"] > cutoff
    forecast["BenchmarkID"] = benchmark_label

    # --- Merge sector info ---
    forecast = forecast.merge(
        meta[["Symbol", "Sector"]].rename(columns={"Symbol": "ID"}),
        on="ID", how="left",
    )

    # --- Merge volatility info ---
    data_path = os.path.join(DATA_DIR, "tickers_data_cleaned.pkl")
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
    display_cols = ["ID", "Sector", "Volatility"] + rank_cols + ["ExpectedRank", "ZScore", "Invest"]
    display_cols = [c for c in display_cols if c in forecast.columns]
    print(forecast[display_cols].to_string(index=False))
    return out_path


if __name__ == "__main__":
    run()
