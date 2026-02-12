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

from config import DATA_DIR, FORECASTS_DIR


def run() -> None:
    # --- Find latest forecast CSV ---
    forecast_files = sorted(
        [f for f in os.listdir(FORECASTS_DIR) if f.startswith("ranked_forecast_") and f.endswith(".csv")
         and f != "ranked_forecast_template.csv"]
    )
    if not forecast_files:
        raise FileNotFoundError("No ranked_forecast_*.csv found. Run Step 06 first.")

    forecast_path = os.path.join(FORECASTS_DIR, forecast_files[-1])
    print(f"[Step 09] Loading forecast: {forecast_path}")
    forecast = pd.read_csv(forecast_path)

    # Drop utility columns
    for col in ["Decision", "Weight"]:
        if col in forecast.columns:
            forecast.drop(columns=[col], inplace=True)

    # --- Expected rank ---
    rank_cols = ["Rank1", "Rank2", "Rank3", "Rank4", "Rank5"]
    forecast["ExpectedRank"] = sum(
        forecast[f"Rank{i}"] * i for i in range(1, 6)
    )

    # --- Z-score vs SPY ---
    spy = forecast.loc[forecast["ID"] == "SPY", "ExpectedRank"]
    if spy.empty:
        print("  ⚠ SPY not found in forecast — skipping Z-score filtering.")
        spy_rank = forecast["ExpectedRank"].median()
    else:
        spy_rank = spy.values[0]

    sigma = forecast["ExpectedRank"].std()
    forecast["ZScore"] = (forecast["ExpectedRank"] - spy_rank) / sigma

    # --- Flag tickers above threshold ---
    z_threshold = 1.0
    cutoff = spy_rank + z_threshold * sigma
    forecast["Invest"] = forecast["ExpectedRank"] > cutoff

    # --- Merge sector info ---
    meta = pd.read_parquet(os.path.join(DATA_DIR, "tickers_metadata.parquet"))
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

    # --- Save ---
    out_path = os.path.join(FORECASTS_DIR, "portfolio_expected_rank_full.csv")
    forecast.to_csv(out_path, index=False)
    print(f"[Step 09] Saved enriched portfolio → {out_path}")

    # --- Display ---
    display_cols = ["ID", "Sector", "Volatility"] + rank_cols + ["ExpectedRank", "ZScore", "Invest"]
    display_cols = [c for c in display_cols if c in forecast.columns]
    print(forecast.sort_values("ExpectedRank", ascending=False)[display_cols].to_string(index=False))


if __name__ == "__main__":
    run()
