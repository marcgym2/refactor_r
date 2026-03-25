from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import pickle
import unittest
from unittest.mock import patch

import pandas as pd

from pipeline import portfolio


class PortfolioTest(unittest.TestCase):
    def test_portfolio_uses_explicit_forecast_path(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            forecasts_dir = tmp_path / "forecasts"
            data_dir = tmp_path / "data"
            forecasts_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            fresh = forecasts_dir / "ranked_forecast_2026-03-25.csv"
            stale = forecasts_dir / "ranked_forecast_2026-03-26_inference.csv"

            fresh.write_text(
                "ID,Rank1,Rank2,Rank3,Rank4,Rank5,Decision\n"
                "ABBV,0.10,0.20,0.20,0.20,0.30,0.01\n"
                "SPY,0.20,0.20,0.20,0.20,0.20,0.01\n"
            )
            stale.write_text(
                "ID,Rank1,Rank2,Rank3,Rank4,Rank5,Decision\n"
                "OLD,0.20,0.20,0.20,0.20,0.20,0.01\n"
                "SPY,0.20,0.20,0.20,0.20,0.20,0.01\n"
            )

            pd.DataFrame(
                {
                    "Symbol": ["ABBV", "SPY"],
                    "Sector": ["M6 Equity", "Broad Market"],
                }
            ).to_parquet(data_dir / "tickers_metadata.parquet", index=False)

            with open(data_dir / "tickers_data_cleaned.pkl", "wb") as handle:
                pickle.dump({}, handle)

            with patch("pipeline.portfolio.FORECASTS_DIR", str(forecasts_dir)):
                with patch("pipeline.portfolio.DATA_DIR", str(data_dir)):
                    output_path = portfolio.run(forecast_path=str(fresh))

            result = pd.read_csv(output_path)
            self.assertEqual(result["ID"].tolist(), ["ABBV", "SPY"])
            self.assertEqual(result.loc[result["ID"] == "ABBV", "Sector"].iloc[0], "M6 Equity")

    def test_portfolio_uses_m6_benchmark_from_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            forecasts_dir = tmp_path / "forecasts"
            data_dir = tmp_path / "data"
            forecasts_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            forecast_path = forecasts_dir / "ranked_forecast_m6.csv"
            forecast_path.write_text(
                "ID,Rank1,Rank2,Rank3,Rank4,Rank5,Decision\n"
                "ABBV,0.10,0.20,0.20,0.20,0.30,0.01\n"
                "IVV,0.20,0.20,0.20,0.20,0.20,0.01\n"
            )

            pd.DataFrame(
                {
                    "Symbol": ["ABBV", "IVV"],
                    "Sector": ["M6 Equity", "M6 Fund"],
                    "Benchmark": [False, True],
                }
            ).to_parquet(data_dir / "tickers_metadata.parquet", index=False)

            with open(data_dir / "tickers_data_cleaned.pkl", "wb") as handle:
                pickle.dump({}, handle)

            with patch("pipeline.portfolio.FORECASTS_DIR", str(forecasts_dir)):
                with patch("pipeline.portfolio.DATA_DIR", str(data_dir)):
                    output_path = portfolio.run(forecast_path=str(forecast_path))

            result = pd.read_csv(output_path)
            self.assertEqual(result["BenchmarkID"].iloc[0], "IVV")
            benchmark_row = result.loc[result["ID"] == "IVV"].iloc[0]
            self.assertAlmostEqual(float(benchmark_row["ZScore"]), 0.0, places=12)

    def test_portfolio_saves_rows_with_invest_true_first_then_rank5_desc(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            forecasts_dir = tmp_path / "forecasts"
            data_dir = tmp_path / "data"
            forecasts_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            forecast_path = forecasts_dir / "ranked_forecast_sort.csv"
            forecast_path.write_text(
                "ID,Rank1,Rank2,Rank3,Rank4,Rank5,Decision\n"
                "FALSEHIGHR5,0.50,0.00,0.00,0.10,0.40,0.01\n"
                "TRUELOWR5,0.00,0.10,0.20,0.45,0.25,0.01\n"
                "MID,0.20,0.20,0.20,0.20,0.20,0.01\n"
                "SPY,0.20,0.20,0.20,0.20,0.20,0.01\n"
            )

            pd.DataFrame(
                {
                    "Symbol": ["FALSEHIGHR5", "TRUELOWR5", "MID", "SPY"],
                    "Sector": ["Test", "Test", "Test", "Broad Market"],
                    "Benchmark": [False, False, False, True],
                }
            ).to_parquet(data_dir / "tickers_metadata.parquet", index=False)

            with open(data_dir / "tickers_data_cleaned.pkl", "wb") as handle:
                pickle.dump({}, handle)

            with patch("pipeline.portfolio.FORECASTS_DIR", str(forecasts_dir)):
                with patch("pipeline.portfolio.DATA_DIR", str(data_dir)):
                    output_path = portfolio.run(forecast_path=str(forecast_path))

            result = pd.read_csv(output_path)
            self.assertEqual(
                result["ID"].tolist(),
                ["TRUELOWR5", "FALSEHIGHR5", "MID", "SPY"],
            )
            self.assertTrue(bool(result.loc[result["ID"] == "TRUELOWR5", "Invest"].iloc[0]))
            self.assertFalse(bool(result.loc[result["ID"] == "FALSEHIGHR5", "Invest"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
