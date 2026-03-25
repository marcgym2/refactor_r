from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline import portfolio


def _price_frame(start: str, end: str, daily_return: float) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    prices = 100.0 * np.cumprod(np.full(len(dates), 1.0 + daily_return, dtype=float))
    return pd.DataFrame({"index": dates, "Adjusted": prices})


class PortfolioTest(unittest.TestCase):
    def test_portfolio_applies_fixed_m6_baseline_weights_and_removes_noise_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            forecasts_dir = tmp_path / "forecasts"
            data_dir = tmp_path / "data"
            features_dir = tmp_path / "features"
            forecasts_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            features_dir.mkdir(parents=True, exist_ok=True)

            forecast_path = forecasts_dir / "ranked_forecast_m6.csv"
            forecast_path.write_text(
                "ID,Rank1,Rank2,Rank3,Rank4,Rank5,Decision\n"
                "ABBV,0.05,0.10,0.15,0.25,0.45,0.00\n"
                "AMZN,0.08,0.12,0.15,0.25,0.40,0.00\n"
                "META,0.42,0.20,0.18,0.08,0.12,0.00\n"
                "TSLA,0.48,0.18,0.14,0.15,0.05,0.00\n"
                "IVV,0.20,0.20,0.20,0.20,0.20,0.00\n"
                "XOM,0.18,0.20,0.22,0.18,0.22,0.00\n"
            )

            pd.DataFrame(
                {
                    "Symbol": ["ABBV", "AMZN", "META", "TSLA", "IVV", "XOM"],
                    "Sector": ["Health", "Consumer", "Tech", "Auto", "ETF", "Energy"],
                }
            ).to_parquet(data_dir / "tickers_metadata.parquet", index=False)

            with open(data_dir / "tickers_data_cleaned.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "ABBV": _price_frame("2026-01-01", "2026-03-31", 0.0010),
                        "AMZN": _price_frame("2026-01-01", "2026-03-31", 0.0009),
                        "META": _price_frame("2026-01-01", "2026-03-31", -0.0007),
                        "TSLA": _price_frame("2026-01-01", "2026-03-31", -0.0010),
                        "IVV": _price_frame("2026-01-01", "2026-03-31", 0.0002),
                        "XOM": _price_frame("2026-01-01", "2026-03-31", 0.0001),
                    },
                    handle,
                )

            with patch("pipeline.portfolio.FORECASTS_DIR", str(forecasts_dir)):
                with patch("pipeline.portfolio.DATA_DIR", str(data_dir)):
                    with patch("pipeline.portfolio.FEATURES_DIR", str(features_dir)):
                        output_path = portfolio.run(forecast_path=str(forecast_path))

            result = pd.read_csv(output_path)

            self.assertEqual(result["ID"].tolist()[:4], ["ABBV", "AMZN", "META", "TSLA"])
            self.assertAlmostEqual(float(result.loc[result["ID"] == "ABBV", "Decision"].iloc[0]), 0.0625)
            self.assertAlmostEqual(float(result.loc[result["ID"] == "AMZN", "Decision"].iloc[0]), 0.0625)
            self.assertAlmostEqual(float(result.loc[result["ID"] == "META", "Decision"].iloc[0]), -0.0625)
            self.assertAlmostEqual(float(result.loc[result["ID"] == "TSLA", "Decision"].iloc[0]), -0.0625)
            self.assertTrue(bool(result.loc[result["ID"] == "META", "Invest"].iloc[0]))
            self.assertEqual(result.loc[result["ID"] == "TSLA", "Position"].iloc[0], "Short")
            self.assertEqual(result.loc[result["ID"] == "ABBV", "Sector"].iloc[0], "Health")
            self.assertNotIn("ExpectedRank", result.columns)
            self.assertNotIn("ZScore", result.columns)
            self.assertNotIn("BenchmarkID", result.columns)

    def test_portfolio_writes_12_month_backtest_from_prediction_history(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            forecasts_dir = tmp_path / "forecasts"
            data_dir = tmp_path / "data"
            features_dir = tmp_path / "features"
            forecasts_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            features_dir.mkdir(parents=True, exist_ok=True)

            forecast_path = forecasts_dir / "ranked_forecast_m6.csv"
            forecast_path.write_text(
                "ID,Rank1,Rank2,Rank3,Rank4,Rank5,Decision\n"
                "LONGA,0.05,0.10,0.10,0.20,0.55,0.00\n"
                "LONGB,0.06,0.10,0.14,0.22,0.48,0.00\n"
                "SHORTA,0.52,0.16,0.14,0.10,0.08,0.00\n"
                "SHORTB,0.45,0.18,0.15,0.12,0.10,0.00\n"
            )

            pd.DataFrame(
                {
                    "Symbol": ["LONGA", "LONGB", "SHORTA", "SHORTB"],
                    "Sector": ["Tech", "Finance", "Retail", "Energy"],
                }
            ).to_parquet(data_dir / "tickers_metadata.parquet", index=False)

            with open(data_dir / "tickers_data_cleaned.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "LONGA": _price_frame("2025-01-01", "2026-03-31", 0.0010),
                        "LONGB": _price_frame("2025-01-01", "2026-03-31", 0.0008),
                        "SHORTA": _price_frame("2025-01-01", "2026-03-31", -0.0008),
                        "SHORTB": _price_frame("2025-01-01", "2026-03-31", -0.0006),
                    },
                    handle,
                )

            rows: list[dict[str, object]] = []
            for idx, start in enumerate(pd.date_range("2025-03-01", periods=12, freq="28D")):
                end = start + pd.Timedelta(days=28)
                rows.extend(
                    [
                        {
                            "Split": "Validation",
                            "Ticker": "LONGA",
                            "IntervalStart": start.strftime("%Y-%m-%d"),
                            "IntervalEnd": end.strftime("%Y-%m-%d"),
                            "Rank1": 0.05,
                            "Rank2": 0.10,
                            "Rank3": 0.10,
                            "Rank4": 0.20,
                            "Rank5": 0.55,
                        },
                        {
                            "Split": "Validation",
                            "Ticker": "LONGB",
                            "IntervalStart": start.strftime("%Y-%m-%d"),
                            "IntervalEnd": end.strftime("%Y-%m-%d"),
                            "Rank1": 0.06,
                            "Rank2": 0.10,
                            "Rank3": 0.14,
                            "Rank4": 0.22,
                            "Rank5": 0.48,
                        },
                        {
                            "Split": "Validation",
                            "Ticker": "SHORTA",
                            "IntervalStart": start.strftime("%Y-%m-%d"),
                            "IntervalEnd": end.strftime("%Y-%m-%d"),
                            "Rank1": 0.52,
                            "Rank2": 0.16,
                            "Rank3": 0.14,
                            "Rank4": 0.10,
                            "Rank5": 0.08,
                        },
                        {
                            "Split": "Validation",
                            "Ticker": "SHORTB",
                            "IntervalStart": start.strftime("%Y-%m-%d"),
                            "IntervalEnd": end.strftime("%Y-%m-%d"),
                            "Rank1": 0.45,
                            "Rank2": 0.18,
                            "Rank3": 0.15,
                            "Rank4": 0.12,
                            "Rank5": 0.10,
                        },
                    ]
                )
            with open(features_dir / "forecast_ranks_all.pkl", "wb") as handle:
                pickle.dump({"meta": pd.DataFrame(rows)}, handle)

            with patch("pipeline.portfolio.FORECASTS_DIR", str(forecasts_dir)):
                with patch("pipeline.portfolio.DATA_DIR", str(data_dir)):
                    with patch("pipeline.portfolio.FEATURES_DIR", str(features_dir)):
                        portfolio.run(forecast_path=str(forecast_path))

            backtest_path = forecasts_dir / "portfolio_m6_backtest_12m.csv"
            summary_path = forecasts_dir / "portfolio_m6_backtest_12m_summary.json"

            self.assertTrue(backtest_path.exists())
            self.assertTrue(summary_path.exists())

            backtest = pd.read_csv(backtest_path)
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(len(backtest), 12)
            self.assertTrue(backtest["MeetsMinExposure"].all())
            self.assertTrue((backtest["GrossExposure"] == 0.25).all())
            self.assertEqual(summary["intervals_evaluated"], 12)
            self.assertEqual(summary["minimum_gross_exposure"], 0.25)


if __name__ == "__main__":
    unittest.main()
