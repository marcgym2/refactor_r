from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import pickle
import unittest
from unittest.mock import patch

import pandas as pd

from pipeline import forecast


class ForecastTest(unittest.TestCase):
    def test_forecast_refreshes_template_from_current_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            features_dir = tmp_path / "features"
            forecasts_dir = tmp_path / "forecasts"
            data_dir.mkdir(parents=True, exist_ok=True)
            features_dir.mkdir(parents=True, exist_ok=True)
            forecasts_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({"Symbol": ["ABBV", "IVV"]}).to_parquet(
                data_dir / "tickers_metadata.parquet",
                index=False,
            )
            with open(data_dir / "tickers_data_cleaned.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "ABBV": pd.DataFrame(
                            {
                                "index": ["2026-03-24", "2026-03-25"],
                                "Adjusted": [100.0, 101.0],
                            }
                        ),
                        "IVV": pd.DataFrame(
                            {
                                "index": ["2026-03-24", "2026-03-25"],
                                "Adjusted": [100.0, 102.0],
                            }
                        ),
                    },
                    handle,
                )

            stale_template = forecasts_dir / "ranked_forecast_template.csv"
            stale_template.write_text(
                "ID,Rank1,Rank2,Rank3,Rank4,Rank5,Decision\n"
                "SPY,0.2,0.2,0.2,0.2,0.2,0.01\n"
            )

            predictions = {
                "meta": pd.DataFrame(
                    {
                        "Split": ["Validation", "Validation"],
                        "Ticker": ["ABBV", "IVV"],
                        "IntervalStart": ["2026-03-24", "2026-03-24"],
                        "IntervalEnd": ["2026-03-25", "2026-03-25"],
                        "Rank1": [0.1, 0.2],
                        "Rank2": [0.2, 0.2],
                        "Rank3": [0.2, 0.2],
                        "Rank4": [0.2, 0.2],
                        "Rank5": [0.3, 0.2],
                    }
                )
            }
            with open(features_dir / "forecast_ranks_all.pkl", "wb") as handle:
                pickle.dump(predictions, handle)

            with patch("pipeline.forecast.DATA_DIR", str(data_dir)):
                with patch("pipeline.forecast.FEATURES_DIR", str(features_dir)):
                    with patch("pipeline.forecast.FORECASTS_DIR", str(forecasts_dir)):
                        output_path = forecast.run()

            refreshed_template = pd.read_csv(stale_template)
            submission = pd.read_csv(output_path)
            metrics_path = Path(str(output_path).replace(".csv", "_m6_metrics.json"))

            self.assertEqual(refreshed_template["ID"].tolist(), ["ABBV", "IVV"])
            self.assertEqual(submission["ID"].tolist(), ["ABBV", "IVV"])
            self.assertTrue(metrics_path.exists())


if __name__ == "__main__":
    unittest.main()
