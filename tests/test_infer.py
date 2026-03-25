from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from pipeline import infer


class _FakeModel:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [
                [0.05, 0.10, 0.15, 0.25, 0.45],  # ABBV
                [0.08, 0.12, 0.15, 0.25, 0.40],  # IVV
                [0.42, 0.20, 0.18, 0.08, 0.12],  # META
                [0.48, 0.18, 0.14, 0.15, 0.05],  # TSLA
            ],
            dtype=torch.float32,
        )


class InferTest(unittest.TestCase):
    def test_infer_exports_submission_without_reordering_template_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            forecasts_dir = Path(tmp) / "forecasts"
            forecasts_dir.mkdir(parents=True, exist_ok=True)

            stock_names = pd.DataFrame({"Symbol": ["TSLA", "ABBV", "IVV", "META"]})
            latest = pd.DataFrame(
                {
                    "Ticker": ["ABBV", "IVV", "META", "TSLA"],
                    "Feature1": [1.0, 2.0, 3.0, 4.0],
                }
            )

            with patch("pipeline.infer._load_candidate_stocks", return_value=(stock_names, {})):
                with patch("pipeline.infer._load_feature_names", return_value=["Feature1"]):
                    with patch("pipeline.infer._latest_available_date", return_value=date(2026, 3, 25)):
                        with patch("pipeline.infer._build_latest_feature_frame", return_value=latest):
                            with patch("pipeline.infer._load_base_model", return_value=_FakeModel()):
                                with patch("pipeline.infer.FORECASTS_DIR", str(forecasts_dir)):
                                    output_path = infer.run()

            submission = pd.read_csv(output_path)
            template = pd.read_csv(forecasts_dir / "ranked_forecast_template.csv")

            self.assertEqual(submission["ID"].tolist(), ["TSLA", "ABBV", "IVV", "META"])
            self.assertEqual(template["ID"].tolist(), ["TSLA", "ABBV", "IVV", "META"])
            self.assertGreater(float(submission["Decision"].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
