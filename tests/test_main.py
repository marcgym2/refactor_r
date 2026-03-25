from __future__ import annotations

from argparse import Namespace
from datetime import date
import unittest
from unittest.mock import patch

import main


class MainSplitFlowTest(unittest.TestCase):
    def test_m6_split_flow_trains_on_m6_and_infers_on_m6_plus_discovery(self) -> None:
        args = Namespace(
            train_universe="m6",
            candidate_file=None,
            discovery_date="2026-03-24",
            top_k=10,
            use_full_candidates=False,
            exclude_spy=False,
        )

        with patch("main.universe.run") as mock_universe_run:
            with patch("main.ingest.run") as mock_ingest_run:
                with patch("main.train.run") as mock_train_run:
                    with patch("main.infer.run", return_value="forecasts/ranked_forecast_2026-03-25_inference.csv") as mock_infer_run:
                        with patch("main.portfolio.run") as mock_portfolio_run:
                            main._run_split_inference_flow(args)

        self.assertEqual(mock_universe_run.call_count, 2)
        self.assertEqual(
            mock_universe_run.call_args_list[0].kwargs,
            {"universe_mode": "m6"},
        )
        self.assertEqual(
            mock_universe_run.call_args_list[1].kwargs,
            {
                "universe_mode": "m6",
                "candidate_file": None,
                "discovery_date": "2026-03-24",
                "top_k": 10,
                "include_spy": False,
                "use_full_candidates": False,
                "merge_candidates_with_base": True,
            },
        )
        self.assertEqual(mock_ingest_run.call_count, 2)
        self.assertEqual(
            mock_ingest_run.call_args_list[0].kwargs,
            {"start_date": date(2022, 1, 1)},
        )
        self.assertEqual(
            mock_ingest_run.call_args_list[1].kwargs,
            {"start_date": date(2022, 1, 1)},
        )
        mock_train_run.assert_called_once_with()
        mock_infer_run.assert_called_once_with()
        mock_portfolio_run.assert_called_once_with(
            forecast_path="forecasts/ranked_forecast_2026-03-25_inference.csv"
        )


if __name__ == "__main__":
    unittest.main()
