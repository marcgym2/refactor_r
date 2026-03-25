from __future__ import annotations

from argparse import Namespace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import main


class MainSplitFlowTest(unittest.TestCase):
    def test_parser_defaults_to_m6_universe(self) -> None:
        args = main.build_parser().parse_args([])

        self.assertEqual(args.train_universe, "m6")

    def test_refresh_discovery_snapshot_uses_today_for_default_m6_run(self) -> None:
        args = Namespace(
            train_universe="m6",
            candidate_file=None,
            discovery_date=None,
            top_k=None,
            use_full_candidates=False,
            exclude_spy=False,
        )

        with patch("main.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 25)
            with patch("main.run_discovery", return_value={"top_candidates_path": "forecasts/discovery/top_candidates_2026-03-25.csv"}) as mock_run:
                with patch("main._discovery_snapshot_is_usable", return_value=True):
                    resolved = main._refresh_discovery_snapshot(args)

        self.assertEqual(resolved.discovery_date, "2026-03-25")
        mock_run.assert_called_once_with(run_date="2026-03-25")

    def test_resolve_latest_discovery_date_skips_invalid_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            discovery_dir = Path(tmp) / "discovery"
            discovery_dir.mkdir(parents=True, exist_ok=True)

            invalid = pd.DataFrame(
                {
                    "symbol": ["US", "EV", "SPX"],
                    "relative_volume": [None, None, None],
                    "dollar_volume": [None, None, None],
                    "market_cap": [None, None, None],
                }
            )
            valid = pd.DataFrame(
                {
                    "symbol": ["VCX", "QBTS", "SOUN"],
                    "relative_volume": [4.8, 3.4, 2.1],
                    "dollar_volume": [12200000, 18300000, 55200000],
                    "market_cap": [850000000, 1400000000, 5200000000],
                }
            )

            invalid.to_csv(discovery_dir / "top_candidates_2026-03-25.csv", index=False)
            valid.to_csv(discovery_dir / "top_candidates_2026-03-24.csv", index=False)

            with patch("main.FORECASTS_DIR", tmp):
                resolved = main._resolve_latest_discovery_date(use_full_candidates=False)

        self.assertEqual(resolved, "2026-03-24")

    def test_apply_default_run_arguments_uses_latest_discovery_for_m6(self) -> None:
        args = Namespace(
            train_universe="m6",
            candidate_file=None,
            discovery_date=None,
            top_k=None,
            use_full_candidates=False,
            exclude_spy=False,
        )

        with patch("main._resolve_latest_discovery_date", return_value="2026-03-24"):
            resolved = main._apply_default_run_arguments(args)

        self.assertEqual(resolved.discovery_date, "2026-03-24")

    def test_main_defaults_to_split_flow_for_m6_with_fresh_discovery(self) -> None:
        args = Namespace(
            train_universe="m6",
            candidate_file=None,
            discovery_date=None,
            top_k=None,
            use_full_candidates=False,
            exclude_spy=False,
        )

        class _Parser:
            def parse_args(self) -> Namespace:
                return args

        with patch("main.build_parser", return_value=_Parser()):
            with patch("main.date") as mock_date:
                mock_date.today.return_value = date(2026, 3, 25)
                with patch("main.run_discovery", return_value={"top_candidates_path": "forecasts/discovery/top_candidates_2026-03-25.csv"}):
                    with patch("main._discovery_snapshot_is_usable", return_value=True):
                        with patch("main._run_split_inference_flow") as mock_split:
                            with patch("main._run_legacy_flow") as mock_legacy:
                                main.main()

        self.assertEqual(args.discovery_date, "2026-03-25")
        mock_split.assert_called_once_with(args)
        mock_legacy.assert_not_called()

    def test_main_falls_back_to_latest_discovery_when_refresh_fails(self) -> None:
        args = Namespace(
            train_universe="m6",
            candidate_file=None,
            discovery_date=None,
            top_k=None,
            use_full_candidates=False,
            exclude_spy=False,
        )

        class _Parser:
            def parse_args(self) -> Namespace:
                return args

        with patch("main.build_parser", return_value=_Parser()):
            with patch("main.date") as mock_date:
                mock_date.today.return_value = date(2026, 3, 25)
                with patch("main.run_discovery", side_effect=RuntimeError("boom")):
                    with patch("main._resolve_latest_discovery_date", return_value="2026-03-24"):
                        with patch("main._run_split_inference_flow") as mock_split:
                            with patch("main._run_legacy_flow") as mock_legacy:
                                main.main()

        self.assertEqual(args.discovery_date, "2026-03-24")
        mock_split.assert_called_once_with(args)
        mock_legacy.assert_not_called()

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
