from __future__ import annotations

from datetime import date
import unittest

import numpy as np
import pandas as pd

from pipeline.m6_metrics import (
    RANK_COLUMNS,
    compute_m6_ir,
    compute_m6_rps,
    compute_tie_aware_rank_probabilities,
)


class M6MetricsTest(unittest.TestCase):
    def test_tie_aware_rank_probabilities_split_across_adjacent_quintiles(self) -> None:
        returns = pd.Series(
            [0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            name="Return",
        )

        probs = compute_tie_aware_rank_probabilities(returns)

        self.assertEqual(probs.loc["A", RANK_COLUMNS].tolist(), [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(probs.loc["B", RANK_COLUMNS].tolist(), [0.5, 0.5, 0.0, 0.0, 0.0])
        self.assertEqual(probs.loc["C", RANK_COLUMNS].tolist(), [0.5, 0.5, 0.0, 0.0, 0.0])
        self.assertEqual(probs.loc["J", RANK_COLUMNS].tolist(), [0.0, 0.0, 0.0, 0.0, 1.0])

    def test_compute_m6_rps_zero_for_perfect_submission(self) -> None:
        ids = [f"A{i}" for i in range(10)]
        hist = pd.DataFrame(
            {
                "date": [date(2026, 1, 1)] * 10 + [date(2026, 1, 2)] * 10,
                "symbol": ids + ids,
                "price": [100.0] * 10 + [100.0 + i for i in range(10)],
            }
        )
        returns = pd.Series(
            [(100.0 + i - 100.0) / 100.0 for i in range(10)],
            index=ids,
            name="Return",
        )
        target = compute_tie_aware_rank_probabilities(returns).reset_index().rename(columns={"index": "ID"})
        target["Decision"] = 0.0

        metrics = compute_m6_rps(
            hist_data=hist,
            submission=target[["ID", *RANK_COLUMNS, "Decision"]],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 2),
        )

        self.assertAlmostEqual(metrics["RPS"], 0.0, places=12)

    def test_compute_m6_ir_matches_manual_log_return_ratio(self) -> None:
        hist = pd.DataFrame(
            {
                "date": [
                    date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3),
                    date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3),
                ],
                "symbol": ["A", "A", "A", "B", "B", "B"],
                "price": [100.0, 110.0, 121.0, 100.0, 100.0, 110.0],
            }
        )
        submission = pd.DataFrame(
            {
                "ID": ["A", "B"],
                "Rank1": [0.0, 0.0],
                "Rank2": [0.0, 0.0],
                "Rank3": [0.0, 0.0],
                "Rank4": [0.0, 0.0],
                "Rank5": [1.0, 1.0],
                "Decision": [0.5, 0.5],
            }
        )

        metrics = compute_m6_ir(
            hist_data=hist,
            submission=submission,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 3),
        )

        expected_daily = [np.log1p(0.05), np.log1p(0.10)]
        expected_ir = sum(expected_daily) / pd.Series(expected_daily).std(ddof=1)

        self.assertAlmostEqual(metrics["IR"], expected_ir, places=12)


if __name__ == "__main__":
    unittest.main()
