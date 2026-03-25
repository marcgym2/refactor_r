from __future__ import annotations

import unittest

import pandas as pd

from pipeline.m6_baseline import build_m6_baseline_submission


class M6BaselineTest(unittest.TestCase):
    def test_baseline_outputs_consistent_decision_position_and_invest_flags(self) -> None:
        frame = pd.DataFrame(
            {
                "ID": ["OVERLAP", "LONG2", "SHORT1", "SHORT2", "FLAT"],
                "Rank1": [0.44, 0.05, 0.52, 0.47, 0.20],
                "Rank2": [0.12, 0.10, 0.16, 0.18, 0.20],
                "Rank3": [0.10, 0.15, 0.14, 0.15, 0.20],
                "Rank4": [0.14, 0.25, 0.10, 0.10, 0.20],
                "Rank5": [0.46, 0.45, 0.08, 0.10, 0.20],
            }
        )

        submission, summary = build_m6_baseline_submission(frame)

        invested = submission.loc[submission["Invest"]]
        self.assertEqual(len(invested), 3)
        self.assertAlmostEqual(float(submission["Decision"].abs().sum()), 1.0)
        self.assertTrue((submission["Decision"] >= 0.0).all())
        self.assertEqual(
            set(submission.loc[submission["Decision"] > 0, "Position"]),
            {"Long"} if (submission["Decision"] > 0).any() else set(),
        )
        self.assertFalse((submission["Decision"] < 0).any())
        self.assertTrue((submission.loc[submission["Decision"] == 0, "Position"] == "Flat").all())
        self.assertTrue((submission["Invest"] == (submission["Decision"] != 0.0)).all())
        self.assertEqual(summary["short_ids"], [])
        self.assertEqual(len(summary["long_ids"]), 3)
        self.assertEqual(len(set(summary["long_ids"]) & set(summary["short_ids"])), 0)

    def test_baseline_clips_low_conviction_exposure_to_minimum(self) -> None:
        frame = pd.DataFrame(
            {
                "ID": ["A", "B", "C", "D", "E"],
                "Rank1": [0.19, 0.18, 0.17, 0.21, 0.22],
                "Rank2": [0.20, 0.20, 0.20, 0.20, 0.20],
                "Rank3": [0.21, 0.21, 0.21, 0.20, 0.20],
                "Rank4": [0.20, 0.20, 0.20, 0.20, 0.19],
                "Rank5": [0.20, 0.21, 0.22, 0.19, 0.19],
            }
        )

        submission, summary = build_m6_baseline_submission(frame)

        self.assertAlmostEqual(float(submission["Decision"].abs().sum()), 0.25)
        self.assertAlmostEqual(float(summary["gross_exposure"]), 0.25)
        self.assertEqual(len(submission.loc[submission["Invest"]]), 3)


if __name__ == "__main__":
    unittest.main()
