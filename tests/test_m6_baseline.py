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
        self.assertGreater(len(invested), 0)
        self.assertLessEqual(float(submission["Decision"].abs().sum()), 1.0)
        self.assertEqual(
            set(submission.loc[submission["Decision"] > 0, "Position"]),
            {"Long"} if (submission["Decision"] > 0).any() else set(),
        )
        self.assertEqual(
            set(submission.loc[submission["Decision"] < 0, "Position"]),
            {"Short"} if (submission["Decision"] < 0).any() else set(),
        )
        self.assertTrue((submission.loc[submission["Decision"] == 0, "Position"] == "Flat").all())
        self.assertTrue((submission["Invest"] == (submission["Decision"] != 0.0)).all())
        self.assertEqual(len(set(summary["long_ids"]) & set(summary["short_ids"])), 0)


if __name__ == "__main__":
    unittest.main()
