from __future__ import annotations

import unittest

import pandas as pd

from pipeline.walk_forward import WalkForwardConfig, build_walk_forward_folds


class WalkForwardFoldTest(unittest.TestCase):
    def test_walk_forward_folds_use_only_strictly_past_intervals(self) -> None:
        rows: list[dict[str, object]] = []
        starts = pd.date_range("2020-01-01", periods=10, freq="28D")
        for start in starts:
            end = start + pd.Timedelta(days=28)
            rows.append(
                {
                    "Interval": f"{start.date()} : {end.date()}",
                    "Shift": 0,
                    "IntervalStart": start,
                    "IntervalEnd": end,
                }
            )

        frame = pd.DataFrame(rows)
        config = WalkForwardConfig(
            evaluation_shift=0,
            train_window_years=3.0,
            min_history_intervals=4,
            retrain_every=2,
            train_fraction=0.5,
            test_fraction=0.25,
        )
        folds = build_walk_forward_folds(frame, config=config)

        self.assertGreater(len(folds), 0)
        for fold in folds:
            history_ends = frame.loc[
                frame["Interval"].isin(
                    list(fold.train_intervals)
                    + list(fold.test_intervals)
                    + list(fold.validation_intervals)
                ),
                "IntervalEnd",
            ]
            self.assertTrue((pd.to_datetime(history_ends) < fold.target_start).all())

        self.assertEqual([fold.retrain for fold in folds[:4]], [True, False, True, False])

    def test_walk_forward_excludes_overlapping_intervals_from_history(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "Interval": "A",
                    "Shift": 0,
                    "IntervalStart": pd.Timestamp("2020-01-01"),
                    "IntervalEnd": pd.Timestamp("2020-01-28"),
                },
                {
                    "Interval": "B",
                    "Shift": 7,
                    "IntervalStart": pd.Timestamp("2020-01-22"),
                    "IntervalEnd": pd.Timestamp("2020-04-10"),
                },
                {
                    "Interval": "C",
                    "Shift": 0,
                    "IntervalStart": pd.Timestamp("2020-02-01"),
                    "IntervalEnd": pd.Timestamp("2020-02-28"),
                },
                {
                    "Interval": "D",
                    "Shift": 0,
                    "IntervalStart": pd.Timestamp("2020-02-29"),
                    "IntervalEnd": pd.Timestamp("2020-03-27"),
                },
                {
                    "Interval": "E",
                    "Shift": 0,
                    "IntervalStart": pd.Timestamp("2020-03-28"),
                    "IntervalEnd": pd.Timestamp("2020-04-24"),
                },
                {
                    "Interval": "F",
                    "Shift": 0,
                    "IntervalStart": pd.Timestamp("2020-04-25"),
                    "IntervalEnd": pd.Timestamp("2020-05-22"),
                },
            ]
        )

        config = WalkForwardConfig(
            evaluation_shift=0,
            train_window_years=3.0,
            min_history_intervals=3,
            retrain_every=1,
            train_fraction=0.34,
            test_fraction=0.33,
        )
        folds = build_walk_forward_folds(frame, config=config)

        target_fold = next(fold for fold in folds if fold.target_interval == "E")
        history = set(target_fold.train_intervals) | set(target_fold.test_intervals) | set(target_fold.validation_intervals)

        self.assertIn("A", history)
        self.assertIn("C", history)
        self.assertIn("D", history)
        self.assertNotIn("B", history)


if __name__ == "__main__":
    unittest.main()
