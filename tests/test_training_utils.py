from __future__ import annotations

import unittest

import pandas as pd

from pipeline.training_utils import standardize


class TrainingUtilsTest(unittest.TestCase):
    def test_standardize_singleton_group_returns_zero(self) -> None:
        result = standardize(pd.Series([42.0]))
        self.assertEqual(result.tolist(), [0.0])

    def test_standardize_constant_group_returns_zero(self) -> None:
        result = standardize(pd.Series([5.0, 5.0, 5.0]))
        self.assertEqual(result.tolist(), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
