from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from pipeline.ingest import _clip_history_start


class IngestTest(unittest.TestCase):
    def test_clip_history_start_filters_older_cached_rows(self) -> None:
        df = pd.DataFrame(
            {
                "index": ["2021-12-31", "2022-01-03", "2022-01-04"],
                "Adjusted": [1.0, 2.0, 3.0],
            }
        )

        clipped = _clip_history_start(df, date(2022, 1, 1))

        self.assertEqual(clipped["index"].tolist(), [date(2022, 1, 3), date(2022, 1, 4)])


if __name__ == "__main__":
    unittest.main()
