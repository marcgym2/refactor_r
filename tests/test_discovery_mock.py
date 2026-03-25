from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from pipeline.discovery.runner import run


class DiscoveryMockPipelineTest(unittest.TestCase):
    def test_mock_run_ranks_abnormal_attention_above_popularity(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outputs = run(
                run_date=date(2026, 3, 24),
                config_path="config/discovery.toml",
                mock_mode=True,
                overrides={
                    "paths": {
                        "data_dir": str(tmp_path / "data"),
                        "output_dir": str(tmp_path / "outputs"),
                        "raw_mentions_dir": str(tmp_path / "data" / "raw_mentions"),
                        "history_path": str(tmp_path / "data" / "history" / "daily_attention_history.parquet"),
                        "normalized_daily_path": str(tmp_path / "data" / "history" / "daily_attention_latest.parquet"),
                    },
                    "sources": {
                        "news": {"enabled": True},
                    },
                },
            )

            top = pd.read_csv(outputs["top_candidates_path"])
            self.assertGreaterEqual(len(top), 5)
            self.assertEqual(top.iloc[0]["symbol"], "VCX")
            self.assertIn("attention_score", top.columns)
            self.assertIn("why_ranked_high", top.columns)
            self.assertFalse({"AND", "IS", "THE"} & set(top["symbol"]))

            ranks = dict(zip(top["symbol"], top["rank"]))
            self.assertLess(ranks["SOUN"], ranks["PLTR"])


if __name__ == "__main__":
    unittest.main()
