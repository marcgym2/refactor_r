from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline import universe


class CandidateUniverseTest(unittest.TestCase):
    def test_mag7_universe_contains_spy_and_seven_names(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch("pipeline.universe.DATA_DIR", str(tmp_path / "data")):
                metadata = universe.run(universe_mode="mags7")
            self.assertEqual(len(metadata), 8)
            self.assertEqual(metadata["Symbol"].tolist()[-1], "SPY")

    def test_candidate_file_builds_universe_with_spy(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_file = tmp_path / "top_candidates_2026-03-24.csv"
            candidate_file.write_text(
                "date,rank,symbol,attention_score\n"
                "2026-03-24,1,VCX,1.1\n"
                "2026-03-24,2,QBTS,0.87\n"
            )

            with patch("pipeline.universe.DATA_DIR", str(tmp_path / "data")):
                metadata = universe.run(
                    candidate_file=str(candidate_file),
                    top_k=1,
                    include_spy=True,
                )

            self.assertEqual(metadata["Symbol"].tolist(), ["VCX", "SPY"])
            self.assertEqual(bool(metadata.loc[metadata["Symbol"] == "SPY", "ETF"].iloc[0]), True)
            self.assertEqual(
                float(metadata.loc[metadata["Symbol"] == "VCX", "AttentionScore"].iloc[0]),
                1.1,
            )

    def test_resolve_candidate_file_from_discovery_date(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            discovery_dir = tmp_path / "discovery"
            discovery_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = discovery_dir / "top_candidates_2026-03-24.csv"
            candidate_path.write_text("symbol\nVCX\n")

            with patch("pipeline.universe.FORECASTS_DIR", str(tmp_path)):
                path = universe.resolve_candidate_file(discovery_date="2026-03-24")
            self.assertEqual(path, str(candidate_path.resolve()))


if __name__ == "__main__":
    unittest.main()
