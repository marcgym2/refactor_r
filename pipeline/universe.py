"""
Step 01 — Generate Ticker Metadata.

Builds either the default ETF universe or a discovery-driven candidate
universe for downstream ingest / training / forecasting.
"""

from __future__ import annotations

import os
from pathlib import Path
from io import StringIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import DATA_DIR, FORECASTS_DIR


SECTOR_ETFS = [
    "SPY", "XLK", "XLF", "XLV", "XLI", "XLY",
    "XLP", "XLE", "XLB", "XLC", "XLU", "XLRE", "SH", "IAU",
]

STOCK_NAMES = pd.DataFrame({
    "Symbol": SECTOR_ETFS,
    "Name": [
        "S&P 500 ETF", "Technology Select Sector", "Financial Select Sector",
        "Health Care Select Sector", "Industrial Select Sector", "Consumer Discretionary",
        "Consumer Staples", "Energy Select Sector", "Materials Select Sector",
        "Communication Services", "Utilities Select Sector", "Real Estate Select Sector",
        "Short S&P 500 ETF", "Gold",
    ],
    "Sector": [
        "Broad Market", "Technology", "Financials", "Health Care", "Industrials",
        "Consumer Discretionary", "Consumer Staples", "Energy", "Materials",
        "Communication Services", "Utilities", "Real Estate", "Broad Market", "Gold",
    ],
    "ETF": True,
})

MAG7_METADATA = pd.DataFrame({
    "Symbol": ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "SPY"],
    "Name": [
        "Apple Inc.",
        "Amazon.com, Inc.",
        "Alphabet Inc. Class A",
        "Meta Platforms, Inc.",
        "Microsoft Corporation",
        "NVIDIA Corporation",
        "Tesla, Inc.",
        "S&P 500 ETF",
    ],
    "Sector": [
        "Technology",
        "Consumer Discretionary",
        "Communication Services",
        "Communication Services",
        "Technology",
        "Technology",
        "Consumer Discretionary",
        "Broad Market",
    ],
    "ETF": [False, False, False, False, False, False, False, True],
})


def _default_metadata() -> pd.DataFrame:
    return STOCK_NAMES.copy()


def _mag7_metadata() -> pd.DataFrame:
    return MAG7_METADATA.copy()


def _fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 refactor-r-universe"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def _sp500_metadata(include_spy: bool = True) -> pd.DataFrame:
    """Fetch the latest S&P 500 constituents from free public sources."""

    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    try:
        html = _fetch_text("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        frames = pd.read_html(StringIO(html))
    except Exception as exc:
        errors.append(f"wikipedia: {exc}")

    source = "wikipedia"
    if not frames:
        try:
            csv_text = _fetch_text("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")
            frames = [pd.read_csv(StringIO(csv_text))]
            source = "datahub"
        except Exception as exc:
            errors.append(f"datahub: {exc}")

    if not frames:
        joined = "; ".join(errors)
        raise RuntimeError(f"Unable to fetch S&P 500 constituents: {joined}")

    table = frames[0].copy()
    rename_map = {
        "Symbol": "Symbol",
        "Security": "Name",
        "GICS Sector": "Sector",
    }
    missing = [column for column in rename_map if column not in table.columns]
    if missing:
        raise RuntimeError(f"S&P 500 source missing required columns: {missing}")

    metadata = table[list(rename_map.keys())].rename(columns=rename_map)
    metadata["Symbol"] = metadata["Symbol"].astype(str).str.replace(".", "-", regex=False)
    metadata["ETF"] = False
    metadata = metadata.drop_duplicates("Symbol").reset_index(drop=True)

    if include_spy and "SPY" not in set(metadata["Symbol"]):
        spy_row = pd.DataFrame({
            "Symbol": ["SPY"],
            "Name": ["S&P 500 ETF"],
            "Sector": ["Broad Market"],
            "ETF": [True],
        })
        metadata = pd.concat([metadata, spy_row], ignore_index=True)

    metadata["UniverseSource"] = source
    return metadata


def resolve_candidate_file(
    *,
    candidate_file: str | None = None,
    discovery_date: str | None = None,
    use_full_candidates: bool = False,
) -> str | None:
    """Resolve a candidate file path from explicit or date-based inputs."""

    if candidate_file:
        resolved = Path(candidate_file).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Candidate file not found: {resolved}")
        return str(resolved)

    if not discovery_date:
        return None

    prefix = "candidates" if use_full_candidates else "top_candidates"
    resolved = Path(FORECASTS_DIR) / "discovery" / f"{prefix}_{discovery_date}.csv"
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Discovery candidate file not found: {resolved}")
    return str(resolved)


def _candidate_metadata(
    *,
    candidate_file: str,
    top_k: int | None = None,
    include_spy: bool = True,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_file)
    if "symbol" not in candidates.columns and "Symbol" in candidates.columns:
        candidates = candidates.rename(columns={"Symbol": "symbol"})
    if "symbol" not in candidates.columns:
        raise ValueError(f"Candidate file must contain a 'symbol' column: {candidate_file}")

    candidates = candidates.copy()
    candidates["symbol"] = candidates["symbol"].astype(str).str.upper().str.strip()
    candidates = candidates.loc[candidates["symbol"] != ""].drop_duplicates("symbol")
    if top_k is not None:
        candidates = candidates.head(top_k).copy()

    metadata = pd.DataFrame({
        "Symbol": candidates["symbol"],
        "Name": candidates["symbol"],
        "Sector": "Discovery",
        "ETF": False,
    })

    if "rank" in candidates.columns:
        metadata["DiscoveryRank"] = candidates["rank"].values
    if "attention_score" in candidates.columns:
        metadata["AttentionScore"] = candidates["attention_score"].values
    if "date" in candidates.columns:
        metadata["DiscoveryDate"] = candidates["date"].values

    if include_spy and "SPY" not in set(metadata["Symbol"]):
        spy_values: dict[str, object] = {}
        for column, dtype in metadata.dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype):
                spy_values[column] = np.nan
            elif pd.api.types.is_bool_dtype(dtype):
                spy_values[column] = False
            else:
                spy_values[column] = None
        spy_values.update({
            "Symbol": "SPY",
            "Name": "S&P 500 ETF",
            "Sector": "Broad Market",
            "ETF": True,
        })
        metadata.loc[len(metadata)] = spy_values

    metadata = metadata.drop_duplicates("Symbol").reset_index(drop=True)
    return metadata


def run(
    *,
    universe_mode: str = "default",
    candidate_file: str | None = None,
    discovery_date: str | None = None,
    top_k: int | None = None,
    include_spy: bool = True,
    use_full_candidates: bool = False,
) -> pd.DataFrame:
    """Generate and save ticker metadata."""

    resolved_candidate_file = None
    if universe_mode == "mags7":
        metadata = _mag7_metadata()
        mode = "Magnificent 7 plus SPY universe"
    elif universe_mode == "sp500":
        metadata = _sp500_metadata(include_spy=include_spy)
        mode = "S&P 500 universe"
    else:
        resolved_candidate_file = resolve_candidate_file(
            candidate_file=candidate_file,
            discovery_date=discovery_date,
            use_full_candidates=use_full_candidates,
        )

    if universe_mode not in {"mags7", "sp500"} and resolved_candidate_file:
        metadata = _candidate_metadata(
            candidate_file=resolved_candidate_file,
            top_k=top_k,
            include_spy=include_spy,
        )
        mode = f"discovery candidates from {resolved_candidate_file}"
    elif universe_mode not in {"mags7", "sp500"}:
        metadata = _default_metadata()
        mode = "default ETF universe"

    out_path = os.path.join(DATA_DIR, "tickers_metadata.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    metadata.to_parquet(out_path, index=False)
    print(f"[Step 01] Saved ticker metadata → {out_path}  ({len(metadata)} tickers, {mode})")
    return metadata


if __name__ == "__main__":
    run()
