"""
Step 01 — Generate Ticker Metadata.

Builds either a standalone universe, a discovery-driven candidate
universe, or a base universe augmented with discovery candidates for
downstream ingest / training / forecasting.
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
    "Benchmark": [True, False, False, False, False, False, False, False, False, False, False, False, False, False],
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
    "Benchmark": [False, False, False, False, False, False, False, True],
})

M6_ASSETS = [
    "ABBV", "ACN", "AEP", "AIZ", "ALLE", "AMAT", "AMP", "AMZN", "AVB", "AVY",
    "AXP", "BDX", "BF-B", "BMY", "BR", "CARR", "CDW", "CE", "CHTR", "CNC",
    "CNP", "COP", "CTAS", "CZR", "DG", "DPZ", "PLD", "DXC", "META", "FTV",
    "GOOG", "GPC", "HIG", "HST", "JPM", "KR", "OGN", "PG", "PPL", "PRU",
    "PYPL", "EG", "ROL", "ROST", "UNH", "URI", "V", "VRSK", "SW", "XOM",
    "IVV", "IWM", "EWU", "EWG", "EWL", "EWQ", "IEUS", "EWJ", "EWT", "MCHI",
    "INDA", "EWY", "EWA", "EWH", "EWZ", "EWC", "IEMG", "LQD", "HYG", "SHY",
    "IEF", "TLT", "SEGA.L", "IEAA.L", "HIGH.L", "JPEA.L", "IAU", "SLV", "GSG", "REET",
    "ICLN", "IXN", "IGF", "IUVL.L", "IUMO.L", "SPMV.L", "IEVL.L", "IEFM.L", "MVEU.L", "XLK",
    "XLF", "XLV", "XLE", "XLY", "XLI", "XLC", "XLU", "XLP", "XLB", "VXX",
]
M6_STOCK_COUNT = 50
STRUCTURAL_METADATA_COLUMNS = {"Name", "Sector", "ETF", "Benchmark", "UniverseSource"}
DISCOVERY_METADATA_COLUMNS = {
    "DiscoveryCandidate",
    "DiscoveryRank",
    "AttentionScore",
    "DiscoveryDate",
}


def _default_metadata() -> pd.DataFrame:
    metadata = STOCK_NAMES.copy()
    metadata["DiscoveryCandidate"] = False
    return metadata


def _mag7_metadata() -> pd.DataFrame:
    metadata = MAG7_METADATA.copy()
    metadata["DiscoveryCandidate"] = False
    return metadata


def _m6_metadata() -> pd.DataFrame:
    stock_symbols = set(M6_ASSETS[:M6_STOCK_COUNT])
    metadata = pd.DataFrame({"Symbol": M6_ASSETS})
    metadata["Name"] = metadata["Symbol"]
    metadata["Sector"] = np.where(
        metadata["Symbol"].isin(stock_symbols),
        "M6 Equity",
        "M6 Fund",
    )
    metadata["ETF"] = ~metadata["Symbol"].isin(stock_symbols)
    metadata["Benchmark"] = metadata["Symbol"] == "IVV"
    metadata["DiscoveryCandidate"] = False
    return metadata


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
    metadata["Benchmark"] = False
    metadata = metadata.drop_duplicates("Symbol").reset_index(drop=True)
    metadata["DiscoveryCandidate"] = False

    if include_spy and "SPY" not in set(metadata["Symbol"]):
        spy_row = pd.DataFrame({
            "Symbol": ["SPY"],
            "Name": ["S&P 500 ETF"],
            "Sector": ["Broad Market"],
            "ETF": [True],
            "Benchmark": [True],
        })
        metadata = pd.concat([metadata, spy_row], ignore_index=True)
    elif "SPY" in set(metadata["Symbol"]):
        metadata.loc[metadata["Symbol"] == "SPY", "Benchmark"] = True

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
        "Benchmark": False,
        "DiscoveryCandidate": True,
    })

    if "rank" in candidates.columns:
        metadata["DiscoveryRank"] = candidates["rank"].values
    if "attention_score" in candidates.columns:
        metadata["AttentionScore"] = candidates["attention_score"].values
    if "date" in candidates.columns:
        metadata["DiscoveryDate"] = candidates["date"].values

    if "SPY" in set(metadata["Symbol"]):
        metadata.loc[metadata["Symbol"] == "SPY", "Name"] = "S&P 500 ETF"
        metadata.loc[metadata["Symbol"] == "SPY", "Sector"] = "Broad Market"
        metadata.loc[metadata["Symbol"] == "SPY", "ETF"] = True
        metadata.loc[metadata["Symbol"] == "SPY", "Benchmark"] = True
    elif include_spy:
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
            "Benchmark": True,
        })
        metadata.loc[len(metadata)] = spy_values

    metadata = metadata.drop_duplicates("Symbol").reset_index(drop=True)
    return metadata


def _merge_base_with_candidates(
    *,
    base_metadata: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
) -> pd.DataFrame:
    base = base_metadata.copy().reset_index(drop=True)
    candidates = candidate_metadata.copy().reset_index(drop=True)

    base["Symbol"] = base["Symbol"].astype(str).str.upper().str.strip()
    candidates["Symbol"] = candidates["Symbol"].astype(str).str.upper().str.strip()

    all_columns = list(dict.fromkeys([*base.columns, *candidates.columns]))

    def _ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for column in all_columns:
            if column in frame.columns:
                continue
            if column == "DiscoveryCandidate":
                frame[column] = False
            elif column in {"DiscoveryRank", "AttentionScore"}:
                frame[column] = np.nan
            else:
                frame[column] = None
        return frame[all_columns]

    base = _ensure_columns(base)
    candidates = _ensure_columns(candidates)

    candidate_lookup = candidates.set_index("Symbol")
    overlapping_symbols = base.loc[
        base["Symbol"].isin(candidate_lookup.index), "Symbol"
    ].tolist()

    for column in all_columns:
        if column == "Symbol":
            continue
        mapped = base["Symbol"].map(candidate_lookup[column])
        if column in STRUCTURAL_METADATA_COLUMNS:
            base[column] = base[column].where(base[column].notna(), mapped)
        elif column == "DiscoveryCandidate":
            base[column] = (
                base[column].astype("boolean").fillna(False).astype(bool)
                | mapped.astype("boolean").fillna(False).astype(bool)
            )
        elif column in DISCOVERY_METADATA_COLUMNS:
            overlap_mask = base["Symbol"].isin(overlapping_symbols)
            base.loc[overlap_mask, column] = mapped.loc[overlap_mask].combine_first(
                base.loc[overlap_mask, column]
            )

    new_rows = candidates.loc[~candidates["Symbol"].isin(set(base["Symbol"]))].copy()
    combined = pd.concat([base, new_rows], ignore_index=True)
    combined = combined.drop_duplicates("Symbol").reset_index(drop=True)
    return combined


def run(
    *,
    universe_mode: str = "default",
    candidate_file: str | None = None,
    discovery_date: str | None = None,
    top_k: int | None = None,
    include_spy: bool = True,
    use_full_candidates: bool = False,
    merge_candidates_with_base: bool = False,
) -> pd.DataFrame:
    """Generate and save ticker metadata."""

    resolved_candidate_file = resolve_candidate_file(
        candidate_file=candidate_file,
        discovery_date=discovery_date,
        use_full_candidates=use_full_candidates,
    )

    if universe_mode == "mags7":
        metadata = _mag7_metadata()
        mode = "Magnificent 7 plus SPY universe"
    elif universe_mode == "m6":
        metadata = _m6_metadata()
        mode = "M6 asset universe"
    elif universe_mode == "sp500":
        metadata = _sp500_metadata(include_spy=include_spy)
        mode = "S&P 500 universe"
    if universe_mode in {"mags7", "m6", "sp500"} and resolved_candidate_file and merge_candidates_with_base:
        candidate_metadata = _candidate_metadata(
            candidate_file=resolved_candidate_file,
            top_k=top_k,
            include_spy=include_spy,
        )
        metadata = _merge_base_with_candidates(
            base_metadata=metadata,
            candidate_metadata=candidate_metadata,
        )
        mode = f"{mode} + discovery candidates from {resolved_candidate_file}"
    elif universe_mode not in {"mags7", "m6", "sp500"} and resolved_candidate_file:
        metadata = _candidate_metadata(
            candidate_file=resolved_candidate_file,
            top_k=top_k,
            include_spy=include_spy,
        )
        mode = f"discovery candidates from {resolved_candidate_file}"
    elif universe_mode not in {"mags7", "m6", "sp500"}:
        metadata = _default_metadata()
        mode = "default ETF universe"

    out_path = os.path.join(DATA_DIR, "tickers_metadata.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    metadata.to_parquet(out_path, index=False)
    print(f"[Step 01] Saved ticker metadata → {out_path}  ({len(metadata)} tickers, {mode})")
    return metadata


if __name__ == "__main__":
    run()
