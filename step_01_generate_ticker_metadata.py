"""
Step 01 — Generate Ticker Metadata.

Defines the universe of sector ETFs and saves metadata to parquet.
"""

import os
import pandas as pd
from config import DATA_DIR


# === Sector ETF Universe ===
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


def run() -> None:
    """Generate and save ticker metadata."""
    out_path = os.path.join(DATA_DIR, "tickers_metadata.parquet")
    STOCK_NAMES.to_parquet(out_path, index=False)
    print(f"[Step 01] Saved ticker metadata → {out_path}  ({len(STOCK_NAMES)} tickers)")


if __name__ == "__main__":
    run()
