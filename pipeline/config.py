"""
config.py — Global constants and directory paths for the stock ranking pipeline.
"""

import os
from datetime import date

# === Pipeline Constants ===
SHIFTS = [0, 7, 14, 21]
SUBMISSION_INTERVALS = 12
TRAIN_START_DATE = date(2000, 1, 1)
UNIVERSE_TRAIN_START_DATES = {
    "m6": date(2022, 1, 1),
}


def resolve_train_start_date(universe_mode: str | None = None) -> date:
    if universe_mode is None:
        return TRAIN_START_DATE
    return UNIVERSE_TRAIN_START_DATES.get(universe_mode, TRAIN_START_DATE)

# === Directory Paths ===
DATA_DIR = "data"
FEATURES_DIR = "features"
FORECASTS_DIR = "forecasts"
TEMP_DIR = "temp"
MARKET_DATA_DB = os.path.join(DATA_DIR, "market_data.sqlite")

# === Ensure Directories Exist ===
for d in [DATA_DIR, FEATURES_DIR, FORECASTS_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)
