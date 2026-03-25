"""
config.py — Global constants and directory paths for the stock ranking pipeline.
"""

import os
from datetime import date

# === Pipeline Constants ===
SHIFTS = [0, 7, 14, 21]
SUBMISSION_INTERVALS = 12
TRAIN_START_DATE = date(2000, 1, 1)

# === Directory Paths ===
DATA_DIR = "data"
FEATURES_DIR = "features"
FORECASTS_DIR = "forecasts"
TEMP_DIR = "temp"
MARKET_DATA_DB = os.path.join(DATA_DIR, "market_data.sqlite")

# === Ensure Directories Exist ===
for d in [DATA_DIR, FEATURES_DIR, FORECASTS_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)
