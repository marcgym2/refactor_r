"""Source adapter helpers."""

from __future__ import annotations

import pandas as pd


MENTION_COLUMNS = [
    "run_date",
    "source",
    "symbol",
    "author",
    "community",
    "mention_time",
    "text_id",
    "content_type",
    "signal_strength",
    "engagement",
    "watchlist_count",
    "body_excerpt",
]


def empty_mentions_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MENTION_COLUMNS)
