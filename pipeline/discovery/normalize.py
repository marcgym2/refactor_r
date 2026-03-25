"""Ticker normalization and extraction helpers."""

from __future__ import annotations

from collections import OrderedDict
import re


CASHTAG_RE = re.compile(r"\$([A-Za-z][A-Za-z\.\-]{0,9})\b")
BARE_TICKER_RE = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z])?\b")


def normalize_symbol(
    raw_symbol: str | None,
    *,
    invalid_tokens: set[str],
    max_len: int,
    min_len: int,
    allow_single_char: bool,
) -> str | None:
    """Normalize a cashtag or bare token into a candidate ticker."""

    if not raw_symbol:
        return None

    symbol = raw_symbol.strip().upper().lstrip("$")
    if "." in symbol:
        base = symbol.split(".", maxsplit=1)[0]
    else:
        base = symbol

    if not base.isalpha():
        return None

    effective_min_len = 1 if allow_single_char else min_len
    if not (effective_min_len <= len(base) <= max_len):
        return None

    if symbol in invalid_tokens or base in invalid_tokens:
        return None

    return symbol


def extract_symbols(text: str, normalization_config: dict) -> list[str]:
    """Extract broad-recall ticker candidates from free-form text."""

    if not text:
        return []

    invalid_tokens = set(normalization_config.get("invalid_tokens", []))
    max_len = int(normalization_config.get("bare_symbol_max_len", 5))
    min_len = int(normalization_config.get("bare_symbol_min_len", 2))

    ordered: OrderedDict[str, None] = OrderedDict()

    for match in CASHTAG_RE.findall(text):
        symbol = normalize_symbol(
            match,
            invalid_tokens=invalid_tokens,
            max_len=max_len,
            min_len=min_len,
            allow_single_char=True,
        )
        if symbol:
            ordered[symbol] = None

    if normalization_config.get("allow_bare_symbols", True):
        for match in BARE_TICKER_RE.findall(text):
            symbol = normalize_symbol(
                match,
                invalid_tokens=invalid_tokens,
                max_len=max_len,
                min_len=min_len,
                allow_single_char=False,
            )
            if symbol:
                ordered[symbol] = None

    return list(ordered.keys())


def normalize_author(author: str | None) -> str:
    author_value = (author or "").strip()
    return author_value if author_value else "unknown"
