"""HTTP helpers with minimal dependencies."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


class SourceFetchError(RuntimeError):
    """Raised when a public source cannot be fetched or parsed."""


def _request(url: str, timeout_seconds: int, user_agent: str) -> bytes:
    try:
        request = Request(url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except Exception as exc:  # pragma: no cover - network failures vary
        raise SourceFetchError(f"{url}: {exc}") from exc


def fetch_json(url: str, timeout_seconds: int, user_agent: str) -> dict:
    payload = _request(url, timeout_seconds=timeout_seconds, user_agent=user_agent)
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise SourceFetchError(f"Invalid JSON from {url}: {exc}") from exc


def fetch_text(url: str, timeout_seconds: int, user_agent: str) -> str:
    payload = _request(url, timeout_seconds=timeout_seconds, user_agent=user_agent)
    return payload.decode("utf-8", errors="ignore")
