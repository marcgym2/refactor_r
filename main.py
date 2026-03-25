"""
main.py — Pipeline Orchestrator.

Runs all steps of the stock ranking pipeline in order.
"""

from __future__ import annotations

from pipeline import forecast, ingest, portfolio, train, universe

def main() -> None:
    print("=" * 60)
    print("  Stock Ranking Pipeline — Python Refactor")
    print("=" * 60)

    universe.run()
    ingest.run()
    train.run()
    forecast.run()
    portfolio.run()

    print("\n[main] ✅ Pipeline complete.")


if __name__ == "__main__":
    main()
