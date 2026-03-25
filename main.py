"""
main.py — Pipeline Orchestrator.

Supports both the legacy single-universe run and a split flow:
- train on a broad universe (`default`, `mags7`, or `sp500`)
- infer only on discovery candidates plus SPY
"""

from __future__ import annotations

import argparse

from pipeline import forecast, ingest, infer, portfolio, train, universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stock ranking pipeline.")
    parser.add_argument(
        "--train-universe",
        choices=["default", "mags7", "sp500"],
        default="default",
        help="Universe used for ingest + training.",
    )
    parser.add_argument(
        "--candidate-file",
        help="CSV file with discovery candidates. Must contain a 'symbol' column.",
    )
    parser.add_argument(
        "--discovery-date",
        help="Load discovery candidates from forecasts/discovery for YYYY-MM-DD.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Keep only the first K discovery candidates before adding SPY.",
    )
    parser.add_argument(
        "--use-full-candidates",
        action="store_true",
        help="Use candidates_<date>.csv instead of top_candidates_<date>.csv when --discovery-date is set.",
    )
    parser.add_argument(
        "--exclude-spy",
        action="store_true",
        help="Do not force SPY into the candidate inference universe.",
    )
    return parser


def _run_split_inference_flow(args: argparse.Namespace) -> None:
    universe.run(universe_mode=args.train_universe)
    ingest.run()
    train.run()

    universe.run(
        universe_mode="default",
        candidate_file=args.candidate_file,
        discovery_date=args.discovery_date,
        top_k=args.top_k,
        include_spy=not args.exclude_spy,
        use_full_candidates=args.use_full_candidates,
    )
    ingest.run()
    infer.run()
    portfolio.run()


def _run_legacy_flow(args: argparse.Namespace) -> None:
    universe.run(
        universe_mode=args.train_universe,
        candidate_file=args.candidate_file,
        discovery_date=args.discovery_date,
        top_k=args.top_k,
        include_spy=not args.exclude_spy,
        use_full_candidates=args.use_full_candidates,
    )
    ingest.run()
    train.run()
    forecast.run()
    portfolio.run()


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 60)
    print("  Stock Ranking Pipeline — Python Refactor")
    print("=" * 60)

    has_candidate_inference = bool(args.candidate_file or args.discovery_date)
    if has_candidate_inference and args.train_universe != "default":
        _run_split_inference_flow(args)
    else:
        _run_legacy_flow(args)

    print("\n[main] ✅ Pipeline complete.")


if __name__ == "__main__":
    main()
