"""
main.py — Pipeline Orchestrator.

Supports both the legacy single-universe run and a split flow:
- train on a configured universe (`default`, `mags7`, `sp500`, or `m6`)
- infer only on discovery candidates plus SPY
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline import forecast, infer_tabicl as infer, ingest, portfolio, train, universe
from pipeline.config import FORECASTS_DIR, resolve_train_start_date
from pipeline.discovery.runner import run as run_discovery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stock ranking pipeline.")
    parser.add_argument(
        "--train-universe",
        choices=["default", "mags7", "sp500", "m6"],
        default="m6",
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


def _discovery_snapshot_is_usable(candidate_path: Path) -> bool:
    if not candidate_path.exists():
        return False

    try:
        candidates = pd.read_csv(candidate_path)
    except Exception:
        return False

    if candidates.empty:
        return False

    required_market_columns = {"relative_volume", "dollar_volume", "market_cap"}
    if not required_market_columns.issubset(candidates.columns):
        return True

    head = candidates.head(10)
    market_ready = head[list(required_market_columns)].notna().all(axis=1)
    minimum_ready = min(3, len(head))
    return int(market_ready.sum()) >= minimum_ready


def _resolve_latest_discovery_date(*, use_full_candidates: bool) -> str | None:
    prefix = "candidates_" if use_full_candidates else "top_candidates_"
    discovery_dir = Path(FORECASTS_DIR) / "discovery"
    if not discovery_dir.exists():
        return None

    candidates = sorted(discovery_dir.glob(f"{prefix}*.csv"), reverse=True)
    for candidate_path in candidates:
        if not _discovery_snapshot_is_usable(candidate_path):
            continue
        latest = candidate_path.stem
        return latest.removeprefix(prefix)
    return None


def _apply_default_run_arguments(args: argparse.Namespace) -> argparse.Namespace:
    if args.train_universe == "m6" and not args.candidate_file and not args.discovery_date:
        args.discovery_date = _resolve_latest_discovery_date(
            use_full_candidates=args.use_full_candidates,
        )
    return args


def _refresh_discovery_snapshot(args: argparse.Namespace) -> argparse.Namespace:
    if args.train_universe != "m6" or args.candidate_file or args.discovery_date:
        return args

    run_date = date.today().isoformat()
    try:
        outputs = run_discovery(run_date=run_date)
        top_candidates_path = Path(outputs["top_candidates_path"])
        if not _discovery_snapshot_is_usable(top_candidates_path):
            raise RuntimeError(f"discovery snapshot failed sanity checks: {top_candidates_path}")
        args.discovery_date = run_date
        print(f"[main] Refreshed discovery candidates → {top_candidates_path}")
    except Exception as exc:
        print(f"[main] Warning: discovery refresh failed for {run_date}: {exc}")
    return args


def _run_split_inference_flow(args: argparse.Namespace) -> None:
    train_start_date = resolve_train_start_date(args.train_universe)
    universe.run(universe_mode=args.train_universe)
    ingest.run(start_date=train_start_date)
    train.run()

    inference_universe_mode = "default"
    include_spy = not args.exclude_spy
    merge_candidates_with_base = False
    if args.train_universe == "m6":
        inference_universe_mode = "m6"
        include_spy = False
        merge_candidates_with_base = True

    universe.run(
        universe_mode=inference_universe_mode,
        candidate_file=args.candidate_file,
        discovery_date=args.discovery_date,
        top_k=args.top_k,
        include_spy=include_spy,
        use_full_candidates=args.use_full_candidates,
        merge_candidates_with_base=merge_candidates_with_base,
    )
    ingest.run(start_date=train_start_date)
    forecast_path = infer.run()
    portfolio.run(forecast_path=forecast_path)


def _run_legacy_flow(args: argparse.Namespace) -> None:
    train_start_date = resolve_train_start_date(args.train_universe)
    universe.run(
        universe_mode=args.train_universe,
        candidate_file=args.candidate_file,
        discovery_date=args.discovery_date,
        top_k=args.top_k,
        include_spy=not args.exclude_spy,
        use_full_candidates=args.use_full_candidates,
    )
    ingest.run(start_date=train_start_date)
    train.run()
    forecast_path = forecast.run()
    portfolio.run(forecast_path=forecast_path)


def main() -> None:
    args = build_parser().parse_args()
    args = _refresh_discovery_snapshot(args)
    args = _apply_default_run_arguments(args)

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
