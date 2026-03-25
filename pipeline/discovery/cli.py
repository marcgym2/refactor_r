"""CLI entrypoint for the discovery pipeline."""

from __future__ import annotations

import argparse

import pandas as pd

from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the end-of-day discovery pipeline.")
    parser.add_argument("--date", help="Run date in YYYY-MM-DD format. Defaults to today.")
    parser.add_argument(
        "--config",
        default="config/discovery.toml",
        help="Path to the discovery TOML config.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use local mock fixtures instead of live public sources.",
    )
    parser.add_argument(
        "--print-head",
        type=int,
        default=10,
        help="Print the top N ranked candidates after the run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run(run_date=args.date, config_path=args.config, mock_mode=args.mock)
    print("[discovery] Full output:", outputs["full_csv_path"])
    print("[discovery] Top candidates:", outputs["top_candidates_path"])
    print("[discovery] Diagnostics:", outputs["diagnostics_path"])

    if args.print_head > 0:
        top = pd.read_csv(outputs["top_candidates_path"]).head(args.print_head)
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
