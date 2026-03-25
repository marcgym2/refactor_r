# refactor_r

This repo now has two distinct flows:

- the existing training / forecasting pipeline driven by `main.py`
- a new end-of-day attention discovery pipeline driven by `discovery_main.py`

The discovery pipeline is a free-for-now MVP that ranks ticker candidates based on abnormal attention growth, then writes daily candidate files for downstream modeling.

## Discovery MVP

Relevant modules:

- `pipeline/discovery/runner.py`: end-to-end orchestration
- `pipeline/discovery/sources/stocktwits.py`: Stocktwits trending + symbol stream adapter
- `pipeline/discovery/sources/reddit.py`: Reddit finance-community adapter
- `pipeline/discovery/sources/news.py`: optional Google News RSS count adapter
- `pipeline/discovery/history.py`: raw mention storage and rolling history
- `pipeline/discovery/scoring.py`: abnormality features and ranking
- `config/discovery.toml`: source toggles, weights, and optional thresholds

Design choices:

- modular adapters per source
- free public endpoints only for the MVP
- broad candidate retention by default
- optional filters are config-driven and disabled by default
- one source failing does not kill the run
- normalized raw mentions are stored for debugging

## Setup

```bash
uv sync
```

Live end-of-day run:

```bash
uv run python -m pipeline.discovery.cli --config config/discovery.toml --date 2026-03-24
```

Mock validation run:

```bash
uv run python discovery_main.py --config config/discovery.toml --date 2026-03-24 --mock --print-head 10
```

The optional news-count source is implemented but disabled by default. Turn it on in `config/discovery.toml` by setting `[sources.news].enabled = true`.

## Running Main On Discovery Candidates

After you have a discovery output, you can run the existing pipeline in two ways.

Legacy reduced-universe mode:

- train and forecast only on the reduced candidate universe
- useful for quick experiments, but not the preferred setup

```bash
uv run python main.py --candidate-file forecasts/discovery/top_candidates_2026-03-24.csv
```

Preferred split mode:

- train on a broader universe
- then switch to `candidates + SPY` for inference only

For the current lightweight setup, train on `mags7 + SPY` and infer on the discovery output for a given date:

```bash
uv run python main.py --train-universe mags7 --discovery-date 2026-03-24
```

The M6 branch supports a merged split flow. This trains on the M6 asset universe and then scores the full M6 100 plus any additional discovery names in one inference CSV:

```bash
uv run python main.py --train-universe m6 --discovery-date 2026-03-24 --top-k 10
```

For this merged M6 inference universe, `IVV` stays the active benchmark and `SPY` is not auto-added.

Run the legacy train + forecast flow only on the M6 asset universe:

```bash
uv run python main.py --train-universe m6
```

The `m6` universe uses the 100 provided assets and syncs market history starting from `2022-01-01`.
Unavailable legacy symbols were replaced with live equivalents for this workflow: `DRE -> PLD`, `RE -> EG`, and `WRK -> SW`.

Or point directly at a candidate file:

```bash
uv run python main.py --train-universe mags7 --candidate-file forecasts/discovery/top_candidates_2026-03-24.csv
```

Limit the inference universe further:

```bash
uv run python main.py --train-universe mags7 --discovery-date 2026-03-24 --top-k 10
```

How it works:

- `main.py` first builds the training universe and runs ingest + train
- if `--train-universe` is not `default` and a discovery candidate input is supplied, the pipeline then rebuilds the universe from `candidates + SPY`
- the second ingest step refreshes only those candidate names
- `pipeline/infer.py` uses the saved base model artifacts to score the candidate universe
- `portfolio.py` then consumes that inference forecast the same way it consumes the legacy forecast output

## Outputs

Run artifacts are written under `forecasts/discovery` and `data/discovery`:

- `data/discovery/raw_mentions/mentions_<date>.parquet`: normalized source-level mention records
- `data/discovery/history/daily_attention_history.parquet`: rolling daily aggregate history
- `forecasts/discovery/candidates_<date>.csv`
- `forecasts/discovery/candidates_<date>.parquet`
- `forecasts/discovery/top_candidates_<date>.csv`
- `forecasts/discovery/diagnostics_<date>.json`

Core output columns:

| column | meaning |
| --- | --- |
| `symbol` | normalized ticker symbol |
| `mention_count_today` | current-day mentions across enabled sources |
| `mention_count_vs_5d_baseline` | `(today + 1) / (prior_5d_mean + 1)` |
| `mention_count_vs_20d_baseline` | `(today + 1) / (prior_20d_mean + 1)` |
| `mention_zscore` | 20-day abnormality score with graceful fallback |
| `mention_acceleration_day_over_day` | today minus prior-day mention count |
| `unique_authors` | unique authors across sources |
| `source_breadth` | number of sources mentioning the ticker |
| `subreddit_breadth` | number of Reddit communities mentioning the ticker |
| `news_count_today` | optional Google News count |
| `price_change_today` | latest daily return from `yfinance` |
| `relative_volume` | latest volume vs prior 20-day average |
| `dollar_volume` | latest close times latest volume |
| `market_cap` | `yfinance.fast_info` market cap when available |
| `attention_score` | weighted percentile-rank score |
| `why_ranked_high` | top contributing reasons for the score |

The smaller model-ingestion file is `top_candidates_<date>.csv`.

## Scheduler

Example cron entry for a weekday run shortly after the US close:

```cron
TZ=America/New_York
10 16 * * 1-5 cd /Users/marcgrayson/Documents/New\ project/refactor_r && /usr/bin/env uv run python discovery_main.py --config config/discovery.toml >> logs/discovery.log 2>&1
```

## Mock Test

Run the regression test:

```bash
uv run python -m unittest tests.test_discovery_mock
```

The committed mock fixtures live in `data/discovery/mock` and simulate a low-base symbol suddenly rising in attention.

## Limitations

- Stocktwits and Reddit public endpoints are free but can be rate-limited or changed without notice.
- The MVP uses daily baselines, not full intraday reconstruction.
- Market-cap enrichment depends on `yfinance.fast_info`, which can be missing for some names.
- Bare ticker extraction on Reddit is intentionally conservative to reduce false positives.
- There is no paid firehose or premium news feed in this version.

## Next Upgrades

- persist intraday snapshots through the session so mention acceleration becomes truly intraday
- add more free community sources or free-tier news endpoints
- feed the top-candidate file directly into the downstream forecasting input contract
- add optional paid-source adapters later if you decide the free public endpoints are too fragile
