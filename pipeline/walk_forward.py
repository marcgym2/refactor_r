"""
Walk-forward backtest with rolling retraining.

Trains the quantile models on a trailing historical window, predicts the next
non-overlapping interval, applies the live allocation rule, and measures the
realized portfolio return. The fold construction is strictly causal: no fold
may train on intervals whose realized window overlaps the target interval.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
import math
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import DATA_DIR, FEATURES_DIR, FORECASTS_DIR, SHIFTS, SUBMISSION_INTERVALS, TEMP_DIR
from .features import TTR_FEATURES, compute_return, is_etf, lag_return, lag_volatility
from .m6_baseline import RANK_COLUMNS, build_m6_baseline_submission
from .models import MetaModel, prepare_base_model
from .training_utils import (
    ConstructFFNN,
    compute_rps_tensor,
    gen_interval_infos,
    gen_stocks_aggr,
    impute_na,
    minibatch_sampler,
    subset_tensor,
    train_model,
)


@dataclass(frozen=True)
class WalkForwardConfig:
    evaluation_shift: int = 0
    train_window_years: float = 3.0
    min_history_intervals: int = 96
    retrain_every: int = 3
    train_fraction: float = 0.7
    test_fraction: float = 0.15
    base_epochs: int = 40
    base_minibatch: int = 200
    base_patience: int = 4
    base_lr: tuple[float, ...] = (0.01,)
    meta_epochs: int = 25
    meta_patience: int = 5
    meta_lr: tuple[float, ...] = (0.01, 0.001, 0.0005)
    meta_minibatch_size: int = 4
    warm_start: bool = True


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    retrain: bool
    target_shift: int
    target_interval: str
    target_start: pd.Timestamp
    target_end: pd.Timestamp
    history_start: pd.Timestamp
    history_end: pd.Timestamp
    train_intervals: tuple[str, ...]
    test_intervals: tuple[str, ...]
    validation_intervals: tuple[str, ...]


def _build_feature_functions(stock_names: pd.DataFrame) -> list:
    return [
        compute_return,
        lambda df, t: lag_volatility(df, t, lags=list(range(1, 8))),
        lambda df, t: lag_return(df, t, lags=list(range(1, 8))),
        lambda df, t: is_etf(df, t, stock_names=stock_names),
    ] + TTR_FEATURES


def _load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    stock_names = pd.read_parquet(os.path.join(DATA_DIR, "tickers_metadata.parquet"))
    with open(os.path.join(DATA_DIR, "tickers_data_cleaned.pkl"), "rb") as handle:
        stocks = pickle.load(handle)
    return stock_names, dict(sorted(stocks.items()))


def _load_or_generate_raw_features(
    *,
    stock_names: pd.DataFrame,
    stocks: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    feature_fns = _build_feature_functions(stock_names)
    interval_infos = gen_interval_infos(submission=SUBMISSION_INTERVALS, shifts=SHIFTS)
    precomputed_path = os.path.join(FEATURES_DIR, "features_raw.parquet")
    metadata_path = os.path.join(DATA_DIR, "tickers_metadata.parquet")
    cleaned_data_path = os.path.join(DATA_DIR, "tickers_data_cleaned.pkl")
    required_cols = {"Interval", "Ticker", "Shift", "IntervalStart", "IntervalEnd"}

    def _generate_and_cache() -> pd.DataFrame:
        print("[WalkForward] Generating aggregated features …")
        generated = gen_stocks_aggr(stocks, interval_infos, feature_fns, check_leakage=False)
        if generated.empty:
            raise RuntimeError("Feature aggregation produced no rows.")
        generated.to_parquet(precomputed_path, index=False)
        return generated

    if os.path.exists(precomputed_path):
        stocks_aggr = pd.read_parquet(precomputed_path)
        missing = sorted(required_cols - set(stocks_aggr.columns))
        cache_stale = any(
            os.path.getmtime(path) > os.path.getmtime(precomputed_path)
            for path in [metadata_path, cleaned_data_path]
            if os.path.exists(path)
        )
        if missing or stocks_aggr.empty or cache_stale:
            print(
                "[WalkForward] Cached features invalid or stale "
                f"(missing: {missing}, rows: {len(stocks_aggr)}, stale: {cache_stale}). Regenerating."
            )
            stocks_aggr = _generate_and_cache()
    else:
        stocks_aggr = _generate_and_cache()
    return stocks_aggr


def _fit_feature_medians(frame: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for column in feature_names:
        value = impute_na(frame[column]).median()
        medians[column] = 0.0 if not np.isfinite(value) else float(value)
    return medians


def _standardize_cross_section(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    result = frame.copy()
    std_features = [feature for feature in feature_names if feature != "ETF"]
    for column in std_features:
        grouped = result.groupby("Interval", observed=False)[column]
        result[column] = grouped.transform(
            lambda values: pd.Series(0.0, index=values.index, dtype=float)
            if (not np.isfinite(values.std(ddof=0))) or values.std(ddof=0) < 1e-8
            else (values - values.mean()) / values.std(ddof=0)
        )
    return result


def _apply_preprocessing(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    medians: dict[str, float],
) -> pd.DataFrame:
    result = frame.copy()
    for column in feature_names:
        result[column] = result[column].replace([np.inf, -np.inf], np.nan).fillna(medians[column])
    return _standardize_cross_section(result, feature_names)


def _build_target_tensor(frame: pd.DataFrame) -> torch.Tensor:
    target_columns = [f"TargetRank{i}" for i in range(1, 6)]
    if set(target_columns).issubset(frame.columns):
        target_probs = frame[target_columns].fillna(0.0).to_numpy(dtype=float)
        return torch.tensor(np.cumsum(target_probs, axis=1), dtype=torch.float32)

    quintiles = frame["ReturnQuintile"].values
    tensor = torch.zeros(len(quintiles), 5)
    for idx, quintile in enumerate(quintiles):
        if not np.isnan(quintile):
            q = int(quintile)
            tensor[idx, q - 1 :] = 1.0
    return tensor


def _build_feature_tensor(frame: pd.DataFrame, feature_names: list[str]) -> torch.Tensor:
    return torch.tensor(frame[feature_names].to_numpy(dtype=float), dtype=torch.float32)


def _build_sparse_ticker_tensor(frame: pd.DataFrame, ticker_categories: list[str]) -> torch.Tensor:
    tickers = pd.Categorical(frame["Ticker"], categories=ticker_categories)
    row_idx = torch.arange(len(tickers), dtype=torch.long)
    col_idx = torch.tensor(tickers.codes, dtype=torch.long)
    indices = torch.stack([row_idx, col_idx])
    values = torch.ones(len(tickers))
    torch.sparse.check_sparse_tensor_invariants.disable()
    return torch.sparse_coo_tensor(
        indices,
        values,
        (len(tickers), len(ticker_categories)),
    ).coalesce()


def _build_allow_meta_structure(base_model: ConstructFFNN) -> OrderedDict:
    allow_meta_structure: OrderedDict[str, torch.Tensor] = OrderedDict()
    n_layers = len(base_model.layers)
    included_keys = set()
    for layer_idx in range(max(0, n_layers - 3), n_layers):
        included_keys.add(f"layers.{layer_idx}.weight")
        included_keys.add(f"layers.{layer_idx}.bias")

    for key, shape in base_model.state_structure.items():
        if key in included_keys:
            allow_meta_structure[key] = torch.ones(shape, dtype=torch.bool)
        else:
            allow_meta_structure[key] = torch.zeros(shape, dtype=torch.bool)
    return allow_meta_structure


def _compute_portfolio_return(
    *,
    stocks: dict[str, pd.DataFrame],
    submission: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float | None:
    selected = submission.loc[submission["Decision"] != 0.0, ["ID", "Decision"]].copy()
    if selected.empty:
        return None

    series_by_ticker: dict[str, pd.Series] = {}
    for row in selected.itertuples(index=False):
        stock = stocks.get(str(row.ID))
        if stock is None or stock.empty:
            return None
        frame = stock[["index", "Adjusted"]].copy()
        frame["index"] = pd.to_datetime(frame["index"])
        frame = frame.loc[(frame["index"] >= start_date) & (frame["index"] <= end_date)].dropna(subset=["Adjusted"])
        if len(frame) < 2:
            return None
        series_by_ticker[str(row.ID)] = (
            frame.drop_duplicates(subset=["index"]).sort_values("index").set_index("index")["Adjusted"].astype(float)
        )

    panel = pd.concat(series_by_ticker.values(), axis=1, keys=series_by_ticker.keys()).sort_index().ffill().dropna()
    if len(panel) < 2:
        return None
    weights = selected.set_index("ID")["Decision"].astype(float).reindex(panel.columns).fillna(0.0)
    daily_portfolio_return = panel.pct_change().iloc[1:].mul(weights, axis=1).sum(axis=1)
    return float(np.prod(1.0 + daily_portfolio_return.to_numpy(dtype=float)) - 1.0)


def build_walk_forward_folds(
    frame: pd.DataFrame,
    *,
    config: WalkForwardConfig,
) -> list[WalkForwardFold]:
    required = {"Interval", "Shift", "IntervalStart", "IntervalEnd"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature frame missing required columns: {missing}")

    intervals = (
        frame[["Interval", "Shift", "IntervalStart", "IntervalEnd"]]
        .drop_duplicates()
        .copy()
    )
    intervals["IntervalStart"] = pd.to_datetime(intervals["IntervalStart"])
    intervals["IntervalEnd"] = pd.to_datetime(intervals["IntervalEnd"])
    intervals = intervals.sort_values(["IntervalStart", "IntervalEnd", "Shift"]).reset_index(drop=True)

    targets = intervals.loc[intervals["Shift"] == config.evaluation_shift].reset_index(drop=True)
    folds: list[WalkForwardFold] = []
    train_window = pd.Timedelta(days=int(round(config.train_window_years * 365.25)))

    for target in targets.itertuples(index=False):
        history = intervals.loc[intervals["IntervalEnd"] < target.IntervalStart].copy()
        history = history.loc[history["IntervalStart"] >= target.IntervalStart - train_window]
        if len(history) < config.min_history_intervals:
            continue

        n = len(history)
        train_end_idx = max(1, int(n * config.train_fraction))
        test_end_idx = max(train_end_idx + 1, int(n * (config.train_fraction + config.test_fraction)))
        if test_end_idx >= n:
            test_end_idx = n - 1
        if train_end_idx >= n - 1:
            continue

        train_intervals = tuple(history.iloc[:train_end_idx]["Interval"].astype(str).tolist())
        test_intervals = tuple(history.iloc[train_end_idx:test_end_idx]["Interval"].astype(str).tolist())
        validation_intervals = tuple(history.iloc[test_end_idx:]["Interval"].astype(str).tolist())
        if not test_intervals or not validation_intervals:
            continue

        folds.append(
            WalkForwardFold(
                fold_index=len(folds),
                retrain=(len(folds) % config.retrain_every == 0),
                target_shift=int(target.Shift),
                target_interval=str(target.Interval),
                target_start=pd.Timestamp(target.IntervalStart),
                target_end=pd.Timestamp(target.IntervalEnd),
                history_start=pd.Timestamp(history["IntervalStart"].min()),
                history_end=pd.Timestamp(history["IntervalEnd"].max()),
                train_intervals=train_intervals,
                test_intervals=test_intervals,
                validation_intervals=validation_intervals,
            )
        )
    return folds


def _train_models_for_fold(
    *,
    history_frame: pd.DataFrame,
    train_intervals: tuple[str, ...],
    test_intervals: tuple[str, ...],
    validation_intervals: tuple[str, ...],
    feature_names: list[str],
    ticker_categories: list[str],
    config: WalkForwardConfig,
    temp_dir: str,
    previous_base_state: dict[str, torch.Tensor] | None,
    previous_meta_state: dict[str, torch.Tensor] | None,
) -> tuple[ConstructFFNN, MetaModel]:
    train_frame = history_frame.loc[history_frame["Interval"].isin(train_intervals)].reset_index(drop=True)
    test_frame = history_frame.loc[history_frame["Interval"].isin(test_intervals)].reset_index(drop=True)
    validation_frame = history_frame.loc[history_frame["Interval"].isin(validation_intervals)].reset_index(drop=True)

    medians = _fit_feature_medians(train_frame, feature_names)
    train_frame = _apply_preprocessing(train_frame, feature_names=feature_names, medians=medians)
    test_frame = _apply_preprocessing(test_frame, feature_names=feature_names, medians=medians)
    validation_frame = _apply_preprocessing(validation_frame, feature_names=feature_names, medians=medians)

    x_train = _build_feature_tensor(train_frame, feature_names)
    y_train = _build_target_tensor(train_frame)
    xtype_train = _build_sparse_ticker_tensor(train_frame, ticker_categories)

    x_test = _build_feature_tensor(test_frame, feature_names)
    y_test = _build_target_tensor(test_frame)
    xtype_test = _build_sparse_ticker_tensor(test_frame, ticker_categories)

    x_validation = _build_feature_tensor(validation_frame, feature_names)
    y_validation = _build_target_tensor(validation_frame)
    xtype_validation = _build_sparse_ticker_tensor(validation_frame, ticker_categories)

    input_size = len(feature_names)
    layer_sizes = [32, 8, 5]
    layer_dropouts = [0.2] * (len(layer_sizes) - 1) + [0.0]
    layer_transforms = [F.leaky_relu] * (len(layer_sizes) - 1) + [
        lambda x: F.softmax(x, dim=1)
    ]

    base_model = ConstructFFNN(input_size, layer_sizes, layer_transforms, layer_dropouts)
    if previous_base_state is not None and config.warm_start:
        base_model.load_state_dict(previous_base_state)
    base_model = prepare_base_model(base_model, x_train)
    fit = train_model(
        model=base_model,
        criterion=compute_rps_tensor,
        train=[y_train, x_train],
        test=[y_test, x_test],
        validation=[y_validation, x_validation],
        epochs=config.base_epochs,
        minibatch=config.base_minibatch,
        temp_dir=temp_dir,
        patience=config.base_patience,
        print_every=10_000,
        lr=list(config.base_lr),
    )
    base_model = fit["model"]

    allow_meta_structure = _build_allow_meta_structure(base_model)
    meta_model = MetaModel(
        base_model,
        xtype_train,
        mesa_parameter_size=1,
        allow_bias=False,
        p_dropout=0.15,
        init_mesa_range=0.01,
        init_meta_range=1.0,
        allow_meta_structure=allow_meta_structure,
    )
    if previous_meta_state is not None and config.warm_start:
        current_meta_state = meta_model.state_dict()
        for key in ["mesa_layer_weight", "meta_layer_weight", "meta_layer_bias"]:
            if key in previous_meta_state and key in current_meta_state:
                current_meta_state[key] = previous_meta_state[key].detach().clone()
        meta_model.load_state_dict(current_meta_state)

    mb_fn = lambda: minibatch_sampler(config.meta_minibatch_size, xtype_train)
    fit_meta = train_model(
        model=meta_model,
        criterion=compute_rps_tensor,
        train=[y_train, x_train, xtype_train],
        test=[y_validation, x_validation, xtype_validation],
        validation=[y_test, x_test, xtype_test],
        epochs=config.meta_epochs,
        minibatch=mb_fn,
        temp_dir=temp_dir,
        patience=config.meta_patience,
        print_every=10_000,
        lr=list(config.meta_lr),
    )
    return base_model, fit_meta["model"]


def run(
    *,
    config: WalkForwardConfig | None = None,
) -> tuple[str, str]:
    started_at = time.time()
    cfg = config or WalkForwardConfig()
    np.random.seed(1)
    torch.manual_seed(1)

    stock_names, stocks = _load_inputs()
    raw_features = _load_or_generate_raw_features(stock_names=stock_names, stocks=stocks).copy()
    raw_features["IntervalStart"] = pd.to_datetime(raw_features["IntervalStart"])
    raw_features["IntervalEnd"] = pd.to_datetime(raw_features["IntervalEnd"])
    raw_features = raw_features.sort_values(["IntervalStart", "Ticker"]).reset_index(drop=True)

    exclude_cols = {
        "Ticker", "Interval", "Return", "Shift",
        "ReturnQuintile", "IntervalStart", "IntervalEnd", *[f"TargetRank{i}" for i in range(1, 6)],
    }
    feature_names = [column for column in raw_features.columns if column not in exclude_cols]
    if not feature_names:
        raise RuntimeError("No feature columns were produced for walk-forward training.")

    ticker_categories = sorted(raw_features["Ticker"].astype(str).unique().tolist())
    folds = build_walk_forward_folds(raw_features, config=cfg)
    if not folds:
        raise RuntimeError("No walk-forward folds available for the current configuration.")

    print(
        "[WalkForward] Prepared "
        f"{len(folds)} folds (shift={cfg.evaluation_shift}, "
        f"window={cfg.train_window_years:.1f}y, retrain_every={cfg.retrain_every})."
    )

    current_base_model: ConstructFFNN | None = None
    current_meta_model: MetaModel | None = None
    current_medians: dict[str, float] | None = None
    previous_base_state: dict[str, torch.Tensor] | None = None
    previous_meta_state: dict[str, torch.Tensor] | None = None
    records: list[dict[str, object]] = []

    for fold in folds:
        target_frame = raw_features.loc[
            (raw_features["Interval"] == fold.target_interval)
            & (raw_features["Shift"] == fold.target_shift)
        ].reset_index(drop=True)
        if target_frame.empty:
            continue

        target_interval_start = fold.target_start
        history_frame = raw_features.loc[
            (raw_features["IntervalEnd"] < target_interval_start)
            & (raw_features["IntervalStart"] >= fold.history_start)
        ].reset_index(drop=True)

        if fold.retrain or current_base_model is None or current_meta_model is None or current_medians is None:
            print(
                "[WalkForward] Retraining fold "
                f"{fold.fold_index + 1}/{len(folds)} "
                f"for target {fold.target_start.date()} → {fold.target_end.date()}"
            )
            current_medians = _fit_feature_medians(
                history_frame.loc[history_frame["Interval"].isin(fold.train_intervals)],
                feature_names,
            )
            current_base_model, current_meta_model = _train_models_for_fold(
                history_frame=history_frame,
                train_intervals=fold.train_intervals,
                test_intervals=fold.test_intervals,
                validation_intervals=fold.validation_intervals,
                feature_names=feature_names,
                ticker_categories=ticker_categories,
                config=cfg,
                temp_dir=TEMP_DIR,
                previous_base_state=previous_base_state,
                previous_meta_state=previous_meta_state,
            )
            previous_base_state = {
                key: value.detach().clone()
                for key, value in current_base_model.state_dict().items()
            }
            previous_meta_state = {
                key: value.detach().clone()
                for key, value in current_meta_model.state_dict().items()
            }

        target_processed = _apply_preprocessing(
            target_frame,
            feature_names=feature_names,
            medians=current_medians,
        )
        x_target = _build_feature_tensor(target_processed, feature_names)
        xtype_target = _build_sparse_ticker_tensor(target_processed, ticker_categories)

        with torch.no_grad():
            base_pred = current_base_model(x_target).numpy()
            meta_pred = current_meta_model(x_target, xtype_target).numpy()

        for model_name, prediction in [("base", base_pred), ("meta", meta_pred)]:
            submission = target_frame[["Ticker"]].copy().rename(columns={"Ticker": "ID"})
            for idx, column in enumerate(RANK_COLUMNS):
                submission[column] = prediction[:, idx]
            submission, allocation_summary = build_m6_baseline_submission(submission)
            period_return = _compute_portfolio_return(
                stocks=stocks,
                submission=submission[["ID", *RANK_COLUMNS, "Decision"]],
                start_date=fold.target_start,
                end_date=fold.target_end,
            )
            records.append(
                {
                    "Model": model_name,
                    "FoldIndex": fold.fold_index,
                    "Retrained": fold.retrain,
                    "TargetShift": fold.target_shift,
                    "PeriodStart": str(fold.target_start.date()),
                    "PeriodEnd": str(fold.target_end.date()),
                    "TrainWindowStart": str(fold.history_start.date()),
                    "TrainWindowEnd": str(fold.history_end.date()),
                    "TrainIntervals": len(fold.train_intervals),
                    "TestIntervals": len(fold.test_intervals),
                    "ValidationIntervals": len(fold.validation_intervals),
                    "Longs": ",".join(allocation_summary["long_ids"]),
                    "Shorts": ",".join(allocation_summary["short_ids"]),
                    "GrossExposure": allocation_summary["gross_exposure"],
                    "PeriodReturn": period_return,
                }
            )

    if not records:
        raise RuntimeError("Walk-forward run produced no portfolio records.")

    backtest = pd.DataFrame(records)
    backtest_path = os.path.join(FORECASTS_DIR, "portfolio_walk_forward_backtest.csv")
    backtest.to_csv(backtest_path, index=False)

    summary: dict[str, object] = {
        "config": asdict(cfg),
        "elapsed_seconds": round(time.time() - started_at, 4),
        "models": {},
    }
    for model_name, model_frame in backtest.groupby("Model", sort=False):
        returns = model_frame["PeriodReturn"].dropna().astype(float)
        if returns.empty:
            metrics = {
                "intervals": 0,
                "annualized_return": None,
                "cumulative_return": None,
                "average_period_return": None,
                "positive_periods": 0,
                "start": None,
                "end": None,
            }
        else:
            metrics = {
                "intervals": int(len(returns)),
                "annualized_return": float(np.prod(1.0 + returns.values) ** (365 / (28 * len(returns))) - 1.0),
                "cumulative_return": float(np.prod(1.0 + returns.values) - 1.0),
                "average_period_return": float(returns.mean()),
                "positive_periods": int((returns > 0).sum()),
                "start": str(model_frame["PeriodStart"].min()),
                "end": str(model_frame["PeriodEnd"].max()),
                "min_period_return": float(returns.min()),
                "max_period_return": float(returns.max()),
            }
        summary["models"][model_name] = metrics

    summary_path = os.path.join(FORECASTS_DIR, "portfolio_walk_forward_backtest_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[WalkForward] Saved backtest → {backtest_path}")
    print(f"[WalkForward] Saved summary → {summary_path}")
    for model_name, metrics in summary["models"].items():
        print(
            "[WalkForward] "
            f"{model_name}: annualized={metrics['annualized_return']:.5f} "
            f"cumulative={metrics['cumulative_return']:.5f} "
            f"positive={metrics['positive_periods']}/{metrics['intervals']}"
        )
    return backtest_path, summary_path


if __name__ == "__main__":
    run()
