"""
TabICL v2 walk-forward backtest using the same causal fold construction
and portfolio-return logic as the FFNN baseline.

Writes:

- forecasts/portfolio_walk_forward_tabicl_backtest.csv
- forecasts/portfolio_walk_forward_tabicl_summary.json
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tabicl import TabICLClassifier

from .config import FORECASTS_DIR
from .m6_baseline import RANK_COLUMNS, build_m6_baseline_submission
from .walk_forward import (
    WalkForwardConfig,
    _apply_preprocessing,
    _compute_portfolio_return,
    _fit_feature_medians,
    _load_inputs,
    _load_or_generate_raw_features,
    build_walk_forward_folds,
)

TABICL_BATCH_SIZE = 256
TABICL_N_ESTIMATORS = 1


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude_cols = {
        "Ticker",
        "Interval",
        "Return",
        "Shift",
        "ReturnQuintile",
        "IntervalStart",
        "IntervalEnd",
        *[f"TargetRank{i}" for i in range(1, 6)],
    }
    return [column for column in frame.columns if column not in exclude_cols]


def _prepare_xy(frame: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = frame[feature_names].to_numpy(dtype=np.float32)
    y = frame["ReturnQuintile"].astype(int).to_numpy(dtype=np.int64) - 1
    return x, y


def _predict_rank_probabilities(
    model: TabICLClassifier,
    frame: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    x = frame[feature_names].to_numpy(dtype=np.float32)
    probabilities = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)

    full = np.zeros((len(frame), 5), dtype=float)
    for idx, class_id in enumerate(classes):
        if 0 <= int(class_id) < 5:
            full[:, int(class_id)] = probabilities[:, idx]
    return full


def _train_model_for_fold(
    *,
    history_frame: pd.DataFrame,
    train_intervals: tuple[str, ...],
    feature_names: list[str],
) -> tuple[TabICLClassifier, dict[str, float], dict[str, object]]:
    train_frame = history_frame.loc[history_frame["Interval"].isin(train_intervals)].reset_index(drop=True)
    medians = _fit_feature_medians(train_frame, feature_names)
    train_processed = _apply_preprocessing(train_frame, feature_names=feature_names, medians=medians)
    x_train, y_train = _prepare_xy(train_processed, feature_names)

    model = TabICLClassifier(
        device="cpu",
        random_state=1,
        n_jobs=1,
        verbose=False,
        batch_size=TABICL_BATCH_SIZE,
        n_estimators=TABICL_N_ESTIMATORS,
    )
    model.fit(x_train, y_train)
    fit_info = {
        "device": "cpu",
        "checkpoint_version": str(getattr(model, "checkpoint_version", "tabicl_v2_default")),
        "train_rows": int(len(train_frame)),
        "batch_size": TABICL_BATCH_SIZE,
        "n_estimators": TABICL_N_ESTIMATORS,
    }
    return model, medians, fit_info


def _build_summary(backtest: pd.DataFrame, cfg: WalkForwardConfig, started_at: float) -> dict[str, object]:
    returns = backtest["PeriodReturn"].dropna().astype(float)
    if returns.empty:
        metrics: dict[str, object] = {
            "intervals": 0,
            "annualized_return": None,
            "cumulative_return": None,
            "average_period_return": None,
            "positive_periods": 0,
            "start": None,
            "end": None,
            "min_period_return": None,
            "max_period_return": None,
        }
    else:
        metrics = {
            "intervals": int(len(returns)),
            "annualized_return": float(np.prod(1.0 + returns.values) ** (365 / (28 * len(returns))) - 1.0),
            "cumulative_return": float(np.prod(1.0 + returns.values) - 1.0),
            "average_period_return": float(returns.mean()),
            "positive_periods": int((returns > 0).sum()),
            "start": str(backtest["PeriodStart"].min()),
            "end": str(backtest["PeriodEnd"].max()),
            "min_period_return": float(returns.min()),
            "max_period_return": float(returns.max()),
            "average_gross_exposure": float(backtest["GrossExposure"].astype(float).mean()),
            "retraining_events": int(backtest["Retrained"].astype(bool).sum()),
            "average_predict_seconds": float(backtest["PredictSeconds"].astype(float).mean()),
        }

    summary: dict[str, object] = {
        "config": asdict(cfg),
        "elapsed_seconds": round(time.time() - started_at, 4),
        "models": {
            "tabicl_v2": metrics,
        },
    }

    baseline_summary_path = Path(FORECASTS_DIR) / "portfolio_walk_forward_backtest_summary.json"
    if baseline_summary_path.exists():
        try:
            baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
            baseline_models = baseline_summary.get("models", {})
            if "base" in baseline_models:
                summary["baseline_reference"] = {
                    "base": baseline_models["base"],
                }
            if "meta" in baseline_models:
                summary.setdefault("baseline_reference", {})["meta"] = baseline_models["meta"]
        except Exception:
            pass

    catboost_summary_path = Path(FORECASTS_DIR) / "portfolio_walk_forward_catboost_summary.json"
    if catboost_summary_path.exists():
        try:
            catboost_summary = json.loads(catboost_summary_path.read_text(encoding="utf-8"))
            summary["catboost_reference"] = catboost_summary.get("models", {}).get("catboost")
        except Exception:
            pass

    return summary


def run(
    *,
    config: WalkForwardConfig | None = None,
) -> tuple[str, str]:
    started_at = time.time()
    cfg = config or WalkForwardConfig()
    np.random.seed(1)

    stock_names, stocks = _load_inputs()
    raw_features = _load_or_generate_raw_features(stock_names=stock_names, stocks=stocks).copy()
    raw_features["IntervalStart"] = pd.to_datetime(raw_features["IntervalStart"])
    raw_features["IntervalEnd"] = pd.to_datetime(raw_features["IntervalEnd"])
    raw_features = raw_features.sort_values(["IntervalStart", "Ticker"]).reset_index(drop=True)

    feature_names = _feature_columns(raw_features)
    if not feature_names:
        raise RuntimeError("No feature columns were produced for TabICL walk-forward training.")

    folds = build_walk_forward_folds(raw_features, config=cfg)
    if not folds:
        raise RuntimeError("No walk-forward folds available for the current configuration.")

    print(
        "[WalkForwardTabICL] Prepared "
        f"{len(folds)} folds (shift={cfg.evaluation_shift}, "
        f"window={cfg.train_window_years:.1f}y, retrain_every={cfg.retrain_every})."
    )

    current_model: TabICLClassifier | None = None
    current_medians: dict[str, float] | None = None
    current_fit_info: dict[str, object] | None = None
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

        if fold.retrain or current_model is None or current_medians is None or current_fit_info is None:
            print(
                "[WalkForwardTabICL] Retraining fold "
                f"{fold.fold_index + 1}/{len(folds)} "
                f"for target {fold.target_start.date()} -> {fold.target_end.date()}"
            )
            current_model, current_medians, current_fit_info = _train_model_for_fold(
                history_frame=history_frame,
                train_intervals=fold.train_intervals,
                feature_names=feature_names,
            )

        target_processed = _apply_preprocessing(
            target_frame,
            feature_names=feature_names,
            medians=current_medians,
        )
        predict_started = time.time()
        probabilities = _predict_rank_probabilities(current_model, target_processed, feature_names)
        predict_seconds = time.time() - predict_started

        submission = target_frame[["Ticker"]].copy().rename(columns={"Ticker": "ID"})
        for idx, column in enumerate(RANK_COLUMNS):
            submission[column] = probabilities[:, idx]
        submission, allocation_summary = build_m6_baseline_submission(submission)

        period_return = _compute_portfolio_return(
            stocks=stocks,
            submission=submission[["ID", *RANK_COLUMNS, "Decision"]],
            start_date=fold.target_start,
            end_date=fold.target_end,
        )
        records.append(
            {
                "Model": "tabicl_v2",
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
                "CheckpointVersion": current_fit_info["checkpoint_version"],
                "Device": current_fit_info["device"],
                "BatchSize": current_fit_info["batch_size"],
                "NEstimators": current_fit_info["n_estimators"],
                "PredictSeconds": predict_seconds,
                "Longs": ",".join(allocation_summary["long_ids"]),
                "Shorts": ",".join(allocation_summary["short_ids"]),
                "GrossExposure": allocation_summary["gross_exposure"],
                "PeriodReturn": period_return,
            }
        )

    if not records:
        raise RuntimeError("TabICL walk-forward run produced no portfolio records.")

    backtest = pd.DataFrame(records)
    backtest_path = os.path.join(FORECASTS_DIR, "portfolio_walk_forward_tabicl_backtest.csv")
    backtest.to_csv(backtest_path, index=False)

    summary = _build_summary(backtest, cfg, started_at)
    summary_path = os.path.join(FORECASTS_DIR, "portfolio_walk_forward_tabicl_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    metrics = summary["models"]["tabicl_v2"]
    print(f"[WalkForwardTabICL] Saved backtest -> {backtest_path}")
    print(f"[WalkForwardTabICL] Saved summary -> {summary_path}")
    print(
        "[WalkForwardTabICL] "
        f"annualized={metrics['annualized_return']:.5f} "
        f"cumulative={metrics['cumulative_return']:.5f} "
        f"positive={metrics['positive_periods']}/{metrics['intervals']}"
    )
    return backtest_path, summary_path


if __name__ == "__main__":
    run()
