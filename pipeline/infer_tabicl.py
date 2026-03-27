"""
Inference-only forecast generation using the walk-forward TabICL v2 protocol.

This mirrors the latest causal walk-forward fold: train on the trailing
history available strictly before the current target interval and score the
latest non-overlapping Shift-0 window.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, FORECASTS_DIR
from .forecast import validate_submission
from .m6_baseline import RANK_COLUMNS, build_m6_baseline_submission
from .walk_forward import (
    WalkForwardConfig,
    _apply_preprocessing,
    _fit_feature_medians,
    _load_inputs,
    _load_or_generate_raw_features,
    build_walk_forward_folds,
)
from .walk_forward_tabicl import (
    TABICL_BATCH_SIZE,
    TABICL_N_ESTIMATORS,
    _feature_columns,
    _predict_rank_probabilities,
    _train_model_for_fold,
)


def _build_template(stock_names: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": stock_names["Symbol"].astype(str).str.upper(),
            "Rank1": 0.2,
            "Rank2": 0.2,
            "Rank3": 0.2,
            "Rank4": 0.2,
            "Rank5": 0.2,
            "Decision": 0.0,
        }
    )


def _prepare_latest_fold(
    *,
    raw_features: pd.DataFrame,
    config: WalkForwardConfig,
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    folds = build_walk_forward_folds(raw_features, config=config)
    if not folds:
        raise RuntimeError("No walk-forward folds available for TabICL live inference.")

    fold = folds[-1]
    target_frame = raw_features.loc[
        (raw_features["Interval"] == fold.target_interval)
        & (raw_features["Shift"] == fold.target_shift)
    ].reset_index(drop=True)
    if target_frame.empty:
        raise RuntimeError(f"Target interval {fold.target_interval} produced no rows.")

    history_frame = raw_features.loc[
        (raw_features["IntervalEnd"] < fold.target_start)
        & (raw_features["IntervalStart"] >= fold.history_start)
    ].reset_index(drop=True)
    if history_frame.empty:
        raise RuntimeError("No historical rows available for latest TabICL fold.")

    return fold, history_frame, target_frame


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
        raise RuntimeError("No feature columns were produced for TabICL inference.")

    fold, history_frame, target_frame = _prepare_latest_fold(raw_features=raw_features, config=cfg)
    train_frame = history_frame.loc[history_frame["Interval"].isin(fold.train_intervals)].reset_index(drop=True)
    medians = _fit_feature_medians(train_frame, feature_names)
    model, _, fit_info = _train_model_for_fold(
        history_frame=history_frame,
        train_intervals=fold.train_intervals,
        feature_names=feature_names,
    )
    target_processed = _apply_preprocessing(target_frame, feature_names=feature_names, medians=medians)

    predict_started = time.time()
    probabilities = _predict_rank_probabilities(model, target_processed, feature_names)
    predict_seconds = time.time() - predict_started

    prediction_rows = pd.DataFrame({"ID": target_frame["Ticker"].astype(str).values})
    for idx, rank_col in enumerate(RANK_COLUMNS):
        prediction_rows[rank_col] = probabilities[:, idx]

    template = _build_template(stock_names)
    template_path = os.path.join(FORECASTS_DIR, "ranked_forecast_template.csv")
    template.to_csv(template_path, index=False)
    print(f"[TabICL Inference] Refreshed template -> {template_path}")

    submission = template[["ID"]].merge(prediction_rows, on="ID", how="left")
    for rank_col in RANK_COLUMNS:
        submission[rank_col] = submission[rank_col].fillna(0.2).astype(float)
    submission, allocation_summary = build_m6_baseline_submission(submission)
    submission = submission[["ID", *RANK_COLUMNS, "Decision"]]
    submission = validate_submission(submission, template, do_round=True)

    run_date = pd.Timestamp(fold.target_end).date().isoformat()
    forecast_path = os.path.join(FORECASTS_DIR, f"ranked_forecast_{run_date}_tabicl_inference.csv")
    submission.to_csv(forecast_path, index=False)

    metadata = {
        "model": "tabicl_v2",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": round(time.time() - started_at, 4),
        "predict_seconds": round(predict_seconds, 4),
        "target_interval": str(fold.target_interval),
        "target_shift": int(fold.target_shift),
        "train_window_start": str(pd.Timestamp(fold.history_start).date()),
        "train_window_end": str(pd.Timestamp(fold.history_end).date()),
        "train_intervals": len(fold.train_intervals),
        "test_intervals": len(fold.test_intervals),
        "validation_intervals": len(fold.validation_intervals),
        "train_rows": int(len(train_frame)),
        "target_rows": int(len(target_frame)),
        "device": fit_info["device"],
        "checkpoint_version": fit_info["checkpoint_version"],
        "batch_size": TABICL_BATCH_SIZE,
        "n_estimators": TABICL_N_ESTIMATORS,
        "allocation": allocation_summary,
    }
    metadata_path = forecast_path.replace(".csv", "_meta.json")
    Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        "[TabICL Inference] Allocation "
        f"(gross exposure={allocation_summary['gross_exposure']:.4f}, "
        f"longs={allocation_summary['long_ids']}, shorts={allocation_summary['short_ids']})"
    )
    print(f"[TabICL Inference] Exported forecast -> {forecast_path}")
    print(f"[TabICL Inference] Exported metadata -> {metadata_path}")
    return forecast_path, metadata_path


if __name__ == "__main__":
    run()
