"""
Step 04 — Train Quantile Models.

Loads cleaned data, engineers features, trains a base FFNN and a MetaModel,
and saves quantile predictions for all splits.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from datetime import date
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import DATA_DIR, FEATURES_DIR, TEMP_DIR, SHIFTS, SUBMISSION_INTERVALS
from .m6_metrics import TARGET_RANK_COLUMNS
from .training_utils import (
    ConstructFFNN,
    compute_rps_tensor,
    compute_rps_tensor_vector,
    gen_interval_infos,
    gen_stocks_aggr,
    impute_features,
    standardize_features,
    subset_tensor,
    train_model,
    minibatch_sampler,
)
from .features import (
    compute_return,
    lag_return,
    lag_volatility,
    is_etf,
    TTR_FEATURES,
)
from .models import MetaModel, prepare_base_model

RESULTS_PATH = "results.json"


def run() -> None:
    """Full training pipeline."""

    run_started_at = time.time()
    np.random.seed(1)
    torch.manual_seed(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    stock_names = pd.read_parquet(os.path.join(DATA_DIR, "tickers_metadata.parquet"))
    with open(os.path.join(DATA_DIR, "tickers_data_cleaned.pkl"), "rb") as f:
        stocks = pickle.load(f)
    # Sort by symbol
    stocks = dict(sorted(stocks.items()))

    # ------------------------------------------------------------------
    # Feature list
    # ------------------------------------------------------------------
    feature_fns = [
        compute_return,
        lambda df, t: lag_volatility(df, t, lags=list(range(1, 8))),
        lambda df, t: lag_return(df, t, lags=list(range(1, 8))),
        lambda df, t: is_etf(df, t, stock_names=stock_names),
    ] + TTR_FEATURES

    # ------------------------------------------------------------------
    # Interval infos & feature aggregation
    # ------------------------------------------------------------------
    interval_infos = gen_interval_infos(
        submission=SUBMISSION_INTERVALS, shifts=SHIFTS
    )

    metadata_path = os.path.join(DATA_DIR, "tickers_metadata.parquet")
    cleaned_data_path = os.path.join(DATA_DIR, "tickers_data_cleaned.pkl")
    precomputed_path = os.path.join(FEATURES_DIR, "features_raw.parquet")
    required_cols = {"Interval", "Ticker", "Shift", "IntervalStart", "IntervalEnd"}

    def _generate_and_cache() -> pd.DataFrame:
        print("[Step 04] Generating aggregated features …")
        generated = gen_stocks_aggr(
            stocks, interval_infos, feature_fns, check_leakage=False
        )
        if generated.empty:
            raise RuntimeError(
                "Feature aggregation produced no rows. Check downloaded ticker data."
            )
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
                "[Step 04] Cached features invalid or stale "
                f"(missing: {missing}, rows: {len(stocks_aggr)}, stale: {cache_stale}). Regenerating."
            )
            stocks_aggr = _generate_and_cache()
    else:
        stocks_aggr = _generate_and_cache()

    # ------------------------------------------------------------------
    # Impute & standardise
    # ------------------------------------------------------------------
    exclude_cols = {
        "Ticker", "Interval", "Return", "Shift",
        "ReturnQuintile", "IntervalStart", "IntervalEnd", *TARGET_RANK_COLUMNS,
    }
    feature_names = [c for c in stocks_aggr.columns if c not in exclude_cols]
    if not feature_names:
        raise RuntimeError("No feature columns were produced for model training.")

    stocks_aggr = impute_features(stocks_aggr, feature_names)
    std_features = [f for f in feature_names if f != "ETF"]
    stocks_aggr = standardize_features(stocks_aggr, std_features)
    stocks_aggr = stocks_aggr.sort_values(["IntervalStart", "Ticker"]).reset_index(drop=True)
    stocks_aggr.to_parquet(
        os.path.join(FEATURES_DIR, "features_standardized.parquet"), index=False
    )

    # ------------------------------------------------------------------
    # Train / Test / Validation split
    # ------------------------------------------------------------------
    train_start = pd.Timestamp("2000-01-01")
    stocks_aggr["IntervalStart"] = pd.to_datetime(stocks_aggr["IntervalStart"])
    stocks_aggr["IntervalEnd"] = pd.to_datetime(stocks_aggr["IntervalEnd"])
    intervals = (
        stocks_aggr.loc[stocks_aggr["IntervalStart"] >= train_start, ["IntervalStart", "IntervalEnd"]]
        .drop_duplicates()
        .sort_values("IntervalStart")
        .reset_index(drop=True)
    )
    n = len(intervals)
    train_end_idx = int(0.8 * n)
    test_end_idx = int(0.9 * n)

    train_end = intervals.loc[train_end_idx - 1, "IntervalEnd"]
    test_start = intervals.loc[train_end_idx, "IntervalStart"]
    test_end = intervals.loc[test_end_idx - 1, "IntervalEnd"]
    val_start = intervals.loc[test_end_idx, "IntervalStart"]
    val_end = intervals.loc[n - 1, "IntervalEnd"]

    sa = stocks_aggr
    train_mask = (sa["IntervalStart"] >= train_start) & (sa["IntervalEnd"] <= train_end)
    test_mask = (sa["IntervalStart"] >= test_start) & (sa["IntervalEnd"] <= test_end)
    val_mask = (sa["IntervalStart"] >= val_start) & (sa["IntervalEnd"] <= val_end)

    train_rows = sa.index[train_mask].tolist()
    test_rows = sa.index[test_mask].tolist()
    val_rows = sa.index[val_mask].tolist()

    print(f"[Step 04] Split — train: {len(train_rows)}  test: {len(test_rows)}  val: {len(val_rows)}")
    print(f"          Validation: {val_start} → {val_end}")

    # ------------------------------------------------------------------
    # Prepare tensors
    # ------------------------------------------------------------------
    if set(TARGET_RANK_COLUMNS).issubset(sa.columns):
        target_probs = sa[TARGET_RANK_COLUMNS].fillna(0.0).to_numpy(dtype=float)
        y_tensor = torch.tensor(np.cumsum(target_probs, axis=1), dtype=torch.float32)
    else:
        y_quintiles = sa["ReturnQuintile"].values
        y_tensor = torch.zeros(len(y_quintiles), 5)
        for i, q in enumerate(y_quintiles):
            if not np.isnan(q):
                q = int(q)
                y_tensor[i, q - 1 :] = 1.0

    x = torch.tensor(sa[feature_names].values, dtype=torch.float32)

    # Sparse ticker indicator
    tickers = pd.Categorical(sa["Ticker"])
    row_idx = torch.arange(len(tickers), dtype=torch.long)
    col_idx = torch.tensor(tickers.codes, dtype=torch.long)
    indices = torch.stack([row_idx, col_idx])
    values = torch.ones(len(tickers))
    torch.sparse.check_sparse_tensor_invariants.disable()
    xtype = torch.sparse_coo_tensor(
        indices,
        values,
        (len(tickers), len(tickers.categories)),
    ).coalesce()

    # Split
    y_train = y_tensor[train_rows]
    x_train = x[train_rows]
    xtype_train = subset_tensor(xtype, train_rows)
    y_test = y_tensor[test_rows]
    x_test = x[test_rows]
    xtype_test = subset_tensor(xtype, test_rows)
    y_val = y_tensor[val_rows]
    x_val = x[val_rows]
    xtype_val = subset_tensor(xtype, val_rows)

    criterion = compute_rps_tensor

    # ------------------------------------------------------------------
    # Train base model
    # ------------------------------------------------------------------
    input_size = len(feature_names)
    layer_sizes = [32, 8, 5]
    layer_dropouts = [0.2] * (len(layer_sizes) - 1) + [0.0]
    layer_transforms = [F.leaky_relu] * (len(layer_sizes) - 1) + [
        lambda x: F.softmax(x, dim=1)
    ]
    base_epochs = 100
    base_minibatch = 200
    base_patience = 5
    base_lr = [0.01]

    base_model = ConstructFFNN(input_size, layer_sizes, layer_transforms, layer_dropouts)
    base_model = prepare_base_model(base_model, x_train)

    print("[Step 04] Training base model …")
    t0 = time.time()
    fit = train_model(
        model=base_model,
        criterion=criterion,
        train=[y_train, x_train],
        test=[y_test, x_test],
        validation=[y_val, x_val],
        epochs=base_epochs,
        minibatch=base_minibatch,
        temp_dir=TEMP_DIR,
        patience=base_patience,
        print_every=1,
        lr=base_lr,
    )
    base_elapsed = time.time() - t0
    print(f"[Step 04] Base model trained in {base_elapsed:.1f}s")

    base_model = fit["model"]
    fit["progress"].to_parquet(os.path.join(FEATURES_DIR, "training_log_base.parquet"), index=False)
    torch.save(base_model.state_dict(), os.path.join(FEATURES_DIR, "model_base.pt"))

    # Evaluate base
    base_model.eval()
    with torch.no_grad():
        y_pred_base = base_model(x_val)
        loss_base = compute_rps_tensor(y_pred_base, y_val).item()
        loss_base_vec = compute_rps_tensor_vector(y_pred_base, y_val).detach().numpy()
    print(f"[Step 04] Base validation loss: {loss_base:.5f}")

    # ------------------------------------------------------------------
    # Build allow_meta_structure (last 3 layers only)
    # ------------------------------------------------------------------
    keys = list(base_model.state_structure.keys())
    allow_meta_structure = OrderedDict()
    n_layers = len(base_model.layers)
    included_keys = set()
    for li in range(max(0, n_layers - 3), n_layers):
        included_keys.add(f"layers.{li}.weight")
        included_keys.add(f"layers.{li}.bias")

    for k, shape in base_model.state_structure.items():
        if k in included_keys:
            allow_meta_structure[k] = torch.ones(shape, dtype=torch.bool)
        else:
            allow_meta_structure[k] = torch.zeros(shape, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Train meta-model
    # ------------------------------------------------------------------
    meta_mesa_parameter_size = 1
    meta_allow_bias = False
    meta_dropout = 0.15
    meta_init_mesa_range = 0.01
    meta_init_meta_range = 1
    meta_epochs = 100
    meta_patience = 10
    meta_lr = [0.01, 0.001, 0.001, 0.0005, 0.0003, 0.0001, 0.00005]
    meta_minibatch_size = 4

    meta_model = MetaModel(
        base_model,
        xtype_train,
        mesa_parameter_size=meta_mesa_parameter_size,
        allow_bias=meta_allow_bias,
        p_dropout=meta_dropout,
        init_mesa_range=meta_init_mesa_range,
        init_meta_range=meta_init_meta_range,
        allow_meta_structure=allow_meta_structure,
    )

    mb_fn = lambda: minibatch_sampler(meta_minibatch_size, xtype_train)
    meta_selection_split = "validation"

    print("[Step 04] Training meta-model …")
    t0 = time.time()
    fit_meta = train_model(
        model=meta_model,
        criterion=criterion,
        train=[y_train, x_train, xtype_train],
        test=[y_val, x_val, xtype_val],
        validation=[y_test, x_test, xtype_test],
        epochs=meta_epochs,
        minibatch=mb_fn,
        temp_dir=TEMP_DIR,
        patience=meta_patience,
        print_every=1,
        lr=meta_lr,
    )
    meta_elapsed = time.time() - t0
    print(f"[Step 04] Meta-model trained in {meta_elapsed:.1f}s")

    meta_model = fit_meta["model"]
    fit_meta["progress"].to_parquet(os.path.join(FEATURES_DIR, "training_log_meta.parquet"), index=False)
    torch.save(meta_model.state_dict(), os.path.join(FEATURES_DIR, "model_meta.pt"))

    meta_model.eval()
    with torch.no_grad():
        y_pred_meta = meta_model(x_val, xtype_val)
        loss_meta = compute_rps_tensor(y_pred_meta, y_val).item()
    print(f"[Step 04] Meta validation loss: {loss_meta:.5f}")

    best_meta_progress = fit_meta["progress"].loc[
        fit_meta["progress"]["loss_validation"].astype(float).idxmin()
    ]
    results = {
        "loss_base": float(loss_base),
        "loss_meta": float(loss_meta),
        "loss_meta_best_progress": float(best_meta_progress["loss_validation"]),
        "elapsed_seconds_total": round(time.time() - run_started_at, 4),
        "elapsed_seconds_base": round(base_elapsed, 4),
        "elapsed_seconds_meta": round(meta_elapsed, 4),
        "config": {
            "layer_sizes": layer_sizes,
            "layer_dropouts": layer_dropouts,
            "base_epochs": base_epochs,
            "base_minibatch": base_minibatch,
            "base_patience": base_patience,
            "base_lr": base_lr,
            "meta_mesa_parameter_size": meta_mesa_parameter_size,
            "meta_allow_bias": meta_allow_bias,
            "meta_dropout": meta_dropout,
            "meta_init_mesa_range": meta_init_mesa_range,
            "meta_init_meta_range": meta_init_meta_range,
            "meta_epochs": meta_epochs,
            "meta_patience": meta_patience,
            "meta_lr": meta_lr,
            "meta_minibatch_size": meta_minibatch_size,
            "meta_selection_split": meta_selection_split,
            "meta_trainable_layer_start": max(0, n_layers - 1),
            "meta_trainable_layer_end": n_layers - 1,
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Step 04] Saved metrics → {RESULTS_PATH}")

    # ------------------------------------------------------------------
    # Generate quantile predictions for all splits
    # ------------------------------------------------------------------
    non_feature_cols = [c for c in sa.columns if c not in feature_names]

    def _pred_df(rows, split_name, model_fn):
        info = sa.iloc[rows][non_feature_cols].copy().reset_index(drop=True)
        info["Split"] = split_name
        with torch.no_grad():
            pred = model_fn(rows)
        pred_np = pred.numpy()
        for i in range(5):
            info[f"Rank{i + 1}"] = pred_np[:, i]
        return info

    base_model.eval()
    meta_model.eval()

    base_preds = pd.concat([
        _pred_df(train_rows, "Train", lambda r: base_model(x[r])),
        _pred_df(test_rows, "Test", lambda r: base_model(x[r])),
        _pred_df(val_rows, "Validation", lambda r: base_model(x[r])),
    ], ignore_index=True)

    meta_preds = pd.concat([
        _pred_df(train_rows, "Train", lambda r: meta_model(x[r], subset_tensor(xtype, r))),
        _pred_df(test_rows, "Test", lambda r: meta_model(x[r], subset_tensor(xtype, r))),
        _pred_df(val_rows, "Validation", lambda r: meta_model(x[r], subset_tensor(xtype, r))),
    ], ignore_index=True)

    predictions = {"base": base_preds, "meta": meta_preds}
    with open(os.path.join(FEATURES_DIR, "forecast_ranks_all.pkl"), "wb") as f:
        pickle.dump(predictions, f)

    print("[Step 04] Saved quantile predictions.")


if __name__ == "__main__":
    run()
