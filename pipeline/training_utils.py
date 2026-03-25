"""
Step 03a — Feature Engineering Helpers.

Interval generation, standardization, imputation, tensor utilities,
neural-network building blocks, and the main aggregation + training loops.
"""

from __future__ import annotations

import os
import math
import copy
from datetime import date, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .m6_metrics import build_group_target_frame


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def augment_stock(df: pd.DataFrame, time_end: date) -> pd.DataFrame:
    """Extend stock DataFrame up to *time_end*, excluding weekends."""
    max_date = pd.Timestamp(df["index"].max())
    time_end_ts = pd.Timestamp(time_end)
    if max_date + timedelta(days=1) < time_end_ts:
        extra_dates = pd.bdate_range(start=max_date + timedelta(days=1), end=time_end_ts)
        extra = pd.DataFrame({"index": extra_dates})
        df = pd.concat([df, extra], ignore_index=True)
    return df


def standardize(x: pd.Series) -> pd.Series:
    """Z-score standardization."""
    std = x.std(ddof=0)
    if not np.isfinite(std) or std < 1e-8:
        return pd.Series(0.0, index=x.index, dtype=float)
    return (x - x.mean()) / std


def compute_quintile(x: pd.Series) -> pd.Series:
    """Map values to quintile buckets 1–5 (NAs stay NaN)."""
    out = pd.Series(np.nan, index=x.index)
    valid = x.dropna()
    if len(valid) == 0:
        return out
    ranks = valid.rank(method="first") / len(valid)
    bins = pd.cut(ranks, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])
    out.loc[valid.index] = bins.astype(float)
    return out


def impute_na(x: pd.Series) -> pd.Series:
    """Replace NaN / Inf with column median."""
    median_val = x.replace([np.inf, -np.inf], np.nan).median()
    return x.replace([np.inf, -np.inf], np.nan).fillna(median_val)


def impute_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Impute NaN / Inf in selected feature columns."""
    df = df.copy()
    for col in feature_names:
        df[col] = impute_na(df[col])
    return df


def standardize_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Standardize selected features per Interval group."""
    df = df.copy()
    for col in feature_names:
        df[col] = df.groupby("Interval", observed=False)[col].transform(standardize)
    return df


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------

def compute_rps_tensor(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Ranked Probability Score (mean)."""
    diff = y_pred.cumsum(dim=1) - y
    return (diff ** 2).sum(dim=1).mean() / 5


def compute_rps_tensor_vector(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Ranked Probability Score (per sample)."""
    diff = y_pred.cumsum(dim=1) - y
    return (diff ** 2).sum(dim=1) / 5


def subset_tensor(
    x: torch.Tensor,
    rows: list[int] | torch.Tensor,
    is_sparse: bool | None = None,
) -> torch.Tensor:
    """Index into *x* along dim-0, handling both dense and sparse tensors."""
    if is_sparse is None:
        try:
            is_sparse = x.is_sparse
        except Exception:
            is_sparse = False
    if is_sparse:
        x_coalesced = x.coalesce()
        indices = x_coalesced.indices()
        values = x_coalesced.values()
        row_idx = indices[0]
        if isinstance(rows, torch.Tensor):
            rows_set = set(rows.tolist())
        else:
            rows_set = set(rows)
        mask = torch.tensor([int(r) in rows_set for r in row_idx.tolist()], dtype=torch.bool)
        new_indices = indices[:, mask].clone()
        # Remap row indices to 0..N-1
        sorted_rows = sorted(rows_set)
        row_map = {old: new for new, old in enumerate(sorted_rows)}
        new_indices[0] = torch.tensor([row_map[int(r)] for r in new_indices[0].tolist()], dtype=torch.long)
        new_values = values[mask]
        return torch.sparse_coo_tensor(
            new_indices,
            new_values,
            (len(sorted_rows), x.size(1)),
        ).coalesce()
    else:
        if isinstance(rows, list):
            rows = torch.tensor(rows, dtype=torch.long)
        return x[rows]


# ---------------------------------------------------------------------------
# Interval generation
# ---------------------------------------------------------------------------

def gen_interval_infos(
    submission: int,
    shifts: list[int] | None = None,
    time_end: date | None = None,
    interval_days: int = 28,
    total_intervals: int = 1000,
) -> list[dict[str, Any]]:
    """Generate interval metadata for each shift."""
    if shifts is None:
        shifts = [0]
    if time_end is None:
        time_end = date.today()

    infos: list[dict] = []
    for shift in shifts:
        shifted_end = time_end - timedelta(days=shift)
        time_start = shifted_end - timedelta(days=interval_days * total_intervals)
        breaks = []
        d = time_start
        while d <= shifted_end:
            breaks.append(d)
            d += timedelta(days=interval_days)
        if breaks[-1] < shifted_end:
            breaks.append(shifted_end)

        starts = [b + timedelta(days=1) for b in breaks[:-1]]
        ends = breaks[1:]
        names = [f"{s} : {e}" for s, e in zip(starts, ends)]
        idx = len(ends) - (12 - submission)
        infos.append({
            "Shift": shift,
            "TimeBreaks": breaks,
            "IntervalStarts": starts,
            "IntervalEnds": ends,
            "IntervalNames": names,
            "Start": starts[0],
            "End": ends[idx - 1] if idx > 0 else ends[-1],
            "CheckLeakageStart": starts[idx - 1] if idx > 0 else starts[-1],
        })
    return infos


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def gen_stocks_aggr(
    stocks: dict[str, pd.DataFrame],
    interval_infos: list[dict],
    feature_fns: list[Callable],
    check_leakage: bool = False,
) -> pd.DataFrame:
    """Compute per-interval feature aggregations for every stock and shift."""
    all_rows: list[pd.DataFrame] = []
    sorted_tickers = sorted(stocks.keys())

    for s, ticker in enumerate(sorted_tickers):
        print(f"[Step 03a] Stock {s + 1}/{len(sorted_tickers)}: {ticker}")
        for info in interval_infos:
            stock = stocks[ticker].copy()
            stock["index"] = pd.to_datetime(stock["index"])
            start_ts = pd.Timestamp(info["Start"])
            end_ts = pd.Timestamp(info["End"])
            mask = (stock["index"] >= start_ts) & (stock["index"] <= end_ts)
            stock = stock.loc[mask].copy()
            stock = augment_stock(stock, info["End"])
            # Ensure index is Timestamp after augment
            stock["index"] = pd.to_datetime(stock["index"])

            # Assign interval labels
            breaks = [pd.Timestamp(b) for b in info["TimeBreaks"]]
            bins = [breaks[0] - timedelta(days=1)] + breaks[1:]
            stock["Interval"] = pd.cut(
                stock["index"],
                bins=bins,
                labels=info["IntervalNames"][: len(breaks) - 1],
                right=True,
            )
            stock["Ticker"] = ticker

            # Compute features
            feature_dfs = []
            for fn in feature_fns:
                feat = fn(stock, ticker)
                if feat is not None:
                    feature_dfs.append(feat)
            if feature_dfs:
                merged = feature_dfs[0]
                for fdf in feature_dfs[1:]:
                    merged = merged.merge(fdf, on="Interval", how="outer")
                merged["Ticker"] = ticker
                merged["Shift"] = info["Shift"]
                all_rows.append(merged)

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)

    # Compute return quintile per interval
    if "Return" in result.columns:
        result["ReturnQuintile"] = result.groupby("Interval", observed=False)["Return"].transform(compute_quintile)
        target_frame = build_group_target_frame(result, group_col="Interval", id_col="Ticker", return_col="Return")
        result = result.merge(target_frame, on=["Interval", "Ticker"], how="left")

    # Parse interval start/end dates
    result["IntervalStart"] = result["Interval"].astype(str).str[:10].apply(
        lambda x: pd.to_datetime(x).date() if x != "nan" else None
    )
    result["IntervalEnd"] = result["Interval"].astype(str).str[13:23].apply(
        lambda x: pd.to_datetime(x).date() if x != "nan" else None
    )

    return result


# ---------------------------------------------------------------------------
# Neural-network building block
# ---------------------------------------------------------------------------

class ConstructFFNN(nn.Module):
    """Feedforward neural network with configurable layers, activations, and dropout."""

    def __init__(
        self,
        input_size: int,
        layer_sizes: list[int],
        layer_transforms: list[Callable],
        layer_dropouts: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.layer_sizes = layer_sizes
        self.layer_transforms = layer_transforms
        all_sizes = [input_size] + layer_sizes
        self.use_dropout = layer_dropouts is not None

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)):
            self.layers.append(nn.Linear(all_sizes[i], all_sizes[i + 1]))

        self.dropouts = nn.ModuleList()
        if self.use_dropout:
            for p in layer_dropouts:
                self.dropouts.append(nn.Dropout(p=p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.layer_transforms[i](layer(x))
            if self.use_dropout:
                x = self.dropouts[i](x)
        return x

    def fforward(self, x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass using an external state dict (for meta-learning)."""
        for i in range(len(self.layers)):
            w = state[f"layers.{i}.weight"]
            b = state[f"layers.{i}.bias"]
            x = self.layer_transforms[i](F.linear(x, w, b))
        return x


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    criterion: Callable,
    train: list[torch.Tensor],
    test: list[torch.Tensor] | None = None,
    validation: list[torch.Tensor] | None = None,
    epochs: int = 10,
    minibatch: int | Callable = float("inf"),
    temp_dir: str | None = None,
    patience: int = 1,
    print_every: int = float("inf"),
    lr: list[float] | float = 0.001,
    weight_decay: float = 0,
    optimizer_type: str = "adam",
    is_sparse: list[bool] | None = None,
) -> dict:
    """Train *model* with early stopping and optional restarts at different LRs."""
    if isinstance(lr, (int, float)):
        lr = [lr]

    best_model = copy.deepcopy(model)
    all_progress: list[pd.DataFrame] = []

    for rs, current_lr in enumerate(lr):
        # Select optimizer
        opt_map = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adadelta": torch.optim.Adadelta,
            "rmsprop": torch.optim.RMSprop,
        }
        opt_cls = opt_map.get(optimizer_type, torch.optim.Adam)
        optimizer = opt_cls(best_model.parameters(), lr=current_lr, weight_decay=weight_decay)

        progress = pd.DataFrame({
            "epoch": range(1, epochs + 1),
            "loss_train": np.inf,
            "loss_test": np.inf,
            "loss_validation": np.inf,
        })

        if is_sparse is None:
            is_sparse_flags = [False, False] + [True] * (len(train) - 2)
        else:
            is_sparse_flags = is_sparse

        mb_value = minibatch

        for e in range(1, epochs + 1):
            # --- Mini-batch training (skip epoch 1 = eval only) ---
            if e > 1:
                best_model.train()
                n = train[1].size(0)
                if callable(mb_value):
                    batches = mb_value()
                else:
                    bs = min(int(mb_value), n)
                    perm = torch.randperm(n)
                    batches = [perm[i : i + bs].tolist() for i in range(0, n, bs)]

                for batch_rows in batches:
                    train_mb = [
                        subset_tensor(t, batch_rows, is_sparse_flags[i])
                        for i, t in enumerate(train)
                    ]
                    optimizer.zero_grad()
                    y_pred = best_model(*train_mb[1:])
                    loss = criterion(y_pred, train_mb[0])
                    loss.backward()
                    optimizer.step()

            # --- Evaluate ---
            best_model.eval()
            with torch.no_grad():
                y_pred_all = best_model(*train[1:])
                progress.loc[e - 1, "loss_train"] = criterion(y_pred_all, train[0]).item()
                if test is not None:
                    y_pred_t = best_model(*test[1:])
                    progress.loc[e - 1, "loss_test"] = criterion(y_pred_t, test[0]).item()
                if validation is not None:
                    y_pred_v = best_model(*validation[1:])
                    progress.loc[e - 1, "loss_validation"] = criterion(y_pred_v, validation[0]).item()

            if e % print_every == 0 or e == 1:
                row = progress.loc[e - 1]
                print(f"  restart {rs}  epoch {e:3d}  "
                      f"train={row.loss_train:.5f}  test={row.loss_test:.5f}  "
                      f"val={row.loss_validation:.5f}")

            # --- Early stopping ---
            if test is not None:
                best_epoch = int(progress.loc[: e - 1, "loss_test"].idxmin()) + 1
                if e == best_epoch and temp_dir is not None:
                    os.makedirs(temp_dir, exist_ok=True)
                    torch.save(best_model.state_dict(), os.path.join(temp_dir, "temp.pt"))
                if e - best_epoch >= patience:
                    progress = progress.iloc[:e]
                    break

        # Reload best checkpoint
        if temp_dir is not None and test is not None:
            ckpt = os.path.join(temp_dir, "temp.pt")
            if os.path.exists(ckpt):
                best_model.load_state_dict(torch.load(ckpt, weights_only=True))
                os.remove(ckpt)

        all_progress.append(progress)

    full_progress = pd.concat(all_progress, ignore_index=True)
    return {"model": best_model, "progress": full_progress}


# ---------------------------------------------------------------------------
# Mini-batch sampler for sparse xtype
# ---------------------------------------------------------------------------

def minibatch_sampler(batch_size: int, xtype_train: torch.Tensor) -> list[list[int]]:
    """Group rows by column index in a sparse xtype tensor and create batches."""
    xtype_c = xtype_train.coalesce()
    indices = xtype_c.indices()
    rows = indices[0].tolist()
    cols = indices[1].tolist()
    unique_cols = list(set(cols))
    np.random.shuffle(unique_cols)
    batches = [unique_cols[i : i + batch_size] for i in range(0, len(unique_cols), batch_size)]
    result = []
    col_to_rows: dict[int, list[int]] = {}
    for r, c in zip(rows, cols):
        col_to_rows.setdefault(c, []).append(r)
    for batch_cols in batches:
        batch_rows = []
        for c in batch_cols:
            batch_rows.extend(col_to_rows.get(c, []))
        result.append(batch_rows)
    return result
