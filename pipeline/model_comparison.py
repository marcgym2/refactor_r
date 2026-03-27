"""
Leak-safe quintile model comparison.

Compares the current FFNN baseline architecture against TabICL v2 and
TabPFN v2 on a chronological, non-overlapping Shift==0 holdout split.
Writes:

- logs/data_audit.log
- logs/results.log
- logs/final_comparison.csv
- logs/REPORT.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from .config import FEATURES_DIR, TEMP_DIR
from .m6_metrics import TARGET_RANK_COLUMNS
from .models import prepare_base_model
from .training_utils import ConstructFFNN, compute_rps_tensor, train_model

LOG_DIR = Path("logs")
DATA_AUDIT_LOG = LOG_DIR / "data_audit.log"
RESULTS_LOG = LOG_DIR / "results.log"
FINAL_COMPARISON_CSV = LOG_DIR / "final_comparison.csv"
REPORT_MD = LOG_DIR / "REPORT.md"
CLASS_LABELS = [1, 2, 3, 4, 5]
RANDOM_SEED = 1


@dataclass(frozen=True)
class SplitBundle:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
    train_intervals: pd.DataFrame
    dev_intervals: pd.DataFrame
    test_intervals: pd.DataFrame


def _seed_everything() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def _ensure_logs_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _json_dump(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=False)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude_cols = {
        "Ticker",
        "Interval",
        "Return",
        "Shift",
        "ReturnQuintile",
        "IntervalStart",
        "IntervalEnd",
        *TARGET_RANK_COLUMNS,
    }
    return [column for column in frame.columns if column not in exclude_cols]


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = Path(FEATURES_DIR) / "features_raw.parquet"
    standardized_path = Path(FEATURES_DIR) / "features_standardized.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw feature cache: {raw_path}")
    if not standardized_path.exists():
        raise FileNotFoundError(f"Missing standardized feature cache: {standardized_path}")
    raw = pd.read_parquet(raw_path)
    standardized = pd.read_parquet(standardized_path)
    raw["IntervalStart"] = pd.to_datetime(raw["IntervalStart"])
    raw["IntervalEnd"] = pd.to_datetime(raw["IntervalEnd"])
    standardized["IntervalStart"] = pd.to_datetime(standardized["IntervalStart"])
    standardized["IntervalEnd"] = pd.to_datetime(standardized["IntervalEnd"])
    return raw, standardized


def _build_comparison_frame(standardized: pd.DataFrame) -> pd.DataFrame:
    frame = standardized.copy()
    frame = frame.loc[frame["Shift"] == 0].copy()
    frame = frame.dropna(subset=["ReturnQuintile"]).copy()
    frame["ReturnQuintile"] = frame["ReturnQuintile"].astype(int)
    frame = frame.sort_values(["IntervalStart", "Ticker"]).reset_index(drop=True)
    return frame


def _build_splits(frame: pd.DataFrame) -> SplitBundle:
    intervals = (
        frame[["Interval", "IntervalStart", "IntervalEnd"]]
        .drop_duplicates()
        .sort_values(["IntervalStart", "IntervalEnd"])
        .reset_index(drop=True)
    )
    n_intervals = len(intervals)
    if n_intervals < 10:
        raise RuntimeError(f"Need at least 10 intervals for train/dev/test; found {n_intervals}.")

    train_end_idx = int(0.8 * n_intervals)
    dev_end_idx = int(0.9 * n_intervals)
    if train_end_idx <= 0 or dev_end_idx <= train_end_idx or dev_end_idx >= n_intervals:
        raise RuntimeError(
            "Invalid interval split boundaries "
            f"(n={n_intervals}, train_end_idx={train_end_idx}, dev_end_idx={dev_end_idx})."
        )

    train_intervals = intervals.iloc[:train_end_idx].copy()
    dev_intervals = intervals.iloc[train_end_idx:dev_end_idx].copy()
    test_intervals = intervals.iloc[dev_end_idx:].copy()

    train_names = set(train_intervals["Interval"])
    dev_names = set(dev_intervals["Interval"])
    test_names = set(test_intervals["Interval"])

    train = frame.loc[frame["Interval"].isin(train_names)].copy()
    dev = frame.loc[frame["Interval"].isin(dev_names)].copy()
    test = frame.loc[frame["Interval"].isin(test_names)].copy()

    return SplitBundle(
        train=train,
        dev=dev,
        test=test,
        train_intervals=train_intervals,
        dev_intervals=dev_intervals,
        test_intervals=test_intervals,
    )


def _class_distribution(frame: pd.DataFrame) -> dict[int, int]:
    counts = frame["ReturnQuintile"].value_counts().sort_index()
    return {label: int(counts.get(label, 0)) for label in CLASS_LABELS}


def _top_null_columns(frame: pd.DataFrame, columns: list[str], top_k: int = 15) -> dict[str, int]:
    counts = frame[columns].isna().sum().sort_values(ascending=False)
    counts = counts[counts > 0].head(top_k)
    return {str(column): int(value) for column, value in counts.items()}


def _cross_shift_overlap_examples(frame: pd.DataFrame, max_examples: int = 5) -> tuple[int, list[dict[str, str | int]]]:
    intervals = frame[["Shift", "IntervalStart", "IntervalEnd"]].drop_duplicates().copy()
    rows = intervals.to_dict("records")
    overlap_count = 0
    examples: list[dict[str, str | int]] = []
    for idx, left in enumerate(rows):
        for right in rows[idx + 1 :]:
            if int(left["Shift"]) == int(right["Shift"]):
                continue
            overlap = max(left["IntervalStart"], right["IntervalStart"]) <= min(left["IntervalEnd"], right["IntervalEnd"])
            if not overlap:
                continue
            overlap_count += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "left_shift": int(left["Shift"]),
                        "left_start": str(pd.Timestamp(left["IntervalStart"]).date()),
                        "left_end": str(pd.Timestamp(left["IntervalEnd"]).date()),
                        "right_shift": int(right["Shift"]),
                        "right_start": str(pd.Timestamp(right["IntervalStart"]).date()),
                        "right_end": str(pd.Timestamp(right["IntervalEnd"]).date()),
                    }
                )
    return overlap_count, examples


def _interval_overlap_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
    count = 0
    for left_row in left.itertuples(index=False):
        for right_row in right.itertuples(index=False):
            if max(left_row.IntervalStart, right_row.IntervalStart) <= min(left_row.IntervalEnd, right_row.IntervalEnd):
                count += 1
    return count


def _audited_leakage_notes(feature_names: list[str]) -> dict[str, object]:
    forbidden_feature_names = {
        "Return",
        "ReturnQuintile",
        "Shift",
        "Interval",
        "IntervalStart",
        "IntervalEnd",
        *TARGET_RANK_COLUMNS,
    }
    forbidden_present = sorted(forbidden_feature_names.intersection(feature_names))
    non_lag_return_features = sorted(
        column
        for column in feature_names
        if "Return" in column and not column.startswith("ReturnLag")
    )
    return {
        "forbidden_feature_columns_present": forbidden_present,
        "non_lag_return_feature_columns": non_lag_return_features,
        "feature_engineering_guards": [
            "Return is excluded from model inputs and kept only as target metadata.",
            "TargetRank1..5 and ReturnQuintile are excluded from model inputs.",
            "Technical indicators in pipeline/features.py are shifted by one interval in ttr_wrapper().",
            "Lagged returns and lagged volatility use prior intervals only.",
        ],
    }


def audit_data(
    *,
    raw_frame: pd.DataFrame,
    standardized_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    splits: SplitBundle,
    feature_names: list[str],
) -> dict[str, object]:
    overlap_count, overlap_examples = _cross_shift_overlap_examples(standardized_frame)
    audit = {
        "raw_feature_shape": [int(raw_frame.shape[0]), int(raw_frame.shape[1])],
        "standardized_feature_shape": [int(standardized_frame.shape[0]), int(standardized_frame.shape[1])],
        "comparison_frame_shape": [int(comparison_frame.shape[0]), int(comparison_frame.shape[1])],
        "comparison_protocol": {
            "reason": "Use Shift == 0 only to avoid cross-shift overlapping interval leakage and to match live inference.",
            "rows": int(len(comparison_frame)),
            "tickers": int(comparison_frame["Ticker"].nunique()),
            "intervals": int(comparison_frame["Interval"].nunique()),
            "feature_count": int(len(feature_names)),
        },
        "split_summary": {
            "train_rows": int(len(splits.train)),
            "dev_rows": int(len(splits.dev)),
            "test_rows": int(len(splits.test)),
            "train_intervals": int(len(splits.train_intervals)),
            "dev_intervals": int(len(splits.dev_intervals)),
            "test_intervals": int(len(splits.test_intervals)),
            "train_start": str(splits.train_intervals["IntervalStart"].min().date()),
            "train_end": str(splits.train_intervals["IntervalEnd"].max().date()),
            "dev_start": str(splits.dev_intervals["IntervalStart"].min().date()),
            "dev_end": str(splits.dev_intervals["IntervalEnd"].max().date()),
            "test_start": str(splits.test_intervals["IntervalStart"].min().date()),
            "test_end": str(splits.test_intervals["IntervalEnd"].max().date()),
        },
        "class_distribution": {
            "full_comparison_frame": _class_distribution(comparison_frame),
            "train": _class_distribution(splits.train),
            "dev": _class_distribution(splits.dev),
            "test": _class_distribution(splits.test),
        },
        "nulls": {
            "raw_total_nulls": int(raw_frame.isna().sum().sum()),
            "standardized_total_nulls": int(standardized_frame.isna().sum().sum()),
            "comparison_total_nulls": int(comparison_frame.isna().sum().sum()),
            "raw_top_null_feature_columns": _top_null_columns(raw_frame, feature_names),
            "comparison_top_null_feature_columns": _top_null_columns(comparison_frame, feature_names),
        },
        "duplicate_keys": {
            "raw_interval_ticker_shift_duplicates": int(
                raw_frame.duplicated(subset=["Interval", "Ticker", "Shift"]).sum()
            ),
            "comparison_interval_ticker_duplicates": int(
                comparison_frame.duplicated(subset=["Interval", "Ticker"]).sum()
            ),
        },
        "leakage_risks": {
            "full_frame_cross_shift_overlap_pairs": int(overlap_count),
            "full_frame_cross_shift_overlap_examples": overlap_examples,
            "comparison_train_dev_interval_overlaps": int(
                _interval_overlap_count(splits.train_intervals, splits.dev_intervals)
            ),
            "comparison_train_test_interval_overlaps": int(
                _interval_overlap_count(splits.train_intervals, splits.test_intervals)
            ),
            "comparison_dev_test_interval_overlaps": int(
                _interval_overlap_count(splits.dev_intervals, splits.test_intervals)
            ),
            **_audited_leakage_notes(feature_names),
        },
    }
    return audit


def _format_confusion_matrix(cm: np.ndarray) -> str:
    rows = []
    for idx, row in enumerate(cm, start=1):
        rows.append(f"class_{idx}: {row.tolist()}")
    return "\n".join(rows)


def _compute_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_seconds: float,
    predict_seconds: float,
    device: str,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    per_class_f1 = f1_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "device": device,
    }
    for idx, score in enumerate(per_class_f1, start=1):
        metrics[f"f1_class_{idx}"] = float(score)
    if details:
        metrics.update(details)
    return metrics


def _build_baseline_target_tensor(frame: pd.DataFrame) -> torch.Tensor:
    if set(TARGET_RANK_COLUMNS).issubset(frame.columns):
        target_probs = frame[TARGET_RANK_COLUMNS].fillna(0.0).to_numpy(dtype=float)
        return torch.tensor(np.cumsum(target_probs, axis=1), dtype=torch.float32)

    quintiles = frame["ReturnQuintile"].to_numpy(dtype=int)
    tensor = torch.zeros(len(quintiles), 5, dtype=torch.float32)
    for idx, quintile in enumerate(quintiles):
        tensor[idx, quintile - 1 :] = 1.0
    return tensor


def _prepare_xy(frame: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = frame[feature_names].to_numpy(dtype=np.float32)
    y = frame["ReturnQuintile"].to_numpy(dtype=int) - 1
    return x, y


def _preferred_devices() -> list[str]:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def _run_baseline(
    *,
    splits: SplitBundle,
    feature_names: list[str],
) -> dict[str, object]:
    _seed_everything()
    x_train = torch.tensor(splits.train[feature_names].to_numpy(dtype=np.float32), dtype=torch.float32)
    x_dev = torch.tensor(splits.dev[feature_names].to_numpy(dtype=np.float32), dtype=torch.float32)
    x_test = torch.tensor(splits.test[feature_names].to_numpy(dtype=np.float32), dtype=torch.float32)
    y_train_tensor = _build_baseline_target_tensor(splits.train)
    y_dev_tensor = _build_baseline_target_tensor(splits.dev)
    y_test_tensor = _build_baseline_target_tensor(splits.test)
    y_test = splits.test["ReturnQuintile"].to_numpy(dtype=int)

    layer_sizes = [32, 8, 5]
    layer_dropouts = [0.2] * (len(layer_sizes) - 1) + [0.0]
    layer_transforms = [F.leaky_relu] * (len(layer_sizes) - 1) + [lambda x: F.softmax(x, dim=1)]

    model = ConstructFFNN(len(feature_names), layer_sizes, layer_transforms, layer_dropouts)
    model = prepare_base_model(model, x_train)

    temp_path = Path(TEMP_DIR) / "model_comparison_baseline"
    if temp_path.exists():
        shutil.rmtree(temp_path)

    train_started = time.time()
    fit = train_model(
        model=model,
        criterion=compute_rps_tensor,
        train=[y_train_tensor, x_train],
        test=[y_dev_tensor, x_dev],
        validation=[y_test_tensor, x_test],
        epochs=100,
        minibatch=200,
        temp_dir=str(temp_path),
        patience=5,
        print_every=10,
        lr=[0.01],
    )
    train_seconds = time.time() - train_started

    best_model = fit["model"]
    best_epoch_idx = fit["progress"]["loss_test"].astype(float).idxmin()
    best_epoch = int(fit["progress"].loc[best_epoch_idx, "epoch"])
    best_dev_loss = float(fit["progress"].loc[best_epoch_idx, "loss_test"])
    final_test_loss = float(fit["progress"]["loss_validation"].astype(float).min())

    best_model.eval()
    predict_started = time.time()
    with torch.no_grad():
        probs = best_model(x_test).detach().cpu().numpy()
    predict_seconds = time.time() - predict_started
    y_pred = probs.argmax(axis=1) + 1

    metrics = _compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        device="cpu",
        details={
            "model_name": "baseline_ffnn",
            "best_dev_epoch": best_epoch,
            "best_dev_loss": best_dev_loss,
            "best_test_loss": final_test_loss,
            "n_train_rows": int(len(splits.train)),
            "n_test_rows": int(len(splits.test)),
        },
    )
    return metrics


def _run_with_device_fallback(
    *,
    model_name: str,
    runner,
) -> dict[str, object]:
    errors: list[str] = []
    for device in _preferred_devices():
        try:
            return runner(device)
        except Exception as exc:
            errors.append(f"{model_name} failed on {device}: {exc}\n{traceback.format_exc()}")
    raise RuntimeError("\n".join(errors))


def _run_tabicl(
    *,
    splits: SplitBundle,
    feature_names: list[str],
) -> dict[str, object]:
    x_train, y_train = _prepare_xy(splits.train, feature_names)
    x_test, _ = _prepare_xy(splits.test, feature_names)
    y_test = splits.test["ReturnQuintile"].to_numpy(dtype=int)

    def _runner(device: str) -> dict[str, object]:
        _seed_everything()
        model = TabICLClassifier(
            device=device,
            random_state=RANDOM_SEED,
            n_jobs=1,
            verbose=False,
        )
        train_started = time.time()
        model.fit(x_train, y_train)
        train_seconds = time.time() - train_started

        predict_started = time.time()
        probs = model.predict_proba(x_test)
        predict_seconds = time.time() - predict_started

        classes = np.asarray(model.classes_)
        y_pred = classes[probs.argmax(axis=1)] + 1
        metrics = _compute_metrics(
            y_true=y_test,
            y_pred=y_pred.astype(int),
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            device=device,
            details={
                "model_name": "tabicl_v2",
                "checkpoint_version": str(getattr(model, "checkpoint_version", "v2_default")),
                "n_train_rows": int(len(splits.train)),
                "n_test_rows": int(len(splits.test)),
            },
        )
        return metrics

    return _run_with_device_fallback(model_name="TabICL v2", runner=_runner)


def _run_catboost(
    *,
    splits: SplitBundle,
    feature_names: list[str],
) -> dict[str, object]:
    _seed_everything()
    x_train, y_train = _prepare_xy(splits.train, feature_names)
    x_dev, y_dev = _prepare_xy(splits.dev, feature_names)
    x_test, _ = _prepare_xy(splits.test, feature_names)
    y_test = splits.test["ReturnQuintile"].to_numpy(dtype=int)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=RANDOM_SEED,
        iterations=1000,
        verbose=False,
        allow_writing_files=False,
        use_best_model=True,
        od_type="Iter",
        od_wait=50,
    )

    train_started = time.time()
    model.fit(x_train, y_train, eval_set=(x_dev, y_dev), verbose=False)
    train_seconds = time.time() - train_started

    predict_started = time.time()
    probs = model.predict_proba(x_test)
    predict_seconds = time.time() - predict_started

    y_pred = probs.argmax(axis=1) + 1
    metrics = _compute_metrics(
        y_true=y_test,
        y_pred=y_pred.astype(int),
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        device="cpu",
        details={
            "model_name": "catboost",
            "best_iteration": int(model.get_best_iteration()),
            "best_dev_score": float(model.get_best_score()["validation"]["TotalF1:average=Macro"]),
            "n_train_rows": int(len(splits.train)),
            "n_test_rows": int(len(splits.test)),
        },
    )
    return metrics


def _run_tabpfn(
    *,
    splits: SplitBundle,
    feature_names: list[str],
) -> dict[str, object]:
    x_train, y_train = _prepare_xy(splits.train, feature_names)
    x_test, _ = _prepare_xy(splits.test, feature_names)
    y_test = splits.test["ReturnQuintile"].to_numpy(dtype=int)

    def _runner(device: str) -> dict[str, object]:
        _seed_everything()
        ignore_limits = device == "cpu"
        fit_mode = "fit_preprocessors" if device != "cpu" else "low_memory"
        model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device=device,
            random_state=RANDOM_SEED,
            n_preprocessing_jobs=1,
            ignore_pretraining_limits=ignore_limits,
            fit_mode=fit_mode,
        )
        train_started = time.time()
        model.fit(x_train, y_train)
        train_seconds = time.time() - train_started

        predict_started = time.time()
        probs = model.predict_proba(x_test)
        predict_seconds = time.time() - predict_started

        classes = np.asarray(model.classes_)
        y_pred = classes[probs.argmax(axis=1)] + 1
        metrics = _compute_metrics(
            y_true=y_test,
            y_pred=y_pred.astype(int),
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            device=device,
            details={
                "model_name": "tabpfn_v2",
                "fit_mode": fit_mode,
                "ignore_pretraining_limits": bool(ignore_limits),
                "n_train_rows": int(len(splits.train)),
                "n_test_rows": int(len(splits.test)),
            },
        )
        return metrics

    return _run_with_device_fallback(model_name="TabPFN v2", runner=_runner)


def _results_section(title: str, metrics: dict[str, object]) -> str:
    lines = [
        f"## {title}",
        f"accuracy: {metrics['accuracy']:.6f}",
        f"macro_f1: {metrics['macro_f1']:.6f}",
        f"train_seconds: {metrics['train_seconds']:.4f}",
        f"predict_seconds: {metrics['predict_seconds']:.4f}",
        f"device: {metrics['device']}",
    ]
    for label in CLASS_LABELS:
        lines.append(f"f1_class_{label}: {metrics[f'f1_class_{label}']:.6f}")
    extra_keys = [
        key
        for key in metrics.keys()
        if key not in {
            "accuracy",
            "macro_f1",
            "train_seconds",
            "predict_seconds",
            "device",
            "confusion_matrix",
            *[f"f1_class_{label}" for label in CLASS_LABELS],
        }
    ]
    for key in sorted(extra_keys):
        lines.append(f"{key}: {metrics[key]}")
    lines.append("confusion_matrix:")
    lines.append(_format_confusion_matrix(np.asarray(metrics["confusion_matrix"], dtype=int)))
    lines.append("")
    return "\n".join(lines)


def _comparison_dataframe(results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "model_name": result["model_name"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "train_seconds": result["train_seconds"],
            "predict_seconds": result["predict_seconds"],
            "device": result["device"],
        }
        for label in CLASS_LABELS:
            row[f"f1_class_{label}"] = result[f"f1_class_{label}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)


def _build_report(
    *,
    audit: dict[str, object],
    comparison: pd.DataFrame,
    results: list[dict[str, object]],
) -> str:
    winner = comparison.iloc[0]
    runner_up = comparison.iloc[1] if len(comparison) > 1 else None
    margin = float(winner["macro_f1"] - runner_up["macro_f1"]) if runner_up is not None else float("nan")
    results_by_name = {result["model_name"]: result for result in results}
    winner_result = results_by_name[str(winner["model_name"])]

    lines = [
        "# Model Comparison Report",
        "",
        "## Winner",
        (
            f"Winner: **{winner['model_name']}** with macro F1 `{winner['macro_f1']:.6f}` "
            f"and accuracy `{winner['accuracy']:.6f}`."
        ),
    ]
    if runner_up is not None:
        lines.append(
            f"It beat `{runner_up['model_name']}` by `{margin:.6f}` macro F1 on the held-out test split."
        )
    lines.extend(
        [
            "",
            "## Why This Protocol",
            "The full feature table contains four shift variants of 28-day windows, and those windows overlap heavily in calendar time.",
            "To avoid leakage, the comparison was restricted to `Shift == 0`, which matches the live inference path and produces non-overlapping intervals.",
            "The baseline is the current live FFNN architecture retrained with the repo's existing loss and hyperparameters, using a middle dev block for early stopping and the final block as the only scorecard.",
            "",
            "## Data Audit Highlights",
            f"- Raw feature table shape: `{audit['raw_feature_shape'][0]} x {audit['raw_feature_shape'][1]}`",
            f"- Standardized feature table shape: `{audit['standardized_feature_shape'][0]} x {audit['standardized_feature_shape'][1]}`",
            f"- Comparison frame shape: `{audit['comparison_frame_shape'][0]} x {audit['comparison_frame_shape'][1]}`",
            f"- Full-frame cross-shift overlap pairs: `{audit['leakage_risks']['full_frame_cross_shift_overlap_pairs']}`",
            f"- Shift-0 split overlap counts train/dev/test: "
            f"`{audit['leakage_risks']['comparison_train_dev_interval_overlaps']}` / "
            f"`{audit['leakage_risks']['comparison_train_test_interval_overlaps']}` / "
            f"`{audit['leakage_risks']['comparison_dev_test_interval_overlaps']}`",
            f"- Comparison-frame total nulls after preprocessing: `{audit['nulls']['comparison_total_nulls']}`",
            "",
            "## Final Table",
            comparison.to_string(index=False),
            "",
            "## Winner Reasoning",
            (
                f"`{winner['model_name']}` is the best choice here because it had the strongest macro F1 on a balanced five-class target, "
                "which is the most relevant metric for quintile prediction."
            ),
            (
                f"Its per-class F1 profile was: "
                + ", ".join(
                    f"class {label} = {winner_result[f'f1_class_{label}']:.4f}" for label in CLASS_LABELS
                )
                + "."
            ),
            "Accuracy is included as a secondary metric, but macro F1 gets priority because the goal is robust rank-bucket discrimination across all quintiles.",
            "",
            "## Caveats",
            "- The original all-shifts table is leaky for naive split-based evaluation because shifted windows overlap in time.",
            "- The baseline FFNN and CatBoost use the dev split for early stopping; TabICL v2 and TabPFN v2 run with fixed defaults and do not tune on that split.",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> pd.DataFrame:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _ensure_logs_dir()
    _seed_everything()

    raw_frame, standardized_frame = _load_frames()
    comparison_frame = _build_comparison_frame(standardized_frame)
    feature_names = _feature_columns(comparison_frame)
    splits = _build_splits(comparison_frame)
    audit = audit_data(
        raw_frame=raw_frame,
        standardized_frame=standardized_frame,
        comparison_frame=comparison_frame,
        splits=splits,
        feature_names=feature_names,
    )

    _write_text(DATA_AUDIT_LOG, _json_dump(audit) + "\n")
    _write_text(
        RESULTS_LOG,
        "\n".join(
            [
                "# Quintile Model Comparison",
                "",
                f"feature_count: {len(feature_names)}",
                f"train_rows: {len(splits.train)}",
                f"dev_rows: {len(splits.dev)}",
                f"test_rows: {len(splits.test)}",
                f"candidate_devices: {_preferred_devices()}",
                "",
            ]
        ),
    )

    results = []
    baseline_metrics = _run_baseline(splits=splits, feature_names=feature_names)
    results.append(baseline_metrics)
    _append_text(RESULTS_LOG, _results_section("Baseline FFNN", baseline_metrics))

    catboost_metrics = _run_catboost(splits=splits, feature_names=feature_names)
    results.append(catboost_metrics)
    _append_text(RESULTS_LOG, _results_section("CatBoost", catboost_metrics))

    tabicl_metrics = _run_tabicl(splits=splits, feature_names=feature_names)
    results.append(tabicl_metrics)
    _append_text(RESULTS_LOG, _results_section("TabICL v2", tabicl_metrics))

    tabpfn_metrics = _run_tabpfn(splits=splits, feature_names=feature_names)
    results.append(tabpfn_metrics)
    _append_text(RESULTS_LOG, _results_section("TabPFN v2", tabpfn_metrics))

    comparison = _comparison_dataframe(results)
    comparison.to_csv(FINAL_COMPARISON_CSV, index=False)
    report = _build_report(audit=audit, comparison=comparison, results=results)
    _write_text(REPORT_MD, report)
    return comparison


if __name__ == "__main__":
    run()
