"""
Step 05 — Ranking & Validation Helpers.

Utilities for submission validation, return-array construction,
and Sharpe-ratio computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

def round_preserve_sum(x: np.ndarray, digits: int = 0) -> np.ndarray:
    """Round each element so that the sum is preserved."""
    factor = 10 ** digits
    scaled = x * factor
    floored = np.floor(scaled)
    remainders = scaled - floored
    deficit = int(round(scaled.sum())) - int(floored.sum())
    # Give the extra units to the elements with largest remainders
    indices = np.argsort(-remainders)[:deficit]
    floored[indices] += 1
    return floored / factor


# ---------------------------------------------------------------------------
# Submission validation
# ---------------------------------------------------------------------------

def validate_submission(
    submission: pd.DataFrame,
    template: pd.DataFrame,
    do_round: bool = False,
) -> pd.DataFrame:
    """Validate a ranked-forecast submission against a template."""
    rank_cols = ["Rank1", "Rank2", "Rank3", "Rank4", "Rank5"]

    if do_round:
        orig = submission.copy()
        for i in range(len(submission)):
            row = submission.loc[i, rank_cols].values.astype(float)
            row = row / row.sum()
            submission.loc[i, rank_cols] = round_preserve_sum(row, digits=5)
        submission["Decision"] = round_preserve_sum(
            submission["Decision"].values.astype(float), digits=5
        )
        max_diff = (orig[rank_cols + ["Decision"]].values - submission[rank_cols + ["Decision"]].values)
        print(f"  Max rounding diff: {np.abs(max_diff).max():.2e}")

    # Checks
    assert list(template["ID"]) == list(submission["ID"]), "ID ordering mismatch"
    assert list(template.columns) == list(submission.columns), "Column mismatch"
    row_sums = submission[rank_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8), "Rank probs don't sum to 1"
    assert (submission[rank_cols] >= 0).all().all(), "Negative probabilities"
    assert (submission[rank_cols] <= 1).all().all(), "Probabilities > 1"
    assert submission["Decision"].abs().sum() > 0, "Decision sum is zero"
    assert submission["Decision"].abs().sum() <= 1, "Decision sum exceeds 1"

    return submission


# ---------------------------------------------------------------------------
# Sharpe tensor
# ---------------------------------------------------------------------------

def compute_sharpe_tensor(
    weights: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    Compute annualised Sharpe ratio of a portfolio.
    weights: (N, K)   y: (N, K, T)
    """
    ret = torch.einsum("nkt,nk->nt", y, weights)
    log_ret = torch.log(ret + 1)
    s_ret = log_ret.sum(dim=1)
    sd = torch.std(log_ret, dim=1, unbiased=True)
    return ((21 * 12) / (252 ** 0.5)) * (1 / 20) * s_ret / (sd + eps)
