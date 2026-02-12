"""
Step 02a — Data Cleaning Helpers.

Provides noisy interpolation for filling NA gaps in stock price series.
"""

import numpy as np
import pandas as pd


def noisy_interpolation(series: pd.Series) -> pd.Series:
    """
    Fill NA gaps by linear interpolation plus realistic micro-noise sampled
    from non-NA segments of the same series (mirrors the R implementation).
    """
    x = series.values.astype(float).copy()
    na_mask = np.isnan(x)

    if not na_mask.any():
        return series.copy()

    # Identify contiguous NA intervals
    intervals: list[list[int]] = []
    current: list[int] = []
    for i, is_na in enumerate(na_mask):
        if is_na:
            current.append(i)
        else:
            if current:
                intervals.append(current)
                current = []
    if current:
        intervals.append(current)

    x_omitted = x[~na_mask]
    noise = np.zeros_like(x)

    for interval in intervals:
        n = len(interval)
        if n <= len(x_omitted):
            start_idx = np.random.randint(0, len(x_omitted) - n + 1)
            x_sub = x_omitted[start_idx : start_idx + n].copy()
            x_sub -= x_sub[0]
            # Remove linear trend so endpoints match interpolation
            ramp = np.arange(n) / max(n - 1, 1) * x_sub[-1] if n > 1 else np.zeros(n)
            x_sub -= ramp
        else:
            x_sub = np.zeros(n)
        noise[interval] = x_sub

    # Linear interpolation of the original series + noise
    interpolated = pd.Series(x).interpolate(method="linear").values
    result = interpolated + noise
    return pd.Series(result, index=series.index, name=series.name)
