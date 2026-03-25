"""
Step 03 — Feature Engineering.

Defines all technical-indicator feature functions and the TTR_FEATURES list
that drives the aggregation pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# We use the `ta` (technical analysis) library for most indicators.
# pip install ta
import ta


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def first_num(x: pd.Series) -> float | None:
    """First non-NaN value."""
    valid = x.dropna()
    return float(valid.iloc[0]) if len(valid) > 0 else np.nan


def last_num(x: pd.Series) -> float | None:
    """Last non-NaN value."""
    valid = x.dropna()
    return float(valid.iloc[-1]) if len(valid) > 0 else np.nan


# ---------------------------------------------------------------------------
# Core feature functions — each receives a stock DataFrame and returns a
# DataFrame indexed by "Interval" with feature columns.
# ---------------------------------------------------------------------------

def compute_return(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Per-interval return = last(Adjusted) / first(Adjusted) - 1."""
    return df.groupby("Interval", observed=True).apply(
        lambda g: pd.Series({"Return": last_num(g["Adjusted"]) / first_num(g["Adjusted"]) - 1
                              if first_num(g["Adjusted"]) else np.nan}),
        include_groups=False,
    ).reset_index()


def lag_return(df: pd.DataFrame, ticker: str, lags: list[int] | None = None) -> pd.DataFrame:
    """Lagged interval returns."""
    if lags is None:
        lags = list(range(1, 8))
    ret = df.groupby("Interval", observed=True).apply(
        lambda g: pd.Series({"_ret": last_num(g["Adjusted"]) / first_num(g["Adjusted"]) - 1
                              if first_num(g["Adjusted"]) else np.nan}),
        include_groups=False,
    ).reset_index()
    for lag in lags:
        ret[f"ReturnLag{lag}"] = ret["_ret"].shift(lag)
    ret = ret.drop(columns=["_ret"])
    return ret


def lag_volatility(df: pd.DataFrame, ticker: str, lags: list[int] | None = None) -> pd.DataFrame:
    """Lagged interval volatility (mean of squared log-returns)."""
    if lags is None:
        lags = list(range(1, 8))

    def _vol(g: pd.DataFrame) -> float:
        adj = g["Adjusted"].dropna()
        if len(adj) < 2:
            return np.nan
        lr = np.diff(np.log(adj.values))
        return float(np.mean(lr ** 2))

    vol = df.groupby("Interval", observed=True).apply(_vol, include_groups=False).reset_index()
    vol.columns = ["Interval", "_vol"]
    for lag in lags:
        vol[f"VolatilityLag{lag}"] = vol["_vol"].shift(lag)
    vol = vol.drop(columns=["_vol"])
    return vol


def is_etf(df: pd.DataFrame, ticker: str, stock_names: pd.DataFrame | None = None) -> pd.DataFrame:
    """Binary ETF flag (constant across intervals)."""
    etf_val = 1.0
    if stock_names is not None and "ETF" in stock_names.columns:
        match = stock_names.loc[stock_names["Symbol"] == ticker, "ETF"]
        etf_val = float(match.iloc[0]) if len(match) else 0.0
    intervals = df.groupby("Interval", observed=True).size().reset_index()[["Interval"]]
    intervals["ETF"] = etf_val
    return intervals


# ---------------------------------------------------------------------------
# Generic TTR wrapper
# ---------------------------------------------------------------------------

def ttr_wrapper(
    df: pd.DataFrame,
    ticker: str,
    *,
    compute_fn: Any,
    normalize: bool | list[bool] = False,
    prefix: str = "",
    suffix: str = "",
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Apply a TA indicator function *compute_fn(df)* → DataFrame of indicator
    columns, optionally normalize by Close, aggregate per interval, and lag
    by one period.
    """
    try:
        indicator_df = compute_fn(df)
    except Exception:
        # Indicator might fail on insufficient data
        return None

    if indicator_df is None or indicator_df.empty:
        return None

    # Normalize
    if isinstance(normalize, bool):
        normalize = [normalize] * indicator_df.shape[1]
    for i, (col, do_norm) in enumerate(zip(indicator_df.columns, normalize)):
        if do_norm:
            indicator_df[col] = indicator_df[col] / df["Close"].values[: len(indicator_df)]

    # Rename columns
    indicator_df.columns = [f"{prefix}{c}{suffix}" for c in indicator_df.columns]

    # Attach interval labels
    indicator_df["Interval"] = df["Interval"].values[: len(indicator_df)]

    # Aggregate per interval
    feat_cols = [c for c in indicator_df.columns if c != "Interval"]
    if aggregation == "mean":
        agg = indicator_df.groupby("Interval", observed=True)[feat_cols].mean().reset_index()
    else:
        agg = indicator_df.groupby("Interval", observed=True)[feat_cols].last().reset_index()

    # Lag by 1 interval
    for col in feat_cols:
        agg[col] = agg[col].shift(1)

    return agg


# ---------------------------------------------------------------------------
# Individual indicator compute functions (using `ta` library)
# ---------------------------------------------------------------------------

def _compute_adx(df: pd.DataFrame) -> pd.DataFrame:
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
    return pd.DataFrame({
        "ADX": adx.adx(), "ADX_pos": adx.adx_pos(), "ADX_neg": adx.adx_neg(),
    })

def _compute_aroon(df: pd.DataFrame) -> pd.DataFrame:
    a = ta.trend.AroonIndicator(df["Close"])
    return pd.DataFrame({"aroonUp": a.aroon_up(), "aroonDown": a.aroon_down(), "aroonOsc": a.aroon_indicator()})

def _compute_atr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=n)
    return pd.DataFrame({"atr": atr.average_true_range()})

def _compute_bbands(df: pd.DataFrame) -> pd.DataFrame:
    bb = ta.volatility.BollingerBands(df["Close"])
    return pd.DataFrame({
        "bb_high": bb.bollinger_hband(),
        "bb_mid": bb.bollinger_mavg(),
        "bb_low": bb.bollinger_lband(),
        "bb_pband": bb.bollinger_pband(),
    })

def _compute_cci(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"cci": ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"]).cci()})

def _compute_chaikin_mf(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"CMF": ta.volume.ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).chaikin_money_flow()})

def _compute_cmo(df: pd.DataFrame) -> pd.DataFrame:
    # Chande Momentum Oscillator — approximate via RSI transform
    rsi = ta.momentum.RSIIndicator(df["Close"]).rsi()
    cmo = (2 * rsi - 100)  # CMO ≈ 2*RSI - 100
    return pd.DataFrame({"CMO": cmo})

def _compute_donchian(df: pd.DataFrame) -> pd.DataFrame:
    dc = ta.volatility.DonchianChannel(df["High"], df["Low"], df["Close"])
    return pd.DataFrame({
        "dc_high": dc.donchian_channel_hband(),
        "dc_low": dc.donchian_channel_lband(),
        "dc_mid": dc.donchian_channel_mband(),
    })

def _compute_ema(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"ema": ta.trend.EMAIndicator(df["Close"], window=window).ema_indicator()})

def _compute_kst(df: pd.DataFrame) -> pd.DataFrame:
    kst = ta.trend.KSTIndicator(df["Close"])
    return pd.DataFrame({"KST": kst.kst(), "KST_sig": kst.kst_sig(), "KST_diff": kst.kst_diff()})

def _compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    m = ta.trend.MACD(df["Close"])
    return pd.DataFrame({"MACD": m.macd(), "MACD_signal": m.macd_signal(), "MACD_diff": m.macd_diff()})

def _compute_mfi(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"MFI": ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).money_flow_index()})

def _compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    return pd.DataFrame({"OBV": obv.diff()})

def _compute_roc(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"ROC": ta.momentum.ROCIndicator(df["Close"]).roc()})

def _compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"RSI": ta.momentum.RSIIndicator(df["Close"]).rsi()})

def _compute_percent_rank(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    close = df["Close"]
    prank = close.rolling(n).apply(lambda w: (w.iloc[-1] > w.iloc[:-1]).mean(), raw=False)
    return pd.DataFrame({"runPercentRank": prank})

def _compute_stoch(df: pd.DataFrame) -> pd.DataFrame:
    s = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    return pd.DataFrame({"SMI_stoch": s.stoch(), "SMI_signal": s.stoch_signal()})

def _compute_trix(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"TRIX": ta.trend.TRIXIndicator(df["Close"]).trix()})

def _compute_uo(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"ultimateOscillator": ta.momentum.UltimateOscillator(df["High"], df["Low"], df["Close"]).ultimate_oscillator()})

def _compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(df["Close"] / df["Close"].shift(1))
    return pd.DataFrame({"volatility": lr.rolling(20).std()})

def _compute_wpr(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"WPR": ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"]).williams_r()})

def _compute_vhf(df: pd.DataFrame) -> pd.DataFrame:
    """Vertical Horizontal Filter — custom implementation."""
    close = df["Close"]
    n = 28
    hcp = close.rolling(n).max()
    lcp = close.rolling(n).min()
    num = (hcp - lcp).abs()
    denom = close.diff().abs().rolling(n).sum()
    return pd.DataFrame({"VHF": num / (denom + 1e-10)})

def _compute_williams_ad(df: pd.DataFrame) -> pd.DataFrame:
    """Williams Accumulation/Distribution — custom diff."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    ad = np.where(
        close > prev_close,
        close - np.minimum(low, prev_close),
        np.where(close < prev_close, close - np.maximum(high, prev_close), 0),
    )
    ad_cumsum = pd.Series(ad, index=close.index).cumsum()
    return pd.DataFrame({"williamsAD": ad_cumsum.diff()})

def _compute_dema(df: pd.DataFrame) -> pd.DataFrame:
    ema1 = df["Close"].ewm(span=10).mean()
    ema2 = ema1.ewm(span=10).mean()
    return pd.DataFrame({"DEMA": 2 * ema1 - ema2})

def _compute_zlema(df: pd.DataFrame) -> pd.DataFrame:
    n = 10
    lag = (n - 1) // 2
    adjusted = 2 * df["Close"] - df["Close"].shift(lag)
    zlema = adjusted.ewm(span=n).mean()
    return pd.DataFrame({"ZLEMA": zlema})

def _compute_hma(df: pd.DataFrame) -> pd.DataFrame:
    n = 10
    half_n = max(n // 2, 1)
    sqrt_n = max(int(np.sqrt(n)), 1)
    wma_half = df["Close"].rolling(half_n).mean()
    wma_full = df["Close"].rolling(n).mean()
    diff = 2 * wma_half - wma_full
    return pd.DataFrame({"HMA": diff.rolling(sqrt_n).mean()})

def _compute_chaikin_volatility(df: pd.DataFrame) -> pd.DataFrame:
    hl_diff = df["High"] - df["Low"]
    ema_hl = hl_diff.ewm(span=10).mean()
    cv = (ema_hl - ema_hl.shift(10)) / (ema_hl.shift(10) + 1e-10)
    return pd.DataFrame({"chaikinVolatility": cv})

def _compute_clv(df: pd.DataFrame) -> pd.DataFrame:
    hl = df["High"] - df["Low"]
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (hl + 1e-10)
    return pd.DataFrame({"CLV": clv})

def _compute_snr(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Signal-to-Noise Ratio — approximate."""
    close = df["Close"]
    signal = close.rolling(n).mean()
    noise = (close - signal).abs().rolling(n).mean()
    return pd.DataFrame({f"SNR_n={n}": signal / (noise + 1e-10)})

def _compute_tdi(df: pd.DataFrame) -> pd.DataFrame:
    """Trend Detection Index."""
    close = df["Close"]
    n = 20
    mom = close.diff(n)
    abs_mom_sum = close.diff().abs().rolling(n).sum()
    return pd.DataFrame({"TDI": mom / (abs_mom_sum + 1e-10)})

def _compute_cti(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation Trend Indicator — Spearman rank of close vs time."""
    close = df["Close"]
    n = 20
    cti = close.rolling(n).apply(
        lambda w: pd.Series(w.values).corr(pd.Series(range(len(w))), method="spearman"),
        raw=False,
    )
    return pd.DataFrame({"CTI": cti})

def _compute_evwma(df: pd.DataFrame) -> pd.DataFrame:
    """Elastic Volume Weighted Moving Average (approx)."""
    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    n = 20
    evwma = np.full(len(close), np.nan, dtype=float)
    if len(close) > n:
        rolling_vol = pd.Series(vol).rolling(n, min_periods=n).sum().to_numpy(dtype=float)
        evwma[n - 1] = np.nanmean(close[:n])
        for i in range(n, len(close)):
            prev = evwma[i - 1]
            if not np.isfinite(prev):
                prev = close[i - 1]
            if not np.isfinite(prev) or not np.isfinite(close[i]):
                continue
            denom = rolling_vol[i]
            weight = 0.0
            if np.isfinite(denom) and denom > 0:
                weight = float(np.clip(vol[i] / denom, 0.0, 1.0))
            evwma[i] = prev + weight * (close[i] - prev)
    return pd.DataFrame({"EVWMA": evwma})

def _compute_pbands(df: pd.DataFrame) -> pd.DataFrame:
    """Price bands (percentage bands around SMA)."""
    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    return pd.DataFrame({
        "PBands_upper": sma + 2 * std,
        "PBands_lower": sma - 2 * std,
        "PBands_pct": (df["Close"] - sma) / (2 * std + 1e-10),
    })

def _compute_gmma(df: pd.DataFrame) -> pd.DataFrame:
    """Guppy Multiple Moving Average (short vs long EMAs)."""
    short_ema = df["Close"].ewm(span=10).mean()
    long30 = df["Close"].ewm(span=30).mean()
    long60 = df["Close"].ewm(span=60).mean()
    return pd.DataFrame({"GMMA_s10": short_ema, "GMMA_l30": long30, "GMMA_l60": long60})

def _compute_chaikin_ad(df: pd.DataFrame) -> pd.DataFrame:
    """Chaikin Accumulation/Distribution."""
    ad = ta.volume.AccDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).acc_dist_index()
    return pd.DataFrame({"chaikinAD": ad.diff()})


# ---------------------------------------------------------------------------
# Master feature list — each entry is a closure producing a feature function.
# ---------------------------------------------------------------------------

TTR_FEATURES: list = [
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_adx, normalize=False, prefix="ADX1_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_aroon, normalize=False, prefix="aroon_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=lambda d: _compute_atr(d, n=14), normalize=True, prefix="ATR_", suffix="_n=14"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=lambda d: _compute_atr(d, n=28), normalize=True, prefix="ATR_", suffix="_n=28"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=lambda d: _compute_atr(d, n=7), normalize=True, prefix="ATR_", suffix="_n=7"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_bbands, normalize=[True, True, True, False], prefix="BBands1_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_cci, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_chaikin_ad, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_chaikin_volatility, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_clv, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_chaikin_mf, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_cmo, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_cti, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_donchian, normalize=True, prefix="DonchianChannel_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_gmma, normalize=True, prefix="GMMA1_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_kst, normalize=False, prefix="KST1_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_macd, normalize=False, prefix="MACD1_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_mfi, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_obv, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_pbands, normalize=True, prefix="PBands_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_roc, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_rsi, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_percent_rank, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_ema, normalize=True),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_dema, normalize=True),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_evwma, normalize=True),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_zlema, normalize=True),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_hma, normalize=True),
    lambda df, t: ttr_wrapper(df, t, compute_fn=lambda d: _compute_snr(d, n=20), normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=lambda d: _compute_snr(d, n=60), normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_stoch, normalize=False, prefix="SMI1_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_tdi, normalize=True, prefix="TDI_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_trix, normalize=False, prefix="TRIX1_"),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_uo, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_vhf, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_volatility, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_williams_ad, normalize=False),
    lambda df, t: ttr_wrapper(df, t, compute_fn=_compute_wpr, normalize=False),
]
