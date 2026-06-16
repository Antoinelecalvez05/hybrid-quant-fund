"""
ai_signal.py
------------
Contextual signal layer — V3 (NLP-enriched, backward-compatible).

Signal version history
----------------------
V1: All columns = 0.0 (neutral mock).
V2: Market-derived price-proxy signals (SPY/TLT momentum, sector momentum, vol regime).
V3: Optional NLP signals blended with V2 price-proxy signals.

Public API
----------
generate_ai_signals(features, prices=None, nlp_records=None)
  - prices=None, nlp_records=None  → V1 neutral fallback (zeros)
  - prices provided, no nlp_records → V2 price-proxy only (unchanged)
  - prices + nlp_records           → V3 blended (V2 proxy + NLP)

Backward compatibility is fully maintained.
The ai_* column names never change; nlp_* columns are appended additively.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import (
    TICKERS,
    CONTEXT_MOMENTUM_WINDOW,
    CONTEXT_VOL_WINDOW,
    CONTEXT_ZSCORE_WINDOW,
)

AI_SIGNAL_COLUMNS = [
    "ai_macro_sentiment",
    "ai_sector_momentum",
    "ai_volatility_regime",
]

# Internal names used to stash V2 proxy values before blending
_V2_MACRO  = "_v2_macro_sentiment"
_V2_SECTOR = "_v2_sector_momentum"
_V2_VOL    = "_v2_vol_regime"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu  = series.rolling(window, min_periods=window // 2).mean()
    sig = series.rolling(window, min_periods=window // 2).std()
    return (series - mu) / sig.replace(0, np.nan)


def _clip(series: pd.Series) -> pd.Series:
    return series.clip(-1.0, 1.0)


# ---------------------------------------------------------------------------
# V2 proxy signal computation
# ---------------------------------------------------------------------------

def _compute_v2_proxies(prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Return (macro_signal, sector_mom_wide, vol_signal) on the price date index.
    All trailing-only; no look-ahead.
    """
    log_ret = np.log(prices / prices.shift(1))

    spy_mom = (log_ret["SPY"].rolling(CONTEXT_MOMENTUM_WINDOW).sum()
               if "SPY" in log_ret.columns
               else pd.Series(0.0, index=prices.index))
    tlt_mom = (log_ret["TLT"].rolling(CONTEXT_MOMENTUM_WINDOW).sum()
               if "TLT" in log_ret.columns
               else pd.Series(0.0, index=prices.index))

    macro_raw    = spy_mom - tlt_mom
    macro_signal = _clip(_rolling_zscore(macro_raw, CONTEXT_ZSCORE_WINDOW))

    mom_63           = log_ret.rolling(CONTEXT_MOMENTUM_WINDOW).sum()
    universe_avg_mom = mom_63.mean(axis=1)
    rel_mom          = mom_63.subtract(universe_avg_mom, axis=0)
    sector_mom_wide  = rel_mom.apply(lambda col: _clip(_rolling_zscore(col, CONTEXT_ZSCORE_WINDOW)))

    ew_ret         = log_ret.mean(axis=1)
    vol_regime_raw = ew_ret.rolling(CONTEXT_VOL_WINDOW).std()
    vol_signal     = _clip(-_rolling_zscore(vol_regime_raw, CONTEXT_ZSCORE_WINDOW))

    return macro_signal, sector_mom_wide, vol_signal


def _apply_v2_to_features(
    features: pd.DataFrame,
    macro_signal: pd.Series,
    sector_mom_wide: pd.DataFrame,
    vol_signal: pd.Series,
) -> pd.DataFrame:
    """Map V2 proxy signals onto the (date, ticker) MultiIndex."""
    result = features.copy()

    dates   = result.index.get_level_values("date")
    tickers = result.index.get_level_values("ticker")

    result["ai_macro_sentiment"] = macro_signal.reindex(dates).values
    result["ai_volatility_regime"] = vol_signal.reindex(dates).values

    sector_values = [
        sector_mom_wide.loc[d, t]
        if (d in sector_mom_wide.index and t in sector_mom_wide.columns)
        else np.nan
        for d, t in zip(dates, tickers)
    ]
    result["ai_sector_momentum"] = sector_values

    for col in AI_SIGNAL_COLUMNS:
        result[col] = result[col].fillna(0.0)

    # Stash V2 proxy values for optional blending step
    result[_V2_MACRO]  = result["ai_macro_sentiment"].copy()
    result[_V2_SECTOR] = result["ai_sector_momentum"].copy()
    result[_V2_VOL]    = result["ai_volatility_regime"].copy()

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ai_signals(
    features: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    nlp_records=None,          # list[NLPSignalRecord] | None
) -> pd.DataFrame:
    """
    Append contextual signal columns to the feature DataFrame.

    Parameters
    ----------
    features    : MultiIndex (date, ticker) feature DataFrame.
    prices      : Optional price DataFrame. Enables V2 proxy computation.
    nlp_records : Optional list of NLPSignalRecord. Enables V3 NLP blending.

    Returns
    -------
    DataFrame with ai_* columns (and nlp_* columns if NLP records provided).

    Fallback hierarchy:
        no prices, no nlp → V1 neutral zeros
        prices only        → V2 proxy signals
        prices + nlp       → V3 blended signals
    """
    # ---- V1 fallback: all zeros ----------------------------------------
    if prices is None and not nlp_records:
        result = features.copy()
        for col in AI_SIGNAL_COLUMNS:
            result[col] = 0.0
        return result

    # ---- V2 proxy signals -----------------------------------------------
    if prices is not None:
        macro_sig, sector_wide, vol_sig = _compute_v2_proxies(prices)
        result = _apply_v2_to_features(features, macro_sig, sector_wide, vol_sig)
    else:
        result = features.copy()
        for col in AI_SIGNAL_COLUMNS:
            result[col] = 0.0
        result[_V2_MACRO]  = 0.0
        result[_V2_SECTOR] = 0.0
        result[_V2_VOL]    = 0.0

    # ---- V3 NLP enrichment (optional) -----------------------------------
    if nlp_records:
        from nlp_signal import build_nlp_features, blend_ai_signals
        result = build_nlp_features(result, nlp_records)
        result = blend_ai_signals(result, v2_prefix="_v2_")

    # Drop internal stash columns (not needed in the model)
    result = result.drop(columns=[_V2_MACRO, _V2_SECTOR, _V2_VOL], errors="ignore")

    return result


def get_ai_signal_columns() -> list[str]:
    return list(AI_SIGNAL_COLUMNS)
