"""
nlp_signal.py
-------------
Convert NLP extraction records into daily features aligned to the price
MultiIndex, then blend them with V2 price-proxy signals.

Leakage rules
-------------
- A source dated D can only affect market dates >= D.
- If D is not a trading day the signal is placed on the next available
  trading day (forward snap, not backward — no look-ahead).
- Signal is then forward-filled with exponential decay for NLP_SIGNAL_DECAY_DAYS.

NLP columns produced:
    nlp_macro_sentiment    [-1, +1]
    nlp_sector_sentiment   [-1, +1]
    nlp_volatility_risk    [-1, +1]
    nlp_inflation_pressure [-1, +1]
    nlp_rates_pressure     [-1, +1]
    nlp_recession_risk     [-1, +1]
    nlp_confidence         [0,  +1]

Blended ai_* columns:
    ai_macro_sentiment   = V2_PROXY_WEIGHT * v2_macro
                         + NLP_WEIGHT * (nlp_macro - 0.5*nlp_recession - 0.5*nlp_rates)
    ai_sector_momentum   = V2_PROXY_WEIGHT * v2_sector
                         + NLP_WEIGHT * nlp_sector
    ai_volatility_regime = V2_PROXY_WEIGHT * v2_vol
                         + NLP_WEIGHT * (-nlp_volatility_risk)
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from nlp_schema import NLPSignalRecord, get_nlp_signal_columns
from config import NLP_SIGNAL_DECAY_DAYS, NLP_WEIGHT, V2_PROXY_WEIGHT

logger = logging.getLogger(__name__)

_SIGNAL_COLS = [
    "nlp_macro_sentiment", "nlp_sector_sentiment", "nlp_volatility_risk",
    "nlp_inflation_pressure", "nlp_rates_pressure", "nlp_recession_risk",
    "nlp_confidence",
]

# Internal V2 proxy stash column names
_V2_MACRO  = "_v2_macro_sentiment"
_V2_SECTOR = "_v2_sector_momentum"
_V2_VOL    = "_v2_vol_regime"


def _records_to_daily_df(
    records: list[NLPSignalRecord],
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Map NLP records onto date_index with forward-snap + exponential decay.

    Forward-snap: if a record's date is not in date_index, find the next
    date in date_index that is >= record.date. This ensures no look-ahead.
    """
    result = pd.DataFrame(0.0, index=date_index, columns=_SIGNAL_COLS)

    if not records:
        return result

    # Build a sorted searchable array of the date_index
    date_arr = np.array(date_index, dtype="datetime64[ns]")

    # Collect (snap_date, record) pairs
    placed: dict[pd.Timestamp, list[NLPSignalRecord]] = {}
    for r in records:
        try:
            src_dt = np.datetime64(r.date, "D").astype("datetime64[ns]")
        except Exception:
            logger.warning("Skipping record with unparseable date: %s", r.date)
            continue
        # Find first date_index entry >= src_dt
        pos = np.searchsorted(date_arr, src_dt, side="left")
        if pos >= len(date_arr):
            logger.info("Source dated %s is after the price series — skipping.", r.date)
            continue
        snap_dt = date_index[pos]
        placed.setdefault(snap_dt, []).append(r)

    if not placed:
        return result

    # Confidence-weighted average for records placed on the same day
    snap_df = pd.DataFrame(0.0, index=date_index, columns=_SIGNAL_COLS)
    snap_df[:] = np.nan  # start with NaN so we can ffill cleanly

    for snap_dt, recs in placed.items():
        if not recs:
            continue
        weights = np.array([r.confidence for r in recs], dtype=float)
        weights = np.where(weights > 0, weights, 1e-6)  # avoid zero-weight

        row: dict[str, float] = {}
        for col, attr in [
            ("nlp_macro_sentiment",    "macro_sentiment"),
            ("nlp_sector_sentiment",   "sector_sentiment"),
            ("nlp_volatility_risk",    "volatility_risk"),
            ("nlp_inflation_pressure", "inflation_pressure"),
            ("nlp_rates_pressure",     "rates_pressure"),
            ("nlp_recession_risk",     "recession_risk"),
        ]:
            vals = np.array([getattr(rec, attr) for rec in recs], dtype=float)
            row[col] = float(np.average(vals, weights=weights))
        row["nlp_confidence"] = float(np.mean([r.confidence for r in recs]))
        snap_df.loc[snap_dt] = row

    # Exponential decay forward-fill
    decay = 0.5 ** (1.0 / max(NLP_SIGNAL_DECAY_DAYS, 1))
    last_values = {col: 0.0 for col in _SIGNAL_COLS}
    has_signal  = False

    for dt in date_index:
        if not pd.isna(snap_df.loc[dt, _SIGNAL_COLS[0]]):
            # New signal placement — reset to the fresh values
            for col in _SIGNAL_COLS:
                last_values[col] = float(snap_df.loc[dt, col])
            has_signal = True
            result.loc[dt] = last_values
        elif has_signal:
            # Decay existing signal
            for col in _SIGNAL_COLS:
                if col != "nlp_confidence":
                    last_values[col] *= decay
                # confidence decays faster
            result.loc[dt] = last_values
        # else: no signal yet → leave as 0.0

    # Final clip
    for col in _SIGNAL_COLS:
        lo, hi = (0.0, 1.0) if col == "nlp_confidence" else (-1.0, 1.0)
        result[col] = result[col].clip(lo, hi)

    return result


def build_nlp_features(
    features: pd.DataFrame,
    records: list[NLPSignalRecord],
) -> pd.DataFrame:
    """Append nlp_* columns to the features DataFrame."""
    date_index = features.index.get_level_values("date").unique().sort_values()
    daily_nlp  = _records_to_daily_df(records, date_index)

    result = features.copy()
    dates  = result.index.get_level_values("date")

    for col in get_nlp_signal_columns():
        result[col] = daily_nlp[col].reindex(dates).fillna(0.0).values

    return result


def blend_ai_signals(
    features: pd.DataFrame,
    v2_prefix: str = "_v2_",
) -> pd.DataFrame:
    """Blend V2 proxy signals and NLP signals into the ai_* columns."""
    result = features.copy()

    has_nlp = all(c in result.columns for c in [
        "nlp_macro_sentiment", "nlp_sector_sentiment", "nlp_volatility_risk",
    ])
    has_v2 = all(c in result.columns for c in [
        f"{v2_prefix}macro_sentiment",
        f"{v2_prefix}sector_momentum",
        f"{v2_prefix}vol_regime",
    ])

    if not has_nlp:
        return result

    def _get(col: str) -> pd.Series:
        return result[col].fillna(0.0) if col in result.columns else pd.Series(0.0, index=result.index)

    def _blend(v2_col: str, nlp_expr: pd.Series) -> pd.Series:
        if has_v2 and v2_col in result.columns:
            return (V2_PROXY_WEIGHT * _get(v2_col) + NLP_WEIGHT * nlp_expr).clip(-1.0, 1.0)
        return nlp_expr.clip(-1.0, 1.0)

    result["ai_macro_sentiment"] = _blend(
        f"{v2_prefix}macro_sentiment",
        _get("nlp_macro_sentiment") - 0.5 * _get("nlp_recession_risk") - 0.5 * _get("nlp_rates_pressure"),
    )
    result["ai_sector_momentum"] = _blend(
        f"{v2_prefix}sector_momentum",
        _get("nlp_sector_sentiment"),
    )
    result["ai_volatility_regime"] = _blend(
        f"{v2_prefix}vol_regime",
        -_get("nlp_volatility_risk"),
    )

    return result
