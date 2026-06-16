"""
regime.py
---------
Rule-based market regime classification (price-derived, V2.1).
V3 addition: optional NLP-aware summary table per regime.

Regimes (diagnostic only — never used as training labels):
  1. Risk-On Trend
  2. Bond-Led Defensive
  3. High-Vol Stress
  4. Defensive / Risk-Off
  5. Neutral / Mixed
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import TICKERS, CONTEXT_MOMENTUM_WINDOW, CONTEXT_VOL_WINDOW
from nlp_schema import get_nlp_signal_columns

STRONG_POSITIVE =  0.50
STRONG_NEGATIVE = -0.50
MILD_NEGATIVE   = -0.20

REGIME_COLORS = {
    "Risk-On Trend":        "#3fb950",
    "Neutral / Mixed":      "#8b949e",
    "Defensive / Risk-Off": "#d29922",
    "Bond-Led Defensive":   "#58a6ff",
    "High-Vol Stress":      "#f85149",
}


def classify_regimes(features: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    feat_spy = (
        features.reset_index()
        .query("ticker == 'SPY'")
        .set_index("date")
        .sort_index()
    )

    macro   = feat_spy["ai_macro_sentiment"]   if "ai_macro_sentiment"   in feat_spy.columns else pd.Series(0.0, index=feat_spy.index)
    vol_reg = feat_spy["ai_volatility_regime"] if "ai_volatility_regime" in feat_spy.columns else pd.Series(0.0, index=feat_spy.index)

    log_ret = np.log(prices / prices.shift(1))
    spy_mom = log_ret["SPY"].rolling(CONTEXT_MOMENTUM_WINDOW).sum() if "SPY" in log_ret else pd.Series(0.0, index=prices.index)
    tlt_mom = log_ret["TLT"].rolling(CONTEXT_MOMENTUM_WINDOW).sum() if "TLT" in log_ret else pd.Series(0.0, index=prices.index)

    date_index = feat_spy.index
    macro   = macro.reindex(date_index).fillna(0.0)
    vol_reg = vol_reg.reindex(date_index).fillna(0.0)
    spy_mom = spy_mom.reindex(date_index).fillna(0.0)
    tlt_mom = tlt_mom.reindex(date_index).fillna(0.0)

    regimes = pd.Series("Neutral / Mixed", index=date_index, name="regime")
    regimes[macro < MILD_NEGATIVE]                                = "Defensive / Risk-Off"
    regimes[(macro < STRONG_NEGATIVE) & (tlt_mom > 0)]           = "Bond-Led Defensive"
    regimes[(macro > STRONG_POSITIVE) & (spy_mom > 0)]           = "Risk-On Trend"
    regimes[vol_reg < STRONG_NEGATIVE]                            = "High-Vol Stress"

    return regimes


def performance_by_regime(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    from config import TRADING_DAYS
    from risk import max_drawdown as mdd_fn, annualised_volatility

    common = strategy_returns.index.intersection(benchmark_returns.index).intersection(regimes.index)
    strat  = strategy_returns.loc[common]
    bench  = benchmark_returns.loc[common]
    reg    = regimes.loc[common]

    rows = []
    for label in sorted(reg.unique()):
        mask = reg == label
        s, b = strat[mask], bench[mask]
        if len(s) == 0:
            continue
        rows.append({
            "Regime":                  label,
            "Days":                    len(s),
            "Strategy Avg Daily Ret":  f"{s.mean():.3%}",
            "Benchmark Avg Daily Ret": f"{b.mean():.3%}",
            "Strategy Ann Vol":        f"{annualised_volatility(s):.2%}",
            "Benchmark Ann Vol":       f"{annualised_volatility(b):.2%}",
            "Strategy Max Drawdown":   f"{mdd_fn(s):.2%}",
            "Benchmark Max Drawdown":  f"{mdd_fn(b):.2%}",
        })
    return pd.DataFrame(rows)


def regime_timeline_data(regimes: pd.Series) -> pd.DataFrame:
    REGIME_ORDER = [
        "Risk-On Trend", "Neutral / Mixed", "Defensive / Risk-Off",
        "Bond-Led Defensive", "High-Vol Stress",
    ]
    code_map = {r: i for i, r in enumerate(REGIME_ORDER)}
    df = regimes.reset_index()
    df.columns = ["date", "regime"]
    df["regime_code"] = df["regime"].map(code_map).fillna(len(REGIME_ORDER))
    return df


def nlp_by_regime(
    features: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    V3: Average NLP signals per regime.
    Returns empty DataFrame if nlp_* columns are absent.
    """
    nlp_cols = [c for c in get_nlp_signal_columns() if c in features.columns]
    if not nlp_cols:
        return pd.DataFrame()

    # Get date-level NLP values (use SPY slice as representative cross-sectional values)
    feat_flat = features.reset_index()
    spy_flat  = feat_flat[feat_flat["ticker"] == "SPY"].set_index("date").sort_index()
    nlp_daily = spy_flat[[c for c in nlp_cols if c in spy_flat.columns]]

    common = regimes.index.intersection(nlp_daily.index)
    if common.empty:
        return pd.DataFrame()

    merged = nlp_daily.loc[common].copy()
    merged["regime"] = regimes.loc[common]

    rows = []
    for regime_label in sorted(merged["regime"].unique()):
        sub = merged[merged["regime"] == regime_label]
        row = {"Regime": regime_label, "Days": len(sub)}
        for col in [c for c in nlp_cols if c in merged.columns]:
            row[col.replace("nlp_", "")] = round(float(sub[col].mean()), 4)
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
