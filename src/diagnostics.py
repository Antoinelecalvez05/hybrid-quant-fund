"""
diagnostics.py
--------------
Feature importance diagnostics, signal correlation analysis, rolling metrics,
and V3 NLP diagnostics.

Includes:
  - permutation importance by period
  - contextual signal correlations with future returns
  - rolling performance metrics
  - NLP source summary
  - NLP record summary
  - NLP coverage by period
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from config import (
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
    TRADING_DAYS,
    RISK_FREE_RATE,
)

from ai_signal import get_ai_signal_columns
from features import prepare_Xy


# ---------------------------------------------------------------------------
# Safe period slicing
# ---------------------------------------------------------------------------

def _slice_features(
    features: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Safely slice a MultiIndex DataFrame indexed by (date, ticker).
    """

    if features is None or len(features) == 0:
        return pd.DataFrame()

    f = features.sort_index()

    if not isinstance(f.index, pd.MultiIndex):
        return pd.DataFrame()

    dates = f.index.get_level_values("date")
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))

    return f.loc[mask].copy().sort_index()


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    clf: RandomForestClassifier,
    features: pd.DataFrame,
    feature_cols: list[str],
    start: str,
    end: str,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance on one period.

    Uses the already-trained model.
    Does not retrain on validation or test.
    """

    period_features = _slice_features(features, start, end)

    if period_features.empty:
        return pd.DataFrame(
            columns=["feature", "mean_importance", "std_importance"]
        )

    X, y = prepare_Xy(period_features, feature_cols)

    if len(X) < 10 or y.nunique() < 2:
        return pd.DataFrame(
            columns=["feature", "mean_importance", "std_importance"]
        )

    result = permutation_importance(
        clf,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="roc_auc",
    )

    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_importance": result.importances_mean,
                "std_importance": result.importances_std,
            }
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_permutation_importance_all_periods(
    clf: RandomForestClassifier,
    features: pd.DataFrame,
    feature_cols: list[str],
    n_repeats: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Compute permutation importance on Train / Validation / Test.
    """

    periods = {
        "Train": (TRAIN_START, TRAIN_END),
        "Validation": (VAL_START, VAL_END),
        "Test": (TEST_START, TEST_END),
    }

    return {
        label: compute_permutation_importance(
            clf=clf,
            features=features,
            feature_cols=feature_cols,
            start=start,
            end=end,
            n_repeats=n_repeats,
        )
        for label, (start, end) in periods.items()
    }


# ---------------------------------------------------------------------------
# Signal correlation with future returns
# ---------------------------------------------------------------------------

def _get_nlp_signal_columns_safe(features: pd.DataFrame | None = None) -> list[str]:
    """
    Safely get NLP signal columns.
    """

    try:
        from nlp_schema import get_nlp_signal_columns

        cols = get_nlp_signal_columns()
    except Exception:
        cols = [
            "nlp_macro_sentiment",
            "nlp_sector_sentiment",
            "nlp_volatility_risk",
            "nlp_inflation_pressure",
            "nlp_rates_pressure",
            "nlp_recession_risk",
            "nlp_confidence",
        ]

    if features is not None:
        cols = [c for c in cols if c in features.columns]

    return cols


def compute_signal_correlations(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: list[int] | None = None,
    extra_signal_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Spearman correlation between contextual signals and future returns.

    Uses future returns for analysis only, never for training.
    """

    if horizons is None:
        horizons = [1, 5, 21]

    if features is None or prices is None or len(features) == 0 or len(prices) == 0:
        return pd.DataFrame()

    ai_cols = [c for c in get_ai_signal_columns() if c in features.columns]
    nlp_cols = _get_nlp_signal_columns_safe(features)

    signal_cols = []
    for col in ai_cols + nlp_cols + (extra_signal_cols or []):
        if col in features.columns and col not in signal_cols:
            signal_cols.append(col)

    if not signal_cols:
        return pd.DataFrame()

    log_returns = np.log(prices / prices.shift(1))
    future_returns = {
        h: log_returns.rolling(h).sum().shift(-h)
        for h in horizons
    }

    feat_flat = features.reset_index()

    if "date" not in feat_flat.columns or "ticker" not in feat_flat.columns:
        return pd.DataFrame()

    feat_flat["date"] = pd.to_datetime(feat_flat["date"])

    periods = {
        "Train": (TRAIN_START, TRAIN_END),
        "Validation": (VAL_START, VAL_END),
        "Test": (TEST_START, TEST_END),
    }

    rows = []

    for period_label, (start, end) in periods.items():
        mask = (
            (feat_flat["date"] >= pd.Timestamp(start))
            & (feat_flat["date"] <= pd.Timestamp(end))
        )

        period_flat = feat_flat.loc[mask].copy()

        if period_flat.empty:
            continue

        for signal_col in signal_cols:
            row = {
                "signal": signal_col,
                "period": period_label,
            }

            for horizon in horizons:
                future_wide = future_returns[horizon]

                future_values = [
                    future_wide.loc[d, t]
                    if d in future_wide.index and t in future_wide.columns
                    else np.nan
                    for d, t in zip(period_flat["date"], period_flat["ticker"])
                ]

                temp = period_flat[[signal_col]].copy()
                temp[f"future_return_{horizon}d"] = future_values

                temp = temp.replace([np.inf, -np.inf], np.nan).dropna()

                if len(temp) < 10:
                    row[f"horizon_{horizon}d"] = np.nan
                else:
                    corr = temp[signal_col].corr(
                        temp[f"future_return_{horizon}d"],
                        method="spearman",
                    )
                    row[f"horizon_{horizon}d"] = (
                        round(float(corr), 4) if pd.notna(corr) else np.nan
                    )

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    column_order = ["signal", "period"] + [f"horizon_{h}d" for h in horizons]

    return result[[c for c in column_order if c in result.columns]]


def format_correlation_table(corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format signal correlation table for Streamlit.
    """

    if corr_df is None or corr_df.empty:
        return pd.DataFrame()

    display = corr_df.copy()

    for col in display.columns:
        if col.startswith("horizon_"):
            display[col] = display[col].map(
                lambda x: f"{x:+.4f}" if pd.notna(x) else "n/a"
            )

    display.columns = [
        c.replace("horizon_", "Corr ")
        .replace("d", "d fwd")
        .replace("_", " ")
        .title()
        if c.startswith("horizon_")
        else c.replace("_", " ").title()
        for c in display.columns
    ]

    return display


# ---------------------------------------------------------------------------
# Rolling performance metrics
# ---------------------------------------------------------------------------

def compute_rolling_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 126,
) -> dict[str, pd.Series]:
    """
    Compute rolling return, volatility, and Sharpe ratio.
    """

    strategy_returns = strategy_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()

    common = strategy_returns.index.intersection(benchmark_returns.index)

    strategy_returns = strategy_returns.loc[common]
    benchmark_returns = benchmark_returns.loc[common]

    def rolling_return(returns: pd.Series) -> pd.Series:
        return (
            (1 + returns)
            .rolling(window)
            .apply(lambda x: x.prod(), raw=True)
            ** (TRADING_DAYS / window)
            - 1
        )

    def rolling_volatility(returns: pd.Series) -> pd.Series:
        return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

    def rolling_sharpe(returns: pd.Series) -> pd.Series:
        rr = rolling_return(returns)
        rv = rolling_volatility(returns)
        return (rr - RISK_FREE_RATE) / rv.replace(0, np.nan)

    return {
        "strat_rolling_return": rolling_return(strategy_returns),
        "bench_rolling_return": rolling_return(benchmark_returns),
        "strat_rolling_vol": rolling_volatility(strategy_returns),
        "bench_rolling_vol": rolling_volatility(benchmark_returns),
        "strat_rolling_sharpe": rolling_sharpe(strategy_returns),
        "bench_rolling_sharpe": rolling_sharpe(benchmark_returns),
    }


# ---------------------------------------------------------------------------
# V3 NLP diagnostics helpers
# ---------------------------------------------------------------------------

def nlp_source_summary(sources_df: pd.DataFrame) -> dict:
    """
    Summarise loaded NLP text sources.
    """

    if sources_df is None or len(sources_df) == 0:
        return {
            "source_count": 0,
            "unique_sources": 0,
            "date_min": None,
            "date_max": None,
            "topics": [],
            "tickers": [],
        }

    df = sources_df.copy()

    for col in ["date", "source", "topic", "ticker"]:
        if col not in df.columns:
            df[col] = ""

    dates = pd.to_datetime(df["date"], errors="coerce")

    return {
        "source_count": int(len(df)),
        "unique_sources": int(df["source"].nunique()),
        "date_min": dates.min().date().isoformat() if dates.notna().any() else None,
        "date_max": dates.max().date().isoformat() if dates.notna().any() else None,
        "topics": sorted(
            [str(x) for x in df["topic"].dropna().unique() if str(x)]
        ),
        "tickers": sorted(
            [str(x) for x in df["ticker"].dropna().unique() if str(x)]
        ),
    }


def nlp_records_summary(records) -> dict:
    """
    Summarise extracted NLP records.

    Works with either:
      - NLPSignalRecord objects
      - dictionaries
    """

    if not records:
        return {
            "record_count": 0,
            "avg_confidence": 0.0,
            "heuristic_count": 0,
            "llm_count": 0,
        }

    rows = []

    for record in records:
        if hasattr(record, "to_dict"):
            rows.append(record.to_dict())
        elif isinstance(record, dict):
            rows.append(record)

    if not rows:
        return {
            "record_count": 0,
            "avg_confidence": 0.0,
            "heuristic_count": 0,
            "llm_count": 0,
        }

    df = pd.DataFrame(rows)

    if "confidence" not in df.columns:
        df["confidence"] = 0.0

    if "extraction_method" not in df.columns:
        df["extraction_method"] = ""

    confidence = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

    return {
        "record_count": int(len(df)),
        "avg_confidence": float(confidence.mean()),
        "heuristic_count": int((df["extraction_method"] == "heuristic").sum()),
        "llm_count": int((df["extraction_method"] == "llm").sum()),
    }


def nlp_coverage_by_period(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NLP signal coverage by period.

    Coverage = fraction of rows where at least one directional NLP signal is non-zero.
    """

    if features is None or len(features) == 0:
        return pd.DataFrame()

    nlp_cols = _get_nlp_signal_columns_safe(features)

    if not nlp_cols:
        return pd.DataFrame()

    directional_cols = [
        c for c in nlp_cols
        if c != "nlp_confidence" and c in features.columns
    ]

    if not directional_cols:
        return pd.DataFrame()

    f = features.sort_index()

    if not isinstance(f.index, pd.MultiIndex):
        return pd.DataFrame()

    dates = f.index.get_level_values("date")

    periods = {
        "Train": (TRAIN_START, TRAIN_END),
        "Validation": (VAL_START, VAL_END),
        "Test": (TEST_START, TEST_END),
    }

    rows = []

    for label, (start, end) in periods.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        period_df = f.loc[mask]

        if period_df.empty:
            continue

        values = period_df[directional_cols].fillna(0.0)
        active = values.abs().sum(axis=1) > 1e-12

        row = {
            "rows": float(len(period_df)),
            "active_coverage": float(active.mean()),
        }

        for col in nlp_cols:
            if col in period_df.columns:
                row[col] = float(period_df[col].fillna(0.0).mean())

        rows.append((label, row))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [row for _, row in rows],
        index=[label for label, _ in rows],
    )
