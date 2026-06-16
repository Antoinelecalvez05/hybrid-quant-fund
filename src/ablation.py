"""
ablation.py
-----------
Three-way ablation: Baseline / V2 Proxy / V3 NLP-Enriched.

Model A — Baseline:
    Financial features only. No ai_* or nlp_* columns.

Model B — V2 Proxy Contextual:
    Financial features + V2 price-proxy ai_* signals.
    Uses the same features as main pipeline without NLP records.

Model C — V3 NLP-Enriched Contextual:
    Financial features + ai_* (blended V2+NLP) + nlp_* columns.
    Only run if NLP records are available.
    If no NLP records, Model C is skipped and a message is returned.

All models use identical RF hyperparameters, train on 2010-2018 only,
and apply transaction costs. This tab is diagnostic only — no parameters
are modified based on results.
"""

from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import (
    TRAIN_START, TRAIN_END,
    VAL_START,   VAL_END,
    TEST_START,  TEST_END,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF,
    RF_RANDOM_STATE, RF_CLASS_WEIGHT,
    TICKERS,
)
from ai_signal import get_ai_signal_columns
from nlp_schema import get_nlp_signal_columns
from features import prepare_Xy
from backtest import run_backtest
from risk import compute_metrics


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RF_RANDOM_STATE,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=-1,
    )


def _slice(features: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    features = features.sort_index()
    dates = features.index.get_level_values("date")
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return features.loc[mask].copy()


def _evaluate_period(
    clf: RandomForestClassifier,
    features: pd.DataFrame,
    feature_cols: list[str],
    prices: pd.DataFrame,
    start: str,
    end: str,
    model_label: str,
    period_label: str,
) -> dict:
    from model import predict_signals
    period_feat = _slice(features, start, end)
    X, _ = prepare_Xy(period_feat, feature_cols)
    if len(X) == 0:
        return {}
    signals = predict_signals(clf, X)
    period_prices = prices.loc[start:end]
    bt = run_backtest(period_prices, signals, tickers=TICKERS)
    m = compute_metrics(bt["strategy_returns"], label=f"{model_label} | {period_label}")
    m["model"]            = model_label
    m["period"]           = period_label
    m["average_turnover"] = bt["average_turnover"]
    m["equity_curve"]     = bt["equity_curve"]
    m["benchmark_curve"]  = bt["benchmark_curve"]
    m["benchmark_returns"]= bt["benchmark_returns"]
    return m


def run_ablation(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    nlp_records=None,   # list[NLPSignalRecord] | None
) -> dict:
    """
    Train and evaluate three models.

    Parameters
    ----------
    features    : full V3 feature DataFrame (includes ai_* and nlp_* columns)
    prices      : raw price DataFrame
    nlp_records : optional NLP records list

    Returns
    -------
    dict with keys: results, comparison_df, baseline_cols, v2_cols, v3_cols,
                    has_nlp (bool), skipped_c (bool if Model C was skipped)
    """
    ai_cols  = get_ai_signal_columns()
    nlp_cols = get_nlp_signal_columns()
    all_cols = [c for c in features.columns if c != "target"]

    # Define feature column sets
    baseline_cols = [c for c in all_cols if c not in ai_cols and c not in nlp_cols]
    v2_cols       = [c for c in all_cols if c not in nlp_cols]   # financial + ai_*
    v3_cols       = all_cols                                       # everything

    has_nlp   = bool(nlp_records)
    skipped_c = not has_nlp

    train_feat = _slice(features, TRAIN_START, TRAIN_END)
    X_base,  y_train = prepare_Xy(train_feat, baseline_cols)
    X_v2,    _       = prepare_Xy(train_feat, v2_cols)
    X_v3,    _       = prepare_Xy(train_feat, v3_cols)

    clf_base = _make_rf(); clf_base.fit(X_base, y_train)
    clf_v2   = _make_rf(); clf_v2.fit(X_v2,   y_train)
    clf_v3   = _make_rf(); clf_v3.fit(X_v3,   y_train)

    periods = [
        ("Train",      TRAIN_START, TRAIN_END),
        ("Validation", VAL_START,   VAL_END),
        ("Test",       TEST_START,  TEST_END),
    ]

    models = [
        ("Baseline",           clf_base, baseline_cols),
        ("V2 Proxy",           clf_v2,   v2_cols),
    ]
    if not skipped_c:
        models.append(("V3 NLP-Enriched", clf_v3, v3_cols))

    results = []
    for period_label, start, end in periods:
        for model_label, clf, cols in models:
            r = _evaluate_period(clf, features, cols, prices, start, end, model_label, period_label)
            if r:
                results.append(r)

    display_keys = [
        "model", "period",
        "total_return", "annualised_return", "annualised_volatility",
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
        "win_rate", "average_turnover",
    ]
    rows = [{k: r.get(k, 0.0) for k in display_keys} for r in results]
    comparison_df = pd.DataFrame(rows)

    return {
        "results":       results,
        "comparison_df": comparison_df,
        "baseline_cols": baseline_cols,
        "v2_cols":       v2_cols,
        "v3_cols":       v3_cols,
        "has_nlp":       has_nlp,
        "skipped_c":     skipped_c,
        "clf_base":      clf_base,
        "clf_v2":        clf_v2,
        "clf_v3":        clf_v3,
    }


def format_comparison_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    pct_cols   = ["total_return", "annualised_return", "annualised_volatility",
                  "max_drawdown", "win_rate", "average_turnover"]
    ratio_cols = ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]
    display = comparison_df.copy()
    display["label"] = display["model"] + " | " + display["period"]
    display = display.set_index("label").drop(columns=["model", "period"], errors="ignore")
    for col in pct_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.2%}")
    for col in ratio_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.2f}")
    display.columns = [c.replace("_", " ").title() for c in display.columns]
    return display


def ablation_interpretation(comparison_df: pd.DataFrame, has_nlp: bool = False) -> str:
    def get_sharpe(model: str, period: str) -> float:
        row = comparison_df[
            (comparison_df["model"] == model) & (comparison_df["period"] == period)
        ]
        return float(row["sharpe_ratio"].iloc[0]) if not row.empty else 0.0

    val_base = get_sharpe("Baseline",  "Validation")
    val_v2   = get_sharpe("V2 Proxy",  "Validation")
    val_v3   = get_sharpe("V3 NLP-Enriched", "Validation") if has_nlp else None
    tst_base = get_sharpe("Baseline",  "Test")
    tst_v2   = get_sharpe("V2 Proxy",  "Test")
    tst_v3   = get_sharpe("V3 NLP-Enriched", "Test") if has_nlp else None

    lines = []
    # V2 vs Baseline
    if val_v2 > val_base and tst_v2 > tst_base:
        lines.append("✅ **V2 Proxy** outperforms Baseline in both validation and test (Sharpe).")
    elif tst_v2 > tst_base:
        lines.append("⚠️ **V2 Proxy** outperforms Baseline in test but not validation — may be regime-specific.")
    elif val_v2 > val_base:
        lines.append("⚠️ **V2 Proxy** outperforms Baseline in validation but not test.")
    else:
        lines.append("❌ **V2 Proxy** does not outperform Baseline in either out-of-sample period.")

    # V3 vs V2
    if has_nlp and val_v3 is not None and tst_v3 is not None:
        if val_v3 > val_v2 and tst_v3 > tst_v2:
            lines.append("✅ **V3 NLP-Enriched** outperforms V2 Proxy in both val and test.")
        elif tst_v3 > tst_v2:
            lines.append("⚠️ **V3 NLP-Enriched** outperforms V2 Proxy in test but not validation.")
        elif val_v3 > val_v2:
            lines.append("⚠️ **V3 NLP-Enriched** outperforms V2 Proxy in validation but not test.")
        else:
            lines.append("❌ **V3 NLP-Enriched** does not outperform V2 Proxy in either out-of-sample period.")
    elif not has_nlp:
        lines.append(
            "ℹ️ **V3 NLP-Enriched model not run** — no NLP records available. "
            "Add text files to `data/text_sources/` to enable Model C."
        )

    lines.append(
        "_These results are diagnostic only. No model parameters were changed based on them._"
    )
    return "  \n".join(lines)
