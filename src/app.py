"""
app.py — V3
-----------
Streamlit front-end for the hybrid-quant-fund research prototype.

Run with:  streamlit run src/app.py

V3 additions vs V2.1:
  - NLP mode selector in sidebar (Off / Heuristic / LLM if available)
  - "🧠 NLP / LLM Signals" tab
  - Ablation tab extended to three models (Baseline / V2 Proxy / V3 NLP-Enriched)
  - Context Signals tab shows V2 vs V3 NLP signals separately
  - Feature Analysis extended with nlp_* column highlighting
  - Governance tab updated to V3

All V2.1 tabs preserved.
"""

import sys
import os
from io import StringIO

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from data_loader  import load_prices
from features     import build_features, get_feature_columns, prepare_Xy
from ai_signal    import generate_ai_signals, get_ai_signal_columns
from nlp_schema   import get_nlp_signal_columns
from text_sources import load_text_sources
from nlp_extractor import extract_all
from model        import (
    train_model, predict_signals, evaluate_model,
    feature_importance_df, confusion_matrix_df,
)
from backtest     import run_backtest
from risk         import compute_metrics, metrics_to_dataframe
from governance   import create_record, DISCLAIMER
from ablation     import run_ablation, format_comparison_table, ablation_interpretation
from diagnostics  import (
    compute_permutation_importance_all_periods,
    compute_signal_correlations, format_correlation_table,
    compute_rolling_metrics,
    nlp_source_summary, nlp_records_summary, nlp_coverage_by_period,
)
from regime       import (
    classify_regimes, performance_by_regime,
    regime_timeline_data, nlp_by_regime,
    REGIME_COLORS,
)

# ---------------------------------------------------------------------------
# Page config & CSS (unchanged from V2.1)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hybrid Quant Fund — Research Prototype",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Inter', 'SF Pro Text', system-ui, sans-serif; }
    .main { background-color: #0d1117; }
    [data-testid="metric-container"] {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8b949e !important; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 0.08em;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #e6edf3 !important; font-size: 1.4rem;
        font-weight: 700; font-variant-numeric: tabular-nums;
    }
    h1 { color: #e6edf3 !important; font-weight: 800; letter-spacing: -0.5px; }
    h2 { color: #e6edf3 !important; font-weight: 700; }
    h3 { color: #8b949e !important; font-weight: 600; font-size: 0.85rem;
         text-transform: uppercase; letter-spacing: 0.1em; }
    .disclaimer { background: #1c1a10; border-left: 4px solid #d29922;
        border-radius: 4px; padding: 14px 18px; color: #d29922;
        font-size: 0.82rem; line-height: 1.6; }
    .v3badge { display: inline-block; background: #1a0a2e; color: #bc8cff;
        border: 1px solid #6e40c9; border-radius: 4px; padding: 1px 8px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
        margin-left: 6px; vertical-align: middle; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
             font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
             text-transform: uppercase; margin-right: 6px; }
    .badge-train { background: #0d4a6e; color: #58a6ff; }
    .badge-val   { background: #1a3a1a; color: #3fb950; }
    .badge-test  { background: #3a1a1a; color: #f85149; }
    hr { border-color: #30363d; }
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

PLOT_BG   = "#0d1117"
GRID_COL  = "#21262d"
TEXT_COL  = "#8b949e"
LIGHT_COL = "#e6edf3"


# ---------------------------------------------------------------------------
# Safe helpers (unchanged)
# ---------------------------------------------------------------------------

def restore_features_from_json(features_json: str) -> pd.DataFrame:
    features = pd.read_json(StringIO(features_json))
    if "date" not in features.columns or "ticker" not in features.columns:
        raise ValueError("Expected 'date' and 'ticker' columns.")
    features["date"] = pd.to_datetime(features["date"])
    return features.set_index(["date", "ticker"]).sort_index()


def slice_feature_period(features: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    features = features.sort_index()
    dates = features.index.get_level_values("date")
    mask  = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return features.loc[mask].copy().sort_index()


# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Downloading price data …")
def cached_load_prices(force: bool = False) -> pd.DataFrame:
    return load_prices(force_download=force)


@st.cache_data(show_spinner="Loading NLP text sources …")
def cached_load_sources(source_dir: str) -> pd.DataFrame:
    return load_text_sources(source_dir)


@st.cache_data(show_spinner="Extracting NLP signals …")
def cached_extract(sources_json: str, mode: str) -> list[dict]:
    """Extract NLP records; cache as list-of-dicts (JSON-serialisable)."""
    sources_df = pd.read_json(StringIO(sources_json))
    records = extract_all(sources_df, mode=mode)
    return [r.to_dict() for r in records]


@st.cache_data(show_spinner="Building features …")
def cached_build_features(prices_json: str, nlp_records_json: str) -> pd.DataFrame:
    from nlp_schema import validate_record
    prices = pd.read_json(StringIO(prices_json))
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    nlp_raw = pd.read_json(StringIO(nlp_records_json), orient="records")
    nlp_records = (
        [validate_record(row) for _, row in nlp_raw.iterrows()]
        if not nlp_raw.empty else []
    )

    features = build_features(prices)
    features = generate_ai_signals(
        features,
        prices=prices,
        nlp_records=nlp_records if nlp_records else None,
    )
    return features.sort_index()


@st.cache_resource(show_spinner="Training Random Forest …")
def cached_train(train_json: str) -> tuple:
    train_features = restore_features_from_json(train_json)
    feature_cols   = get_feature_columns(train_features)
    X_train, y_train = prepare_Xy(train_features, feature_cols)
    clf = train_model(X_train, y_train)
    return clf, feature_cols, evaluate_model(clf, X_train, y_train, "Train")


@st.cache_data(show_spinner="Running ablation study …")
def cached_ablation(features_json: str, prices_json: str, nlp_records_json: str) -> dict:
    from nlp_schema import validate_record
    features = restore_features_from_json(features_json)
    prices   = pd.read_json(StringIO(prices_json))
    prices.index = pd.to_datetime(prices.index)
    nlp_raw  = pd.read_json(StringIO(nlp_records_json), orient="records")
    nlp_records = (
        [validate_record(row) for _, row in nlp_raw.iterrows()]
        if not nlp_raw.empty else []
    )
    result = run_ablation(features, prices, nlp_records=nlp_records)
    # Remove non-serialisable Series
    slim = []
    for r in result["results"]:
        slim.append({k: v for k, v in r.items() if not isinstance(v, pd.Series)})
    return {
        "comparison_df": result["comparison_df"],
        "slim_results":  slim,
        "baseline_cols": result["baseline_cols"],
        "v2_cols":       result["v2_cols"],
        "v3_cols":       result["v3_cols"],
        "has_nlp":       result["has_nlp"],
        "skipped_c":     result["skipped_c"],
    }


@st.cache_resource(show_spinner="Running ablation — training models (equity curves) …")
def cached_ablation_full(features_json: str, prices_json: str, nlp_records_json: str):
    from nlp_schema import validate_record
    features = restore_features_from_json(features_json)
    prices   = pd.read_json(StringIO(prices_json))
    prices.index = pd.to_datetime(prices.index)
    nlp_raw  = pd.read_json(StringIO(nlp_records_json), orient="records")
    nlp_records = (
        [validate_record(row) for _, row in nlp_raw.iterrows()]
        if not nlp_raw.empty else []
    )
    return run_ablation(features, prices, nlp_records=nlp_records)


@st.cache_data(show_spinner="Computing permutation importance …")
def cached_perm_importance(train_json: str, features_json: str) -> dict:
    clf, feature_cols, _ = cached_train(train_json)
    features = restore_features_from_json(features_json)
    return compute_permutation_importance_all_periods(clf, features, feature_cols, n_repeats=5)


@st.cache_data(show_spinner="Computing signal correlations …")
def cached_signal_corr(features_json: str, prices_json: str) -> pd.DataFrame:
    features = restore_features_from_json(features_json)
    prices   = pd.read_json(StringIO(prices_json))
    prices.index = pd.to_datetime(prices.index)
    return compute_signal_correlations(features, prices)


@st.cache_data(show_spinner="Classifying regimes …")
def cached_regimes(features_json: str, prices_json: str) -> pd.Series:
    features = restore_features_from_json(features_json)
    prices   = pd.read_json(StringIO(prices_json))
    prices.index = pd.to_datetime(prices.index)
    return classify_regimes(features, prices)


# ---------------------------------------------------------------------------
# Chart helpers (unchanged from V2.1, plus new nlp ones)
# ---------------------------------------------------------------------------

def _base_layout(title: str = "", height: int = 380) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color=LIGHT_COL)),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COL, size=11),
        xaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False, tickfont=dict(color=TEXT_COL)),
        yaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False, tickfont=dict(color=TEXT_COL)),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1, font=dict(color=LIGHT_COL)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=height,
    )


def equity_curve_chart(strategy, benchmark, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy.index, y=strategy.values, name="Strategy (net)",
        line=dict(color="#58a6ff", width=2), hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.3f}<extra>Strategy</extra>"))
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, name="Benchmark (EW)",
        line=dict(color="#8b949e", width=1.5, dash="dot"), hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.3f}<extra>Benchmark</extra>"))
    layout = _base_layout(title, height=360)
    layout["yaxis"]["title"] = "Portfolio value ($1 invested)"
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def turnover_chart(turnover, title="Daily Turnover"):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=turnover.index, y=turnover.values, marker_color="#388bfd", opacity=0.7,
        name="Daily turnover", hovertemplate="%{x|%Y-%m-%d}<br>Turnover: %{y:.2%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=turnover.index, y=turnover.rolling(21).mean().values,
        name="21d avg", line=dict(color="#f0883e", width=1.5)))
    layout = _base_layout(title, height=220)
    layout["yaxis"]["tickformat"] = ".1%"
    layout["showlegend"] = True
    fig.update_layout(**layout)
    return fig


def returns_histogram(returns, label):
    fig = go.Figure(go.Histogram(x=returns.values*100, nbinsx=60,
        marker_color="#388bfd", opacity=0.8))
    layout = _base_layout(f"Return distribution — {label}", height=240)
    layout["xaxis"]["title"] = "Daily return (%)"
    layout["yaxis"]["title"] = "Count"
    fig.update_layout(**layout)
    return fig


def feature_importance_chart(imp_df, ai_cols, nlp_cols=None):
    top = imp_df.head(15)
    nlp_cols = nlp_cols or []
    def color(f):
        if f in nlp_cols:  return "#bc8cff"
        if f in ai_cols:   return "#f0883e"
        return "#58a6ff"
    colors = [color(f) for f in top["feature"]]
    fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h",
        marker_color=colors, hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    layout = _base_layout("", height=420)
    layout["xaxis"]["title"] = "Importance"
    layout["yaxis"]["autorange"] = "reversed"
    layout["yaxis"]["tickfont"] = dict(size=10)
    fig.update_layout(**layout)
    return fig


def permutation_importance_chart(perm_df, ai_cols, title, nlp_cols=None):
    top = perm_df.head(15)
    nlp_cols = nlp_cols or []
    def color(f):
        if f in nlp_cols:  return "#bc8cff"
        if f in ai_cols:   return "#f0883e"
        return "#3fb950"
    colors = [color(f) for f in top["feature"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["mean_importance"], y=top["feature"], orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=top["std_importance"].tolist(), visible=True),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    layout = _base_layout(title, height=380)
    layout["xaxis"]["title"] = "Mean AUC decrease"
    layout["yaxis"]["autorange"] = "reversed"
    layout["yaxis"]["tickfont"] = dict(size=10)
    fig.update_layout(**layout)
    return fig


def signal_time_series(series, label, color):
    fig = go.Figure()
    fig.add_hrect(y0=-0.2, y1=0.2, fillcolor="#30363d", opacity=0.2, line_width=0)
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name=label,
        line=dict(color=color, width=1.5),
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:.3f}}<extra>{label}</extra>"))
    fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", line_width=1)
    layout = _base_layout(label, height=200)
    layout["yaxis"]["range"] = [-1.1, 1.1]
    layout["yaxis"]["title"] = "Signal [-1, +1]"
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def confusion_heatmap(cm_df, title):
    fig = go.Figure(go.Heatmap(z=cm_df.values, x=cm_df.columns.tolist(),
        y=cm_df.index.tolist(), colorscale=[[0,"#0d1117"],[1,"#1f6feb"]],
        text=cm_df.values, texttemplate="%{text}", showscale=False))
    layout = _base_layout(title, height=220)
    layout["xaxis"]["side"] = "bottom"
    fig.update_layout(**layout)
    return fig


def ablation_equity_chart(results, period):
    model_colors = {"Baseline": "#f0883e", "V2 Proxy": "#58a6ff", "V3 NLP-Enriched": "#bc8cff"}
    fig = go.Figure()
    bench_added = False
    for r in results:
        if r.get("period") != period:
            continue
        model = r.get("model", "")
        ec = r.get("equity_curve")
        bc = r.get("benchmark_curve")
        if ec is not None and isinstance(ec, pd.Series):
            fig.add_trace(go.Scatter(x=ec.index, y=ec.values, name=model,
                line=dict(color=model_colors.get(model, "#8b949e"), width=2),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>$%{{y:.3f}}<extra>{model}</extra>"))
        if bc is not None and isinstance(bc, pd.Series) and not bench_added:
            fig.add_trace(go.Scatter(x=bc.index, y=bc.values, name="Benchmark (EW)",
                line=dict(color="#8b949e", width=1.5, dash="dot")))
            bench_added = True
    layout = _base_layout(f"{period} — Ablation Equity Curves", height=320)
    layout["yaxis"]["title"] = "Portfolio value ($1)"
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def rolling_chart(rolling, metric_key_strat, metric_key_bench, title, y_fmt=".1%"):
    fig = go.Figure()
    s = rolling.get(metric_key_strat)
    b = rolling.get(metric_key_bench)
    if s is not None:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Strategy",
            line=dict(color="#58a6ff", width=1.8)))
    if b is not None:
        fig.add_trace(go.Scatter(x=b.index, y=b.values, name="Benchmark",
            line=dict(color="#8b949e", width=1.5, dash="dot")))
    fig.add_hline(y=0, line_dash="dot", line_color="#30363d", line_width=1)
    layout = _base_layout(title, height=220)
    layout["yaxis"]["tickformat"] = y_fmt
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def regime_timeline_chart(regime_df):
    fig = go.Figure()
    for regime_label, color in REGIME_COLORS.items():
        subset = regime_df[regime_df["regime"] == regime_label]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(x=subset["date"], y=[1]*len(subset), mode="markers",
            marker=dict(color=color, size=4, symbol="square"), name=regime_label,
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{regime_label}<extra></extra>"))
    layout = _base_layout("Market Regime Timeline", height=180)
    layout["yaxis"]["visible"] = False
    layout["yaxis"]["range"] = [0.5, 1.5]
    layout["hovermode"] = "x"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Metric helpers (unchanged)
# ---------------------------------------------------------------------------

def show_metrics_row(strat_m, bench_m):
    primary = [
        ("Total Return","total_return",True), ("Ann. Return","annualised_return",True),
        ("Ann. Volatility","annualised_volatility",True), ("Sharpe","sharpe_ratio",False),
        ("Max Drawdown","max_drawdown",True), ("Sortino","sortino_ratio",False),
    ]
    cols = st.columns(len(primary))
    for col, (label, key, is_pct) in zip(cols, primary):
        sv, bv = strat_m.get(key, 0.0), bench_m.get(key, 0.0)
        delta  = sv - bv
        col.metric(label, f"{sv:.1%}" if is_pct else f"{sv:.2f}",
                   f"{delta:+.1%} vs EW" if is_pct else f"{delta:+.2f} vs EW")


def show_extra_metrics(strat_m):
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [
        ("Win Rate","win_rate"),("Best Day","best_day"),
        ("Worst Day","worst_day"),("Downside Vol","downside_volatility"),
    ]):
        col.metric(label, f"{strat_m.get(key, 0.0):.2%}")


def eval_period(features, clf, feature_cols, prices, start, end, label):
    pf = slice_feature_period(features, start, end)
    X, y = prepare_Xy(pf, feature_cols)
    if len(X) == 0:
        return None, None, None, None
    sigs = predict_signals(clf, X)
    mm   = evaluate_model(clf, X, y, label)
    bt   = run_backtest(prices.loc[start:end], sigs)
    sm   = compute_metrics(bt["strategy_returns"], f"{label} — Strategy")
    return sigs, mm, bt, sm


def render_period_tab(label, bt, strat_m, bench_m, model_metrics, period_title, caption_text):
    st.markdown(f"### {period_title}")
    st.caption(caption_text)
    if bt is None:
        st.warning("No data available.")
        return
    show_metrics_row(strat_m, bench_m)
    show_extra_metrics(strat_m)
    st.caption(
        f"Avg daily turnover: **{bt.get('average_turnover',0):.1%}** · "
        f"Transaction cost: **{config.TRANSACTION_COST_BPS} bps/unit** · "
        f"Returns shown are **net of costs**"
    )
    st.plotly_chart(equity_curve_chart(bt["equity_curve"], bt["benchmark_curve"],
                                       f"{label} Equity Curve"), use_container_width=True)
    st.plotly_chart(turnover_chart(bt["turnover"], f"{label} Daily Turnover"), use_container_width=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(returns_histogram(bt["strategy_returns"], "Strategy (net)"), use_container_width=True)
    with col_r:
        st.plotly_chart(returns_histogram(bt["benchmark_returns"], "Benchmark"), use_container_width=True)
    with st.expander(f"Model diagnostics — {label}"):
        if model_metrics:
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
            c2.metric("ROC-AUC",  f"{model_metrics['auc']:.4f}")
            cm_df = confusion_matrix_df(model_metrics["confusion"])
            col_cm, col_rpt = st.columns([1, 2])
            with col_cm:
                st.markdown("**Confusion matrix**")
                st.plotly_chart(confusion_heatmap(cm_df, ""), use_container_width=True)
            with col_rpt:
                st.markdown("**Classification report**")
                st.code(model_metrics["report"], language="text")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        force_download = st.button("🔄 Re-download data")
        st.divider()
        st.markdown("### Universe")
        st.write(", ".join(config.TICKERS))
        st.divider()
        st.markdown("### Temporal Split")
        st.markdown(f"""
        <span class="badge badge-train">Train</span>
        {config.TRAIN_START} → {config.TRAIN_END}<br><br>
        <span class="badge badge-val">Val</span>
        {config.VAL_START} → {config.VAL_END}<br><br>
        <span class="badge badge-test">Test</span>
        {config.TEST_START} → {config.TEST_END}
        """, unsafe_allow_html=True)
        st.divider()
        st.markdown(f"### Transaction costs\n`{config.TRANSACTION_COST_BPS} bps` per unit turnover")
        st.divider()
        st.markdown("### NLP Mode")
        # Detect available sources to set a sensible default
        default_mode = "heuristic"  # always safe
        nlp_mode = st.radio(
            "Extraction mode",
            options=["off", "heuristic", "llm"],
            index=["off", "heuristic", "llm"].index(default_mode),
            help=(
                "off: No NLP, use V2 proxy signals only.  \n"
                "heuristic: Keyword-based extraction (no API needed).  \n"
                "llm: LLM extraction (requires OPENAI_API_KEY or ANTHROPIC_API_KEY in .env)."
            ),
        )
        st.divider()
        st.markdown(
            '<div style="color:#484f58;font-size:0.72rem;">Research prototype — not investment advice</div>',
            unsafe_allow_html=True,
        )

    # ---- Header ------------------------------------------------------------
    st.markdown("# 📊 Hybrid Quant Fund")
    st.markdown(
        '<p style="color:#8b949e;margin-top:-12px;font-size:0.9rem;">'
        'Random Forest + Contextual Proxy + NLP Signals'
        '<span class="v3badge">V3</span>'
        ' &nbsp;·&nbsp; Research Prototype</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ <strong>DISCLAIMER</strong> — Research prototype only. "
        "Not a live trading system. Not investment advice. Not a regulated fund. "
        "All results are hypothetical. The NLP layer reads only curated local sources "
        "and does not make trading decisions. The Random Forest is the sole decision layer."
        "</div>", unsafe_allow_html=True,
    )
    st.markdown("")

    # ---- Load prices -------------------------------------------------------
    prices = cached_load_prices(force=force_download).sort_index()
    prices_json = prices.to_json(date_format="iso")

    with st.expander("📥 Price data", expanded=False):
        st.dataframe(prices.tail(10).style.format("{:.2f}"), use_container_width=True)
        st.caption(f"{len(prices):,} trading days · {prices.columns.tolist()}")

    # ---- Load NLP sources --------------------------------------------------
    sources_df   = cached_load_sources(config.NLP_SOURCE_DIR)
    sources_json = sources_df.to_json(orient="records", date_format="iso") if not sources_df.empty else "[]"

    has_sources = not sources_df.empty
    effective_nlp_mode = nlp_mode if has_sources and nlp_mode != "off" else "off"

    nlp_record_dicts = (
        cached_extract(sources_json, effective_nlp_mode)
        if effective_nlp_mode != "off" else []
    )
    nlp_records_json = pd.DataFrame(nlp_record_dicts).to_json(orient="records") if nlp_record_dicts else "[]"

    # Reconstruct typed records for diagnostics
    from nlp_schema import validate_record, NLPSignalRecord
    nlp_records_typed: list[NLPSignalRecord] = (
        [validate_record(d) for d in nlp_record_dicts] if nlp_record_dicts else []
    )
    has_nlp = bool(nlp_records_typed)

    # ---- Build features ----------------------------------------------------
    features = cached_build_features(prices_json, nlp_records_json).sort_index()
    features_json = features.reset_index().to_json(date_format="iso")
    feature_cols  = get_feature_columns(features)
    ai_cols       = get_ai_signal_columns()
    nlp_cols      = [c for c in get_nlp_signal_columns() if c in features.columns]

    # ---- Train main model --------------------------------------------------
    train_features = slice_feature_period(features, config.TRAIN_START, config.TRAIN_END)
    train_json     = train_features.reset_index().to_json(date_format="iso")
    clf, _, train_metrics = cached_train(train_json)

    # ---- Evaluate periods --------------------------------------------------
    train_sigs, train_mm, train_bt, train_sm = eval_period(
        features, clf, feature_cols, prices, config.TRAIN_START, config.TRAIN_END, "Train")
    val_sigs, val_mm, val_bt, val_sm = eval_period(
        features, clf, feature_cols, prices, config.VAL_START, config.VAL_END, "Validation")
    test_sigs, test_mm, test_bt, test_sm = eval_period(
        features, clf, feature_cols, prices, config.TEST_START, config.TEST_END, "Test")

    train_bm = compute_metrics(train_bt["benchmark_returns"], "Train — Benchmark") if train_bt else {}
    val_bm   = compute_metrics(val_bt["benchmark_returns"],   "Val — Benchmark")   if val_bt   else {}
    test_bm  = compute_metrics(test_bt["benchmark_returns"],  "Test — Benchmark")  if test_bt  else {}

    # ---- V2.1/V3 cached computations ---------------------------------------
    ablation_meta  = cached_ablation(features_json, prices_json, nlp_records_json)
    ablation_full  = cached_ablation_full(features_json, prices_json, nlp_records_json)
    perm_imp       = cached_perm_importance(train_json, features_json)
    signal_corr_df = cached_signal_corr(features_json, prices_json)
    regimes        = cached_regimes(features_json, prices_json)

    # ====================================================================
    #  TABS
    # ====================================================================
    (tab_train, tab_val, tab_test,
     tab_ablation, tab_rolling, tab_regime,
     tab_context, tab_nlp, tab_features, tab_gov) = st.tabs([
        "🟦 Training (2010-2018)",
        "🟩 Validation (2019-2020)",
        "🟥 Test (2021-2023)",
        "🧪 Ablation Test",
        "📈 Rolling Diagnostics",
        "🌍 Regime Analysis",
        "🧭 Context Signals",
        "🧠 NLP / LLM Signals",
        "🔬 Feature Analysis",
        "📋 Governance",
    ])

    # ---- Training ----------------------------------------------------------
    with tab_train:
        render_period_tab("Training", train_bt, train_sm or {}, train_bm, train_mm,
            "Training Period — 2010 to 2018",
            "In-sample. ⚠️ High accuracy often reflects overfitting — val/test matter more.")

    # ---- Validation --------------------------------------------------------
    with tab_val:
        render_period_tab("Validation", val_bt, val_sm or {}, val_bm, val_mm,
            "Validation Period — 2019 to 2020",
            "Out-of-sample. No hyperparameters changed after viewing these results.")

    # ---- Test --------------------------------------------------------------
    with tab_test:
        render_period_tab("Test", test_bt, test_sm or {}, test_bm, test_mm,
            "Test Period — 2021 to 2023",
            "Final held-out evaluation. Viewed after modelling decisions are locked.")
        st.divider()
        st.markdown("### Summary — All Periods")
        strat_list = [m for m in [train_sm, val_sm, test_sm] if m]
        bench_list = [m for m in [train_bm, val_bm, test_bm] if m]
        if strat_list:
            st.markdown("**Strategy — net of transaction costs**")
            st.dataframe(metrics_to_dataframe(strat_list), use_container_width=True)
        if bench_list:
            st.markdown("**Benchmark — equal-weight, frictionless**")
            st.dataframe(metrics_to_dataframe(bench_list), use_container_width=True)

    # ---- Ablation ----------------------------------------------------------
    with tab_ablation:
        st.markdown("### 🧪 Ablation Test — Baseline / V2 Proxy / V3 NLP-Enriched")
        n_models = 3 if not ablation_meta["skipped_c"] else 2
        st.info(
            f"**Model A (Baseline):** Financial features only.  \n"
            f"**Model B (V2 Proxy):** Financial features + V2 price-proxy ai_* signals.  \n"
            f"**Model C (V3 NLP-Enriched):** {'All features including nlp_* signals.' if not ablation_meta['skipped_c'] else 'Skipped — no NLP records available. Add files to `data/text_sources/` to enable.'}  \n"
            f"All models use identical RF hyperparameters, train on 2010-2018, apply transaction costs."
        )

        comp_df = ablation_meta["comparison_df"]
        interp  = ablation_interpretation(comp_df, has_nlp=ablation_meta["has_nlp"])
        for line in interp.split("  \n"):
            if line.startswith("✅"):       st.success(line)
            elif line.startswith("⚠️"):    st.warning(line)
            elif line.startswith("❌"):    st.error(line)
            elif line.startswith("ℹ️"):   st.info(line)
            else:                           st.caption(line)

        st.markdown("#### Performance Summary")
        st.dataframe(format_comparison_table(comp_df), use_container_width=True)

        st.markdown("#### Equity Curves by Period")
        for period in ["Train", "Validation", "Test"]:
            st.plotly_chart(
                ablation_equity_chart(ablation_full["results"], period),
                use_container_width=True,
            )

    # ---- Rolling Diagnostics -----------------------------------------------
    with tab_rolling:
        st.markdown("### 📈 Rolling Performance Diagnostics")
        st.caption("6-month rolling window (≈ 126 trading days).")
        ROLLING_WINDOW = 126

        def full_series(bt_list, key):
            parts = [bt[key] for bt in bt_list if bt is not None]
            return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)

        full_strat = full_series([train_bt, val_bt, test_bt], "strategy_returns")
        full_bench = full_series([train_bt, val_bt, test_bt], "benchmark_returns")

        if len(full_strat) > ROLLING_WINDOW:
            rolling = compute_rolling_metrics(full_strat, full_bench, window=ROLLING_WINDOW)
            st.markdown("#### Rolling 6-Month Return")
            st.plotly_chart(rolling_chart(rolling, "strat_rolling_return", "bench_rolling_return",
                "Rolling 6-Month Annualised Return", ".1%"), use_container_width=True)
            st.markdown("#### Rolling 6-Month Volatility")
            st.plotly_chart(rolling_chart(rolling, "strat_rolling_vol", "bench_rolling_vol",
                "Rolling 6-Month Annualised Volatility", ".1%"), use_container_width=True)
            st.markdown("#### Rolling 6-Month Sharpe Ratio")
            st.plotly_chart(rolling_chart(rolling, "strat_rolling_sharpe", "bench_rolling_sharpe",
                "Rolling 6-Month Sharpe Ratio", ".2f"), use_container_width=True)
            st.divider()
            st.markdown("#### Rolling Sharpe by Period")
            period_cols = st.columns(3)
            for (period_label, bt, col) in [
                ("Train", train_bt, period_cols[0]),
                ("Validation", val_bt, period_cols[1]),
                ("Test", test_bt, period_cols[2]),
            ]:
                with col:
                    st.markdown(f"**{period_label}**")
                    if bt and len(bt["strategy_returns"]) > ROLLING_WINDOW:
                        r = compute_rolling_metrics(bt["strategy_returns"], bt["benchmark_returns"], ROLLING_WINDOW)
                        fig = go.Figure()
                        s, b = r["strat_rolling_sharpe"], r["bench_rolling_sharpe"]
                        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Strategy", line=dict(color="#58a6ff", width=1.5)))
                        fig.add_trace(go.Scatter(x=b.index, y=b.values, name="Benchmark", line=dict(color="#8b949e", width=1.2, dash="dot")))
                        fig.add_hline(y=0, line_dash="dot", line_color="#30363d", line_width=1)
                        layout = _base_layout(f"{period_label} Rolling Sharpe", height=200)
                        layout["yaxis"]["tickformat"] = ".2f"
                        layout["showlegend"] = False
                        fig.update_layout(**layout)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("Not enough data for rolling window.")
        else:
            st.warning("Not enough data.")

    # ---- Regime Analysis ---------------------------------------------------
    with tab_regime:
        st.markdown("### 🌍 Market Regime Analysis")
        st.info("Regimes use price-derived signals only. Diagnostic only — never used to train the model.")

        regime_df = regime_timeline_data(regimes)
        st.markdown("#### Regime Timeline")
        st.plotly_chart(regime_timeline_chart(regime_df), use_container_width=True)

        regime_counts = regimes.value_counts().reset_index()
        regime_counts.columns = ["Regime", "Days"]
        col_c, col_t = st.columns([1, 2])
        with col_c:
            st.markdown("#### Days per Regime")
            st.dataframe(regime_counts, use_container_width=True, hide_index=True)

        if train_bt and val_bt and test_bt:
            all_strat = pd.concat([train_bt["strategy_returns"], val_bt["strategy_returns"], test_bt["strategy_returns"]]).sort_index()
            all_bench = pd.concat([train_bt["benchmark_returns"], val_bt["benchmark_returns"], test_bt["benchmark_returns"]]).sort_index()
            with col_t:
                st.markdown("#### Performance by Regime")
                st.dataframe(performance_by_regime(all_strat, all_bench, regimes), use_container_width=True, hide_index=True)

        if has_nlp:
            st.divider()
            st.markdown("#### Average NLP Signals by Regime (V3)")
            nlp_regime_table = nlp_by_regime(features, regimes)
            if not nlp_regime_table.empty:
                st.dataframe(nlp_regime_table, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Regime Distribution by Period")
        cols3 = st.columns(3)
        for (period_label, start, end), col in zip([
            ("Train", config.TRAIN_START, config.TRAIN_END),
            ("Validation", config.VAL_START, config.VAL_END),
            ("Test", config.TEST_START, config.TEST_END),
        ], cols3):
            with col:
                mask = (regimes.index >= pd.Timestamp(start)) & (regimes.index <= pd.Timestamp(end))
                counts = regimes[mask].value_counts()
                st.markdown(f"**{period_label}**")
                st.dataframe(counts.rename("Days").to_frame(), use_container_width=True)

    # ---- Context Signals ---------------------------------------------------
    with tab_context:
        st.markdown("### 🧭 Contextual Signals")
        v3_active = has_nlp and effective_nlp_mode != "off"
        if v3_active:
            st.success(
                f"**V3 NLP-enriched mode active** (mode: `{effective_nlp_mode}`). "
                "The `ai_*` columns below are a blend of V2 price proxies + NLP signals. "
                "See the 🧠 NLP tab for raw NLP signal time series."
            )
        else:
            st.info("**V2 price-proxy mode** — no NLP sources active. `ai_*` columns are V2 proxies.")

        feat_reset = features.reset_index()
        try:
            macro_series = feat_reset[feat_reset["ticker"]=="SPY"].set_index("date")["ai_macro_sentiment"].sort_index()
            vol_series   = feat_reset[feat_reset["ticker"]=="SPY"].set_index("date")["ai_volatility_regime"].sort_index()

            st.markdown("#### `ai_macro_sentiment`")
            st.caption("V2+NLP blended risk-on / risk-off signal." if v3_active else "V2 price-proxy: SPY − TLT 63d momentum, z-scored.")
            st.plotly_chart(signal_time_series(macro_series, "ai_macro_sentiment", "#58a6ff"), use_container_width=True)

            st.markdown("#### `ai_volatility_regime`")
            st.caption("V2+NLP blended vol regime signal." if v3_active else "V2 price-proxy: rolling vol, z-scored and negated.")
            st.plotly_chart(signal_time_series(vol_series, "ai_volatility_regime", "#f85149"), use_container_width=True)

            st.markdown("#### `ai_sector_momentum`")
            sector_wide = feat_reset[["date","ticker","ai_sector_momentum"]].pivot(index="date",columns="ticker",values="ai_sector_momentum").sort_index()
            fig_sec = go.Figure()
            palette = ["#58a6ff","#3fb950","#f0883e","#bc8cff","#f85149","#39d353"]
            for i, ticker in enumerate(config.TICKERS):
                if ticker in sector_wide.columns:
                    fig_sec.add_trace(go.Scatter(x=sector_wide.index, y=sector_wide[ticker], name=ticker,
                        line=dict(color=palette[i%len(palette)], width=1.4)))
            fig_sec.add_hline(y=0, line_dash="dot", line_color="#8b949e", line_width=1)
            layout = _base_layout("", height=280)
            layout["yaxis"]["range"] = [-1.1, 1.1]
            layout["hovermode"] = "x unified"
            fig_sec.update_layout(**layout)
            st.plotly_chart(fig_sec, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render context signals: {e}")

        # Signal correlations
        st.divider()
        st.markdown("#### Signal Correlation with Future Returns")
        st.caption("Spearman correlation. Future returns for analysis only — never for training.")
        if not signal_corr_df.empty:
            st.dataframe(format_correlation_table(signal_corr_df), use_container_width=True)
        else:
            st.warning("No correlation data available.")

    # ---- NLP / LLM Signals tab (V3 new) ------------------------------------
    with tab_nlp:
        st.markdown("### 🧠 NLP / LLM Signals — V3")
        st.info(
            "**V3 NLP architecture:**  \n"
            "Curated local source files → keyword/LLM extractor → validated JSON schema → "
            "numerical `nlp_*` features → blended with V2 proxies into `ai_*` features → Random Forest.  \n"
            "The LLM/NLP layer **does not** make trading decisions. "
            "The **Random Forest** is the sole decision-making layer."
        )

        # Status
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("NLP Mode",       effective_nlp_mode if effective_nlp_mode != "off" else "Off")
        c2.metric("Source Files",   str(len(sources_df)))
        c3.metric("NLP Records",    str(len(nlp_records_typed)))
        c4.metric("Avg Confidence", f"{np.mean([r.confidence for r in nlp_records_typed]):.2%}" if nlp_records_typed else "n/a")

        if not has_sources:
            st.warning(
                f"No source files found in `{config.NLP_SOURCE_DIR}`.  \n"
                "The app is running in V2 proxy-only mode.  \n"
                "To enable V3 NLP signals, add `.txt`, `.md`, or `.csv` files to that folder."
            )
        else:
            # Source table
            st.markdown("#### Loaded Source Documents")
            display_sources = sources_df[["date","source","title","ticker","topic"]].copy()
            st.dataframe(display_sources, use_container_width=True, hide_index=True)

            if nlp_records_typed:
                # NLP records table
                st.markdown("#### Extracted NLP Records")
                rec_rows = [r.to_dict() for r in nlp_records_typed]
                rec_df = pd.DataFrame(rec_rows)[
                    ["date","source","extraction_method","confidence",
                     "macro_sentiment","volatility_risk","rates_pressure","recession_risk","explanation"]
                ]
                st.dataframe(rec_df.style.format({
                    "confidence": "{:.2%}",
                    "macro_sentiment": "{:+.3f}",
                    "volatility_risk": "{:+.3f}",
                    "rates_pressure":  "{:+.3f}",
                    "recession_risk":  "{:+.3f}",
                }), use_container_width=True, hide_index=True)

                # NLP signal time series
                if nlp_cols:
                    feat_reset_nlp = features.reset_index()
                    spy_nlp = feat_reset_nlp[feat_reset_nlp["ticker"] == "SPY"].set_index("date").sort_index()

                    nlp_pairs = [
                        ("nlp_macro_sentiment",    "#58a6ff"),
                        ("nlp_volatility_risk",    "#f85149"),
                        ("nlp_rates_pressure",     "#d29922"),
                        ("nlp_recession_risk",     "#f0883e"),
                    ]
                    for col, color in nlp_pairs:
                        if col in spy_nlp.columns:
                            st.markdown(f"#### `{col}`")
                            s = spy_nlp[col].dropna()
                            if not s.empty:
                                st.plotly_chart(signal_time_series(s, col, color), use_container_width=True)

                # Coverage table
                st.markdown("#### NLP Signal Coverage by Period")
                coverage_df = nlp_coverage_by_period(features)
                if not coverage_df.empty:
                    st.dataframe(coverage_df.style.format("{:+.4f}"), use_container_width=True)

        st.divider()
        st.markdown("#### V3 Design Principles")
        st.markdown("""
| Principle | Detail |
|-----------|--------|
| Source control | Only files in `data/text_sources/` are read |
| No browsing | LLM never accesses the internet |
| No memory | LLM reads only the provided document text |
| No decisions | LLM/NLP output is numerical features only |
| Schema validation | All output validated against `nlp_schema.py` |
| Safe fallback | Invalid LLM output → heuristic extraction → V2 proxy → neutral 0.0 |
| No leakage | Source dated D only affects market dates ≥ D |
| Decay | Signal forward-filled with exponential decay over {decay}d |
        """.format(decay=config.NLP_SIGNAL_DECAY_DAYS))

    # ---- Feature Analysis --------------------------------------------------
    with tab_features:
        st.markdown("### 🔬 Feature Importance")
        st.markdown(
            "🔵 Blue = financial features  &nbsp;·&nbsp; "
            "🟠 Orange = V2 price-proxy ai_* signals  &nbsp;·&nbsp; "
            "🟣 Purple = V3 nlp_* signals"
        )

        imp_df = feature_importance_df(clf, feature_cols)
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            st.plotly_chart(feature_importance_chart(imp_df, ai_cols, nlp_cols), use_container_width=True)
        with col_table:
            st.dataframe(imp_df.style.format({"importance": "{:.4f}"}), use_container_width=True, height=440)

        ai_imp  = imp_df[imp_df["feature"].isin(ai_cols)]
        nlp_imp = imp_df[imp_df["feature"].isin(nlp_cols)]
        c1, c2 = st.columns(2)
        with c1:
            if not ai_imp.empty:
                total_ai = ai_imp["importance"].sum()
                st.metric("ai_* (V2 proxy) total importance", f"{total_ai:.1%}")
        with c2:
            if not nlp_imp.empty:
                total_nlp = nlp_imp["importance"].sum()
                st.metric("nlp_* (V3 NLP) total importance",  f"{total_nlp:.1%}")
            elif nlp_cols:
                st.metric("nlp_* (V3 NLP) total importance", "< 0.1%")

        st.divider()
        st.markdown("#### Permutation Importance by Period")
        st.caption(
            "ROC-AUC drop when each feature is shuffled — using the already-trained model, no retraining.  \n"
            "🔵 Financial · 🟠 ai_* (V2 proxy) · 🟣 nlp_* (V3 NLP)"
        )
        perm_tabs = st.tabs(["Train", "Validation", "Test"])
        for tab, period_label in zip(perm_tabs, ["Train", "Validation", "Test"]):
            with tab:
                perm_df = perm_imp.get(period_label, pd.DataFrame())
                if perm_df.empty:
                    st.caption("No data.")
                else:
                    pc, pt = st.columns([2, 1])
                    with pc:
                        st.plotly_chart(
                            permutation_importance_chart(perm_df, ai_cols, f"Perm. Importance — {period_label}", nlp_cols),
                            use_container_width=True)
                    with pt:
                        st.dataframe(perm_df.style.format({
                            "mean_importance": "{:.4f}", "std_importance": "{:.4f}"}),
                            use_container_width=True, height=380)

        st.divider()
        st.markdown("#### Signal Layer Roadmap")
        st.markdown("""
| Version | Layer | Status |
|---------|-------|--------|
| V1 | Neutral mock | ✅ Done |
| V2 | Market price-proxy signals | ✅ Done |
| V2.1 | Research diagnostics | ✅ Done |
| V3 | Curated-source NLP/LLM signals | ✅ Current |
| V4 | Broader sources, streaming ingestion | 🔲 Planned |
        """)

    # ---- Governance --------------------------------------------------------
    with tab_gov:
        st.markdown("### Model Governance Report")
        record = create_record()
        st.markdown(record.to_markdown())
        st.divider()
        with st.expander("Raw governance JSON"):
            st.code(record.to_json(), language="json")
        st.divider()
        st.markdown("### Full Disclaimer")
        st.markdown(DISCLAIMER)


if __name__ == "__main__":
    main()
