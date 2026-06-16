"""
risk.py
-------
Performance and risk analytics for backtest output.

All metrics follow standard quantitative finance conventions:
  - Returns are assumed to be *daily* simple returns (not log-returns).
  - Annualisation uses TRADING_DAYS (252 by default).
  - Risk-free rate is annual (from config.py).

V1 metrics:
  total_return, annualised_return, annualised_volatility,
  sharpe_ratio, max_drawdown, calmar_ratio

V2 additions:
  downside_volatility  — volatility of negative returns only (semi-deviation)
  sortino_ratio        — excess return / downside volatility
  win_rate             — fraction of days with positive return
  best_day             — single best daily return
  worst_day            — single worst daily return
"""

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE, TRADING_DAYS


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def total_return(returns: pd.Series) -> float:
    """Cumulative return over the full period."""
    return float((1 + returns).prod() - 1)


def annualised_return(returns: pd.Series) -> float:
    """CAGR: compound annual growth rate."""
    n_days = len(returns)
    if n_days == 0:
        return 0.0
    n_years = n_days / TRADING_DAYS
    cum = (1 + returns).prod()
    if cum <= 0 or n_years <= 0:
        return 0.0
    return float(cum ** (1 / n_years) - 1)


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series) -> float:
    """
    Annualised Sharpe ratio.
    Sharpe = (annualised_return − risk_free) / annualised_volatility
    """
    ann_vol = annualised_volatility(returns)
    if ann_vol == 0:
        return 0.0
    return float((annualised_return(returns) - RISK_FREE_RATE) / ann_vol)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (always ≤ 0)."""
    if len(returns) == 0:
        return 0.0
    cum = (1 + returns).cumprod()
    rolling_peak = cum.cummax()
    return float((cum / rolling_peak - 1).min())


def calmar_ratio(returns: pd.Series) -> float:
    """Annualised return divided by the absolute maximum drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(annualised_return(returns) / abs(mdd))


# ---------------------------------------------------------------------------
# V2 additions
# ---------------------------------------------------------------------------

def downside_volatility(returns: pd.Series) -> float:
    """
    Annualised downside deviation: std of *negative* daily returns × √252.
    Used in the Sortino ratio denominator.
    """
    negative = returns[returns < 0]
    if len(negative) < 2:
        return 0.0
    return float(negative.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series) -> float:
    """
    Annualised Sortino ratio.
    Sortino = (annualised_return − risk_free) / downside_volatility
    """
    down_vol = downside_volatility(returns)
    if down_vol == 0:
        return 0.0
    return float((annualised_return(returns) - RISK_FREE_RATE) / down_vol)


def win_rate(returns: pd.Series) -> float:
    """Fraction of trading days with a positive return."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def best_day(returns: pd.Series) -> float:
    """Single best daily return."""
    return float(returns.max()) if len(returns) else 0.0


def worst_day(returns: pd.Series) -> float:
    """Single worst daily return."""
    return float(returns.min()) if len(returns) else 0.0


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series, label: str = "") -> dict:
    """
    Compute the full set of V2 performance metrics for a return series.

    Returns
    -------
    dict with keys:
      label, total_return, annualised_return, annualised_volatility,
      sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
      downside_volatility, win_rate, best_day, worst_day
    """
    return {
        "label":                 label,
        "total_return":          total_return(returns),
        "annualised_return":     annualised_return(returns),
        "annualised_volatility": annualised_volatility(returns),
        "sharpe_ratio":          sharpe_ratio(returns),
        "sortino_ratio":         sortino_ratio(returns),
        "max_drawdown":          max_drawdown(returns),
        "calmar_ratio":          calmar_ratio(returns),
        "downside_volatility":   downside_volatility(returns),
        "win_rate":              win_rate(returns),
        "best_day":              best_day(returns),
        "worst_day":             worst_day(returns),
    }


def metrics_to_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Convert a list of metrics dicts into a display-ready DataFrame.
    Numbers formatted as % or plain decimals as appropriate.
    """
    df = pd.DataFrame(metrics_list).set_index("label")

    pct_cols = [
        "total_return", "annualised_return", "annualised_volatility",
        "max_drawdown", "downside_volatility", "win_rate", "best_day", "worst_day",
    ]
    ratio_cols = ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]

    display = pd.DataFrame(index=df.index)
    for col in pct_cols:
        if col in df.columns:
            display[col] = df[col].map(lambda x: f"{x:.2%}")
    for col in ratio_cols:
        if col in df.columns:
            display[col] = df[col].map(lambda x: f"{x:.2f}")

    display.columns = [c.replace("_", " ").title() for c in display.columns]
    return display
