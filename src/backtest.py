"""
backtest.py
-----------
Long-only, equal-weight backtest engine with transaction costs.

Rules:
  - Each day, go long every ticker whose model signal ≥ SIGNAL_THRESHOLD.
  - If no ticker passes the threshold, hold cash (return = 0 for that day).
  - All positions within a day receive equal weight (1 / n_long).
  - Positions are based exclusively on the *previous* day's signal —
    yesterday's signal → today's weight → applied to today's return.
  - No short selling, no leverage.

V2 additions:
  - Transaction costs deducted from daily returns.
  - Turnover computed as sum of absolute weight changes each day.
  - Cost = turnover × TRANSACTION_COST_BPS / 10_000.
  - Returns both gross and net (after-cost) strategy returns.

The benchmark is a daily-rebalanced equal-weight portfolio across all
tickers.  No transaction costs are applied to the benchmark (it acts as
a frictionless reference).

Output keys:
  gross_strategy_returns  : daily returns before costs
  strategy_returns        : daily returns after costs (use this for metrics)
  benchmark_returns       : equal-weight benchmark daily returns
  turnover                : daily turnover Series
  average_turnover        : scalar mean daily turnover
  positions               : DataFrame (date × ticker, weights, lagged by 1d)
  equity_curve            : compounded strategy (after costs)
  benchmark_curve         : compounded benchmark
"""

import numpy as np
import pandas as pd

from config import SIGNAL_THRESHOLD, TICKERS, TRANSACTION_COST_BPS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    tickers: list[str] = TICKERS,
    signal_threshold: float = SIGNAL_THRESHOLD,
    transaction_cost_bps: int = TRANSACTION_COST_BPS,
) -> dict:
    """
    Execute the backtest and return daily return series plus metadata.

    Parameters
    ----------
    prices               : price DataFrame (DatetimeIndex × ticker columns)
    signals              : MultiIndex Series (date, ticker) → float [0,1]
    tickers              : list of ticker symbols
    signal_threshold     : minimum signal to take a long position
    transaction_cost_bps : one-way cost per unit of turnover in basis points

    Returns
    -------
    dict — see module docstring for keys.
    """
    # --- Daily simple returns ----------------------------------------------
    daily_ret = prices[tickers].pct_change()

    # --- Unstack signals → date × ticker -----------------------------------
    signal_wide = signals.unstack(level="ticker").reindex(columns=tickers)

    # --- Binary long flag ---------------------------------------------------
    long_flag = (signal_wide >= signal_threshold).astype(float)

    # --- Equal-weight within long positions --------------------------------
    n_long  = long_flag.sum(axis=1).replace(0, np.nan)
    weights = long_flag.div(n_long, axis=0).fillna(0.0)

    # --- Lag weights by 1 day (no look-ahead) ------------------------------
    weights_lagged = weights.shift(1).fillna(0.0)

    # --- Align dates -------------------------------------------------------
    common_dates = daily_ret.index.intersection(weights_lagged.index)
    ret_aligned  = daily_ret.loc[common_dates]
    wt_aligned   = weights_lagged.loc[common_dates]

    # --- Gross strategy return ---------------------------------------------
    gross_strat_ret = (ret_aligned * wt_aligned).sum(axis=1)

    # --- Turnover (V2) -----------------------------------------------------
    # Daily turnover = sum of absolute changes in weights.
    # On day 0, all weights are 0 → first turnover = sum of initial weights.
    weight_change = wt_aligned.diff().abs()
    turnover = weight_change.sum(axis=1)
    turnover.iloc[0] = wt_aligned.iloc[0].sum()   # entry cost on first day

    # --- Transaction cost deduction ----------------------------------------
    cost_per_day = turnover * (transaction_cost_bps / 10_000)
    net_strat_ret = gross_strat_ret - cost_per_day

    # --- Benchmark ---------------------------------------------------------
    bench_ret = ret_aligned.mean(axis=1)

    # --- Equity curves (start at $1) --------------------------------------
    equity_curve    = (1 + net_strat_ret).cumprod()
    benchmark_curve = (1 + bench_ret).cumprod()

    return {
        "gross_strategy_returns": gross_strat_ret,
        "strategy_returns":       net_strat_ret,        # after costs — primary metric
        "benchmark_returns":      bench_ret,
        "turnover":               turnover,
        "average_turnover":       float(turnover.mean()),
        "positions":              wt_aligned,
        "equity_curve":           equity_curve,
        "benchmark_curve":        benchmark_curve,
    }
