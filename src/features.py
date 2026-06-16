"""
features.py
-----------
Builds the feature matrix used to train and run the Random Forest model.

Features computed per ticker per day:
  1. Rolling returns   — log-returns over several momentum windows
  2. Volatility        — rolling std of daily log-returns
  3. MA ratio          — short MA / long MA
  4. Drawdown          — current price / rolling max
  5. AI contextual     — appended later by ai_signal.py

Label:
  Binary: 1 if the ticker's next-day log-return > 0, else 0.

Important V2 fix:
  The final day has no known next-day return, so its target is set to NaN,
  not fake 0.
"""

import numpy as np
import pandas as pd

from config import (
    MOMENTUM_WINDOWS,
    VOLATILITY_WINDOW,
    MA_SHORT,
    MA_LONG,
    DRAWDOWN_WINDOW,
)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a stacked feature DataFrame from a prices DataFrame.

    Parameters
    ----------
    prices : DataFrame
        DatetimeIndex as rows, ticker symbols as columns.

    Returns
    -------
    combined : DataFrame
        MultiIndex: (date, ticker)
        Columns: financial features + target.
    """

    log_ret = np.log(prices / prices.shift(1))

    frames = []

    for ticker in prices.columns:
        ret = log_ret[ticker]
        price = prices[ticker]

        df = pd.DataFrame(index=prices.index)
        df["ticker"] = ticker

        # Momentum features
        for window in MOMENTUM_WINDOWS:
            df[f"mom_{window}d"] = ret.rolling(window).sum()

        # Realised volatility
        df["vol_21d"] = ret.rolling(VOLATILITY_WINDOW).std()

        # Moving-average ratio
        ma_short = price.rolling(MA_SHORT).mean()
        ma_long = price.rolling(MA_LONG).mean()
        df["ma_ratio"] = ma_short / ma_long - 1.0

        # Drawdown from rolling high
        rolling_max = price.rolling(DRAWDOWN_WINDOW).max()
        df["drawdown"] = price / rolling_max - 1.0

        # Target: next-day direction
        # Important: keep final unknown target as NaN.
        future_return = ret.shift(-1)
        df["target"] = np.where(
            future_return.notna(),
            (future_return > 0).astype(int),
            np.nan,
        )

        frames.append(df)

    combined = pd.concat(frames)
    combined.index.name = "date"

    combined = combined.reset_index().set_index(["date", "ticker"])
    combined = combined.sort_index()

    # Drop rows where feature values are missing.
    # Do not drop target here except in prepare_Xy().
    feature_cols = [c for c in combined.columns if c != "target"]
    combined = combined.dropna(subset=feature_cols)

    return combined


def get_feature_columns(features: pd.DataFrame) -> list[str]:
    """
    Return model input column names.
    Excludes only target.
    """

    return [c for c in features.columns if c != "target"]


def prepare_Xy(
    features: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the feature DataFrame into X and y.
    Drops rows where target is missing.
    """

    valid = features.dropna(subset=["target"])

    X = valid[feature_cols]
    y = valid["target"].astype(int)

    return X, y