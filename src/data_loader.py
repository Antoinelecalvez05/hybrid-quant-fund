"""
data_loader.py
--------------
Downloads and caches historical adjusted-close price data for the ETF
universe using yfinance.

Design decisions:
  - Data is downloaded once and saved to data/prices.csv.
  - On subsequent runs the cache is used unless force_download=True.
  - Only adjusted-close prices are kept (splits and dividends corrected).
  - Rows with all NaN are dropped; remaining NaN are forward-filled then
    back-filled so every ticker starts on the same grid.
"""

import os
import logging

import pandas as pd
import yfinance as yf

from config import TICKERS, TRAIN_START, TEST_END, DATA_DIR, PRICES_FILE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_prices(
    tickers: list[str] = TICKERS,
    start: str = TRAIN_START,
    end: str = TEST_END,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily adjusted-close prices.

    Columns : ticker symbols (str)
    Index   : pd.DatetimeIndex (business days)

    Parameters
    ----------
    tickers        : list of Yahoo Finance ticker symbols
    start / end    : date strings "YYYY-MM-DD" (inclusive range)
    force_download : if True, ignore the local cache and re-download
    """
    cache_path = os.path.join(DATA_DIR, PRICES_FILE)

    if not force_download and os.path.exists(cache_path):
        logger.info("Loading prices from cache: %s", cache_path)
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return prices

    logger.info("Downloading price data from Yahoo Finance …")
    os.makedirs(DATA_DIR, exist_ok=True)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusted prices (splits + dividends)
        progress=False,
    )

    # yfinance returns a MultiIndex when >1 ticker is requested.
    # We only need the "Close" level.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker: yfinance returns flat columns
        prices = raw[["Close"]]
        prices.columns = tickers

    # Clean up
    prices = prices.dropna(how="all")
    prices = prices.ffill().bfill()

    # Ensure column order is deterministic
    prices = prices[tickers]

    prices.to_csv(cache_path)
    logger.info("Prices saved to %s  (%d rows × %d cols)", cache_path, *prices.shape)

    return prices


def split_prices(
    prices: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Slice the price DataFrame into three non-overlapping periods.

    Returns (train_prices, val_prices, test_prices).
    """
    train = prices.loc[train_start:train_end]
    val   = prices.loc[val_start:val_end]
    test  = prices.loc[test_start:test_end]
    return train, val, test
