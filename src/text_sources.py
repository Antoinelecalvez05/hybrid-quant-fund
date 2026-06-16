"""
text_sources.py
---------------
Load curated local text sources from data/text_sources/.

Supported formats: .txt, .md, .csv

Returned DataFrame columns (always present, missing fields = empty string):
    date    : str "YYYY-MM-DD" (inferred from filename if not in document)
    source  : str filename
    title   : str
    text    : str full document text
    ticker  : str ("ALL" if not specified)
    topic   : str ("macro" if not specified)

No web scraping. No API calls. Everything local and auditable.

Date inference from filename
----------------------------
If the filename starts with a date pattern YYYY-MM-DD the loader uses that
as the document date. Otherwise the date field is left empty.

CSV format
----------
Required: at least a "text" column.
Optional: date, source, title, ticker, topic.
Missing columns are filled with defaults.
Each row in the CSV becomes one record.
"""

from __future__ import annotations
import os
import re
import logging
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)

_DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})")

_REQUIRED_COLS  = ["date", "source", "title", "text", "ticker", "topic"]
_COL_DEFAULTS   = {
    "date":   "",
    "source": "",
    "title":  "",
    "text":   "",
    "ticker": "ALL",
    "topic":  "macro",
}


def _infer_date_from_filename(filename: str) -> str:
    """Extract YYYY-MM-DD from the start of a filename, or return ''."""
    base = os.path.basename(filename)
    m = _DATE_PATTERN.match(base)
    return m.group(1) if m else ""


def _normalise(df: pd.DataFrame, source_filename: str) -> pd.DataFrame:
    """Ensure all required columns exist, fill missing with defaults."""
    for col, default in _COL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    # Fill empty source with filename
    df["source"] = df["source"].fillna("").replace("", os.path.basename(source_filename))
    # Fill empty ticker and topic
    df["ticker"] = df["ticker"].fillna("ALL").replace("", "ALL")
    df["topic"]  = df["topic"].fillna("macro").replace("", "macro")
    return df[_REQUIRED_COLS].copy()


def _load_txt(filepath: str) -> pd.DataFrame:
    """Load a .txt or .md file as a single record."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        logger.warning("Could not read %s: %s", filepath, e)
        return pd.DataFrame()

    date  = _infer_date_from_filename(filepath)
    title = os.path.basename(filepath).replace("_", " ").rsplit(".", 1)[0]
    row   = {
        "date":   date,
        "source": os.path.basename(filepath),
        "title":  title,
        "text":   text.strip(),
        "ticker": "ALL",
        "topic":  "macro",
    }
    return pd.DataFrame([row])


def _load_csv(filepath: str) -> pd.DataFrame:
    """Load a .csv file; each row becomes one NLP record."""
    try:
        df = pd.read_csv(filepath, dtype=str).fillna("")
    except Exception as e:
        logger.warning("Could not read CSV %s: %s", filepath, e)
        return pd.DataFrame()

    if "text" not in df.columns:
        logger.warning("CSV %s has no 'text' column — skipping.", filepath)
        return pd.DataFrame()

    # If date column missing, infer from filename
    if "date" not in df.columns or df["date"].eq("").all():
        df["date"] = _infer_date_from_filename(filepath)

    return _normalise(df, filepath)


def load_text_sources(source_dir: str) -> pd.DataFrame:
    """
    Load all supported files from source_dir.

    Parameters
    ----------
    source_dir : path to the folder containing curated source documents.

    Returns
    -------
    DataFrame with columns: date, source, title, text, ticker, topic.
    Empty DataFrame if directory is missing or no supported files found.
    """
    if not os.path.isdir(source_dir):
        logger.info("NLP source directory not found: %s — returning empty sources.", source_dir)
        return pd.DataFrame(columns=_REQUIRED_COLS)

    records = []
    for fname in sorted(os.listdir(source_dir)):
        fpath = os.path.join(source_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = fname.lower().rsplit(".", 1)[-1]
        if ext in ("txt", "md"):
            df = _load_txt(fpath)
        elif ext == "csv":
            df = _load_csv(fpath)
        else:
            continue
        if not df.empty:
            records.append(df)

    if not records:
        logger.info("No supported source files found in %s.", source_dir)
        return pd.DataFrame(columns=_REQUIRED_COLS)

    result = pd.concat(records, ignore_index=True)
    result = _normalise(result, source_dir)  # final normalisation pass
    # Sort by date
    result = result.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d text source records from %s.", len(result), source_dir)
    return result
