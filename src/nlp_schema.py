"""
nlp_schema.py
-------------
Strict schema for NLP/LLM extracted signals.

Every record produced by nlp_extractor.py must validate against this
schema before entering the feature pipeline.

Fields
------
date              : str "YYYY-MM-DD" — date the source document relates to
source            : str — filename or source label
title             : str — document title
ticker            : str — specific ticker (or "ALL" for cross-sectional)
topic             : str — e.g. "macro", "monetary_policy", "sector"
macro_sentiment   : float [-1, +1]  — overall market tone (positive = risk-on)
sector_sentiment  : float [-1, +1]  — sector-specific tone
volatility_risk   : float [-1, +1]  — market stress (positive = more stress)
inflation_pressure: float [-1, +1]  — price pressure (positive = higher inflation)
rates_pressure    : float [-1, +1]  — rate trajectory (positive = tighter/higher)
recession_risk    : float [-1, +1]  — recession probability (positive = higher risk)
confidence        : float [0, 1]    — extraction confidence
explanation       : str — short rationale for the scores
extraction_method : str "heuristic" | "llm"

Validation rules
----------------
- All numeric fields are clamped to their allowed ranges.
- Missing or NaN values are replaced with neutral defaults (0.0 for signals, 0.5 for confidence).
- extraction_method must be "heuristic" or "llm"; defaults to "heuristic" otherwise.
- Returns are always plain Python dicts (JSON-serialisable).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Neutral defaults
# ---------------------------------------------------------------------------
NEUTRAL_SIGNAL    = 0.0
NEUTRAL_CONFIDENCE = 0.5
DEFAULT_METHOD    = "heuristic"


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    """Clamp a value to [lo, hi]; return default if value is non-numeric or NaN."""
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Schema dataclass
# ---------------------------------------------------------------------------

@dataclass
class NLPSignalRecord:
    date:               str   = ""
    source:             str   = ""
    title:              str   = ""
    ticker:             str   = "ALL"
    topic:              str   = "macro"
    macro_sentiment:    float = NEUTRAL_SIGNAL
    sector_sentiment:   float = NEUTRAL_SIGNAL
    volatility_risk:    float = NEUTRAL_SIGNAL
    inflation_pressure: float = NEUTRAL_SIGNAL
    rates_pressure:     float = NEUTRAL_SIGNAL
    recession_risk:     float = NEUTRAL_SIGNAL
    confidence:         float = NEUTRAL_CONFIDENCE
    explanation:        str   = ""
    extraction_method:  str   = DEFAULT_METHOD

    def validate(self) -> "NLPSignalRecord":
        """Return a validated copy with all values clamped and normalised."""
        return NLPSignalRecord(
            date               = str(self.date) if self.date else "",
            source             = str(self.source),
            title              = str(self.title),
            ticker             = str(self.ticker) if self.ticker else "ALL",
            topic              = str(self.topic) if self.topic else "macro",
            macro_sentiment    = _clamp(self.macro_sentiment,    -1.0, 1.0, NEUTRAL_SIGNAL),
            sector_sentiment   = _clamp(self.sector_sentiment,   -1.0, 1.0, NEUTRAL_SIGNAL),
            volatility_risk    = _clamp(self.volatility_risk,    -1.0, 1.0, NEUTRAL_SIGNAL),
            inflation_pressure = _clamp(self.inflation_pressure, -1.0, 1.0, NEUTRAL_SIGNAL),
            rates_pressure     = _clamp(self.rates_pressure,     -1.0, 1.0, NEUTRAL_SIGNAL),
            recession_risk     = _clamp(self.recession_risk,     -1.0, 1.0, NEUTRAL_SIGNAL),
            confidence         = _clamp(self.confidence,          0.0, 1.0, NEUTRAL_CONFIDENCE),
            explanation        = str(self.explanation)[:500],  # cap length
            extraction_method  = (
                self.extraction_method
                if self.extraction_method in ("heuristic", "llm")
                else DEFAULT_METHOD
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation of raw dicts (from JSON, LLM output, etc.)
# ---------------------------------------------------------------------------

NUMERIC_FIELDS_SIGNAL = [
    "macro_sentiment", "sector_sentiment", "volatility_risk",
    "inflation_pressure", "rates_pressure", "recession_risk",
]
NUMERIC_BOUNDS = {f: (-1.0, 1.0) for f in NUMERIC_FIELDS_SIGNAL}
NUMERIC_BOUNDS["confidence"] = (0.0, 1.0)


def validate_record(raw: dict) -> NLPSignalRecord:
    """
    Parse and validate a raw dict into a NLPSignalRecord.
    Missing keys receive neutral defaults. Invalid values are clamped.
    Never raises — always returns a safe record.
    """
    try:
        rec = NLPSignalRecord(
            date               = raw.get("date", ""),
            source             = raw.get("source", ""),
            title              = raw.get("title", ""),
            ticker             = raw.get("ticker", "ALL"),
            topic              = raw.get("topic", "macro"),
            macro_sentiment    = raw.get("macro_sentiment",    NEUTRAL_SIGNAL),
            sector_sentiment   = raw.get("sector_sentiment",   NEUTRAL_SIGNAL),
            volatility_risk    = raw.get("volatility_risk",    NEUTRAL_SIGNAL),
            inflation_pressure = raw.get("inflation_pressure", NEUTRAL_SIGNAL),
            rates_pressure     = raw.get("rates_pressure",     NEUTRAL_SIGNAL),
            recession_risk     = raw.get("recession_risk",     NEUTRAL_SIGNAL),
            confidence         = raw.get("confidence",         NEUTRAL_CONFIDENCE),
            explanation        = raw.get("explanation",        ""),
            extraction_method  = raw.get("extraction_method",  DEFAULT_METHOD),
        )
        return rec.validate()
    except Exception:
        # Ultimate fallback — neutral record
        return NLPSignalRecord().validate()


# NLP signal column names exposed to the feature pipeline
NLP_SIGNAL_COLUMNS = [
    "nlp_macro_sentiment",
    "nlp_sector_sentiment",
    "nlp_volatility_risk",
    "nlp_inflation_pressure",
    "nlp_rates_pressure",
    "nlp_recession_risk",
    "nlp_confidence",
]


def get_nlp_signal_columns() -> list[str]:
    return list(NLP_SIGNAL_COLUMNS)
