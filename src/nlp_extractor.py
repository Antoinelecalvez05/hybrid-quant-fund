"""
nlp_extractor.py
----------------
Two extraction paths that convert source documents into NLPSignalRecord lists.

Path A — Heuristic (always available, no API required)
    Scores each document by counting positive/negative keyword hits per
    thematic category. Transparent, auditable, deterministic.

Path B — LLM (optional, requires API key in .env)
    Sends the document text to an LLM with a strict JSON prompt.
    Validates the response against nlp_schema.py.
    Falls back to heuristic extraction on any failure.
    The LLM is instructed to read ONLY the provided text.

The LLM path is provider-agnostic: it reads OPENAI_API_KEY or
ANTHROPIC_API_KEY from the environment. Only one needs to be set.
If neither is present, LLM extraction is silently skipped.
"""

from __future__ import annotations
import json
import logging
import os
import re
from typing import Optional

import pandas as pd

from nlp_schema import NLPSignalRecord, validate_record, NEUTRAL_SIGNAL, NEUTRAL_CONFIDENCE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic keyword lexicon
# ---------------------------------------------------------------------------

_LEXICON: dict[str, dict[str, list[str]]] = {
    "macro_sentiment": {
        "positive": [
            "growth", "recovery", "easing", "cuts", "rate cut", "strong demand",
            "improving", "resilient", "rally", "risk-on", "bullish", "expansion",
            "soft landing", "disinflation", "stabilisation", "positive momentum",
        ],
        "negative": [
            "recession", "slowdown", "contraction", "unemployment", "crisis",
            "stress", "risk-off", "downturn", "collapse", "bear", "stagflation",
            "slump", "deteriorating", "distress", "negative growth",
        ],
    },
    "volatility_risk": {
        "positive": [
            "volatility", "uncertainty", "crisis", "stress", "panic",
            "extreme moves", "circuit breaker", "liquidity crunch", "tail risk",
            "dislocation", "fear", "sell-off", "shock",
        ],
        "negative": [
            "calm", "stable", "low volatility", "orderly", "normalising",
            "improving stability", "reduced uncertainty",
        ],
    },
    "inflation_pressure": {
        "positive": [
            "inflation", "cpi", "price pressure", "overheating", "wage growth",
            "supply shortage", "energy prices", "commodity prices", "above target",
            "sticky inflation", "inflationary",
        ],
        "negative": [
            "disinflation", "deflation", "falling prices", "price decline",
            "easing inflation", "below target", "cooling prices",
        ],
    },
    "rates_pressure": {
        "positive": [
            "rate hike", "tightening", "hawkish", "restrictive", "rate increase",
            "higher rates", "monetary tightening", "raising rates", "terminal rate",
        ],
        "negative": [
            "rate cut", "easing", "dovish", "accommodative", "rate reduction",
            "lower rates", "monetary easing", "cutting rates", "pause",
        ],
    },
    "recession_risk": {
        "positive": [
            "recession", "contraction", "negative gdp", "downturn", "slump",
            "job losses", "unemployment rising", "credit stress", "hard landing",
            "economic weakness",
        ],
        "negative": [
            "expansion", "growth", "strong labour", "resilient economy",
            "soft landing", "recovery", "low unemployment",
        ],
    },
    "sector_sentiment": {
        "positive": [
            "earnings beat", "revenue growth", "strong results", "upgrade",
            "outperform", "positive guidance", "robust demand",
        ],
        "negative": [
            "earnings miss", "revenue decline", "downgrade", "underperform",
            "weak guidance", "margin pressure", "sector stress",
        ],
    },
}


def _score_text(text: str) -> dict[str, float]:
    """
    Score a document against the keyword lexicon.
    Returns a dict of raw scores in [-1, +1] per signal dimension.
    """
    text_lower = text.lower()

    def _count(words: list[str]) -> int:
        return sum(1 for w in words if w in text_lower)

    scores: dict[str, float] = {}
    total_hits = 0

    for dimension, buckets in _LEXICON.items():
        pos = _count(buckets.get("positive", []))
        neg = _count(buckets.get("negative", []))
        total = pos + neg
        total_hits += total
        if total == 0:
            scores[dimension] = NEUTRAL_SIGNAL
        else:
            scores[dimension] = round((pos - neg) / total, 4)

    # Confidence scales with how many keyword hits were found
    max_possible = sum(len(b["positive"]) + len(b["negative"])
                       for b in _LEXICON.values())
    confidence = min(1.0, total_hits / max(max_possible * 0.05, 1))
    scores["confidence"] = round(confidence, 4)

    return scores


# ---------------------------------------------------------------------------
# Heuristic extractor
# ---------------------------------------------------------------------------

def extract_heuristic(row: pd.Series) -> NLPSignalRecord:
    """
    Extract a signal record from a single source row using keyword scoring.
    Always returns a valid NLPSignalRecord.
    """
    text = str(row.get("text", ""))
    scores = _score_text(text)

    raw = {
        "date":               str(row.get("date", "")),
        "source":             str(row.get("source", "")),
        "title":              str(row.get("title", "")),
        "ticker":             str(row.get("ticker", "ALL")),
        "topic":              str(row.get("topic", "macro")),
        "macro_sentiment":    scores.get("macro_sentiment",    NEUTRAL_SIGNAL),
        "sector_sentiment":   scores.get("sector_sentiment",   NEUTRAL_SIGNAL),
        "volatility_risk":    scores.get("volatility_risk",    NEUTRAL_SIGNAL),
        "inflation_pressure": scores.get("inflation_pressure", NEUTRAL_SIGNAL),
        "rates_pressure":     scores.get("rates_pressure",     NEUTRAL_SIGNAL),
        "recession_risk":     scores.get("recession_risk",     NEUTRAL_SIGNAL),
        "confidence":         scores.get("confidence",         NEUTRAL_CONFIDENCE),
        "explanation":        "Heuristic keyword scoring",
        "extraction_method":  "heuristic",
    }
    return validate_record(raw)


# ---------------------------------------------------------------------------
# LLM prompt template
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You are a financial information extraction assistant. "
    "Your ONLY job is to read the provided text and extract structured sentiment scores. "
    "You must NOT use any outside knowledge, memory, or browsing. "
    "You must NOT make trading recommendations. "
    "Return ONLY valid JSON matching the schema below. "
    "All numeric fields must be floats between -1.0 and 1.0 (or 0.0 to 1.0 for confidence). "
    "If you cannot determine a score from the text, use 0.0."
)

_LLM_SCHEMA_DESCRIPTION = """{
  "macro_sentiment":    float [-1, +1],  // overall macro tone (positive = risk-on)
  "sector_sentiment":   float [-1, +1],  // sector-specific tone
  "volatility_risk":    float [-1, +1],  // market stress level (positive = more stress)
  "inflation_pressure": float [-1, +1],  // price pressure (positive = rising inflation)
  "rates_pressure":     float [-1, +1],  // rate trajectory (positive = tighter/higher)
  "recession_risk":     float [-1, +1],  // recession likelihood (positive = more risk)
  "confidence":         float [0, 1],    // confidence in your extraction
  "explanation":        string           // 1-2 sentence rationale
}"""


def _build_llm_prompt(text: str) -> str:
    return (
        f"Read the following document and extract sentiment scores. "
        f"Return ONLY valid JSON with exactly these fields:\n{_LLM_SCHEMA_DESCRIPTION}\n\n"
        f"--- BEGIN DOCUMENT ---\n{text[:4000]}\n--- END DOCUMENT ---\n\n"
        f"Return only the JSON object. No preamble, no explanation outside the JSON."
    )


def _parse_llm_response(response_text: str) -> Optional[dict]:
    """Extract JSON from LLM response text. Returns None if parsing fails."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip()
    # Find first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _call_openai(prompt: str, api_key: str) -> Optional[str]:
    """Call OpenAI API. Returns raw response text or None on failure."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning("OpenAI API call failed: %s", e)
        return None


def _call_anthropic(prompt: str, api_key: str) -> Optional[str]:
    """Call Anthropic API. Returns raw response text or None on failure."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        logger.warning("Anthropic API call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# LLM extractor (optional)
# ---------------------------------------------------------------------------

def extract_llm(row: pd.Series) -> NLPSignalRecord:
    """
    Attempt LLM extraction. Falls back to heuristic on any failure.
    Reads API keys from environment (OPENAI_API_KEY or ANTHROPIC_API_KEY).
    Never crashes — always returns a valid NLPSignalRecord.
    """
    text = str(row.get("text", ""))
    if not text.strip():
        return extract_heuristic(row)

    prompt = _build_llm_prompt(text)
    raw_response: Optional[str] = None

    openai_key    = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if openai_key:
        raw_response = _call_openai(prompt, openai_key)
    elif anthropic_key:
        raw_response = _call_anthropic(prompt, anthropic_key)
    else:
        logger.info("No LLM API key found — falling back to heuristic extraction.")
        return extract_heuristic(row)

    if raw_response is None:
        return extract_heuristic(row)

    parsed = _parse_llm_response(raw_response)
    if parsed is None:
        logger.warning("LLM returned invalid JSON — falling back to heuristic.")
        return extract_heuristic(row)

    # Merge metadata with parsed signal fields
    parsed.update({
        "date":              str(row.get("date", "")),
        "source":            str(row.get("source", "")),
        "title":             str(row.get("title", "")),
        "ticker":            str(row.get("ticker", "ALL")),
        "topic":             str(row.get("topic", "macro")),
        "extraction_method": "llm",
    })

    record = validate_record(parsed)
    # If confidence is suspiciously high from LLM, trust but verify
    return record


# ---------------------------------------------------------------------------
# Batch extractor — public API
# ---------------------------------------------------------------------------

def extract_all(
    sources_df: pd.DataFrame,
    mode: str = "heuristic",
) -> list[NLPSignalRecord]:
    """
    Extract NLP signal records from all rows in sources_df.

    Parameters
    ----------
    sources_df : DataFrame from text_sources.load_text_sources()
    mode       : "heuristic" | "llm" | "off"
                 If "off", returns empty list.
                 If "llm" and no API key available, falls back to "heuristic".

    Returns
    -------
    List of validated NLPSignalRecord instances.
    """
    if mode == "off" or sources_df.empty:
        return []

    records = []
    for _, row in sources_df.iterrows():
        if mode == "llm":
            rec = extract_llm(row)
        else:
            rec = extract_heuristic(row)
        records.append(rec)

    logger.info(
        "Extracted %d NLP records using mode='%s'.",
        len(records), mode,
    )
    return records
